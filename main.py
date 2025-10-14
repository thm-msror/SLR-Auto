# main.py
import time
from pathlib import Path
import json
from datetime import datetime
from src.utils import (
    load_json,
    save_json,
    clean_papers,
    clean_bullets,
    deduplicate_papers_by_link,
    save_md
)
from src.fetch_arxiv import fetch_papers as fetch_arvix
from src.fetch_crossref import fetch_papers as fetch_crossref
from src.enrich_openalex import enrich
from src.llm_screener_bullets import screen_papers
from src.highly_relevant import (
    filter_highly_relevant_papers_from_list,
    extract_bullets_for_highly_relevant_text,
)
from src.summarizer import summarize_no_llm
from src.parse_bullets_to_markdown import generate_markdown, parse_bullets_file
from src.marker_convert import convert_pdfs_in_directory, run_marker_batch
import config as config

# ---------------- Directories ----------------
SCREENED_ARTICLES_DIR = Path(config.SCREENED_PAPERS_FOLDER)
CHECKPOINT_DIR = SCREENED_ARTICLES_DIR / "checkpoints"
SCREENED_ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
Path(config.FETCHED_PAPERS_FOLDER).mkdir(parents=True, exist_ok=True)
Path(config.SUMMARY_FOLDER).mkdir(parents=True, exist_ok=True)
Path(config.PDF_PAPERS_FOLDER).mkdir(parents=True, exist_ok=True)
Path(config.MARKDOWN_PAPERS_FOLDER).mkdir(parents=True, exist_ok=True)

ALL_JSON_FILE = (
    Path(config.all_screened_papers_path)
    if getattr(config, "all_screened_papers_path", "")
    else SCREENED_ARTICLES_DIR / "all_screened_papers.json"
)
ALL_BULLETS_FILE = (
    Path(config.all_screened_bullets_path)
    if getattr(config, "all_screened_bullets_path", "")
    else SCREENED_ARTICLES_DIR / "all_screened_bullets.txt"
)
HIGH_JSON_PATH = Path(config.SCREENED_PAPERS_FOLDER) / "highly_relevant_papers.json"
HIGH_BULLETS_PATH = Path(config.SCREENED_PAPERS_FOLDER) / "highly_relevant_bullets.txt"

# ---------------- Helper ----------------
def get_latest_checkpoint(track_dir):
    checkpoints = sorted(Path(track_dir).glob("papers_remaining_*.json"))
    return checkpoints[-1] if checkpoints else None

# ---------------- Main ----------------
if __name__ == "__main__":

    start_time = time.time()

    # ---------------- FETCHING ARTICLES ----------------
    if config.all_fetched_path:
        all_fetched_papers = load_json(config.all_fetched_path)
    else:
        raw_arvix_fetch = fetch_arvix(
            config.QUERIES,
            max_results=config.MAX_QUERIES,
            track=f"{config.FETCHED_PAPERS_FOLDER}/raw_fetch/checkpoints"
        )
        raw_crossref_fetch = fetch_crossref(
            config.QUERIES,
            max_results=config.MAX_QUERIES,
            track=f"{config.FETCHED_PAPERS_FOLDER}/raw_fetch/checkpoints"
        )
        combined_fetch = raw_crossref_fetch + raw_arvix_fetch
        all_fetched_papers = enrich(
            combined_fetch,
            track=f"{config.FETCHED_PAPERS_FOLDER}/enrich/checkpoints"
        )
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        fetched_path = Path(config.FETCHED_PAPERS_FOLDER) / f"fetched_{len(all_fetched_papers)}_{ts}.json"
        save_json(all_fetched_papers, str(fetched_path))

    # ---------------- LLM SCREENING ----------------
    if ALL_JSON_FILE.exists():
        all_screened_papers = load_json(ALL_JSON_FILE)
    else:
        remaining_papers = all_fetched_papers
        all_screened_papers = screen_papers(
            remaining_papers,
            batch_size=100,
            model="Fanar",
            prompt_txt_path=config.LLM_SCREENING_PROMPT_TXT,
            save_to_files=False,
        )

    # ---------------- DEDUPLICATE BY LINK ----------------
    deduped_papers = deduplicate_papers_by_link(all_screened_papers)
    deduped_papers = clean_papers(deduped_papers, remove_duplicates_only=False)

    # ---------------- SAVE JSON (OVERWRITE) ----------------
    save_json(deduped_papers, ALL_JSON_FILE)
    print(f"\n Final JSON saved (deduplicated): {ALL_JSON_FILE}")

    # ---------------- SAVE BULLETS (OVERWRITE) ----------------
    bullet_text = clean_bullets(deduped_papers)
    with open(ALL_BULLETS_FILE, "w", encoding="utf-8") as f:
        f.write(bullet_text)
    print(f" Bullets TXT saved (deduplicated): {ALL_BULLETS_FILE}")

    # ---------------- FILTER HIGHLY RELEVANT (PURE + SAVE HERE) ----------------
    highly_relevant_papers = filter_highly_relevant_papers_from_list(deduped_papers)
    HIGH_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HIGH_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(highly_relevant_papers, f, indent=2, ensure_ascii=False)
    with open(ALL_BULLETS_FILE, "r", encoding="utf-8") as f:
        all_bullets_text = f.read()
    highly_relevant_bullets = extract_bullets_for_highly_relevant_text(
        highly_relevant_papers,
        all_bullets_text,
    )
    with open(HIGH_BULLETS_PATH, "w", encoding="utf-8") as f:
        f.write(highly_relevant_bullets)

    # ---------------- NO-LLM PAPER TABLE (HIGHLY RELEVANT) ----------------
    t0 = time.time()
    print("\n Paper Table of Highly Relevant Papers:\n")
    no_llm_summary = summarize_no_llm(HIGH_JSON_PATH)
    # Save summary table (skip if existing summary files already present and skipping is enabled)
    summary_stem = f"paper_summary_{config.MAX_QUERIES}"
    existing_summaries = list(Path(config.SUMMARY_FOLDER).glob(f"{summary_stem}_*.md"))
    if not (getattr(config, "SKIP_SUMMARIZATION_IF_EXISTS", False) and existing_summaries):
        save_md(no_llm_summary, folder=config.SUMMARY_FOLDER, filename=summary_stem)
    
    # ---------------- NO-LLM MARKDOWN (HIGHLY RELEVANT) ----------------
    
    # Parse the highly relevant bullets txt
    papers = parse_bullets_file(HIGH_BULLETS_PATH)

    # Generate Markdown summary (with datasets, methods, notable papers, and top applications)
    no_llm_summary = generate_markdown(papers)

    # Save to SUMMARY_FOLDER (skip if existing summary files already present and skipping is enabled)
    existing_summaries = list(Path(config.SUMMARY_FOLDER).glob(f"{summary_stem}_*.md"))
    if not (getattr(config, "SKIP_SUMMARIZATION_IF_EXISTS", False) and existing_summaries):
        save_md(no_llm_summary, folder=config.SUMMARY_FOLDER, filename=f"paper_summary_{config.MAX_QUERIES}.md")

    print(f" Non-LLM summary saved to {config.SUMMARY_FOLDER}/paper_summary_{config.MAX_QUERIES}.md")

    # ---------------- PDF → MARKDOWN CONVERSION (USER-PROVIDED PDFs) ----------------
    print("\n Converting PDFs in configured folder to Markdown...")
    try:
        if getattr(config, "SKIP_PDF_CONVERSION_IF_UP_TO_DATE", False):
            pdf_names = {p.stem for p in Path(config.PDF_PAPERS_FOLDER).glob("*.pdf")}
            md_names = {p.stem for p in Path(config.MARKDOWN_PAPERS_FOLDER).glob("*.md")}
            missing = sorted(list(pdf_names - md_names))
            if not missing:
                print(" All PDFs already converted. Skipping PDF→Markdown step.")
            else:
                print(f" Will convert {len(missing)} missing PDFs…")
                run_marker_batch(config.PDF_PAPERS_FOLDER, config.MARKDOWN_PAPERS_FOLDER)
        else:
            pdf_to_md_map = convert_pdfs_in_directory(config.PDF_PAPERS_FOLDER)
            for filename, md_text in pdf_to_md_map.items():
                out_path = Path(config.MARKDOWN_PAPERS_FOLDER) / (Path(filename).stem + ".md")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(md_text)
            print(f" Converted {len(pdf_to_md_map)} PDFs to Markdown in {config.MARKDOWN_PAPERS_FOLDER}")
    except ImportError as e:
        print("   PDF→Markdown step skipped due to missing deps. To enable, run:")
        print("   pip install -r requirements.txt")
        print("   or at least: pip install 'transformers>=4.45,<5' 'tokenizers>=0.22,<0.24' surya-ocr marker-pdf")

