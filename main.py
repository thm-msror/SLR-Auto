# main.py
import time
from pathlib import Path
import json
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
from src.highly_relevant import filter_highly_relevant_papers, extract_bullets_for_highly_relevant
from src.summarizer import summarize_no_llm
from src.parse_bullets_to_markdown import generate_markdown, parse_bullets_file
import config as config

# ---------------- Directories ----------------
SCREENED_ARTICLES_DIR = Path(config.SCREENED_PAPERS_FOLDER)
CHECKPOINT_DIR = SCREENED_ARTICLES_DIR / "checkpoints"
SCREENED_ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

ALL_JSON_FILE = Path(config.all_screened_papers_path)
ALL_BULLETS_FILE = Path(config.all_screened_bullets_path)
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
        save_json(
            all_fetched_papers,
            folder=config.FETCHED_PAPERS_FOLDER,
            filename=f"fetched_{len(all_fetched_papers)}_"
        )

    # ---------------- LLM SCREENING ----------------
    if ALL_JSON_FILE.exists():
        all_screened_papers = load_json(ALL_JSON_FILE)
    else:
        latest_checkpoint = get_latest_checkpoint(CHECKPOINT_DIR)
        if latest_checkpoint:
            print(f"🔄 Resuming from checkpoint: {latest_checkpoint}")
            remaining_papers = load_json(latest_checkpoint)
        else:
            remaining_papers = all_fetched_papers

        all_screened_papers = screen_papers(
            remaining_papers,
            batch_size=100,
            model="Fanar",
            prompt_txt_path=config.LLM_SCREENING_PROMPT_TXT,
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

    # ---------------- FILTER HIGHLY RELEVANT ----------------
    highly_relevant_papers = filter_highly_relevant_papers(
        ALL_JSON_FILE,
        output_json_path=HIGH_JSON_PATH
    )

    highly_relevant_bullets = extract_bullets_for_highly_relevant(
        highly_relevant_papers,
        bullets_txt_path=ALL_BULLETS_FILE,
        output_txt_path=HIGH_BULLETS_PATH
    )

    # ---------------- NO-LLM PAPER TABLE (HIGHLY RELEVANT) ----------------
    t0 = time.time()
    print("\n Paper Table of Highly Relevant Papers:\n")
    no_llm_summary = summarize_no_llm(HIGH_JSON_PATH)
    save_md(no_llm_summary, folder=config.SUMMARY_FOLDER, filename=f"paper_summary_{config.MAX_QUERIES}")
    
    # ---------------- NO-LLM MARKDOWN (HIGHLY RELEVANT) ----------------
    
    # Parse the highly relevant bullets txt
    papers = parse_bullets_file(HIGH_BULLETS_PATH)

    # Generate Markdown summary (with datasets, methods, notable papers, and top applications)
    no_llm_summary = generate_markdown(papers)

    # Save to SUMMARY_FOLDER
    save_md(no_llm_summary, folder=config.SUMMARY_FOLDER, filename=f"paper_summary_{config.MAX_QUERIES}.md")

    print(f"✅ Non-LLM summary saved to {config.SUMMARY_FOLDER}/paper_summary_{config.MAX_QUERIES}.md")

