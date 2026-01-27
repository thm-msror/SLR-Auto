# main.py
import time
from pathlib import Path
import json
from datetime import datetime
from src.utils import *
from src.fetch_arxiv import fetch_papers as fetch_arvix
from src.fetch_crossref import fetch_papers as fetch_crossref
from src.enrich_openalex import enrich
from src.llm_screener_bullets import screen_papers
from src.filter_papers import filter_top_papers, paper_table
from src.marker_convert import run_marker_batch
from src.llm_reader import read_paper_mds, table_summary
from src.paper_reviews import generate_gap_reviews, gap_df
from src.pdf_downloader import download_pdfs   # 🔹 NEW
import config as config

# ---------------- Directories ----------------
FETCHED_PAPERS_FOLDER = Path(config.FETCHED_PAPERS_FOLDER)
SCREENED_PAPERS_FOLDER = Path(config.SCREENED_PAPERS_FOLDER)
TOP_PAPERS_FOLDER = Path(config.TOP_PAPERS_FOLDER)
READ_PAPERS_FOLDER = Path(config.READ_PAPERS_FOLDER)
SUMMARY_FOLDER = Path(config.SUMMARY_FOLDER)

saved_arvix_fetch = getattr(config, "saved_arvix_fetch", "")
saved_crossref_fetch = getattr(config, "saved_crossref_fetch", "")
saved_enriched_papers = getattr(config, "saved_enriched_papers", "")
saved_screened_papers = getattr(config, "saved_screened_papers", "")
saved_top_papers = getattr(config, "saved_top_papers", "")
skip_md_conversion = getattr(config, "skip_md_conversion", False)
saved_read_papers = getattr(config, "saved_read_papers", "")
saved_gap_reviews = getattr(config, "saved_gap_reviews", "")

# ---------------- Helper ----------------
def get_latest_checkpoint(track_dir):
    checkpoints = sorted(Path(track_dir).glob("papers_remaining_*"))
    return checkpoints[-1] if checkpoints else None

# ---------------- Main ----------------
if __name__ == "__main__":

    start_time = time.time()

    # ---------------- FETCHING ENRICHED ARTICLES ----------------
    if saved_enriched_papers:
        enrich_papers = load_json(saved_enriched_papers)
    else:
        t0 = time.time()

        if saved_arvix_fetch:
            arvix_papers = load_json(config.saved_arvix_fetch)
        else:
            t1 = time.time()
            arvix_papers = fetch_arvix(
                config.QUERIES,
                max_results=config.MAX_QUERIES,
                track=FETCHED_PAPERS_FOLDER / "checkpoints" / "raw_fetch"
            )
            print_time(t1, "Arvix fetching")

        if saved_crossref_fetch:
            crossref_papers = load_json(config.saved_crossref_fetch)
        else:
            t1 = time.time()
            crossref_papers = fetch_crossref(
                config.QUERIES,
                max_results=config.MAX_QUERIES,
                track=FETCHED_PAPERS_FOLDER / "checkpoints" / "raw_fetch"
            )
            print_time(t1, "Crossref fetching")

        combined_fetch = arvix_papers + crossref_papers
        all_fetched_papers = deduplicate_papers_by_title_authors(
            combined_fetch, paper_type="fetched"
        )

        enrich_papers = enrich(
            all_fetched_papers,
            track=FETCHED_PAPERS_FOLDER / "checkpoints" / "enriching"
        )

        saved_enriched_papers = save_json(
            enrich_papers,
            FETCHED_PAPERS_FOLDER,
            f"enriched_{len(enrich_papers)}_papers"
        )

    # ---------------- LLM SCREENING ----------------
    if saved_screened_papers:
        screened_papers = load_json(saved_screened_papers)
    else:
        screened_papers = screen_papers(
            enrich_papers,
            prompt_txt_path=config.LLM_SCREENING_PROMPT_TXT,
            criteria=config.CRITERIA,
            batch=2,
            track=SCREENED_PAPERS_FOLDER / "checkpoints"
        )

        screened_papers = deduplicate_papers_by_title_authors(
            screened_papers, paper_type="screened"
        )

        save_json(
            screened_papers,
            SCREENED_PAPERS_FOLDER,
            f"screened_{len(screened_papers)}_papers"
        )

    # ---------------- TOP PAPERS ----------------
    if saved_top_papers:
        top_papers = load_json(saved_top_papers)
    else:
        top_papers = filter_top_papers(screened_papers, [5, 6, 7, 8])
        save_json(top_papers, TOP_PAPERS_FOLDER, f"top_{len(top_papers)}_papers")
        save_md(
            paper_table(top_papers),
            folder=TOP_PAPERS_FOLDER,
            filename=f"top_{len(top_papers)}_papers"
        )

    # ================== 🔹 NEW AUTOMATIC PDF DOWNLOAD ==================
    print("\n📥 Downloading PDFs (OA + UDST Library Proxy)...")
    download_pdfs(
        papers=top_papers,
        output_dir=TOP_PAPERS_FOLDER / "pdf_papers"
    )
    # ===================================================================

    # ---------------- PDF → MARKDOWN ----------------
    if skip_md_conversion:
        print(f"PDF to MD conversions are in {TOP_PAPERS_FOLDER/'markdown_papers'}")
    else:
        print("Converting PDFs in configured folder to Markdown...")
        # ❌ OLD MANUAL STEP (COMMENTED OUT)
        # print(f"!!! MAKE SURE TO MANUALLY PUT THE FOLDERS IN {TOP_PAPERS_FOLDER/'pdf_papers'} !!!")

        try:
            run_marker_batch(
                TOP_PAPERS_FOLDER / "pdf_papers",
                TOP_PAPERS_FOLDER / "markdown_papers"
            )
        except ImportError:
            print("⚠️ Skipping PDF→Markdown — missing dependencies.")

    # ---------------- READING ----------------
    if not saved_read_papers:
        read_papers = read_paper_mds(
            prompt_path=config.LLM_FULL_READ_PROMPT_TXT,
            papers_folder=TOP_PAPERS_FOLDER / "markdown_papers",
            output_path=READ_PAPERS_FOLDER / "full_read.json",
            model_name="gemini-2.5-pro"
        )

        table_summary(
            papers=read_papers,
            csv_output=READ_PAPERS_FOLDER / "table_summary.csv",
            md_output=READ_PAPERS_FOLDER / "table_summary.md",
        )

    # ---------------- GAP REVIEWS ----------------
    if not saved_gap_reviews:
        generate_gap_reviews(
            gap_answers=gap_df(read_papers, config.GAPS),
            output_json=SUMMARY_FOLDER / "paper_reviews.json",
            output_md=SUMMARY_FOLDER / "paper_reviews.md",
        )
