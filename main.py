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
import config as config

# ---------------- Directories ----------------
FETCHED_PAPERS_FOLDER = Path(config.FETCHED_PAPERS_FOLDER)
SCREENED_PAPERS_FOLDER = Path(config.SCREENED_PAPERS_FOLDER)
TOP_PAPERS_FOLDER = Path(config.TOP_PAPERS_FOLDER)
saved_arvix_fetch = getattr(config, "saved_arvix_fetch", "")
saved_crossref_fetch = getattr(config, "saved_crossref_fetch", "")
saved_enriched_papers = getattr(config, "saved_enriched_papers", "")
saved_screened_papers = getattr(config, "saved_screened_papers", "")
saved_top_papers = getattr(config, "saved_top_papers", "")

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

        # Fetching arvix if not skipped
        if saved_arvix_fetch:
            arvix_papers = load_json(config.saved_arvix_fetch)
        else: 
            t1 = time.time()
            arvix_papers = fetch_arvix(
                config.QUERIES,
                max_results=config.MAX_QUERIES,
                track= FETCHED_PAPERS_FOLDER / "checkpoints" / "raw_fetch"
                )
            print_time(t1, "Arvix fetching")
            
        # Fetching crossref if not skipped
        if saved_crossref_fetch:
            crossref_papers = load_json(config.saved_crossref_fetch)
        else:
            t1 = time.time()
            crossref_papers = fetch_crossref(
                config.QUERIES,
                max_results=config.MAX_QUERIES,
                track= FETCHED_PAPERS_FOLDER / "checkpoints" / "raw_fetch"
            )
            print_time(t1, "Crossref fetching")
        
        # Enriching combined fetched papers
        combined_fetch = arvix_papers + crossref_papers

        print(f"\nLoaded {len(arvix_papers)} arXiv and {len(crossref_papers)} Crossref papers.")
        all_fetched_papers = deduplicate_papers_by_title_authors(combined_fetch, paper_type="fetched")

        print("\nEnriching papers via OpenAlex (metadata, authors, DOIs)...")
        enrich_papers = enrich(
            all_fetched_papers,
            track= FETCHED_PAPERS_FOLDER / "checkpoints" / "enriching"
        )

        saved_enriched_papers = save_json(enrich_papers, FETCHED_PAPERS_FOLDER, f"enriched_{len(enrich_papers)}_papers")
        print(f"\nTotal unique papers ready for screening: {len(enrich_papers)}")
        print_time(t1, "Fetching and enriching")


    # ---------------- LLM SCREENING ----------------
    if saved_screened_papers:
        screened_papers = load_json(saved_screened_papers)
    else:
        t0 = time.time()
        screened_papers = screen_papers(
            enrich_papers,
            batch_size=100,
            model="Fanar",
            prompt_txt_path=config.LLM_SCREENING_PROMPT_TXT,
            save_to_files=False,
        )
        screened_papers = deduplicate_papers_by_title_authors(screened_papers, paper_type="screened")
        save_json(screened_papers, SCREENED_PAPERS_FOLDER, f"screened_{len(screened_papers)}_papers")
        print_time(t0, "LLM Screening")


    # ---------------- READING TOP PAPERS ----------------
    if saved_top_papers: 
        top_papers = load_json(saved_screened_papers)
    else:
        t0 = time.time()
        top_papers = filter_top_papers(screened_papers, [7, 8])
        save_json(top_papers, TOP_PAPERS_FOLDER, f"top_{len(top_papers)}_papers")
        save_md(paper_table(top_papers), folder=TOP_PAPERS_FOLDER, filename=f"top_{len(top_papers)}_papers")
        print_time (t0, "Filtering top papers")

    print("Converting PDFs in configured folder to Markdown...")
    print(f"!!! MAKE SURE TO MANUALLY PUT THE FOLDERS IN {TOP_PAPERS_FOLDER/"pdf_papers"} !!!")
    try:
        t0 = time.time()
        # Skips already converted pdfs
        run_marker_batch(TOP_PAPERS_FOLDER/"pdf_papers", TOP_PAPERS_FOLDER/"markdown_papers")
        print_time (t0, "Converting PDFs -> Markdowns")
    except ImportError:
        print("⚠️ Skipping PDF→Markdown — missing dependencies.")
