import os
import time
from src.fetch_arxiv import fetch_papers as fetch_arvix
from src.utils import save_json, clean_papers, save_md
from src.llm_screener import screen_papers
from src.filter_papers import summarize_screened
import config as config


def run():
    """Full pipeline for arXiv."""
    name = "arXiv"
    fetched_path = os.path.join(config.FETCHED_PAPERS_FOLDER, f"{name}_{config.MAX_QUERIES}_latest.json")
    screened_path = os.path.join(config.SCREENED_PAPERS_FOLDER, f"{name}_{config.MAX_QUERIES}_screened.json")

    # --- Fetch ---
    if os.path.exists(fetched_path):
        print(f"⏭️ {name} fetched file exists: {fetched_path}")
    else:
        print(f"⚡ Fetching {name} papers...")
        raw = fetch_arvix(config.QUERIES, max_results=config.MAX_QUERIES)
        fetched_path = save_json(clean_papers(raw), folder=config.FETCHED_PAPERS_FOLDER,
                                 filename=f"{name}_{config.MAX_QUERIES}_latest.json")
        print(f"✅ {name} fetch complete → {fetched_path}")

    # --- Screening ---
    if os.path.exists(screened_path):
        print(f"⏭️ {name} screened file exists: {screened_path}")
    else:
        print(f"⚡ Screening {name} papers...")
        screened = screen_papers(fetched_path, config.LLM_SCREENING_PROMPT_TXT)
        screened_path = save_json(screened, folder=config.SCREENED_PAPERS_FOLDER,
                                  filename=f"{name}_{config.MAX_QUERIES}_screened.json")
        print(f"✅ {name} screening complete → {screened_path}")

    # --- Summarization ---
    print(f"\n📊 Summarizing {name} screened papers...")
    t0 = time.time()
    summary = summarize_screened(screened_path)
    save_md(summary, folder=config.SUMMARY_FOLDER, filename=f"{name}_summary_{config.MAX_QUERIES}_latest.md")
    print(f"✅ {name} summarization done (⏱️ {time.time() - t0:.2f} sec)")
