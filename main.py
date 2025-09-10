import time
from src.fetch_arxiv import fetch_papers as fetch_arvix
from src.fetch_crossref import fetch_papers as fetch_crossref
from src.utils import save_json, clean_papers
from src.llm_screener import screen_papers
from src.summarizer import summarize_screened
from src.utils import save_json, save_md
import config as config

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


if __name__ == "__main__":
    
    start_time = time.time()  # Start timer

    try:
        screened_arvix_path = config.screened_arvix_path
        screened_crossref_path = config.screened_crossref_path
    except:
        screened_arvix_path, screened_crossref_path = None, None

    if not screened_arvix_path and not screened_crossref_path:
        # Fetch papers
        t0 = time.time()
        raw_results = fetch_arvix(config.QUERIES, max_results= config.MAX_QUERIES)
        saved_arvix_path = save_json(clean_papers(raw_results), folder= config.FETCHED_PAPERS_FOLDER, filename= f"arXiv_{config.MAX_QUERIES}_")
        print(f"⏱️ arXiv fetch + save took {time.time() - t0:.2f} sec")

        # LLM screening
        t0 = time.time()
        screened_papers = screen_papers(saved_arvix_path, config.LLM_SCREENING_PROMPT_TXT)
        screened_arvix_path = save_json(screened_papers, folder= config.SCREENED_PAPERS_FOLDER, filename= f"arXiv_{config.MAX_QUERIES}_")
        print(f"⏱️ arXiv screening took {time.time() - t0:.2f} sec")

        # Fetch Crossref papers
        t0 = time.time()
        raw_crossref_results = fetch_crossref(config.QUERIES, max_results=config.MAX_QUERIES)
        saved_crossref_path = save_json(
            clean_papers(raw_crossref_results),
            folder=config.FETCHED_PAPERS_FOLDER,
            filename=f"crossref_{config.MAX_QUERIES}_"
        )
        print(f"⏱️ Crossref fetch + save took {time.time() - t0:.2f} sec")

        # LLM screening for Crossref
        t0 = time.time()
        screened_crossref = screen_papers(saved_crossref_path, config.LLM_SCREENING_PROMPT_TXT)
        screened_crossref_path = save_json(
            screened_crossref,
            folder=config.SCREENED_PAPERS_FOLDER,
            filename=f"crossref_{config.MAX_QUERIES}_"
        )
        print(f"⏱️ Crossref screening took {time.time() - t0:.2f} sec")
        
        print(f"✅ Pipeline complete. Results saved in {screened_arvix_path} and {screened_crossref_path}")
    else: 
        print(f"⏭️ Skipping fetching process. screened_arvix_path and screened_crossref_path detected. ")

    # LLM summarization
    # Summarization for arXiv
    t0 = time.time()
    summary = summarize_screened(screened_arvix_path)
    print("\n📊 LLM Summary of Relevant arXiv Papers:\n")
    print(summary)
    print(f"⏱️ arXiv summarization took {time.time() - t0:.2f} sec")
    
    # Save arXiv summary
    save_md(summary, folder=config.SUMMARY_FOLDER, filename=f"arXiv_summary_{config.MAX_QUERIES}_")

    # Summarization for Crossref
    t0 = time.time()
    summary_crossref = summarize_screened(screened_crossref_path)
    print("\n📊 LLM Summary of Relevant Crossref Papers:\n")
    print(summary_crossref)
    print(f"⏱️ Crossref summarization took {time.time() - t0:.2f} sec")

    # Save Crossref summary
    save_md(summary_crossref, folder=config.SUMMARY_FOLDER, filename=f"crossref_summary_{config.MAX_QUERIES}_")
    
    # --- Total runtime ---
    total_time = time.time() - start_time
    print(f"\n🚀 Total pipeline runtime: {total_time:.2f} sec")
