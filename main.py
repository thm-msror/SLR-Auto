import time
from src.utils import *
from src.fetch_arxiv import fetch_papers as fetch_arvix
from src.fetch_crossref import fetch_papers as fetch_crossref
from src.enrich_openalex import enrich
from src.llm_screener import screen_papers
from src.summarizer import summarize_screened
import config as config

if __name__ == "__main__":
    
    start_time = time.time()  # Start timer

    # FETCHING ARTICLES FROM ARVIX, CROSSREF, AND OPENALEX 
    if config.all_fetched_path:
        all_fetched_papers = load_json(config.all_fetched_path)
    else:
        # Fetch arXiv 
        if config.arvix_fetch_path:
            raw_arvix_fetch = load_json(config.arvix_fetch_path)
        else:
            t0 = time.time()
            raw_arvix_fetch = fetch_arvix(config.QUERIES, 
                                        max_results=config.MAX_QUERIES, 
                                        track=f"{config.FETCHED_PAPERS_FOLDER}/raw_fetch/checkpoints")
            arvix_fetch_path = save_json(raw_arvix_fetch, 
                                        folder= f"{config.FETCHED_PAPERS_FOLDER}/raw_fetch/" , 
                                        filename= f".arXiv_{config.MAX_QUERIES}_")
            print(f"⏱️ ArXiv fetch + save took {time.time() - t0:.2f} sec")
            
        # Fetch crossref 
        if config.crossref_fetch_path:
            raw_crossref_fetch = load_json(config.crossref_fetch_path)
        else:
            t0 = time.time()
            raw_crossref_fetch = fetch_crossref(config.QUERIES, 
                                                max_results=config.MAX_QUERIES,
                                                track=f"{config.FETCHED_PAPERS_FOLDER}/raw_fetch/checkpoints")
            crossref_fetch_path = save_json(raw_crossref_fetch, 
                                        folder=f"{config.FETCHED_PAPERS_FOLDER}/raw_fetch/" ,
                                        filename=f".crossref_{config.MAX_QUERIES}_")
            print(f"⏱️ Crossref fetch + save took {time.time() - t0:.2f} sec")

        # Combine and enrich all fetches
        t0 = time.time()
        older_fetch_pathes = []
        for file_path in config.older_fetch_pathes:
            older_fetch_pathes.extend(load_json(file_path))
        
        combined_fetch = clean_papers(
            older_fetch_pathes
            + raw_crossref_fetch 
            + raw_arvix_fetch
        )
        all_fetched_papers = enrich(combined_fetch, track=f"{config.FETCHED_PAPERS_FOLDER}/enrich/checkpoints")
        all_fetched_path = save_json(all_fetched_papers, folder=config.FETCHED_PAPERS_FOLDER, filename=f"fetched_{len(all_fetched_papers)}_")
        print(f"Cleaning + enriching took {time.time() - t0:.2f} sec")


    # LLM SCREENING 
    if config.all_screened_path:
        all_screened_papers = load_json(config.all_screened_path)
    else:
        t0 = time.time()
        all_screened_papers = screen_papers(all_fetched_papers,
                                            config.LLM_SCREENING_PROMPT_TXT,
                                            track=f"{config.SCREENED_PAPERS_FOLDER}/checkpoints",
                                            batch_size=5)
        all_screened_path = save_json(all_screened_papers, folder= config.SCREENED_PAPERS_FOLDER, filename= f"screened_{len(all_screened_papers)}_")
        print(f"⏱️ LLM screening took {time.time() - t0:.2f} sec")



    # LLM summarization
    # Summarization for arXiv
    t0 = time.time()
    summary = summarize_screened(all_screened_path)
    print("\n📊 LLM Summary of Relevant Papers:\n")
    print(summary)
    print(f"⏱️ summarization took {time.time() - t0:.2f} sec")
    
    # Save arXiv summary
    save_md(summary, folder=config.SUMMARY_FOLDER, filename=f"Summary_{config.MAX_QUERIES}_")