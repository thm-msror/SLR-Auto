from src.fetch_arxiv import fetch_papers
from src.utils import save_json, clean_papers
from src.llm_screener import screen_papers

import config as config

if __name__ == "__main__":

    # Fetch papers
    raw_results = fetch_papers(config.QUERIES, max_results= config.MAX_QUERIES)
    saved_arvix_path = save_json(clean_papers(raw_results), folder= config.FETCHED_PAPERS_FOLDER, filename= f"arXiv_{config.MAX_QUERIES}_")

    # LLM screening
    screened_papers = screen_papers(saved_arvix_path, config.LLM_SCREENING_PROMPT_TXT)
    screened_arvix_path = save_json(screened_papers, folder= config.SCREENED_PAPERS_FOLDER, filename= f"arXiv_{config.MAX_QUERIES}_")

    # LLM summarization
    # still TODO 
     
    print(f"✅ Pipeline complete. Results saved in {screened_arvix_path}")
