import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fetch_arxiv import fetch_papers as fetch_arvix
from src.fetch_crossref import fetch_papers as fetch_crossref
from src.utils import clean_papers, save_json

if __name__ == "__main__":
    # --- Fetch arXiv papers ---
    results = fetch_arvix(["graph neural networks healthcare"], max_results=20)
    cleaned = clean_papers(results)

    save_json(cleaned, filename='arxiv')
    
    # --- Fetch Crossref papers ---
    crossref_results = fetch_crossref(["graph neural networks healthcare"], max_results=20)
    cleaned_crossref = clean_papers(crossref_results)

    save_json(cleaned_crossref, filename='crossref')


'''
🔍 Querying arXiv for: graph neural networks healthcare
✅ Retrieved 20 papers (total so far: 20)
🧹 Cleaned 20 unique papers.
💾 Saved JSON: data/save_articles/articles_2025-09-03T20-05-11.json
'''