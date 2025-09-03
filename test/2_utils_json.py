import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fetch_arxiv import fetch_papers
from src.utils import clean_papers, save_json

if __name__ == "__main__":
    results = fetch_papers(["graph neural networks healthcare"], max_results=20)
    cleaned = clean_papers(results)

    save_json(cleaned, filename='arxiv')

'''
🔍 Querying arXiv for: graph neural networks healthcare
✅ Retrieved 20 papers (total so far: 20)
🧹 Cleaned 20 unique papers.
💾 Saved JSON: data/save_articles/articles_2025-09-03T20-05-11.json
'''