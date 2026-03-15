import requests
import time
from typing import List, Dict, Optional
import os

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

# Allowed DOI prefixes for peer-reviewed publishers
ALLOWED_PREFIXES = [
    "10.1109",   # IEEE
    "10.1145",   # ACM
    "10.1016",   # Elsevier journals (Scopus indexed)
    "10.1007",   # Springer
    "10.1038",   # Nature
    "10.48550",  # arXiv
]

def fetch_papers(queries: List[str], max_results: int = 50, delay: float = 1.0) -> List[Dict]:
    """
    Fetches papers from Semantic Scholar API using the bulk search endpoint.
    """
    all_papers = []
    
    total_q = len(queries)
    for i, q in enumerate(queries):
        print(f"({i+1}/{total_q}) Querying Semantic Scholar for: {q}")
        params = {
            "query": q,
            "limit": min(max_results, 50),
            "fields": "title,authors,abstract,year,externalIds,venue,url"
        }
        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            if response.status_code == 429:
                print(f"  Rate limit hit for {q}. Skipping.")
                continue
            response.raise_for_status()
            data = response.json()
            data_list = data.get("data", [])
            for item in data_list:
                doi = (item.get("externalIds") or {}).get("DOI", "")
                if doi and not any(doi.startswith(prefix) for prefix in ALLOWED_PREFIXES):
                    continue
                all_papers.append({
                    "title": item.get("title"),
                    "authors": [auth.get("name") for auth in item.get("authors", [])],
                    "abstract": item.get("abstract"),
                    "published": str(item.get("year", "")),
                    "publisher": item.get("venue") or "Semantic Scholar",
                    "doi": doi,
                    "link": item.get("url"),
                    "from_query": q,
                    "metadata": {"venue": item.get("venue"), "s2_id": item.get("paperId")}
                })
            
            if i < total_q - 1:
                time.sleep(delay)
                
        except Exception as e:
            print(f"  Error fetching Semantic Scholar for {q}: {e}")
            
    print(f"Semantic Scholar fetch complete. Total: {len(all_papers)}")
    return all_papers
