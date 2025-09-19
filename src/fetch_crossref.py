import os
import requests
from random import uniform  
from time import sleep
from src.utils import *

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

BASE_URL = "https://api.crossref.org/works"

# Allowed DOI prefixes for peer-reviewed publishers
ALLOWED_PREFIXES = [
    "10.1109",   # IEEE
    "10.1145",   # ACM
    "10.1016",   # Elsevier journals (Scopus indexed)
    "10.48550",  # arXiv
]

# Setup a requests session with retries
def create_retry_session(total_retries=3, backoff_factor=1):
    session = requests.Session()
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

session = create_retry_session()

def fetch_papers(queries, max_results=100, per_page=20, delay=3, track=False):
    """
    Fetches papers from Crossref API given a list of queries,
    filters to peer-reviewed sources (IEEE, ACM, Elsevier, Springer).
    """
    all_papers = []
    
    track_dir = None
    if track:
        track_dir = "track" if track is True else str(track)
        os.makedirs(track_dir, exist_ok=True)
        queries_all_dir = save_json(queries, track_dir, ".all_crossref_queries") 

    for i, q in enumerate(queries):
        print(f"🔍 Querying Crossref for: {q}")
        fetched = 0
        cursor = "*"  # initial cursor for first request

        while fetched < max_results:
            params = {
                "query": q,
                "rows": min(per_page, max_results - fetched),
                "cursor": cursor
            }

            try:
                response = session.get(BASE_URL, params=params, timeout=30)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"⏳ Retrying after error: {e}")
                sleep(uniform(delay*5 -2 , delay*5 + 2)) 
                continue  # Try the same request again

            data = response.json()
            items = data.get("message", {}).get("items", [])
            cursor = data.get("message", {}).get("next-cursor")

            if not items:
                print("⚠️ No more entries found.")
                break

            for item in items:
                doi = item.get("DOI", "")
                if not any(doi.startswith(prefix) for prefix in ALLOWED_PREFIXES):
                    continue  # 🚫 Skip non-peer-reviewed (TechRxiv, SSRN, etc.)

                paper = {
                    "title": item.get("title", ["No title"])[0],
                    "authors": [
                        f"{a.get('given', '')} {a.get('family', '')}".strip()
                        for a in item.get("author", [])
                    ] if "author" in item else [],
                    "published": item.get("created", {}).get("date-time"),
                    "publisher": item.get("publisher"),
                    "doi": doi,
                    "link": f"https://doi.org/{doi}" if doi else None,
                    "from_query": q,
                    # TODO: keep all metadata other
                }
                all_papers.append(paper)

            count = len(items)
            fetched += count
            print(f"> ✅ Retrieved {count} papers (total kept so far: {len(all_papers)})")

            if not cursor:
                break
            sleep(uniform(delay -2 , delay + 2)) 

        # Save backup per query
        if track_dir:
            backup_path = save_checkpoint(all_papers, track_dir, ".crossref_backup")
            queries_dir = save_checkpoint(queries[i:], track_dir, ".crossref_queries_remaining")

    return all_papers