import os
import requests
import feedparser
from urllib.parse import urlencode
from time import sleep
from src.utils import *


BASE_URL = "http://export.arxiv.org/api/query"


def fetch_papers(queries, max_results=100, start=0, per_query=50, delay=3, track=False):
    """
    Fetches papers from arXiv API given a list of queries.

    Args:
        queries (list[str]): Search terms (keywords).
        max_results (int): Max total papers per query.
        start (int): Start index for pagination.
        per_query (int): How many results to fetch per API call (max=2000).
        delay (int): Seconds to wait between API calls (to respect rate limits).
        track (bool|str): Directory path to store raw fetches if True/str.

    Returns:
        list[dict]: List of papers with metadata.
    """
    all_papers = []

    # Handle track as a directory
    track_dir = None
    if track:
        track_dir = "track" if track is True else str(track)
        os.makedirs(track_dir, exist_ok=True)
        # Save a timestamped checkpoint of the input queries for traceability
        save_checkpoint(queries, track_dir, ".all_arvix_queries") 

    for i, q in enumerate(queries):
        print(f" Querying arXiv for: {q}")
        fetched = 0

        while fetched < max_results:
            params = {
                "search_query": f"all:{q}",
                "start": start + fetched,
                "max_results": min(per_query, max_results - fetched),
                "sortBy": "relevance",
                "sortOrder": "descending"
            }

            url = f"{BASE_URL}?{urlencode(params)}"
            response = requests.get(url, timeout=30)

            if response.status_code != 200:
                print(f" Failed request (status {response.status_code})")
                break

            feed = feedparser.parse(response.text)

            if not feed.entries:
                print(" No more entries found.")
                break

            for entry in feed.entries:
                doi = arxiv_to_doi(entry.link)
                paper = {
                    "title": entry.title,
                    "authors": [author.name for author in entry.authors],
                    "abstract": entry.summary,
                    "published": entry.published,
                    "updated": entry.updated,
                    "doi": doi,
                    "link": f"https://doi.org/{doi}" if doi else None,
                    "from_query": q,
                    # "raw_entry": entry  # keep all metadata
                }
                all_papers.append(paper)

            count = len(feed.entries)
            fetched += count
            print(f">  Retrieved {count} papers (total so far: {fetched})")

            sleep(delay)

        # Save backup per query
        if track_dir:
            backup_path = save_checkpoint(all_papers, track_dir, ".arxiv_backup")
            queries_dir = save_checkpoint(queries[i:], track_dir, ".arxiv_queries_remaining")
        
    return all_papers

def arxiv_to_doi(arxiv_url: str) -> str | None:
    """
    Convert an arXiv URL (with or without version suffix) into a valid DOI.

    Example: http://arxiv.org/abs/1906.02497v2 → 10.48550/arXiv.1906.02497
    """
    if not arxiv_url or "arxiv.org/abs/" not in arxiv_url:
        return None

    try:
        # Extract the arXiv ID part
        arxiv_id = arxiv_url.split("arxiv.org/abs/")[-1]
        # Remove version suffix if present (e.g., v2 → "")
        arxiv_id = arxiv_id.split("v")[0]

        return f"10.48550/arXiv.{arxiv_id}"
    except Exception:
        return None
