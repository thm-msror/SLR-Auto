import os
import requests
import time
from typing import List, Dict
from src.utils import save_checkpoint, iso_now

BASE_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
QUOTA_FILE = ".ieee_quota_exhausted"

def check_ieee_quota() -> bool:
    """Check if the IEEE quota was marked as exhausted today."""
    if not os.path.exists(QUOTA_FILE):
        return True
    try:
        with open(QUOTA_FILE, "r") as f:
            date_str = f.read().strip()
        if date_str == iso_now()[:10]: # Compare YYYY-MM-DD
            return False
    except:
        pass
    return True

def mark_ieee_quota_exhausted():
    """Mark the IEEE quota as exhausted for today."""
    try:
        with open(QUOTA_FILE, "w") as f:
            f.write(iso_now()[:10])
    except:
        pass

# IEEE Xplore API rate limit: 10 requests/second (as of 2024 plan)
# Set conservative defaults: 1 req/sec with exponential backoff on 429
IEEE_DEFAULT_DELAY = 1.0
IEEE_MAX_RETRIES = 5


def _rate_limited_get(url: str, params: dict, delay: float, max_retries: int = IEEE_MAX_RETRIES) -> requests.Response:
    """
    Perform a GET request with IEEE-specific rate limiting and exponential backoff.

    If the API returns HTTP 429 (Too Many Requests), the request is retried
    with an exponentially increasing wait time up to max_retries attempts.

    Args:
        url: The endpoint URL.
        params: Query parameters.
        delay: Base delay in seconds between normal requests.
        max_retries: Maximum number of retry attempts on rate limit errors.

    Returns:
        The HTTP response object.

    Raises:
        requests.exceptions.HTTPError: If the request still fails after all retries.
    """
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 429:
                wait = IEEE_DEFAULT_DELAY * (2 ** attempt)
                print(f"  IEEE rate limit hit (429). Waiting {wait:.1f}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait)
                continue
            if response.status_code == 403:
                print("  IEEE Quota Exceeded (403). Stopping IEEE queries for today.")
                mark_ieee_quota_exhausted()
                return response
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            if attempt < max_retries:
                wait = IEEE_DEFAULT_DELAY * (2 ** attempt)
                print(f"  IEEE request error: {e}. Retrying in {wait:.1f}s ({attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                wait = IEEE_DEFAULT_DELAY * (2 ** attempt)
                print(f"  IEEE connection error: {e}. Retrying in {wait:.1f}s ({attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise requests.exceptions.HTTPError(f"IEEE request failed after {max_retries} retries.")


def fetch_papers(queries: List[str], api_key: str, max_results: int = 50, start_index: int = 1, delay: float = IEEE_DEFAULT_DELAY, track=False) -> List[Dict]:
    """
    Fetches papers from IEEE Xplore API with rate limiting and exponential backoff.

    Args:
        queries: List of search queries.
        api_key: IEEE Xplore API key.
        max_results: Maximum results per query.
        start_index: Starting record index (1-based).
        delay: Base delay between requests in seconds. Defaults to 1.0s to
               respect IEEE's rate limit of ~10 req/sec with a safety margin.
        track: Path or bool to track progress to disk.

    Returns:
        List of paper dictionaries with normalized fields.
    """
    if not check_ieee_quota():
        print("IEEE Quota was already hit today. Skipping IEEE search.")
        return []

    all_papers = []

    track_dir = None
    if track:
        track_dir = "track" if track is True else str(track)
        os.makedirs(track_dir, exist_ok=True)
        save_checkpoint(queries, track_dir, ".all_ieee_queries")

    for i, q in enumerate(queries):
        print(f"({i+1}/{len(queries)}) Querying IEEE for: {q}")

        params = {
            "apikey": api_key,
            "querytext": q,
            "max_records": max_results,
            "start_record": start_index,
            "format": "json",
            "sort_order": "descending",
            "sort_field": "publication_year"
        }

        try:
            response = _rate_limited_get(BASE_URL, params, delay)
            if response.status_code == 403:
                break # Stop processing further queries
            data = response.json()

            articles = data.get("articles", [])
            for art in articles:
                paper = {
                    "title": art.get("title"),
                    "authors": [auth.get("full_name") for auth in art.get("authors", {}).get("authors", [])],
                    "abstract": art.get("abstract"),
                    "published": str(art.get("publication_year", "")),
                    "publisher": "IEEE",
                    "doi": art.get("doi"),
                    "link": art.get("html_url"),
                    "from_query": q,
                    "metadata": {
                        "publication_title": art.get("publication_title"),
                        "content_type": art.get("content_type"),
                        "citing_paper_count": art.get("citing_paper_count")
                    }
                }
                all_papers.append(paper)

            print(f"  Retrieved {len(articles)} papers from IEEE.")

        except Exception as e:
            print(f"  Error fetching from IEEE: {e}")

        # Enforce delay between every query to stay within rate limits
        if i < len(queries) - 1:
            time.sleep(delay)

        if track_dir:
            save_checkpoint(all_papers, track_dir, ".ieee_backup")

    print(f"IEEE fetch complete. Total papers collected: {len(all_papers)}")
    return all_papers
