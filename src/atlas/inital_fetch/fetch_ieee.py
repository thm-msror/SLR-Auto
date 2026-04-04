import os
import requests
import time
from typing import List, Dict
from atlas.utils.utils import save_checkpoint, iso_now

BASE_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
QUOTA_FILE = ".ieee_quota_exhausted"
IEEE_API_PAGE_SIZE = 200

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


def _coerce_total_records(data: dict) -> int | None:
    for key in ("total_records", "total_records_searched", "totalfound"):
        value = data.get(key)
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def fetch_papers(
    queries: List[str],
    api_key: str,
    max_results: int | None = 50,
    start_index: int = 1,
    delay: float = IEEE_DEFAULT_DELAY,
    track=False,
) -> List[Dict]:
    """
    Fetches papers from IEEE Xplore API with rate limiting and exponential backoff.

    Args:
        queries: List of search queries.
        api_key: IEEE Xplore API key.
        max_results: Maximum results per query. ``None`` fetches all available pages.
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
        query_count = 0
        start_record_for_query = start_index
        total_available = None

        while True:
            remaining = IEEE_API_PAGE_SIZE if max_results is None else max_results - query_count
            if remaining <= 0:
                break

            params = {
                "apikey": api_key,
                "querytext": q,
                "max_records": min(IEEE_API_PAGE_SIZE, remaining),
                "start_record": start_record_for_query,
                "format": "json",
                "sort_order": "descending",
                "sort_field": "publication_year",
            }

            try:
                response = _rate_limited_get(BASE_URL, params, delay)
                if response.status_code == 403:
                    break  # Stop processing further queries
                data = response.json()
                total_available = _coerce_total_records(data)

                articles = data.get("articles", [])
                if not articles:
                    print("  Retrieved 0 papers from IEEE.")
                    break

                for art in articles:
                    arnumber = art.get("article_number") or art.get("arnumber")
                    paper = {
                        "title": art.get("title"),
                        "authors": [auth.get("full_name") for auth in art.get("authors", {}).get("authors", [])],
                        "abstract": art.get("abstract"),
                        "published": str(art.get("publication_year", "")),
                        "publisher": "IEEE",
                        "doi": art.get("doi"),
                        "link": art.get("html_url"),
                        "from_query": q,
                        "ieee_arnumber": arnumber,
                        "metadata": {
                            "publication_title": art.get("publication_title"),
                            "content_type": art.get("content_type"),
                            "citing_paper_count": art.get("citing_paper_count"),
                            "ieee_total_available": total_available,
                        },
                    }
                    all_papers.append(paper)

                query_count += len(articles)
                start_record_for_query += len(articles)

                if total_available is not None:
                    print(
                        f"  Retrieved {len(articles)} papers from IEEE "
                        f"({query_count}/{total_available} for this query)."
                    )
                else:
                    print(f"  Retrieved {len(articles)} papers from IEEE ({query_count} total for this query).")

                if len(articles) < params["max_records"]:
                    break
                if total_available is not None and query_count >= total_available:
                    break
                if track_dir:
                    save_checkpoint(all_papers, track_dir, ".ieee_backup")
                time.sleep(delay)

            except Exception as e:
                print(f"  Error fetching from IEEE: {e}")
                break

        # Enforce delay between queries to stay within rate limits
        if i < len(queries) - 1:
            time.sleep(delay)

        if track_dir:
            save_checkpoint(all_papers, track_dir, ".ieee_backup")

    print(f"IEEE fetch complete. Total papers collected: {len(all_papers)}")
    return all_papers
