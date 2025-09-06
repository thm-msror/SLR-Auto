import requests
from time import sleep

BASE_URL = "https://api.crossref.org/works"

def fetch_papers(queries, max_results=100, per_page=20, delay=1):
    """
    Fetches papers from Crossref API given a list of queries.

    Args:
        queries (list[str]): Search terms (keywords).
        max_results (int): Maximum total papers per query.
        per_page (int): Number of results per API call (max 1000 for Crossref).
        delay (float): Seconds to wait between API calls (rate limit).

    Returns:
        list[dict]: List of papers with metadata.
    """
    all_papers = []

    for q in queries:
        print(f"🔍 Querying Crossref for: {q}")
        fetched = 0
        cursor = "*"  # initial cursor for first request

        while fetched < max_results:
            params = {
                "query": q,
                "rows": min(per_page, max_results - fetched),
                "cursor": cursor
            }

            response = requests.get(BASE_URL, params=params, timeout=30)
            if response.status_code != 200:
                print(f"⚠️ Failed request (status {response.status_code})")
                break

            data = response.json()
            items = data.get("message", {}).get("items", [])
            cursor = data.get("message", {}).get("next-cursor")  # update cursor for next call

            if not items:
                print("⚠️ No more entries found.")
                break

            for item in items:
                paper = {
                    "title": item.get("title", ["No title"])[0],
                    "authors": [
                        f"{a.get('given', '')} {a.get('family', '')}".strip()
                        for a in item.get("author", [])
                    ] if "author" in item else [],
                    "published": item.get("created", {}).get("date-time"),
                    "doi": item.get("DOI"),
                    "publisher": item.get("publisher"),
                    "link": f"https://doi.org/{item.get('DOI')}" if item.get("DOI") else None,
                    "from_query": q
                }
                all_papers.append(paper)

            count = len(items)
            fetched += count
            print(f"> ✅ Retrieved {count} papers (total so far: {fetched})")

            if not cursor:
                break  # no more pages
            sleep(delay)

    return all_papers