import requests
import feedparser
from urllib.parse import urlencode
from time import sleep


BASE_URL = "http://export.arxiv.org/api/query"


def fetch_papers(queries, max_results=100, start=0, per_query=50, delay=3):
    """
    Fetches papers from arXiv API given a list of queries.

    Args:
        queries (list[str]): Search terms (keywords).
        max_results (int): Max total papers per query.
        start (int): Start index for pagination.
        per_query (int): How many results to fetch per API call (max=2000).
        delay (int): Seconds to wait between API calls (to respect rate limits).

    Returns:
        list[dict]: List of papers with metadata.
    """
    all_papers = []

    for q in queries:
        print(f"🔍 Querying arXiv for: {q}")
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
                print(f"⚠️ Failed request (status {response.status_code})")
                break

            feed = feedparser.parse(response.text)

            if not feed.entries:
                print("⚠️ No more entries found.")
                break

            for entry in feed.entries:
                print(entry, "\n\n\n\n\n\n\n")
                paper = {
                    "title": entry.title,
                    "authors": [author.name for author in entry.authors],
                    "summary": entry.summary,
                    "published": entry.published,
                    "updated": entry.updated,
                    # "id": entry.id,   # usually arXiv ID/URL
                    "link": entry.link,
                }
                all_papers.append(paper)

            count = len(feed.entries)
            fetched += count
            print(f"✅ Retrieved {count} papers (total so far: {fetched})")

            sleep(delay)

    return all_papers