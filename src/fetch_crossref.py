import requests
from time import sleep

BASE_URL = "https://api.crossref.org/works"

# Allowed DOI prefixes for peer-reviewed publishers
ALLOWED_PREFIXES = [
    "10.1109",   # IEEE
    "10.1145",   # ACM
    "10.1016",   # Elsevier journals (Scopus indexed)
    "10.1007",   # Springer
]

def fetch_papers(queries, max_results=100, per_page=20, delay=1):
    """
    Fetches papers from Crossref API given a list of queries,
    filters to peer-reviewed sources (IEEE, ACM, Elsevier, Springer).
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
                    "doi": doi,
                    "publisher": item.get("publisher"),
                    "link": f"https://doi.org/{doi}" if doi else None,
                    "from_query": q
                }
                all_papers.append(paper)

            count = len(items)
            fetched += count
            print(f"> ✅ Retrieved {count} papers (total kept so far: {len(all_papers)})")

            if not cursor:
                break
            sleep(delay)

    return all_papers
