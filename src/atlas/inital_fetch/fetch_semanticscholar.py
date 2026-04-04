import requests
import time
from typing import List, Dict

from atlas.inital_fetch.gpt_research_q import tokenize

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
S2_PAGE_SIZE = 100
S2_FIELDS = "title,authors,abstract,year,externalIds,venue,url,paperId"

# Allowed DOI prefixes for peer-reviewed publishers
ALLOWED_PREFIXES = [
    "10.1109",   # IEEE
    "10.1145",   # ACM
    "10.1016",   # Elsevier journals (Scopus indexed)
    "10.1007",   # Springer
    "10.1038",   # Nature
    "10.48550",  # arXiv
]

def _to_semantic_scholar_query(query: str) -> str:
    parts: List[str] = []
    for token in tokenize(query):
        upper = token.upper()
        if upper == "AND":
            parts.append("+")
        elif upper == "OR":
            parts.append("|")
        elif upper == "NOT":
            parts.append("-")
        else:
            parts.append(token)

    normalized = " ".join(parts)
    normalized = normalized.replace("( ", "(").replace(" )", ")")
    normalized = normalized.replace("+ -", "+ -")
    return normalized


def fetch_papers(queries: List[str], max_results: int = 50, delay: float = 1.0) -> List[Dict]:
    """
    Fetches papers from Semantic Scholar API using the bulk search endpoint.
    """
    all_papers = []
    
    total_q = len(queries)
    for i, q in enumerate(queries):
        s2_query = _to_semantic_scholar_query(q)
        print(f"({i+1}/{total_q}) Querying Semantic Scholar for: {q}")
        fetched = 0
        token = None

        while fetched < max_results:
            params = {
                "query": s2_query,
                "limit": min(S2_PAGE_SIZE, max_results - fetched),
                "fields": S2_FIELDS,
            }
            if token:
                params["token"] = token

            try:
                response = requests.get(BASE_URL, params=params, timeout=30)
                if response.status_code == 429:
                    print(f"  Rate limit hit for {q}. Skipping remaining Semantic Scholar pages.")
                    break
                response.raise_for_status()
                data = response.json()
                data_list = data.get("data", [])
                if not data_list:
                    print("  Retrieved 0 papers from Semantic Scholar.")
                    break

                kept = 0
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
                        "metadata": {
                            "venue": item.get("venue"),
                            "s2_id": item.get("paperId"),
                            "s2_query": s2_query,
                        },
                    })
                    kept += 1

                fetched += len(data_list)
                token = data.get("token")
                print(
                    f"  Retrieved {len(data_list)} papers from Semantic Scholar "
                    f"({kept} kept after publisher filter)."
                )

                if not token:
                    break
                if fetched < max_results:
                    time.sleep(delay)

            except Exception as e:
                detail = ""
                try:
                    detail = f" Response: {response.text[:300]}"
                except Exception:
                    pass
                print(f"  Error fetching Semantic Scholar for {q}: {e}{detail}")
                break

        if i < total_q - 1:
            time.sleep(delay)
            
    print(f"Semantic Scholar fetch complete. Total: {len(all_papers)}")
    return all_papers
