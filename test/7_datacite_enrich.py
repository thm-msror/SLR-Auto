import requests
import urllib.parse

OPENALEX_URL = "https://api.openalex.org/works/"
DATACITE_URL = "https://api.datacite.org/dois/"

def enrich(papers):
    enriched = []
    for p in papers:
        work_data = None
        doi = p.get("doi")

        # --- 1. Try OpenAlex by DOI ---
        if doi:
            url = OPENALEX_URL + f"doi:{doi}"
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    work_data = r.json()
                    p["openalex_lookup"] = "doi"
            except Exception as e:
                print(f"⚠️ OpenAlex DOI lookup failed for {doi}: {e}")

        # --- 2. Fallback: OpenAlex by title ---
        if not work_data:
            title = p.get("title", "")
            if title:
                query_url = OPENALEX_URL + f"?filter=title.search:{urllib.parse.quote(title)}"
                try:
                    r = requests.get(query_url, timeout=15)
                    if r.status_code == 200:
                        results = r.json().get("results", [])
                        if results:
                            work_data = results[0]   # take top match
                            p["openalex_lookup"] = "title"
                except Exception as e:
                    print(f"⚠️ OpenAlex title search failed for '{title}': {e}")

        # --- 3. Fallback: DataCite ---
        if not work_data and doi:
            try:
                url = DATACITE_URL + urllib.parse.quote(doi)
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    data = r.json().get("data", {})
                    attributes = data.get("attributes", {})
                    # Map DataCite fields into paper
                    p["abstract"] = attributes.get("descriptions", [{}])[0].get("description")
                    p["publisher"] = attributes.get("publisher")
                    p["datacite_url"] = attributes.get("url")
                    p["datacite_lookup"] = True
            except Exception as e:
                print(f"⚠️ DataCite fetch failed for {doi}: {e}")

        # --- If OpenAlex returned something, enrich from it ---
        if work_data:
            p["abstract"] = work_data.get("abstract_inverted_index") or p.get("abstract")
            p["citations"] = work_data.get("cited_by_count")
            p["reference_count"] = len(work_data.get("referenced_works", []))
            p["openalex_id"] = work_data.get("id")

            # Convert abstract_inverted_index → readable string
            if isinstance(p["abstract"], dict):
                words = sorted(
                    (pos, word)
                    for word, positions in p["abstract"].items()
                    for pos in positions
                )
                p["abstract"] = " ".join(word for _, word in words)

        enriched.append(p)

    return enriched
