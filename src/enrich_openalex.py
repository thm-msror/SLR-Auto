import requests

OPENALEX_URL = "https://api.openalex.org/works/"

def enrich_with_openalex(papers):
    enriched = []
    for p in papers:
        if not p.get("doi"):
            enriched.append(p)
            continue
        doi = p["doi"]
        url = OPENALEX_URL + f"doi:{doi}"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                p["abstract"] = data.get("abstract_inverted_index")
                p["openalex_title"] = data.get("title")
                p["openalex_citations"] = data.get("cited_by_count")
                p["openalex_reference_count"] = len(data.get("referenced_works", []))
                p["fields_of_study"] = [concept["display_name"] for concept in data.get("concepts", [])]
                # Convert abstract_inverted_index → readable abstract
                if isinstance(p["abstract"], dict):
                    words = sorted([(pos, word) for word, positions in p["abstract"].items() for pos in positions])
                    p["abstract"] = " ".join(word for _, word in words)
        except Exception as e:
            print(f"⚠️ OpenAlex fetch failed for {doi}: {e}")
        enriched.append(p)
    return enriched
