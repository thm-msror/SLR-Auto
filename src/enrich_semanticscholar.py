import requests

SEM_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/"

def enrich_with_semantic_scholar(papers, fields="title,abstract,authors,citationCount,referenceCount,fieldsOfStudy"):
    enriched = []
    for p in papers:
        if not p.get("doi"):
            enriched.append(p)
            continue
        doi = p["doi"]
        url = SEM_SCHOLAR_URL + f"DOI:{doi}"
        params = {"fields": fields}
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                s2_data = r.json()
                p["abstract"] = s2_data.get("abstract")
                p["semanticScholar_citations"] = s2_data.get("citationCount")
                p["semanticScholar_refs"] = s2_data.get("referenceCount")
                p["fields_of_study"] = s2_data.get("fieldsOfStudy", [])
        except Exception as e:
            print(f"⚠️ Semantic Scholar fetch failed for {doi}: {e}")
        enriched.append(p)
    return enriched