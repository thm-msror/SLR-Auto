import requests
import urllib.parse
import os
from src.utils import *

OPENALEX_URL = "https://api.openalex.org/works/"
DATACITE_URL = "https://api.datacite.org/dois/"
CROSSREF_URL = "https://api.crossref.org/works/"

def enrich_crossref(paper):
    """Try to enrich paper using CrossRef API (free, no key required)."""
    try:
        doi = paper.get("doi")
        if not doi:
            return None
            
        url = CROSSREF_URL + doi
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            message = data.get("message", {})
            return {
                "abstract": message.get("abstract", ""),
                "citations": message.get("is-referenced-by-count", 0),
                "reference_count": message.get("reference-count", 0),
                "crossref_lookup": True,
                "publisher": message.get("publisher", ""),
                "subject": message.get("subject", [])
            }
    except Exception as e:
        print(f" CrossRef enrichment failed: {e}")
    return None

def enrich(papers, track=None):
    enriched = []
    
    track_dir = None
    if track:
        track_dir = "track" if track is True else str(track)
        os.makedirs(track_dir, exist_ok=True)

    totalp= len(papers)
    for i, p in enumerate(papers):
        print(f'  Enriching paper {i+1} of {totalp}: {p.get("title")} {p.get("doi")}')
        work_data = None
        doi = p.get("doi")
        has_openalex_data = False

        # --- 1. Try OpenAlex by DOI ---
        if doi:
            url = OPENALEX_URL + f"doi:{doi}"
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    work_data = r.json()
                    p["openalex_lookup"] = "doi"
                    has_openalex_data = True
            except Exception as e:
                print(f"  OpenAlex DOI lookup failed for {doi}: {e}")

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
                            has_openalex_data = True
                except Exception as e:
                    print(f"  OpenAlex title search failed for '{title}': {e}")

        # --- 3. Fallback: DataCite ---
        if not work_data and doi:
            try:
                url = DATACITE_URL + urllib.parse.quote(doi)
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    data = r.json().get("data", {})
                    attributes = data.get("attributes", {})
                    p["abstract"] = attributes.get("descriptions", [{}])[0].get("description")
                    p["publisher"] = attributes.get("publisher")
                    p["datacite_url"] = attributes.get("url")
                    p["datacite_lookup"] = True
            except Exception as e:
                print(f"  DataCite fetch failed for {doi}: {e}")

        # --- 4. Fallback: CrossRef ---
        if not has_openalex_data:
            crossref_data = enrich_crossref(p)
            if crossref_data:
                p.update(crossref_data)
                print("  Enriched via CrossRef")

        # --- If OpenAlex returned something, enrich from it ---
        if work_data:
            p["abstract"] = work_data.get("abstract_inverted_index") or p.get("abstract")
            p["citations"] = work_data.get("cited_by_count")
            p["reference_count"] = len(work_data.get("referenced_works", []))
            p["openalex_id"] = work_data.get("id")

            if isinstance(p["abstract"], dict):
                words = sorted(
                    (pos, word)
                    for word, positions in p["abstract"].items()
                    for pos in positions
                )
                p["abstract"] = " ".join(word for _, word in words)

        enriched.append(p)

        if track_dir and i % 100 == 0:
            save_checkpoint(enriched, track_dir, ".all_papers_enriched")

    return enriched
