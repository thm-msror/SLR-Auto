import requests
import urllib.parse
import os
import re
from bs4 import BeautifulSoup
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

def enrich_ieee_web(paper):
    """Try to enrich IEEE papers by scraping IEEE Xplore public pages."""
    try:
        # Check if it's an IEEE paper by URL or DOI pattern
        link = paper.get("link", "")
        doi = paper.get("doi", "")
        
        is_ieee = ("ieeexplore.ieee.org" in link or 
                  doi.startswith("10.1109") or 
                  doi.startswith("10.1108"))
        
        if not is_ieee:
            return None
            
        # Try to get IEEE Xplore URL
        ieee_url = None
        if "ieeexplore.ieee.org" in link:
            ieee_url = link
        elif doi.startswith("10.1109"):
            # Convert DOI to IEEE Xplore URL
            ieee_url = f"https://ieeexplore.ieee.org/document/{doi.split('/')[-1]}"
        
        if ieee_url:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            r = requests.get(ieee_url, headers=headers, timeout=15)
            if r.status_code == 200:
                soup = BeautifulSoup(r.content, 'html.parser')
                
                # Extract abstract
                abstract_elem = soup.find('div', class_='abstract-text')
                abstract = abstract_elem.get_text().strip() if abstract_elem else ""
                
                # Extract citation count (if available)
                citations = 0
                citation_elem = soup.find('span', class_='citation-count')
                if citation_elem:
                    citation_text = citation_elem.get_text()
                    citation_match = re.search(r'(\d+)', citation_text)
                    if citation_match:
                        citations = int(citation_match.group(1))
                
                if abstract:
                    return {
                        "abstract": abstract,
                        "citations": citations,
                        "ieee_web_lookup": True,
                        "ieee_url": ieee_url
                    }
    except Exception as e:
        print(f" IEEE web enrichment failed: {e}")
    return None

def enrich_elsevier_web(paper):
    """Try to enrich Elsevier papers by scraping ScienceDirect public pages."""
    try:
        # Check if it's an Elsevier paper by URL or DOI pattern
        link = paper.get("link", "")
        doi = paper.get("doi", "")
        
        is_elsevier = ("sciencedirect.com" in link or 
                       "elsevier.com" in link or
                       doi.startswith("10.1016"))
        
        if not is_elsevier:
            return None
            
        # Try to get ScienceDirect URL
        sciencedirect_url = None
        if "sciencedirect.com" in link:
            sciencedirect_url = link
        elif doi.startswith("10.1016"):
            # Convert DOI to ScienceDirect URL
            sciencedirect_url = f"https://www.sciencedirect.com/science/article/pii/{doi.split('/')[-1]}"
        
        if sciencedirect_url:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            r = requests.get(sciencedirect_url, headers=headers, timeout=15)
            if r.status_code == 200:
                soup = BeautifulSoup(r.content, 'html.parser')
                
                # Extract abstract
                abstract_elem = soup.find('div', class_='abstract')
                if not abstract_elem:
                    abstract_elem = soup.find('div', class_='abstract-content')
                abstract = abstract_elem.get_text().strip() if abstract_elem else ""
                
                # Extract citation count (if available)
                citations = 0
                citation_elem = soup.find('span', class_='citation-count')
                if citation_elem:
                    citation_text = citation_elem.get_text()
                    citation_match = re.search(r'(\d+)', citation_text)
                    if citation_match:
                        citations = int(citation_match.group(1))
                
                if abstract:
                    return {
                        "abstract": abstract,
                        "citations": citations,
                        "elsevier_web_lookup": True,
                        "sciencedirect_url": sciencedirect_url
                    }
    except Exception as e:
        print(f" Elsevier web enrichment failed: {e}")
    return None

def enrich(papers, track=None):
    enriched = []
    
    track_dir = None
    if track:
        track_dir = "track" if track is True else str(track)
        os.makedirs(track_dir, exist_ok=True)
        # all_papers_dir = save_json(papers, track_dir, ".all_papers_to_enrich") 

    # Import config to check fallback settings
    try:
        import config as config
        enable_crossref = getattr(config, "ENABLE_CROSSREF_FALLBACK", True)
        enable_ieee_web = getattr(config, "ENABLE_IEEE_WEB_FALLBACK", True)
        enable_elsevier_web = getattr(config, "ENABLE_ELSEVIER_WEB_FALLBACK", True)
    except Exception:
        enable_crossref = True
        enable_ieee_web = True
        enable_elsevier_web = True

    for i, p in enumerate(papers):

        print(f'Enriching paper {i+1}: {p.get("title")} {p.get("doi")}')
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
                print(f" OpenAlex DOI lookup failed for {doi}: {e}")

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
                    print(f" OpenAlex title search failed for '{title}': {e}")

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
                print(f" DataCite fetch failed for {doi}: {e}")

        # --- 4. Enhanced Fallbacks: Free alternatives ---
        if not has_openalex_data:
            # Try CrossRef (free API, no key required)
            if enable_crossref:
                crossref_data = enrich_crossref(p)
                if crossref_data:
                    p.update(crossref_data)
                    print("  ✓ Enriched via CrossRef")
            
            # Try IEEE web scraping for IEEE papers
            if enable_ieee_web:
                ieee_data = enrich_ieee_web(p)
                if ieee_data:
                    p.update(ieee_data)
                    print("  ✓ Enriched via IEEE web")
            
            # Try Elsevier web scraping for Elsevier papers
            if enable_elsevier_web:
                elsevier_data = enrich_elsevier_web(p)
                if elsevier_data:
                    p.update(elsevier_data)
                    print("  ✓ Enriched via Elsevier web")

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

        # Save backup per query
        if track_dir and i % 100 == 0:
            save_checkpoint(papers[i:], track_dir, ".all_papers_to_enrich_remaining")
            save_checkpoint(enriched, track_dir, ".all_papers_enriched")

    return enriched
