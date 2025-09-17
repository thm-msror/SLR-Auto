import pytest
from src.enrich_openalex import enrich_with_openalex

# Sample papers list with a known DOI that exists in OpenAlex
# You can swap this with any DOI from your dataset
sample_papers = [
    {"doi": "10.1109/siu55565.2022.9864748"}
]

def test_openalex_enrichment():
    enriched = enrich_with_openalex(sample_papers)
    p = enriched[0]

    # Ensure required fields exist
    assert "abstract" in p
    assert isinstance(p["abstract"], str) or p["abstract"] is None

    assert "openalex_title" in p
    assert "openalex_citations" in p
    assert "openalex_references" in p
    assert "fields_of_study" in p

    # Optional: print out values for manual verification
    print("\n--- OpenAlex Enrichment Output ---")
    print(f"Title: {p.get('openalex_title')}")
    print(f"Abstract snippet: {str(p.get('abstract'))[:200]}")
    print(f"Citations: {p.get('openalex_citations')}")
    print(f"Fields of study: {p.get('fields_of_study')[:5]}")

