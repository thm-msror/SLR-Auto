import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fetch_crossref import fetch_papers
from src.enrich_semanticscholar import enrich_with_semantic_scholar
from src.fetch_utils import resumable_fetch, load_json
import shutil

def test_resumable_fetch():
    queries = ["video question answering"]
    test_folder = "tests/temp_fetch"  # temporary folder for testing
    os.makedirs(test_folder, exist_ok=True)
    
    print("Testing resumable_fetch with fetch_crossref...")

    # --- Run resumable fetch ---
    fetched_path = resumable_fetch(
        fetch_fn=fetch_papers,
        queries=queries,
        save_folder=test_folder,
        save_name_prefix="test_crossref",
        max_results=5,
        per_query_save=1,
        enrich_fn=enrich_with_semantic_scholar
    )

    # --- Load and print results ---
    papers = load_json(fetched_path)
    print(f"\nResumable fetch saved {len(papers)} papers:")
    for i, p in enumerate(papers, 1):
        print(f"{i}. {p['title']} | Citations: {p.get('semanticScholar_citations')} | DOI: {p.get('doi')}")

    # --- Clean up temporary folder ---
    shutil.rmtree(test_folder)
    print("\nTemporary test folder removed.")

if __name__ == "__main__":
    test_resumable_fetch()