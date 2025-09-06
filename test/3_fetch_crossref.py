import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fetch_crossref import fetch_papers

if __name__ == "__main__":
    queries = ["Automatic Clip Retrieval and Multimodal Analysis in Long-Form Content"]
    results = fetch_papers(queries, max_results=10)
    print(f"Fetched {len(results)} papers.\n")
    for i, paper in enumerate(results[:3], 1):
        print(f"{i}. {paper['title']} ({paper.get('link')})")

