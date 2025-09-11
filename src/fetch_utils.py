# src/fetch_utils.py
import os
import time
import json
from requests.exceptions import ReadTimeout, RequestException

def save_json(data, folder, filename):
    """Save JSON to folder and filename."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path

def load_json(path):
    """Load JSON file if it exists."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def resumable_fetch(fetch_fn, queries, save_folder, save_name_prefix, max_results=100, per_query_save=10, retries=3, delay=1, enrich_fn=None):
    """
    Incremental/resumable fetch utility for Crossref or other sources.

    Arguments:
    - fetch_fn: function to fetch papers per query
    - queries: list of query strings
    - save_folder: folder to save intermediate JSON
    - save_name_prefix: prefix for saved JSON file
    - max_results: maximum results per query
    - per_query_save: save JSON every N queries
    - retries: number of retries per query
    - delay: seconds to wait between queries
    - enrich_fn: optional function to enrich fetched papers (e.g., Semantic Scholar)
    """
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"{save_name_prefix}_latest.json")

    # Load existing partial results if any
    all_papers = load_json(save_path) or []
    start_idx = len(all_papers)
    print(f"ℹ️ Starting fetch from query {start_idx + 1}/{len(queries)}")

    for i, query in enumerate(queries[start_idx:], start=start_idx):
        attempt = 0
        while attempt < retries:
            try:
                new_papers = fetch_fn([query], max_results=max_results)
                print(f"> ✅ Fetched {len(new_papers)} papers for query {i+1}: {query}")
                all_papers.extend(new_papers)
                break
            except (ReadTimeout, RequestException) as e:
                attempt += 1
                print(f"⚠️ Timeout/Error for query {i+1} attempt {attempt}/{retries}: {e}")
                time.sleep(delay)
        else:
            print(f"❌ Failed query after {retries} attempts: {query}")

        # Partial save every per_query_save queries
        if (i + 1) % per_query_save == 0:
            save_json(all_papers, save_folder, f"{save_name_prefix}_latest.json")
            print(f"💾 Partial save after query {i+1}: {save_name_prefix}_latest.json")

        time.sleep(delay)  # optional small delay to avoid API throttling

    # Final save
    if enrich_fn:
        print(f"⚡ Applying enrichment for {len(all_papers)} papers...")
        all_papers = enrich_fn(all_papers)

    final_path = save_json(all_papers, save_folder, f"{save_name_prefix}_latest.json")
    print(f"✅ Fetch complete → {final_path} (Total papers: {len(all_papers)})")
    return final_path
