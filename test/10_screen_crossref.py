# src/screen_crossref.py
import os
import time
import random
import json
from src.llm_screener import screen_papers_batch  # batch version
from src.fetch_utils import save_json, load_json

def screen_sequential(
    input_json_path,
    output_json_path,
    prompt_path,
    batch_size=10,
    min_delay=2,
    max_delay=5,
    save_every_batch=True
):
    """
    Sequential LLM screening of enriched Crossref papers with
    terminal printing, deduplication, batch processing, raw output parsing,
    and incremental saving.
    """

    # --- Load enriched papers ---
    papers = load_json(input_json_path) or []
    print(f"ℹ️ Loaded {len(papers)} papers from {input_json_path}")

    # --- Load already screened papers if present ---
    screened_results = load_json(output_json_path) or []

    # --- Deduplicate already-screened papers ---
    screened_map = {}
    for d in screened_results:
        key = d["paper"].get("doi", d["paper"].get("title"))
        screened_map[key] = d
    print(f"ℹ️ Already screened papers (after dedup): {len(screened_map)}")

    # --- Filter papers that still need screening ---
    missing_papers = [p for p in papers if p.get("doi", p.get("title")) not in screened_map]
    total_papers = len(missing_papers)
    if total_papers == 0:
        print("✅ All papers already screened — skipping.")
        return output_json_path

    print(f"⚡ Screening {total_papers} papers sequentially...")

    processed_count = 0
    total_time = 0.0

    for i in range(0, total_papers, batch_size):
        batch = missing_papers[i:i + batch_size]
        batch_start_time = time.time()

        # --- Screen batch ---
        try:
            llm_results = screen_papers_batch(batch, prompt_path)
        except Exception as e:
            print(f"⚠️ Batch screening failed: {e}")
            llm_results = [{"error": True, "raw_output": str(e)} for _ in batch]

        # --- Process each paper in batch ---
        for paper, llm_result in zip(batch, llm_results):
            print(f"🎯 Screening paper: {paper.get('title', 'Unknown title')}")

            # --- Parse raw output if error or failed JSON ---
            if llm_result.get("error") and "raw_output" in llm_result:
                try:
                    parsed = json.loads(llm_result["raw_output"])
                    llm_result = {k: parsed.get(k) for k in [
                        "review_paper",
                        "included",
                        "relevance",
                        "reason_of_relevance",
                        "key_technologies",
                        "modalities",
                        "task_type",
                        "datasets",
                        "application",
                        "limitations",
                        "notes"
                    ]}
                except Exception:
                    llm_result = {
                        "review_paper": False,
                        "included": False,
                        "relevance": 0,
                        "reason_of_relevance": "LLM failed and raw output could not be parsed.",
                        "key_technologies": [],
                        "modalities": [],
                        "task_type": None,
                        "datasets": [],
                        "application": [],
                        "limitations": [],
                        "notes": ""
                    }


            # --- Merge enriched fields ---
            enriched_keys = [
                "title", "authors", "published", "doi", "publisher",
                "link", "from_query", "abstract", "semanticScholar_citations",
                "semanticScholar_refs", "fields_of_study"
            ]
            merged_result = {"paper": {}, "llm_screening": llm_result}
            for key in enriched_keys:
                if key in paper:
                    merged_result["paper"][key] = paper[key]

            # --- Add to map for deduplication ---
            key = merged_result["paper"].get("doi", merged_result["paper"].get("title"))
            screened_map[key] = merged_result

        # --- Incremental save ---
        if save_every_batch:
            save_json(list(screened_map.values()), os.path.dirname(output_json_path), os.path.basename(output_json_path))
            print(f"💾 Saved {len(screened_map)} papers so far (deduplicated).")

        # --- ETA ---
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        processed_count += len(batch)
        total_time += batch_duration
        avg_time_per_paper = total_time / processed_count
        remaining_papers = total_papers - processed_count
        eta_seconds = int(avg_time_per_paper * remaining_papers)
        eta_hours = eta_seconds // 3600
        eta_minutes = (eta_seconds % 3600) // 60
        eta_secs = eta_seconds % 60
        print(f"⏱️ Progress: {processed_count}/{total_papers} papers "
              f"(ETA ~ {eta_hours}h {eta_minutes}m {eta_secs}s)")

    # --- Final save ---
    save_json(list(screened_map.values()), os.path.dirname(output_json_path), os.path.basename(output_json_path))
    print(f"✅ Screening complete! Total screened papers: {len(screened_map)}")

    return output_json_path