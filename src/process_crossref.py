# src/process_crossref.py
import os
from src.fetch_utils import load_json, save_json
from src.fetch_crossref import fetch_papers as fetch_crossref
from src.fetch_utils import resumable_fetch
from src.enrich_parallel import enrich_parallel
from src.screen_parallel import screen_parallel
from src.summarizer import summarize_screened
from src.retry_failed_screening import retry_failed_screening
from src.summarize_crossref_md import summarize_crossref
import config

def process_crossref():
    """Full Crossref pipeline: fetch, enrich, screen, verify, deduplicate, retry failed, summarize."""

    # --- Paths ---
    fetched_path = os.path.join(config.FETCHED_PAPERS_FOLDER, "Crossref_latest.json")
    enriched_path = os.path.join(config.FETCHED_PAPERS_FOLDER, "Crossref_enriched.json")
    screened_path = os.path.join(config.SCREENED_PAPERS_FOLDER, "Crossref_screened.json")
    summary_path = os.path.join(config.SUMMARY_FOLDER, "Crossref_summary.md")

    # --- Step 1: Fetch Crossref papers (resumable) ---
    if not os.path.exists(fetched_path):
        print("⚡ Fetching Crossref papers...")
        fetched_path = resumable_fetch(
            fetch_fn=fetch_crossref,
            queries=config.QUERIES,
            save_folder=config.FETCHED_PAPERS_FOLDER,
            save_name_prefix="Crossref_latest",
            max_results=config.MAX_QUERIES,
            per_query_save=5,
            enrich_fn=None
        )
    else:
        print(f"⏭️ Fetched file exists: {fetched_path}")

    # --- Deduplicate fetched papers ---
    print("⚡ Deduplicating fetched papers...")
    papers = load_json(fetched_path)
    unique_papers = {p.get("doi", p.get("title")): p for p in papers}.values()
    save_json(list(unique_papers), os.path.dirname(fetched_path), os.path.basename(fetched_path))
    print(f"✅ Deduplicated fetched papers: {len(papers)} → {len(unique_papers)}")

    # --- Step 2: Parallel enrichment ---
    if not os.path.exists(enriched_path):
        print("⚡ Enriching Crossref papers in parallel...")
        enrich_parallel(
            input_json_path=fetched_path,
            output_json_path=enriched_path,
            batch_size=50,
            num_threads=4,
            save_every_batch=True
        )
    else:
        print(f"⏭️ Enriched file exists: {enriched_path}")

    # --- Deduplicate enriched papers ---
    print("⚡ Deduplicating enriched papers...")
    papers = load_json(enriched_path)
    unique_papers = {p.get("doi", p.get("title")): p for p in papers}.values()
    save_json(list(unique_papers), os.path.dirname(enriched_path), os.path.basename(enriched_path))
    print(f"✅ Deduplicated enriched papers: {len(papers)} → {len(unique_papers)}")

    # --- Step 3: Parallel LLM screening (resume if partial) ---
    print("⚡ Screening Crossref papers with LLM in parallel (resume if needed)...")
    screen_parallel(
        input_json_path=enriched_path,
        output_json_path=screened_path,
        prompt_path=config.LLM_SCREENING_PROMPT_TXT,
        batch_size=30,
        num_threads=3,
        save_every_batch=True,
        min_delay=3,
        max_delay=8
    )

    # --- Step 3a: Verify all enriched papers were screened ---
    print("🔎 Verifying that all enriched papers were screened...")
    enriched = load_json(enriched_path)
    screened = load_json(screened_path)

    enriched_dict = {p.get("doi", p.get("title")): p for p in enriched if p.get("title")}
    if screened and "paper" in screened[0]:
        screened_keys = {d["paper"].get("doi", d["paper"].get("title")) for d in screened if d["paper"].get("title")}
    else:
        screened_keys = {d.get("doi", d.get("title")) for d in screened if d.get("title")}

    missing_keys = set(enriched_dict.keys()) - screened_keys
    if missing_keys:
        print(f"⚠️ {len(missing_keys)} enriched papers missing from screened output → rescreening them now...")
        missing_papers = [enriched_dict[k] for k in missing_keys]

        # Temporary file for missing papers
        missing_input_path = os.path.join(config.FETCHED_PAPERS_FOLDER, "Crossref_missing_enriched.json")
        save_json(missing_papers, os.path.dirname(missing_input_path), os.path.basename(missing_input_path))

        # Rescreen only the missing papers
        screen_parallel(
            input_json_path=missing_input_path,
            output_json_path=screened_path,  # append to existing screened
            prompt_path=config.LLM_SCREENING_PROMPT_TXT,
            batch_size=20,
            num_threads=2,
            save_every_batch=True,
            min_delay=3,
            max_delay=8
        )
    else:
        print("✅ All enriched papers have been screened.")

    # --- Step 3b: Deduplicate screened papers ---
    print("⚡ Deduplicating screened papers before retrying failed ones...")
    screened = load_json(screened_path)
    if screened and "paper" in screened[0]:
        unique_screened = {
            d["paper"].get("doi", d["paper"].get("title")): d
            for d in screened
        }.values()
    else:
        unique_screened = {
            d.get("doi", d.get("title")): d
            for d in screened
        }.values()

    save_json(list(unique_screened), os.path.dirname(screened_path), os.path.basename(screened_path))
    print(f"✅ Deduplicated screened papers: {len(screened)} → {len(unique_screened)}")

    # --- Step 3c: Retry failed papers ---
    # retry_failed_screening(screened_path, batch_size=20, num_threads=2) # retrying failed screenings with errors did not work so commenting this out
    
    # --- Step 3d: Final log & dedup check before summarization ---
    print("🔍 Final verification before summarization...")

    # Load enriched and screened
    enriched = load_json(enriched_path) or []
    screened = load_json(screened_path) or []

    print(f"ℹ️ Total enriched papers: {len(enriched)}")
    print(f"ℹ️ Total screened papers: {len(screened)}")

    # Deduplicate screened by DOI or title
    final_unique_screened_map = {}
    for d in screened:
        if "paper" in d:
            key = d["paper"].get("doi") or d["paper"].get("title")
        else:
            key = d.get("doi") or d.get("title")
        if key:
            final_unique_screened_map[key] = d

    final_unique_screened = list(final_unique_screened_map.values())
    print(f"✅ Deduplicated screened papers internally: {len(screened)} → {len(final_unique_screened)}")

    # Save cleaned screened JSON
    save_json(final_unique_screened, os.path.dirname(screened_path), os.path.basename(screened_path))


    # --- Step 4: Summarization with LLM ---
    print("📊 Summarizing screened Crossref papers...")
    summary = summarize_screened(screened_path)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"✅ Crossref summary saved → {summary_path}")
    
    # --- Step 5: Summarization without LLM ---
    print("📊 Summarizing screened Crossref papers without LLM...")
    summarize_crossref(
        input_file=screened_path,
        output_file=summary_path.replace(".md", "_noLLM.md")
    )