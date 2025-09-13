# src/process_crossref.py
import os
from src.fetch_utils import load_json, save_json
from src.fetch_crossref import fetch_papers as fetch_crossref
from src.fetch_utils import resumable_fetch
from src.enrich_parallel import enrich_parallel
from src.screen_crossref import screen_sequential
from src.summarizer import summarize_screened
from src.summarize_crossref_md import summarize_crossref
import config 

def process_crossref():
    """Full Crossref pipeline: fetch, enrich, sequential screening with deduplication, fix raw outputs, summarize."""

    # --- Paths ---
    fetched_path = os.path.join(config.FETCHED_PAPERS_FOLDER, "Crossref_latest_latest.json")
    enriched_path = os.path.join(config.FETCHED_PAPERS_FOLDER, "Crossref_enriched.json")
    screened_path = os.path.join(config.SCREENED_PAPERS_FOLDER, "screened_crossref.json")
    summary_path = os.path.join(config.SUMMARY_FOLDER, "Crossref_summary.md")

    # --- Step 1: Fetch Crossref papers (resumable) ---
    print("\n⚡ Step 1: Fetching Crossref papers...")
    if os.path.exists(fetched_path):
        print(f"⏭️ Fetched file exists: {fetched_path}")
        papers = load_json(fetched_path)
        print(f"Loaded {len(papers)} papers from fetched file.")
    else:
        papers = resumable_fetch(
            fetch_fn=fetch_crossref,
            queries=config.QUERIES,
            save_folder=config.FETCHED_PAPERS_FOLDER,
            save_name_prefix="Crossref_latest",
            max_results=config.MAX_QUERIES,
            per_query_save=5,
            enrich_fn=None
        )
        print(f"✅ Fetch complete → {fetched_path} ({len(papers)} papers)")

    # --- Step 2: Deduplicate fetched papers ---
    print("\n⚡ Step 2: Deduplicating fetched papers...")
    unique_papers = {p.get("doi", p.get("title")): p for p in (papers or [])}.values()
    save_json(list(unique_papers), os.path.dirname(fetched_path), os.path.basename(fetched_path))
    print(f"✅ Deduplicated fetched papers: {len(papers or [])} → {len(unique_papers)}")

    # --- Step 3: Parallel enrichment ---
    print("\n⚡ Step 3: Enriching papers...")
    if os.path.exists(enriched_path):
        print(f"⏭️ Enriched file exists: {enriched_path}")
        enriched_papers = load_json(enriched_path)
        print(f"Loaded {len(enriched_papers)} enriched papers.")
    else:
        enrich_parallel(
            input_json_path=fetched_path,
            output_json_path=enriched_path,
            batch_size=50,
            num_threads=4,
            save_every_batch=True
        )
        enriched_papers = load_json(enriched_path) or []
        print(f"✅ Enrichment complete → {enriched_path} ({len(enriched_papers)} papers)")

    # --- Step 4: Sequential LLM screening ---
    print("\n⚡ Step 4: Sequential LLM screening...")
    screen_sequential(
        input_json_path=enriched_path,
        output_json_path=screened_path,
        prompt_path=os.path.join("data", "screening_prompt.txt"),
        batch_size=10
    )

    # --- Step 4.1: Verify/load screened papers ---
    if os.path.exists(screened_path):
        screened_papers = load_json(screened_path)
        print(f"✅ Loaded {len(screened_papers)} screened papers.")
    else:
        screened_papers = []
        print("⚠️ No screened papers found.")

    # --- Step 5: Summarization with LLM ---
    print("\n📊 Step 5: Summarizing screened papers with LLM...")
    if os.path.exists(screened_path):
        summary = summarize_screened(screened_path)
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"✅ LLM summary saved → {summary_path}")
    else:
        print("⚠️ No screened file available for LLM summarization.")

    # --- Step 6: Summarization without LLM ---
    print("\n📊 Step 6: Summarizing screened papers without LLM...")
    summarize_crossref(
        input_file=screened_path,
        output_file=summary_path.replace(".md", "_noLLM.md")
    )
    print(f"✅ Summary without LLM saved → {summary_path.replace('.md', '_noLLM.md')}\n")