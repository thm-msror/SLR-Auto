# src/process_crossref.py
import os
from pathlib import Path
from src.fetch_utils import load_json, save_json, resumable_fetch
from src.fetch_crossref import fetch_papers as fetch_crossref
from src.enrich_openalex import enrich_papers
from src.screen_crossref import screen_sequential
from src.filter_papers import summarize_screened
from src.summarize_crossref_md import summarize_crossref
import config

def process_crossref():
    """Full Crossref pipeline: fetch (Scopus/IEEE/ACM/ArXiv), enrich with OpenAlex, sequential screening, summarization."""

    # --- Paths ---
    latest_path = os.path.join(config.FETCHED_PAPERS_FOLDER, "Crossref_multi_latest.json")
    enriched_path = os.path.join(config.FETCHED_PAPERS_FOLDER, "Crossref_openalex_enriched.json")
    screened_path = os.path.join(config.SCREENED_PAPERS_FOLDER, "screened_crossref_multi.json")
    summary_path = os.path.join(config.SUMMARY_FOLDER, "Crossref_multi_summary.md")

    # --- Step 1: Fetch Crossref papers ---
    print("\nStep 1: Fetching Crossref papers (Scopus, IEEE, ACM, ArXiv)...")
    if os.path.exists(latest_path):
        print(f"Loaded existing latest fetch: {latest_path}")
        papers = load_json(latest_path)
        print(f"Loaded {len(papers)} papers.")
    else:
        papers = resumable_fetch(
            fetch_fn=fetch_crossref,
            queries=config.QUERIES,
            save_folder=config.FETCHED_PAPERS_FOLDER,
            save_name_prefix="Crossref_multi_latest",
            max_results=config.MAX_QUERIES,
            per_query_save=5,
            enrich_fn=None
        )
        print(f"Fetch complete -> {latest_path} ({len(papers)} papers)")

    # --- Step 2: Deduplicate ---
    print("\nStep 2: Deduplicating fetched papers...")
    # Inspect type of first items for sanity
    for i, p in enumerate(papers[:5]):
        print(f"Sample {i}: {type(p)} | {str(p)[:80]}")

    # Deduplicate based on DOI or title
    if papers and isinstance(papers[0], dict):
        unique_papers = {p.get("doi") or p.get("title"): p for p in papers if isinstance(p, dict)}.values()
    else:
        unique_papers = list(set(papers))  # fallback, unlikely with proper JSON

    papers = list(unique_papers)
    save_json(papers, os.path.dirname(latest_path), os.path.basename(latest_path))
    print(f"Deduplicated: {len(papers)} papers saved -> {latest_path}")

    # --- Step 3: Enrich with OpenAlex ---
    print("\nStep 3: Enriching with OpenAlex...")
    if os.path.exists(enriched_path):
        enriched_papers = load_json(enriched_path)
        print(f"Loaded {len(enriched_papers)} enriched papers.")
    else:
        enrich(
            input_json_path=latest_path,
            output_json_path=enriched_path
        )
        enriched_papers = load_json(enriched_path) or []
        print(f"Enrichment complete -> {enriched_path} ({len(enriched_papers)} papers)")

    # --- Step 4: Sequential screening ---
    print("\nStep 4: Sequential LLM screening...")
    screen_sequential(
        input_json_path=enriched_path,
        output_json_path=screened_path,
        prompt_path=os.path.join("data", "screening_prompt.txt"),
        batch_size=10
    )

    # --- Step 5: Summarization ---
    print("\nStep 5: Summarizing screened papers...")
    if os.path.exists(screened_path):
        summary = summarize_screened(screened_path)
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Summary saved -> {summary_path}")
    else:
        print("No screened papers found for summarization.")

    # --- Step 6: No-LLM summary ---
    summarize_crossref(
        input_file=screened_path,
        output_file=summary_path.replace(".md", "_noLLM.md")
    )
    print(f"No-LLM summary saved -> {summary_path.replace('.md', '_noLLM.md')}\n")
