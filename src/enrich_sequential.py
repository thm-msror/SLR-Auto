# src/enrich_sequential.py
import os
from src.enrich_openalex import enrich_with_openalex
from src.fetch_utils import save_json, load_json

def enrich(input_json_path, output_json_path, save_every=50):
    """
    Sequentially enrich a JSON of papers using OpenAlex.

    Args:
        input_json_path (str): Path to raw Crossref JSON.
        output_json_path (str): Path to save enriched JSON.
        save_every (int): Save progress every N papers.
    """
    papers = load_json(input_json_path) or []
    print(f"ℹ️ Loaded {len(papers)} papers from {input_json_path}")

    enriched_results = load_json(output_json_path) or []
    start_idx = len(enriched_results)
    if start_idx > 0:
        print(f"ℹ️ Resuming enrichment from paper #{start_idx}")
    papers_to_process = papers[start_idx:]

    for i, paper in enumerate(papers_to_process, start=start_idx + 1):
        try:
            enriched_batch = enrich_with_openalex([paper])
            enriched_results.extend(enriched_batch)
        except Exception as e:
            print(f"⚠️ Error enriching paper #{i} (DOI={paper.get('doi')}): {e}")

        # Incremental save
        if i % save_every == 0 or i == len(papers):
            save_json(
                enriched_results,
                os.path.dirname(output_json_path),
                os.path.basename(output_json_path)
            )
            print(f"💾 Saved {len(enriched_results)} papers so far...")

    # Final save
    save_json(
        enriched_results,
        os.path.dirname(output_json_path),
        os.path.basename(output_json_path)
    )
    print(f"✅ Sequential enrichment complete! Total enriched papers: {len(enriched_results)}")

    return output_json_path
