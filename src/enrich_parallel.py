# src/enrich_parallel.py
import os
import json
from threading import Thread, Lock
from queue import Queue
from src.enrich_semanticscholar import enrich_with_semantic_scholar
from src.fetch_utils import save_json, load_json

def enrich_parallel(
    input_json_path,
    output_json_path,
    batch_size=50,
    num_threads=4,
    save_every_batch=True
):
    """
    Incrementally enrich a large JSON of papers using Semantic Scholar in parallel.

    Args:
        input_json_path (str): Path to raw Crossref JSON.
        output_json_path (str): Path to save enriched JSON.
        batch_size (int): Number of papers per enrichment batch.
        num_threads (int): Number of parallel threads.
        save_every_batch (bool): Whether to save after each batch.
    """
    # Load existing papers
    papers = load_json(input_json_path) or []
    print(f"ℹ️ Loaded {len(papers)} papers from {input_json_path}")

    # If output exists, load enriched results and skip already enriched
    enriched_results = load_json(output_json_path) or []
    start_idx = len(enriched_results)
    if start_idx > 0:
        print(f"ℹ️ Resuming enrichment from paper #{start_idx}")
    papers_to_process = papers[start_idx:]

    # Split papers into batches
    batches = [papers_to_process[i:i+batch_size] for i in range(0, len(papers_to_process), batch_size)]
    print(f"ℹ️ Total batches to process: {len(batches)} (batch_size={batch_size})")

    # Queue for dynamic batch assignment
    batch_queue = Queue()
    for batch in batches:
        batch_queue.put(batch)

    save_lock = Lock()  # For thread-safe saving

    def worker():
        while not batch_queue.empty():
            batch = batch_queue.get()
            try:
                enriched_batch = enrich_with_semantic_scholar(batch)
                
                # Incremental save
                with save_lock:
                    enriched_results.extend(enriched_batch)
                    if save_every_batch:
                        save_json(enriched_results, os.path.dirname(output_json_path),
                                  os.path.basename(output_json_path))
                        print(f"💾 Saved {len(enriched_results)} papers so far...")

            except Exception as e:
                print(f"⚠️ Error enriching batch: {e}")
            finally:
                batch_queue.task_done()

    # Start threads
    threads = []
    for _ in range(num_threads):
        t = Thread(target=worker)
        t.start()
        threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join()

    # Final save
    save_json(enriched_results, os.path.dirname(output_json_path), os.path.basename(output_json_path))
    print(f"✅ Enrichment complete! Total enriched papers: {len(enriched_results)}")
    return output_json_path
