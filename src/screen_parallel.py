import os
import time
import random
from threading import Thread, Lock
from queue import Queue
from src.llm_screener import screen_papers_batch
from src.fetch_utils import save_json, load_json


def screen_parallel(
    input_json_path,
    output_json_path,
    prompt_path,
    batch_size=20,
    num_threads=2,
    save_every_batch=True,
    min_delay=3,
    max_delay=8,
):
    """
    Parallel LLM screening of papers with resume, retries, and ETA.

    - Loads enriched papers and any existing screened results.
    - Resumes from where screening left off (based on DOI/title).
    - Runs batches in multiple threads with retry + backoff.
    - Saves progress incrementally (thread-safe).
    - Prints progress and dynamic ETA.

    Args:
        input_json_path (str): Path to enriched JSON.
        output_json_path (str): Path to save screened JSON.
        prompt_path (str): Screening prompt for the LLM.
        batch_size (int): Papers per batch.
        num_threads (int): Parallel threads.
        save_every_batch (bool): Save after each batch.
        min_delay (int): Min retry delay (s).
        max_delay (int): Max retry delay (s).

    Returns:
        str: Path to final screened JSON file.
    """

    # Load enriched papers
    papers = load_json(input_json_path) or []
    print(f"ℹ️ Loaded {len(papers)} papers from {input_json_path}")

    # Load already screened results
    screened_results = load_json(output_json_path) or []

    # Build lookup for already screened
    screened_map = {
        d["paper"].get("doi", d["paper"].get("title")): d
        for d in screened_results if "paper" in d
    }

    # Filter missing
    missing_papers = [p for p in papers if p.get("doi", p.get("title")) not in screened_map]

    if not missing_papers:
        print("✅ All papers already screened — skipping screening step.")
        return output_json_path

    print(f"ℹ️ Resuming screening: {len(missing_papers)} papers left out of {len(papers)} total.")

    # Split into batches
    total_papers = len(missing_papers)
    batches = [missing_papers[i:i + batch_size] for i in range(0, total_papers, batch_size)]
    print(f"ℹ️ Total batches to process: {len(batches)} (batch_size={batch_size})")

    batch_queue = Queue()
    for batch in batches:
        batch_queue.put(batch)

    save_lock = Lock()
    time_lock = Lock()
    processed_count = [0]
    total_time = [0.0]

    def worker(thread_id):
        while not batch_queue.empty():
            batch = batch_queue.get()
            batch_start = time.time()
            screened_batch = []

            for paper in batch:
                success = False
                retries = 0
                while not success and retries < 5:
                    try:
                        screened_result = screen_papers_batch([paper], prompt_path)[0]
                        screened_batch.append(screened_result)
                        success = True
                    except Exception as e:
                        retries += 1
                        print(f"⚠️ Thread-{thread_id} error on paper '{paper.get('title')}', retry {retries}: {e}")
                        time.sleep(random.uniform(min_delay, max_delay))
                if not success:
                    screened_batch.append({"paper": paper, "llm_screening": {"error": str(e)}})

            batch_end = time.time()
            batch_duration = batch_end - batch_start

            # Save thread-safe
            with save_lock:
                for result in screened_batch:
                    key = result["paper"].get("doi", result["paper"].get("title"))
                    screened_map[key] = result
                screened_results = list(screened_map.values())
                if save_every_batch:
                    save_json(screened_results, os.path.dirname(output_json_path), os.path.basename(output_json_path))
                    print(f"💾 Thread-{thread_id} saved {len(screened_results)} papers so far...")

            # Update ETA
            with time_lock:
                processed_count[0] += len(batch)
                total_time[0] += batch_duration
                avg_time_per_paper = total_time[0] / processed_count[0]
                remaining_papers = total_papers - processed_count[0]
                eta_seconds = avg_time_per_paper * remaining_papers
                eta_hours = int(eta_seconds // 3600)
                eta_minutes = int((eta_seconds % 3600) // 60)
                eta_seconds = int(eta_seconds % 60)
                print(f"⏱️ Progress: {processed_count[0]}/{total_papers} papers "
                      f"(ETA ~ {eta_hours}h {eta_minutes}m {eta_seconds}s)")

            batch_queue.task_done()

    threads = []
    for i in range(num_threads):
        t = Thread(target=worker, args=(i + 1,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # Final save
    save_json(list(screened_map.values()), os.path.dirname(output_json_path), os.path.basename(output_json_path))
    print(f"✅ Screening complete! Total screened papers: {len(screened_map)}")
    return output_json_path
