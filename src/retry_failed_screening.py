import os
import time
import random
from threading import Thread, Lock
from queue import Queue
from src.screen_parallel import screen_papers_batch
from src.fetch_utils import save_json, load_json
import config


def retry_failed_screening(screened_json_path, batch_size=20, num_threads=2, min_delay=3, max_delay=8, max_retries=5):
    """
    Retry LLM screening only for failed papers.

    - Loads screened results and finds papers with errors.
    - Retries in parallel with backoff and limited retries.
    - Updates JSON in place with corrected results.
    - Logs progress and ETA per batch.

    Args:
        screened_json_path (str): Path to screened JSON.
        batch_size (int): Papers per batch.
        num_threads (int): Parallel threads.
        min_delay (int): Min retry delay (s).
        max_delay (int): Max retry delay (s).
        max_retries (int): Max attempts per paper.

    Returns:
        None: Saves updated results back to JSON.
    """
    screened = load_json(screened_json_path) or []

    # Collect failed
    failed_papers = []
    for d in screened:
        if "llm_screening" not in d:
            failed_papers.append(d["paper"])
        elif isinstance(d["llm_screening"], dict) and "error" in d["llm_screening"]:
            failed_papers.append(d["paper"])

    # Deduplicate
    failed_map = {}
    for p in failed_papers:
        key = p.get("doi", p.get("title"))
        failed_map[key] = p
    failed_papers = list(failed_map.values())

    if not failed_papers:
        print("✅ No failed papers to retry.")
        return

    print(f"⚡ Rescreening {len(failed_papers)} failed papers...")

    # Split into batches
    batches = [failed_papers[i:i + batch_size] for i in range(0, len(failed_papers), batch_size)]
    batch_queue = Queue()
    for batch in batches:
        batch_queue.put(batch)

    save_lock = Lock()
    screened_map = {d["paper"].get("doi", d["paper"].get("title")): d for d in screened}
    batch_times = []

    def worker(thread_id):
        while not batch_queue.empty():
            batch = batch_queue.get()
            start_time = time.time()
            screened_batch = []

            for paper in batch:
                retries = 0
                success = False
                while not success and retries < max_retries:
                    try:
                        screened_result = screen_papers_batch([paper], config.LLM_SCREENING_PROMPT_TXT)[0]
                        screened_batch.append(screened_result)
                        success = True
                    except Exception as e:
                        retries += 1
                        print(f"⚠️ Thread-{thread_id} error on '{paper.get('title')}', retry {retries}: {e}")
                        time.sleep(random.uniform(min_delay, max_delay))
                if not success:
                    screened_batch.append({"paper": paper, "llm_screening": {"error": "Failed after retries"}})

            # Save thread-safe
            with save_lock:
                for result in screened_batch:
                    key = result["paper"].get("doi", result["paper"].get("title"))
                    screened_map[key] = result
                save_json(list(screened_map.values()), 
                          os.path.dirname(screened_json_path), 
                          os.path.basename(screened_json_path))

            # ETA
            end_time = time.time()
            batch_times.append(end_time - start_time)
            avg_batch_time = sum(batch_times) / len(batch_times)
            remaining_batches = batch_queue.qsize()
            eta_sec = avg_batch_time * remaining_batches
            eta_min, eta_sec = divmod(eta_sec, 60)
            print(f"💾 Thread-{thread_id} finished batch. ETA ≈ {int(eta_min)}m {int(eta_sec)}s, remaining batches: {remaining_batches}")

            batch_queue.task_done()

    threads = []
    for i in range(num_threads):
        t = Thread(target=worker, args=(i + 1,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"✅ Rescreening complete. Updated results saved to {screened_json_path}")
