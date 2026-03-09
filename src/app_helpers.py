from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

from src.utils import iso_now
from src.gpt_screener_initial import screen_paper
from src.gpt_screener_full import (
    build_prompt as build_full_prompt,
    call_gpt_pdf_from_path,
    parse_tagged_output,
)

RUNS_DIR = Path(__file__).resolve().parent.parent / "_runs"
RUN_FILE = "_run.json"


def save_run(run: dict, run_path: Path) -> None:
    run["updated_at"] = iso_now()
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f" Saved run: {run_path}")


def new_run() -> dict:
    return {
        "created_at": iso_now(),
        "updated_at": iso_now(),
        "stage": "init",
        "inputs": {},
        "stats": {"timings_sec": {}, "counts": {}},
        "papers_by_id": {},
        "categories": {},
        "top_paper_ids": {},
        "errors": [],
        "steps": {},
    }


def ensure_run_shape(run: dict) -> None:
    run.setdefault("inputs", {})
    run.setdefault("stats", {})
    run["stats"].setdefault("timings_sec", {})
    run["stats"].setdefault("counts", {})
    run.setdefault("papers_by_id", {})
    run.setdefault("categories", {})
    run.setdefault("top_paper_ids", {})
    run.setdefault("errors", [])
    run.setdefault("steps", {})


def resolve_run_path(arg: str) -> Path:
    if not arg:
        raise ValueError("Missing run id or path.")

    p = Path(arg)
    if p.exists():
        if p.is_dir():
            candidate = p / RUN_FILE
            if candidate.exists():
                return candidate
            alt = p / "run.json"
            if alt.exists():
                return alt
        return p

    candidate = RUNS_DIR / arg / RUN_FILE
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Run not found: {arg}")


def summarize_run(run: dict) -> None:
    print("Resume summary:")
    print(f" Stage: {run.get('stage')}")
    rq = (run.get("inputs") or {}).get("research_questions") or ""
    if rq:
        first = rq.splitlines()[0]
        print(f" Research Q: {first[:120]}{'...' if len(first) > 120 else ''}")
    counts = (run.get("stats") or {}).get("counts") or {}
    if counts:
        print(f" Counts: {counts}")


def run_step(run: dict, steps: dict, name: str, func, updated_keys, *args, **kwargs):
    t0 = time.time()
    result = func(*args, **kwargs)
    elapsed = round(time.time() - t0, 2)

    run["updated_at"] = iso_now()
    run.setdefault("stats", {}).setdefault("timings_sec", {})[name] = elapsed

    steps[name] = {
        "done": True,
        "updated_keys": updated_keys,
        "elapsed_sec": elapsed,
        "ts": iso_now(),
    }
    return result, steps[name]


def paper_id_from(paper: dict) -> str:
    doi = (paper.get("doi") or "").strip()
    if doi:
        return doi
    title = (paper.get("title") or "").strip().lower()
    authors = paper.get("authors") or []
    if isinstance(authors, list):
        authors = ",".join([str(a).strip().lower() for a in authors])
    else:
        authors = str(authors).strip().lower()
    base = f"{title}|{authors}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    return f"sha1_{digest}"


def select_top_ids(papers_by_id: dict, max_n: int = 50) -> list[str]:
    buckets = {}
    for pid, paper in papers_by_id.items():
        score = (paper.get("screening") or {}).get("relevance_score")
        if score is None:
            continue
        buckets.setdefault(score, []).append(pid)

    selected = []
    for score in sorted(buckets.keys(), reverse=True):
        bucket = buckets[score]
        bucket.sort(key=lambda x: (papers_by_id[x].get("title") or ""))
        if len(selected) + len(bucket) <= max_n:
            selected.extend(bucket)
        elif not selected:
            selected.extend(bucket[:max_n])
            break
        else:
            break
    return selected


def update_counts(run: dict, **kwargs) -> None:
    counts = run.setdefault("stats", {}).setdefault("counts", {})
    for k, v in kwargs.items():
        counts[k] = v


def run_initial_screening(
    run: dict,
    run_path: Path,
    criteria: list[str],
    prompt_txt_path: str = "prompts/screen_initial.txt",
) -> int:
    papers_by_id = run.get("papers_by_id") or {}
    to_screen = [
        (pid, paper)
        for pid, paper in papers_by_id.items()
        if not paper.get("screening")
    ]
    total = len(to_screen)
    if total == 0:
        return 0

    max_workers = int(os.getenv("GPT_MAX_WORKERS") or 0) or min(8, os.cpu_count() or 4)
    print(f" Screening {total} papers with {max_workers} workers...")

    completed = 0
    t0 = time.time()

    def task(pid, paper):
        result = screen_paper(paper, criteria, prompt_txt_path=prompt_txt_path)
        return pid, result

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(task, pid, paper): pid for pid, paper in to_screen}
        for fut in as_completed(futures):
            pid = futures[fut]
            paper = papers_by_id.get(pid, {})
            title = paper.get("title") or pid
            try:
                _, result = fut.result()
                paper["screening"] = result
                score = result.get("relevance_score")
                print(f" {title} — score {score}")
                completed += 1
            except Exception as exc:
                run.setdefault("errors", []).append(
                    {"stage": "initial_screening", "paper_id": pid, "error": str(exc)}
                )
                print(f" ERROR screening {title}: {exc}")

            if completed % 20 == 0:
                save_run(run, run_path)

    elapsed = round(time.time() - t0, 2)
    run.setdefault("stats", {}).setdefault("timings_sec", {})["initial_screening"] = elapsed
    total_screened = sum(1 for p in papers_by_id.values() if p.get("screening"))
    update_counts(run, screened_total=total_screened, errors=len(run.get("errors", [])))
    return completed


def run_full_screening(
    run: dict,
    run_path: Path,
    prompt_txt_path: str = "prompts/screen_full.txt",
) -> int:
    categories = run.get("categories") or {}
    if not categories:
        print(" Skipping full screening: categories are empty.")
        return 0

    question = (run.get("inputs") or {}).get("research_questions") or ""
    if not question.strip():
        print(" Skipping full screening: research question missing.")
        return 0

    category_names = list(categories.keys())
    prompt = build_full_prompt(question, categories)

    top_ids = list((run.get("top_paper_ids") or {}).keys())
    if not top_ids:
        return 0

    completed = 0
    t0 = time.time()

    for pid in top_ids:
        paper = (run.get("papers_by_id") or {}).get(pid) or {}
        top_entry = (run.get("top_paper_ids") or {}).get(pid) or {}
        pdf_path = top_entry.get("pdf_path")
        if not pdf_path:
            continue
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            run.setdefault("errors", []).append(
                {"stage": "full_screening", "paper_id": pid, "error": "PDF missing"}
            )
            continue

        title = paper.get("title") or pid
        print(f" Full screening: {title}")
        try:
            raw = call_gpt_pdf_from_path(prompt, pdf_path)
            parsed = parse_tagged_output(raw, category_names)
            paper["full_screening"] = parsed
            top_entry["full_screening"] = parsed
            run["top_paper_ids"][pid] = top_entry
            completed += 1
        except Exception as exc:
            run.setdefault("errors", []).append(
                {"stage": "full_screening", "paper_id": pid, "error": str(exc)}
            )
            print(f" ERROR full screening {title}: {exc}")

        if completed % 10 == 0:
            save_run(run, run_path)

    elapsed = round(time.time() - t0, 2)
    run.setdefault("stats", {}).setdefault("timings_sec", {})["full_screening"] = elapsed
    total_full = 0
    for pid in top_ids:
        paper = (run.get("papers_by_id") or {}).get(pid) or {}
        if paper.get("full_screening"):
            total_full += 1
    update_counts(run, full_screened=total_full, errors=len(run.get("errors", [])))
    return completed
