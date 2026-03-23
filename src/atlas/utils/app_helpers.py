from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from atlas.utils.utils import iso_now
from atlas.inital_screen.gpt_screener_initial import screen_paper
from atlas.results.gpt_synthesis import synthesize_category
from atlas.read_paper.gpt_screener_full import (
    build_prompt as build_full_prompt,
    call_gpt_pdf_from_path,
    parse_tagged_output,
)

RUNS_DIR = Path(__file__).resolve().parents[3] / "data" / "runs"
RUN_FILE = "log.py"


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
        # PRISMA 2020 flow counts — updated throughout the pipeline
        "prisma": {
            "identification": {"ieee": 0, "crossref": 0},
            "after_dedup": 0,
            "screened": 0,
            "excluded_screening": 0,
            "sought_retrieval": 0,
            "not_retrieved": 0,
            "assessed_eligibility": 0,
            "excluded_eligibility": 0,
            "included": 0,
        },
        "papers_by_id": {},
        "categories": {},
        "syntheses": {"categories": {}},
        "top_paper_ids": {},
        "errors": [],
        "steps": {},
    }


def ensure_run_shape(run: dict) -> None:
    run.setdefault("inputs", {})
    run.setdefault("stats", {})
    run["stats"].setdefault("timings_sec", {})
    run["stats"].setdefault("counts", {})
    p = run.setdefault("prisma", {})
    p.setdefault("identification", {"ieee": 0, "crossref": 0})
    p["identification"].setdefault("ieee", 0)
    p["identification"].setdefault("crossref", 0)
    p.setdefault("after_dedup", 0)
    p.setdefault("screened", 0)
    p.setdefault("excluded_screening", 0)
    p.setdefault("sought_retrieval", 0)
    p.setdefault("not_retrieved", 0)
    p.setdefault("assessed_eligibility", 0)
    p.setdefault("excluded_eligibility", 0)
    p.setdefault("included", 0)
    run.setdefault("papers_by_id", {})
    run.setdefault("categories", {})
    run.setdefault("syntheses", {})
    run["syntheses"].setdefault("categories", {})
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


def select_top_ids(papers_by_id: dict, max_n: int = 50, min_score: int = 3) -> list[str]:
    buckets = {}
    for pid, paper in papers_by_id.items():
        score = (paper.get("screening") or {}).get("relevance_score")
        if score is None:
            continue
        # Filter by minimum relevance threshold
        if score < min_score:
            continue
        buckets.setdefault(score, []).append(pid)

    selected = []
    # Sort scores descending
    for score in sorted(buckets.keys(), reverse=True):
        bucket = buckets[score]
        # Secondary sort by title
        bucket.sort(key=lambda x: (papers_by_id[x].get("title") or ""))
        
        if len(selected) + len(bucket) <= max_n:
            selected.extend(bucket)
        else:
            # Fill remaining slots up to max_n
            needed = max_n - len(selected)
            selected.extend(bucket[:needed])
            break
    return selected


def update_counts(run: dict, **kwargs) -> None:
    counts = run.setdefault("stats", {}).setdefault("counts", {})
    for k, v in kwargs.items():
        counts[k] = v


def update_prisma(run: dict, **kwargs) -> None:
    """Update PRISMA 2020 flow counts stored in run['prisma'].

    Supported keys:
        ieee (int): Papers identified from IEEE.
        crossref (int): Papers identified from Crossref.
        after_dedup (int): Papers remaining after deduplication.
        screened (int): Papers sent to initial LLM screening.
        excluded_screening (int): Papers excluded at screening stage.
        sought_retrieval (int): Papers sought for full-text retrieval.
        not_retrieved (int): Papers not retrieved (no PDF found).
        assessed_eligibility (int): Papers assessed for full eligibility.
        excluded_eligibility (int): Papers excluded at eligibility stage.
        included (int): Final papers included in the review.
    """
    prisma = run.setdefault("prisma", {})
    ident = prisma.setdefault("identification", {})
    for k, v in kwargs.items():
        if k in ("ieee", "crossref"):
            ident[k] = v
        else:
            prisma[k] = v


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
                print(f" {title} - score {score}")
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
    # Count papers excluded at screening (relevance_score <= 0)
    excluded_screening = sum(
        1 for p in papers_by_id.values()
        if p.get("screening") and (p["screening"].get("relevance_score") or 0) <= 0
    )
    update_counts(run, screened_total=total_screened, errors=len(run.get("errors", [])))
    update_prisma(
        run,
        screened=total_screened,
        excluded_screening=excluded_screening,
        sought_retrieval=total_screened - excluded_screening,
    )
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
        if top_entry.get("full_screening"):
            continue
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
            try:
                parsed = parse_tagged_output(raw, category_names)
                top_entry["full_screening"] = parsed
                run["top_paper_ids"][pid] = top_entry
                completed += 1
            except Exception as parse_exc:
                # Save debug info
                debug_dir = run_path.parent / "debug_screening_failures"
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_file = debug_dir / f"{pid[:8]}_raw.txt"
                debug_file.write_text(raw, encoding="utf-8")
                raise ValueError(f"{parse_exc} (Raw output saved to {debug_file.name})")
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
        top_entry = (run.get("top_paper_ids") or {}).get(pid) or {}
        if top_entry.get("full_screening"):
            total_full += 1
    excluded_eligibility = sum(
        1 for pid in top_ids
        if (run.get("top_paper_ids") or {}).get(pid, {}).get("full_screening", {}).get("included") is False
    )
    included = total_full - excluded_eligibility
    
    update_counts(run, full_screened=total_full, errors=len(run.get("errors", [])))
    update_prisma(
        run,
        assessed_eligibility=total_full,
        excluded_eligibility=excluded_eligibility,
        included=included,
    )
    return completed


def _extract_year(published: str) -> str:
    if not published:
        return ""
    published = published.strip()
    if len(published) >= 4 and published[:4].isdigit():
        return published[:4]
    match = re.search(r"\b(19|20)\d{2}\b", published)
    return match.group(0) if match else ""


def run_category_synthesis(
    run: dict,
    run_path: Path,
    prompt_path: str = "prompts/synthesize_category.txt",
    model_name: str | None = None,
) -> int:
    categories = run.get("categories") or {}
    if not categories:
        print(" Skipping category synthesis: categories are empty.")
        return 0

    top_papers = run.get("top_paper_ids") or {}
    if not top_papers:
        print(" Skipping category synthesis: no top papers.")
        return 0

    papers_by_id = run.get("papers_by_id") or {}
    syntheses = run.setdefault("syntheses", {}).setdefault("categories", {})

    completed = 0
    for category_name in categories.keys():
        input_strings: List[str] = []

        for paper_id, entry in top_papers.items():
            full = entry.get("full_screening") or {}
            if full.get("included") is False:
                continue

            cat_block = (full.get("categories") or {}).get(category_name) or {}
            paragraph = (cat_block.get("paragraph") or "").strip()
            quotes = cat_block.get("quotes") or []
            if not quotes:
                continue

            if not isinstance(quotes, list):
                quotes = [str(quotes)]
            cleaned_quotes = [str(q).strip() for q in quotes if str(q).strip()]
            if not cleaned_quotes:
                continue

            paper = papers_by_id.get(paper_id) or {}
            title = entry.get("title") or paper.get("title") or paper_id
            published = (paper.get("published") or "").strip()
            year = _extract_year(published)
            title_line = f"\"{title} ({year})\"" if year else f"\"{title}\""

            lines = [title_line, paragraph, "Supporting quotes:"]
            lines.extend([f"- {q}" for q in cleaned_quotes])
            input_strings.append("\n".join(lines).strip())

        if not input_strings:
            continue

        synthesis = synthesize_category(
            items=input_strings,
            category_name=category_name,
            prompt_path=prompt_path,
            model_name=model_name,
        )
        syntheses[category_name] = synthesis
        completed += 1

        if completed % 3 == 0:
            save_run(run, run_path)

    return completed
