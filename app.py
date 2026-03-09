import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import config as config
from src.fetch_arxiv import fetch_papers as fetch_arxiv
from src.fetch_crossref import fetch_papers as fetch_crossref
from src.enrich_openalex import enrich
from src.gpt_client import call_gpt_chat_stream
from src.gpt_criteria import build_criteria_from_question, criteria_to_list
from src.gpt_research_q import (
    build_boolean_query_from_questions,
    clause_to_query,
    dnf_clauses,
    parse_boolean_query,
)
from src.gpt_categories import build_taxonomy_categories
from src.gpt_screener_full import (
    build_prompt as build_full_prompt,
    call_gpt_pdf_from_path,
    parse_tagged_output,
)
from src.gpt_screener_initial import (
    build_prompt as build_initial_prompt,
    count_answers,
    parse_screening_answers,
    relevance_score,
)
from src.utils import deduplicate_papers_by_title_authors, safe_filename


STAGE_ORDER = [
    "init",
    "queries_ready",
    "criteria_done",
    "fetch_enrich_done",
    "initial_screening",
    "top_selection",
    "download_pdfs",
    "categories",
    "full_screening",
    "done",
]


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Dict) -> None:
    data["updated_at"] = now_iso()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def ensure_keys(run_data: Dict) -> None:
    run_data.setdefault("created_at", now_iso())
    run_data.setdefault("updated_at", now_iso())
    run_data.setdefault("stage", "init")
    run_data.setdefault("inputs", {})
    run_data.setdefault("stats", {"timings_sec": {}, "counts": {}})
    run_data.setdefault("papers_by_id", {})
    run_data.setdefault("top_paper_ids", [])
    run_data.setdefault("errors", [])


def stage_index(stage: str) -> int:
    try:
        return STAGE_ORDER.index(stage)
    except ValueError:
        return 0


def advance_stage(run_data: Dict, new_stage: str) -> None:
    if stage_index(new_stage) > stage_index(run_data.get("stage", "init")):
        run_data["stage"] = new_stage


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def compute_paper_id(paper: Dict) -> str:
    doi = str(paper.get("doi", "")).strip()
    if doi:
        return doi.lower()
    title = normalize_text(paper.get("title", ""))
    authors = paper.get("authors", "")
    if isinstance(authors, list):
        authors = ", ".join(authors)
    authors = normalize_text(authors)
    raw = f"{title}|{authors}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def read_multiline_input(prompt: str) -> str:
    print(prompt)
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def safe_input(prompt: str, default: str = "") -> str:
    try:
        return input(prompt)
    except EOFError:
        return default


def expand_boolean_query(boolean_query: str) -> List[str]:
    ast = parse_boolean_query(boolean_query)
    clauses = dnf_clauses(ast)
    queries = [clause_to_query(pos, neg) for pos, neg in clauses]
    seen = set()
    deduped: List[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            deduped.append(q)
    return deduped


def is_valid_pdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 5000:
        return False
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"%PDF"
    except Exception:
        return False


def normalize_criteria(criteria: List[str]) -> List[str]:
    cleaned: List[str] = []
    for item in criteria:
        text = (item or "").strip()
        if not text:
            continue
        upper = text.upper()
        if upper.startswith("INCLUDE:") or upper.startswith("EXCLUDE:"):
            cleaned.append(text)
        else:
            cleaned.append(f"INCLUDE: {text}")
    return cleaned


def fetch_and_enrich(queries: List[str]) -> Tuple[List[Dict], Dict[str, int], Dict[str, float]]:
    timings: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    t0 = time.time()
    arvix_papers = fetch_arxiv(
        queries,
        max_results=getattr(config, "MAX_QUERIES", 10),
        track=False,
    )
    timings["fetch_arxiv"] = time.time() - t0
    counts["fetched_arxiv"] = len(arvix_papers)

    t0 = time.time()
    crossref_papers = fetch_crossref(
        queries,
        max_results=getattr(config, "MAX_QUERIES", 10),
        track=False,
    )
    timings["fetch_crossref"] = time.time() - t0
    counts["fetched_crossref"] = len(crossref_papers)

    combined = arvix_papers + crossref_papers
    deduped = deduplicate_papers_by_title_authors(combined, paper_type="fetched")
    counts["deduped_total"] = len(deduped)

    t0 = time.time()
    enriched = enrich(deduped, track=False)
    timings["enrich_openalex"] = time.time() - t0
    counts["enriched_total"] = len(enriched)

    return enriched, counts, timings


def screen_one_paper(
    paper_id: str, paper: Dict, criteria: List[str]
) -> Tuple[str, Dict]:
    prompt = build_initial_prompt(paper, criteria, "prompts/screen_initial.txt")
    raw = call_gpt_chat_stream(
        messages=[
            {"role": "system", "content": "You are an expert SLR screener assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=800,
    )
    parsed = parse_screening_answers(raw, criteria)
    score = relevance_score(parsed)
    counts = count_answers(parsed)
    return paper_id, {
        "relevance_score": score,
        "counts": counts,
        "answers": parsed,
        "raw": raw,
    }


def select_top_papers(papers_by_id: Dict[str, Dict], max_total: int = 50) -> List[str]:
    buckets: Dict[int, List[str]] = {}
    ordered_ids = list(papers_by_id.keys())
    for pid in ordered_ids:
        paper = papers_by_id[pid]
        screening = paper.get("screening") or {}
        score = screening.get("relevance_score")
        if score is None:
            continue
        buckets.setdefault(int(score), []).append(pid)

    sorted_scores = sorted(buckets.keys(), reverse=True)
    if not sorted_scores:
        return []

    top_ids: List[str] = []
    first_bucket = buckets[sorted_scores[0]]
    if len(first_bucket) > max_total:
        return first_bucket[:max_total]

    for score in sorted_scores:
        bucket = buckets[score]
        if len(top_ids) + len(bucket) <= max_total:
            top_ids.extend(bucket)
        else:
            break
    return top_ids


def run_app() -> None:
    runs_dir = Path("_RUNS")
    runs_dir.mkdir(exist_ok=True)
    run_path: Path | None = None
    run_data: Dict | None = None
    lock = threading.Lock()

    def save():
        if run_path is None or run_data is None:
            return
        with lock:
            save_json(run_path, run_data)

    try:
        resume = safe_input("Resume from existing run? (y/n): ", "n").strip().lower()
        if resume.startswith("y"):
            path_input = safe_input("Path to run.json: ").strip()
            run_path = Path(path_input)
            if not run_path.exists():
                print(f"Run file not found: {run_path}")
                return
            run_data = load_json(run_path)
        else:
            run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            run_path = run_dir / "run.json"
            run_data = {
                "created_at": now_iso(),
                "updated_at": now_iso(),
                "stage": "init",
                "inputs": {},
                "stats": {"timings_sec": {}, "counts": {}},
                "papers_by_id": {},
                "top_paper_ids": [],
                "errors": [],
            }
            save_json(run_path, run_data)

        ensure_keys(run_data)
        run_dir = run_path.parent

        print("")
        print(f"Run file: {run_path}")
        if resume.startswith("y"):
            rq = run_data.get("inputs", {}).get("research_questions", "")
            print(f"Stage: {run_data.get('stage')}")
            if rq:
                print("Research questions:")
                print(rq)
                print("")

        if not run_data["inputs"].get("research_questions"):
            questions = read_multiline_input(
                "Paste research question(s) (end with empty line):"
            )
            if not questions:
                print("No research questions provided. Exiting.")
                return
            run_data["inputs"]["research_questions"] = questions
            advance_stage(run_data, "queries_ready")
            save()

        if not run_data["inputs"].get("boolean_query_used"):
            t0 = time.time()
            suggested = build_boolean_query_from_questions(
                run_data["inputs"]["research_questions"]
            )
            run_data["inputs"]["boolean_query_suggested"] = suggested
            print("")
            print("Suggested Boolean query:")
            print(suggested)
            edited = safe_input("Edit Boolean query (press Enter to keep): ").strip()
            boolean_query = edited or suggested

            while True:
                try:
                    queries = expand_boolean_query(boolean_query)
                except Exception as exc:
                    print(f"Invalid Boolean query: {exc}")
                    boolean_query = safe_input("Edit Boolean query: ").strip()
                    if not boolean_query:
                        boolean_query = suggested
                    continue
                if len(queries) <= 50:
                    break
                print("")
                print(
                    f"Sorry, we can only do max of 50 queries. Current count: {len(queries)}"
                )
                edited = safe_input("Edit the Boolean query (press Enter to keep): ").strip()
                boolean_query = edited or boolean_query

            run_data["inputs"]["boolean_query_used"] = boolean_query
            run_data["inputs"]["queries"] = queries
            run_data["stats"]["timings_sec"]["generate_boolean_query"] = time.time() - t0
            run_data["stats"]["counts"]["queries"] = len(queries)
            advance_stage(run_data, "queries_ready")
            save()

        fetch_thread = None
        if (not run_data["papers_by_id"]) and stage_index(run_data["stage"]) < stage_index("fetch_enrich_done"):
            def fetch_worker():
                enriched, counts, timings = fetch_and_enrich(
                    run_data["inputs"]["queries"]
                )
                papers_by_id: Dict[str, Dict] = {}
                for paper in enriched:
                    pid = compute_paper_id(paper)
                    if pid in papers_by_id:
                        continue
                    paper["paper_id"] = pid
                    papers_by_id[pid] = paper
                with lock:
                    run_data["papers_by_id"] = papers_by_id
                    run_data["stats"]["counts"].update(counts)
                    run_data["stats"]["timings_sec"].update(timings)
                    advance_stage(run_data, "fetch_enrich_done")
                    save_json(run_path, run_data)

            fetch_thread = threading.Thread(target=fetch_worker, daemon=True)
            fetch_thread.start()

        if not run_data["inputs"].get("criteria_used"):
            t0 = time.time()
            suggested = build_criteria_from_question(
                run_data["inputs"]["research_questions"]
            )
            suggested_list = normalize_criteria(criteria_to_list(suggested))
            run_data["inputs"]["criteria_suggested"] = suggested_list
            print("")
            print("Suggested inclusion/exclusion criteria:")
            for line in suggested_list:
                print(f"- {line}")

            edited = read_multiline_input(
                "\nPaste edited criteria (one per line, end with empty line).\n"
                "Press Enter on empty line to keep suggested:"
            )
            if edited.strip():
                parsed = criteria_to_list(edited)
                if not parsed:
                    parsed = [line.strip() for line in edited.splitlines() if line.strip()]
                criteria = normalize_criteria(parsed)
            else:
                criteria = suggested_list

            if not criteria:
                print("No criteria provided. Exiting.")
                return

            run_data["inputs"]["criteria_used"] = criteria
            run_data["stats"]["timings_sec"]["build_criteria"] = time.time() - t0
            advance_stage(run_data, "criteria_done")
            save()

        if fetch_thread is not None:
            fetch_thread.join()
            run_data = load_json(run_path)
            ensure_keys(run_data)

        needs_screening = any(
            "screening" not in paper for paper in run_data["papers_by_id"].values()
        )
        if needs_screening:
            criteria = run_data["inputs"].get("criteria_used", [])
            if not criteria:
                print("Missing criteria. Aborting.")
                return

            t0 = time.time()
            workers = os.getenv("GPT_MAX_WORKERS")
            if workers and workers.isdigit():
                max_workers = max(1, int(workers))
            else:
                cpu = os.cpu_count() or 2
                max_workers = min(8, max(2, cpu))

            paper_items = list(run_data["papers_by_id"].items())
            pending = [item for item in paper_items if "screening" not in item[1]]
            errors = 0

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(screen_one_paper, pid, paper, criteria): (pid, paper)
                    for pid, paper in pending
                }
                for future in as_completed(futures):
                    pid, paper = futures[future]
                    try:
                        _, screening = future.result()
                        with lock:
                            run_data["papers_by_id"][pid]["screening"] = screening
                            save_json(run_path, run_data)
                        title = paper.get("title", "N/A")
                        print(f"{title} - {screening.get('relevance_score')}")
                    except Exception as exc:
                        errors += 1
                        run_data["errors"].append(
                            {"stage": "initial_screening", "paper_id": pid, "error": str(exc)}
                        )
                        save()

            run_data["stats"]["timings_sec"]["initial_screening"] = time.time() - t0
            run_data["stats"]["counts"]["screened_total"] = len(run_data["papers_by_id"])
            if errors:
                run_data["stats"]["counts"]["errors"] = (
                    run_data["stats"]["counts"].get("errors", 0) + errors
                )
            advance_stage(run_data, "initial_screening")
            save()

        if not run_data.get("top_paper_ids"):
            t0 = time.time()
            top_ids = select_top_papers(run_data["papers_by_id"], max_total=50)
            run_data["top_paper_ids"] = top_ids
            run_data["stats"]["timings_sec"]["top_selection"] = time.time() - t0
            run_data["stats"]["counts"]["top_selected"] = len(top_ids)
            advance_stage(run_data, "top_selection")
            save()

        needs_pdfs = False
        for pid in run_data["top_paper_ids"]:
            paper = run_data["papers_by_id"].get(pid, {})
            pdf_path = paper.get("pdf_path")
            if not pdf_path or not is_valid_pdf(Path(pdf_path)):
                needs_pdfs = True
                break
        if needs_pdfs:
            t0 = time.time()
            top_papers = [
                {"paper": run_data["papers_by_id"][pid]}
                for pid in run_data["top_paper_ids"]
                if pid in run_data["papers_by_id"]
            ]
            if top_papers:
                try:
                    from src.pdf_downloader import download_pdfs as _download_pdfs
                except Exception as exc:
                    run_data["errors"].append(
                        {"stage": "download_pdfs", "error": f"PDF download unavailable: {exc}"}
                    )
                    save()
                else:
                    _download_pdfs(top_papers, output_dir=run_dir)

            downloaded = 0
            for pid in run_data["top_paper_ids"]:
                paper = run_data["papers_by_id"].get(pid, {})
                title = paper.get("title", "unknown")
                pdf_path = run_dir / f"{safe_filename(title)}.pdf"
                if is_valid_pdf(pdf_path):
                    paper["pdf_path"] = str(pdf_path)
                    downloaded += 1
            run_data["stats"]["timings_sec"]["download_pdfs"] = time.time() - t0
            run_data["stats"]["counts"]["pdf_downloaded"] = downloaded
            advance_stage(run_data, "download_pdfs")
            save()

        needs_categories = False
        for pid in run_data["top_paper_ids"]:
            if "categories" not in run_data["papers_by_id"].get(pid, {}):
                needs_categories = True
                break

        if needs_categories:
            t0 = time.time()
            question = run_data["inputs"].get("research_questions", "").strip()
            abstracts: List[str] = []
            for pid in run_data["top_paper_ids"]:
                paper = run_data["papers_by_id"].get(pid, {})
                abstract = paper.get("abstract") or ""
                if abstract.strip():
                    abstracts.append(abstract.strip())

            if question and abstracts:
                categories = build_taxonomy_categories(question, abstracts)
                for pid in run_data["top_paper_ids"]:
                    run_data["papers_by_id"][pid]["categories"] = categories
                run_data["stats"]["timings_sec"]["build_categories"] = time.time() - t0
                advance_stage(run_data, "categories")
                save()

        needs_full = False
        for pid in run_data["top_paper_ids"]:
            paper = run_data["papers_by_id"].get(pid, {})
            if paper.get("pdf_path") and "full_screening" not in paper:
                needs_full = True
                break
        if needs_full:
            t0 = time.time()
            question = run_data["inputs"].get("research_questions", "").strip()
            for pid in run_data["top_paper_ids"]:
                paper = run_data["papers_by_id"].get(pid, {})
                if "full_screening" in paper:
                    continue
                pdf_path = paper.get("pdf_path")
                categories = paper.get("categories") or {}
                if not pdf_path or not categories:
                    continue
                try:
                    prompt = build_full_prompt(question, categories)
                    raw = call_gpt_pdf_from_path(prompt, Path(pdf_path))
                    parsed = parse_tagged_output(raw, list(categories.keys()))
                    paper["full_screening"] = parsed
                    save()
                except KeyboardInterrupt:
                    save()
                    print("")
                    print("Interrupted during full screening. Progress saved.")
                    return
                except Exception as exc:
                    run_data["errors"].append(
                        {"stage": "full_screening", "paper_id": pid, "error": str(exc)}
                    )
                    save()

            run_data["stats"]["timings_sec"]["full_screening"] = time.time() - t0
            run_data["stats"]["counts"]["full_screened"] = len(
                [pid for pid in run_data["top_paper_ids"]
                 if "full_screening" in run_data["papers_by_id"].get(pid, {})]
            )
            advance_stage(run_data, "full_screening")
            save()

        advance_stage(run_data, "done")
        save()
        print("")
        print("Run complete.")
    except KeyboardInterrupt:
        save()
        print("")
        print("Interrupted. Progress saved.")
        return


if __name__ == "__main__":
    run_app()

'''
**App.py Plan (Original Flow + Option A JSON Structure)**

**Summary**
Implement `app.py` as the interactive orchestrator. Use `_RUNS/<run_id>/run.json` as the single source of truth, with all papers stored in `papers_by_id`, and top selection tracked by `top_paper_ids`. PDFs are saved in the same run folder.

**Run JSON Structure (core)**
```json
{
  "created_at": "2026-03-09T12:34:56",
  "updated_at": "2026-03-09T12:40:12",
  "stage": "initial_screening",

  "inputs": {
    "research_questions": "Line 1...\nLine 2...\nLine 3...",
    "boolean_query_suggested": "\"long video\" AND (retrieval OR search) AND (multimodal OR audio)",
    "boolean_query_used": "\"long video\" AND (retrieval OR search) AND (multimodal OR audio)",
    "queries": [
      "\"long video\" retrieval multimodal",
      "\"long video\" search audio"
    ],
    "criteria_suggested": [
      "INCLUDE: Does the study focus on AI systems for long-form video?",
      "EXCLUDE: Is it only short-form video?"
    ],
    "criteria_used": [
      "INCLUDE: Does the study focus on AI systems for long-form video?",
      "EXCLUDE: Is it only short-form video?"
    ]
  },

  "stats": {
    "timings_sec": {
      "generate_boolean_query": 6.2,
      "fetch_arxiv": 120.4,
      "fetch_crossref": 95.7,
      "enrich_openalex": 210.1,
      "initial_screening": 980.5,
      "top_selection": 0.4,
      "download_pdfs": 140.2,
      "build_categories": 18.7,
      "full_screening": 1880.6
    },
    "counts": {
      "queries": 12,
      "fetched_arxiv": 850,
      "fetched_crossref": 630,
      "deduped_total": 1260,
      "enriched_total": 1260,
      "screened_total": 1260,
      "top_selected": 50,
      "pdf_downloaded": 47,
      "full_screened": 47,
      "errors": 3
    }
  },

  "papers_by_id": {
    "10.1234/abc": {
      "paper_id": "10.1234/abc",
      "title": "Paper title",
      "authors": ["A. Author", "B. Author"],
      "abstract": "Abstract...",
      "published": "2024-05-01",
      "publisher": "Publisher",
      "doi": "10.1234/abc",
      "link": "https://doi.org/10.1234/abc",
      "from_query": "\"long video\" retrieval multimodal",

      "enrich": {
        "citations": 12,
        "openalex_id": "https://openalex.org/W123",
        "crossref_lookup": true
      },

      "screening": {
        "relevance_score": 6,
        "answers": [
          { "criterion": "INCLUDE: ...", "answer": "YES" },
          { "criterion": "EXCLUDE: ...", "answer": "NO" }
        ],
        "raw": "C1: YES\nC2: NO\n..."
      },

      "pdf_path": "_RUNS/2026-03-09T12-34-56/Paper title.pdf",

      "categories": {
        "Category 1": "Explanation...",
        "Category 2": "Explanation..."
      },

      "full_screening": {
        "included": true,
        "categories": {
          "Category 1": {
            "paragraph": "Summary paragraph...",
            "quotes": ["Quote 1", "Quote 2"]
          }
        }
      }
    }
  },

  "top_paper_ids": [
    "10.1234/abc"
  ],

  "errors": [
    {
      "stage": "download_pdfs",
      "paper_id": "10.1234/abc",
      "error": "PDF not found"
    }
  ]
}
```

**Chronological Process**
1. **Startup / Resume**
   - Ask if user wants to resume from an existing `run.json`.
   - If yes: load it, show summary (research question, stage, counts), continue from next stage.
   - If no: create `_RUNS/<timestamp>/run.json` with `stage="init"`.

2. **Research Question**
   - Prompt multiline research question(s).
   - Save to `inputs.research_questions`.

3. **Boolean Query**
   - Generate boolean query with `src/gpt_research_q.build_boolean_query_from_questions`.
   - Show it, allow edit. If edited, use the new one.
   - Expand with `boolean_to_queries`.
   - If >50 queries and user presses Enter, reprompt until ≤50.
   - Save suggested and used boolean queries plus query list.

4. **Fetch + Enrich (background)**
   - In background: fetch arXiv + Crossref, dedupe, enrich via OpenAlex.
   - Save results into `papers_by_id` as they arrive (all updates in place).

5. **Inclusion / Exclusion Criteria**
   - Generate criteria via `src/gpt_criteria.build_criteria_from_question`.
   - Show line-based criteria and allow edits (empty input keeps).
   - Save suggested and used criteria lists.

6. **Initial Screening (parallel GPT streaming)**
   - For each paper in `papers_by_id`, run GPT initial screening in parallel (workers via `GPT_MAX_WORKERS` or CPU default).
   - Print `title — score` when each paper completes.
   - Store results in place at `papers_by_id[id].screening`.

7. **Top Selection (≤50)**
   - Bucket by `relevance_score` descending.
   - Add full buckets while total ≤50.
   - If top bucket alone >50, cap to first 50.
   - Save IDs to `top_paper_ids`.

8. **Download PDFs**
   - Download PDFs only for `top_paper_ids`.
   - Save `pdf_path` in each top paper entry.

9. **Categories**
   - Use abstracts from top papers to build taxonomy via `gpt_categories`.
   - Save categories map on top papers.

10. **Full PDF Screening**
   - For each top paper with PDF, call `gpt_screener_full`.
   - Store output in place at `papers_by_id[id].full_screening`.

11. **Finalize**
   - Set `stage="done"` and finalize `stats`.

**Test Plan**
1. New run end-to-end with ≤50 queries.
2. >50 queries triggers reprompt until ≤50.
3. Resume mid-screening continues correctly.
4. Top bucket >50 caps to 50.
5. PDFs only for top papers, and `pdf_path` saved.
6. Full screening writes results into the same objects.

**Assumptions**
1. `paper_id = DOI if available; else hash(title+authors)`.
2. Only `papers_by_id` is authoritative; top set is represented by `top_paper_ids`.
3. PDFs and `run.json` live in the same run folder.

'''