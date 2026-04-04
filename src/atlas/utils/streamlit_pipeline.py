from __future__ import annotations

import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable

from atlas.inital_fetch.enrich_openalex import enrich as enrich_openalex
from atlas.inital_fetch.fetch_crossref import fetch_papers as fetch_crossref
from atlas.inital_fetch.fetch_semanticscholar import fetch_papers as fetch_semanticscholar
from atlas.inital_screen.gpt_screener_initial import screen_paper
from atlas.read_paper.ieee_client import fetch_ieee_papers as fetch_ieee
from atlas.read_paper.pdf_downloader import download_pdfs
from atlas.utils.app_helpers import (
    save_run,
    select_top_ids,
    set_timing,
    sync_prisma_from_top_papers,
    update_counts,
    update_prisma,
)
from atlas.utils.streamlit_helpers import build_initial_results_df
from atlas.utils.utils import deduplicate_papers_by_title_authors, safe_filename

IEEE_HARDCODED_QUERY = (
    '((All:"semantic video retrieval") OR (All:"video clip retrieval") OR '
    '(All:"video question answering") OR (All:"video QA") OR '
    '(All:"multimodal video") OR (All:"long video retrieval") OR '
    '(All:"video search"))AND ((All:"action recognition") OR '
    '(All:"context-aware") OR (All:"object detection") OR '
    '(All:"audio-visual") OR (All:"speech recognition")) AND '
    '((All:"natural language query") OR (All:"semantic query") OR '
    '(All:"language model") OR (All:LLM) OR (All:NLP))'
)


class StreamlitLogSink(io.StringIO):
    def __init__(self, placeholder=None):
        super().__init__()
        self.placeholder = placeholder

    def write(self, s: str) -> int:
        written = super().write(s)
        if self.placeholder and s:
            lines = self.getvalue().splitlines()[-8:]
            self.placeholder.code("\n".join(lines), language="text")
        return written


def run_initial_screening_live(
    run: dict[str, Any],
    run_path: str | Path,
    criteria: list[str],
    table_callback: Callable[[Any], None] | None = None,
    save_callback: Callable[[dict[str, Any]], None] | None = None,
    max_workers: int | None = None,
) -> bool:
    papers_by_id = run.get("papers_by_id") or {}
    to_screen = [
        (pid, paper)
        for pid, paper in papers_by_id.items()
        if not paper.get("screening")
    ]
    if not to_screen:
        return True

    t0 = time.perf_counter()
    completed = 0
    max_workers = max_workers or min(8, os.cpu_count() or 4)

    def task(pid, paper):
        return pid, screen_paper(paper, criteria)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(task, pid, paper): pid for pid, paper in to_screen}
        for future in as_completed(futures):
            pid = futures[future]
            paper = papers_by_id.get(pid, {})
            try:
                _, result = future.result()
                paper["screening"] = result
                completed += 1
            except Exception as exc:
                run.setdefault("errors", []).append(
                    {"stage": "initial_screening", "paper_id": pid, "error": str(exc)}
                )

            if table_callback and (completed % 5 == 0 or completed == len(to_screen)):
                table_callback(build_initial_results_df(papers_by_id))
            if save_callback and completed % 20 == 0:
                save_callback(run)

    total_screened = sum(1 for paper in papers_by_id.values() if paper.get("screening"))
    excluded_screening = sum(
        1
        for paper in papers_by_id.values()
        if paper.get("screening") and (paper["screening"].get("relevance_score") or 0) <= 0
    )
    update_counts(run, screened_total=total_screened, errors=len(run.get("errors", [])))
    update_prisma(
        run,
        screened=total_screened,
        excluded_screening=excluded_screening,
        sought_retrieval=total_screened - excluded_screening,
    )
    set_timing(run, "initial_screening", time.perf_counter() - t0)
    run["stage"] = "screening_done"
    if save_callback:
        save_callback(run)
    else:
        save_run(run, Path(run_path))
    return True


def fetch_and_enrich(
    queries: list[str],
    run: dict[str, Any],
    app_limits: dict[str, Any],
    log_placeholder=None,
) -> tuple[list[dict[str, Any]], list[str]]:
    sink = StreamlitLogSink(log_placeholder)
    with redirect_stdout(sink):
        fetch_t0 = time.perf_counter()
        per_query_results = app_limits["per_query_results"]
        ieee_max_results = app_limits.get("ieee_max_results")
        expanded_queries = (run.get("inputs") or {}).get("queries") or queries

        source_queries = {
            "ieee": [IEEE_HARDCODED_QUERY],
            "crossref": list(expanded_queries),
            "semanticscholar": list(expanded_queries),
        }
        run.setdefault("inputs", {})["source_queries"] = source_queries

        t0 = time.perf_counter()
        ieee_papers = fetch_ieee(source_queries["ieee"], max_results=ieee_max_results)
        set_timing(run, "fetch_ieee", time.perf_counter() - t0)

        t0 = time.perf_counter()
        crossref_papers = fetch_crossref(
            source_queries["crossref"],
            max_results=per_query_results,
        )
        set_timing(run, "fetch_crossref", time.perf_counter() - t0)

        t0 = time.perf_counter()
        s2_papers = fetch_semanticscholar(
            source_queries["semanticscholar"],
            max_results=per_query_results,
        )
        set_timing(run, "fetch_semanticscholar", time.perf_counter() - t0)

        ident = run.setdefault("prisma", {}).setdefault("identification", {})
        ident["ieee"] = len(ieee_papers)
        ident["crossref"] = len(crossref_papers)
        ident["semanticscholar"] = len(s2_papers)

        update_counts(
            run,
            fetched_ieee=len(ieee_papers),
            fetched_crossref=len(crossref_papers),
            fetched_s2=len(s2_papers),
        )

        # Fetch from all configured sources first, then deduplicate once across the
        # combined pool, and only then enrich the merged records.
        combined = ieee_papers + crossref_papers + s2_papers

        t0 = time.perf_counter()
        deduped = deduplicate_papers_by_title_authors(combined, paper_type="fetched")
        set_timing(run, "dedup", time.perf_counter() - t0)
        update_prisma(run, after_dedup=len(deduped))
        update_counts(run, deduped_total=len(deduped))

        t0 = time.perf_counter()
        enriched = enrich_openalex(deduped)
        set_timing(run, "enrich", time.perf_counter() - t0)
        update_counts(run, enriched_total=len(enriched))
        set_timing(run, "fetch_total", time.perf_counter() - fetch_t0)

    return enriched, sink.getvalue().splitlines()


def run_full_text_step(
    run: dict[str, Any],
    run_path: str | Path,
    app_limits: dict[str, int],
) -> list[str]:
    sink = StreamlitLogSink()
    run_path = Path(run_path)
    with redirect_stdout(sink):
        total_t0 = time.perf_counter()
        papers_by_id = run.get("papers_by_id") or {}

        t0 = time.perf_counter()
        top_ids = select_top_ids(
            papers_by_id,
            max_n=app_limits["top_n"],
            min_score=3,
        )
        set_timing(run, "top_selection", time.perf_counter() - t0)

        run["top_paper_ids"] = {
            pid: {"title": papers_by_id[pid].get("title") or pid}
            for pid in top_ids
        }
        update_counts(run, top_selected=len(top_ids))
        save_run(run, run_path)

        pdf_dir = run_path.parent / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        download_list = [{"paper": papers_by_id[pid]} for pid in top_ids]
        if download_list:
            t0 = time.perf_counter()
            download_pdfs(download_list, pdf_dir)
            set_timing(run, "pdf_download", time.perf_counter() - t0)
        else:
            set_timing(run, "pdf_download", 0.0)

        not_retrieved = 0
        for pid in top_ids:
            title = papers_by_id[pid].get("title") or pid
            pdf_path = pdf_dir / f"{safe_filename(title)}.pdf"
            if pdf_path.exists():
                run["top_paper_ids"][pid]["pdf_path"] = str(pdf_path)
            else:
                not_retrieved += 1

        update_prisma(run, sought_retrieval=len(top_ids), not_retrieved=not_retrieved)
        sync_prisma_from_top_papers(run)
        set_timing(run, "pdf_retrieval_total", time.perf_counter() - total_t0)
        run["stage"] = "proxy_download_done"
        save_run(run, run_path)

    return sink.getvalue().splitlines()
