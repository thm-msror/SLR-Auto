from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any


def _clean_line(line: str) -> str:
    s = (line or "").strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1].strip()
    return s


def _split_lines(text: str) -> List[str]:
    if not text:
        return []
    lines = []
    for raw in str(text).splitlines():
        cleaned = _clean_line(raw)
        if cleaned:
            lines.append(cleaned)
    return lines


def _md_escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _md_list_in_cell(items: List[str]) -> str:
    if not items:
        return ""
    return "".join([f"- {_md_escape(i)}" for i in items])


def _table(rows: List[List[str]], headers: List[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _prisma_summary(counts: Dict[str, Any]) -> str:
    fetched_arxiv = int(counts.get("fetched_arxiv") or 0)
    fetched_crossref = int(counts.get("fetched_crossref") or 0)
    identified = fetched_arxiv + fetched_crossref
    deduped = int(counts.get("deduped_total") or 0)
    screened = int(counts.get("screened_total") or 0)
    top_selected = int(counts.get("top_selected") or 0)
    pdf_downloaded = int(counts.get("pdf_downloaded") or 0)
    full_screened = int(counts.get("full_screened") or 0)
    errors = int(counts.get("errors") or 0)

    parts = []
    parts.append(
        f"Identified {identified} records (arXiv={fetched_arxiv}, Crossref={fetched_crossref})."
    )
    if deduped:
        parts.append(f"After deduplication, {deduped} unique records remained.")
    if screened:
        parts.append(f"Screened {screened} records at the initial stage.")
    if top_selected:
        parts.append(f"Selected {top_selected} top papers for full screening.")
    if pdf_downloaded:
        parts.append(f"Downloaded {pdf_downloaded} PDFs.")
    if full_screened:
        parts.append(f"Completed full screening for {full_screened} papers.")
    if errors:
        parts.append(f"Logged {errors} errors during the pipeline.")
    return " ".join(parts).strip()


def generate_run_report(run_path: Path, output_path: Path | None = None) -> Path:
    run = json.loads(run_path.read_text(encoding="utf-8"))

    inputs = run.get("inputs") or {}
    stats = run.get("stats") or {}
    timings = stats.get("timings_sec") or {}
    counts = stats.get("counts") or {}
    papers_by_id = run.get("papers_by_id") or {}
    top_papers = run.get("top_paper_ids") or {}
    syntheses = (run.get("syntheses") or {}).get("categories") or {}

    if output_path is None:
        output_path = run_path.parent / "report.md"

    md: List[str] = []

    # Research Questions (start of document)
    md.append("# Research Questions")
    for q in _split_lines(inputs.get("research_questions") or ""):
        md.append(f"- {q}")

    # Boolean queries in separate code blocks
    suggested = inputs.get("boolean_query_suggested") or ""
    used = inputs.get("boolean_query_used") or ""

    md.append("")
    md.append("# Boolean Queries")
    md.append("## Suggested")
    md.append("```text")
    md.append(str(suggested).strip())
    md.append("```")
    md.append("## Used")
    md.append("```text")
    md.append(str(used).strip())
    md.append("```")

    # Criteria table (one row, two columns)
    crit_suggested = inputs.get("criteria_suggested") or []
    crit_used = inputs.get("criteria_used") or []
    md.append("")
    md.append("# Criteria")
    criteria_table = _table(
        [[_md_list_in_cell(crit_suggested), _md_list_in_cell(crit_used)]],
        ["Suggested Criteria", "Used Criteria"],
    )
    md.append(criteria_table)

    # Pipeline statistics
    md.append("")
    md.append("# Pipeline Statistics")
    run_meta_rows = [
        ["Created At", _md_escape(run.get("created_at") or "")],
        ["Updated At", _md_escape(run.get("updated_at") or "")],
        ["Stage", _md_escape(run.get("stage") or "")],
        ["Done", "Yes" if run.get("stage") == "done" else "No"],
    ]
    md.append(_table(run_meta_rows, ["Field", "Value"]))

    if timings:
        md.append("")
        md.append("## Timings (seconds)")
        timing_rows = [[_md_escape(k), _md_escape(v)] for k, v in timings.items()]
        md.append(_table(timing_rows, ["Step", "Seconds"]))

    # Counts
    md.append("")
    md.append("# Counts")
    if counts:
        count_rows = [[_md_escape(k), _md_escape(v)] for k, v in counts.items()]
        md.append(_table(count_rows, ["Metric", "Value"]))
    else:
        md.append("_No counts available._")

    # PRISMA-style flow summary (text only)
    md.append("")
    md.append("# PRISMA-Style Flow Summary")
    md.append(_prisma_summary(counts))

    # Top papers table
    md.append("")
    md.append("# Top Papers")
    if top_papers:
        rows = []
        for pid, entry in top_papers.items():
            paper = papers_by_id.get(pid) or {}
            title = entry.get("title") or paper.get("title") or pid
            published = paper.get("published") or ""
            link = paper.get("link") or ""
            score = ""
            screening = paper.get("screening") or {}
            if screening.get("relevance_score") is not None:
                score = str(screening.get("relevance_score"))
            rows.append(
                [
                    _md_escape(title),
                    _md_escape(published),
                    _md_escape(link),
                    _md_escape(score),
                ]
            )

        md.append(
            _table(rows, ["Title", "Published", "Link", "Relevance Score"])
        )
    else:
        md.append("_No top papers available._")

    # Category syntheses
    md.append("")
    md.append("# Category Syntheses")
    if syntheses:
        for name, text in syntheses.items():
            md.append(f"## {name}")
            md.append((text or "").strip() or "_No synthesis text._")
            md.append("")
    else:
        md.append("_No category syntheses available._")

    output_path.write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")
    print(f" Saved report: {output_path}")
    return output_path
