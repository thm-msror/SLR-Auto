from __future__ import annotations

import io
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from atlas.utils.app_helpers import RUN_FILE
from atlas.utils.utils import safe_filename


def build_initial_run_dir_name(run_id: str, app_mode: str) -> str:
    suffix = "-fast" if app_mode == "fast" else ""
    return f".{run_id}{suffix}"


def rename_run_folder_with_title(run: dict[str, Any], run_path: str | Path, title: str) -> Path:
    cleaned_title = safe_filename(title or "", max_len=80).strip()
    current_run_path = Path(run_path)
    if not cleaned_title:
        return current_run_path

    old_run_dir = current_run_path.parent
    target_dir = old_run_dir.parent / f".{_run_timestamp_from_run(run)}-{cleaned_title}"
    if target_dir == old_run_dir:
        return current_run_path

    candidate_dir = target_dir
    suffix = 2
    while candidate_dir.exists():
        candidate_dir = old_run_dir.parent / f"{target_dir.name}-{suffix}"
        suffix += 1

    old_run_dir.rename(candidate_dir)
    new_run_path = candidate_dir / RUN_FILE
    _rewrite_run_local_paths(run, old_run_dir, candidate_dir)
    return new_run_path


def theme_dict_to_text(categories: dict[str, str]) -> str:
    lines = []
    for name, desc in categories.items():
        line = name.strip()
        if desc:
            line = f"{line}: {desc.strip()}"
        if line:
            lines.append(line)
    return "\n".join(lines)


def empty_report_syntheses() -> dict[str, Any]:
    return {
        "title": "",
        "abstract": "",
        "keywords": [],
        "introduction": "",
        "methodology": "",
        "references": "",
        "results": "",
        "discussion": "",
        "conclusion": "",
        "draft_report": "",
        "draft_report_path": "",
        "ieee_html": "",
        "ieee_html_path": "",
        "ieee_tex": "",
        "ieee_tex_path": "",
        "prisma_svg": "",
        "prisma_svg_path": "",
    }


def build_initial_results_df(papers_by_id: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for paper in papers_by_id.values():
        link = paper.get("link")
        if not link and paper.get("doi"):
            link = f"https://doi.org/{paper.get('doi')}"
        rows.append(
            {
                "RS": (paper.get("screening") or {}).get("relevance_score"),
                "article title": paper.get("title") or "",
                "publisher": paper.get("publisher") or "",
                "URL": link or "",
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["_sort_score"] = df["RS"].fillna(-1)
        df = df.sort_values(by="_sort_score", ascending=False, kind="mergesort").drop(columns=["_sort_score"])
    return df


def build_top_papers_df(run: dict[str, Any]) -> pd.DataFrame:
    papers_by_id = run.get("papers_by_id") or {}
    top_paper_ids = run.get("top_paper_ids") or {}
    rows = []

    for pid, entry in top_paper_ids.items():
        paper = papers_by_id.get(pid, {})
        link = paper.get("link")
        if not link and paper.get("doi"):
            link = f"https://doi.org/{paper.get('doi')}"
        pdf_path = entry.get("pdf_path")
        rows.append(
            {
                "RS": (paper.get("screening") or {}).get("relevance_score"),
                "article title": entry.get("title") or paper.get("title") or pid,
                "publisher": paper.get("publisher") or "",
                "download status": "Retrieved" if pdf_path and Path(pdf_path).exists() else "Not Retrieved",
                "URL": link or "",
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["_sort_score"] = df["RS"].fillna(-1)
        df = df.sort_values(by="_sort_score", ascending=False, kind="mergesort").drop(columns=["_sort_score"])
    return df


def build_proxy_helper_zip() -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        with open("scripts/get_session.py", "r", encoding="utf-8") as script_file:
            zf.writestr("get_session.py", script_file.read())
        with open("scripts/proxy_helper_instructions.txt", "r", encoding="utf-8") as instruction_file:
            zf.writestr("proxy_helper_instructions.txt", instruction_file.read())
    return zip_buffer.getvalue()


def _run_timestamp_from_run(run: dict[str, Any]) -> str:
    created_at = (run.get("created_at") or "").strip()
    if created_at:
        return created_at.replace(":", "-")
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def _rewrite_run_local_paths(run: dict[str, Any], old_dir: Path, new_dir: Path) -> None:
    syntheses = run.setdefault("syntheses", {})
    for key in ["draft_report_path", "ieee_html_path", "ieee_tex_path", "prisma_svg_path"]:
        value = syntheses.get(key)
        if value:
            syntheses[key] = _replace_path_prefix(value, old_dir, new_dir)

    for entry in (run.get("top_paper_ids") or {}).values():
        pdf_path = entry.get("pdf_path")
        if pdf_path:
            entry["pdf_path"] = _replace_path_prefix(pdf_path, old_dir, new_dir)


def _replace_path_prefix(path_value: str, old_dir: Path, new_dir: Path) -> str:
    path = Path(path_value)
    try:
        relative = path.relative_to(old_dir)
        return str(new_dir / relative)
    except ValueError:
        return str(path)
