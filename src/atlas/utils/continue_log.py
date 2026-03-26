from __future__ import annotations

import json
from typing import Any, Dict

from atlas.utils.app_helpers import ensure_run_shape


def load_run_from_json_bytes(raw_bytes: bytes) -> dict[str, Any]:
    try:
        data = json.loads(raw_bytes.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError("The uploaded file is not valid UTF-8 JSON.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("The uploaded file is not valid JSON.") from exc

    if not isinstance(data, dict):
        raise ValueError("The uploaded log must be a JSON object.")

    ensure_run_shape(data)
    return data


def derive_continue_state(run: dict[str, Any], has_proxy_session_file: bool = False) -> Dict[str, Any]:
    ensure_run_shape(run)
    inputs = run.get("inputs") or {}
    syntheses = run.get("syntheses") or {}
    papers_by_id = run.get("papers_by_id") or {}
    categories = run.get("categories") or {}
    top_paper_ids = run.get("top_paper_ids") or {}

    research_question = (inputs.get("research_questions") or "").strip()
    search_queries = (
        (inputs.get("boolean_query_used") or "").strip()
        or (inputs.get("boolean_query_suggested") or "").strip()
    )
    criteria_used = inputs.get("criteria_used") or []
    criteria_suggested = inputs.get("criteria_suggested") or []
    research_themes = (
        (inputs.get("research_themes_used") or "").strip()
        or (inputs.get("research_themes_suggested") or "").strip()
        or _theme_dict_to_text(categories)
    )

    queries_generated = bool((inputs.get("boolean_query_suggested") or "").strip() or search_queries)
    queries_confirmed = bool((inputs.get("boolean_query_used") or "").strip() and (inputs.get("queries") or []))
    criteria_generated = bool(criteria_suggested or criteria_used)
    criteria_confirmed = bool(criteria_used)
    fetching_done = bool(papers_by_id)
    screening_done = bool(papers_by_id) and all((paper.get("screening") or {}) for paper in papers_by_id.values())
    full_text_done = bool(top_paper_ids)
    themes_generated = bool((inputs.get("research_themes_suggested") or "").strip() or categories)
    themes_confirmed = bool((inputs.get("research_themes_used") or "").strip() or categories)
    report_generated = bool(inputs.get("report_generated") or (syntheses.get("draft_report") or "").strip())

    if full_text_done:
        proxy_confirmed = True
        proxy_upload_ready = True
        proxy_authorized = True
    else:
        proxy_confirmed = False
        proxy_upload_ready = bool(has_proxy_session_file)
        proxy_authorized = bool(has_proxy_session_file)

    return {
        "started": bool(research_question),
        "queries_confirmed": queries_confirmed,
        "criteria_confirmed": criteria_confirmed,
        "proxy_confirmed": proxy_confirmed,
        "themes_confirmed": themes_confirmed,
        "report_generated": report_generated,
        "research_question": research_question,
        "search_queries": search_queries,
        "screening_criteria": _join_lines(criteria_used or criteria_suggested),
        "research_themes": research_themes,
        "queries_generated": queries_generated,
        "criteria_generated": criteria_generated,
        "themes_generated": themes_generated,
        "fetching_done": fetching_done,
        "screening_done": screening_done,
        "proxy_upload_ready": proxy_upload_ready,
        "proxy_authorized": proxy_authorized,
        "full_text_done": full_text_done,
        "fetch_log": [],
        "download_log": [],
        "full_report": syntheses.get("draft_report") or "",
        "full_report_html": syntheses.get("ieee_html") or "",
        "full_report_tex": syntheses.get("ieee_tex") or "",
        "query_error": "",
        "continue_notice": _build_continue_notice(
            report_generated=report_generated,
            themes_confirmed=themes_confirmed,
            full_text_done=full_text_done,
            screening_done=screening_done,
            criteria_confirmed=criteria_confirmed,
            queries_confirmed=queries_confirmed,
            started=bool(research_question),
        ),
        "continue_error": "",
    }


def _build_continue_notice(
    *,
    report_generated: bool,
    themes_confirmed: bool,
    full_text_done: bool,
    screening_done: bool,
    criteria_confirmed: bool,
    queries_confirmed: bool,
    started: bool,
) -> str:
    if report_generated:
        return "Previous run loaded. The draft is already available."
    if themes_confirmed:
        return "Previous run loaded. The next missing step is draft generation."
    if full_text_done:
        return "Previous run loaded. The next missing step is theme confirmation."
    if screening_done:
        return "Previous run loaded. The next missing step is proxy-assisted full-text retrieval."
    if criteria_confirmed:
        return "Previous run loaded. The next missing step is initial screening."
    if queries_confirmed:
        return "Previous run loaded. The next missing step is criteria confirmation."
    if started:
        return "Previous run loaded. The next missing step is query confirmation."
    return "Previous run loaded."


def _join_lines(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    if value is None:
        return ""
    return str(value).strip()


def _theme_dict_to_text(categories: dict[str, Any]) -> str:
    lines = []
    for name, desc in categories.items():
        line = str(name).strip()
        if desc:
            line = f"{line}: {str(desc).strip()}"
        if line:
            lines.append(line)
    return "\n".join(lines)
