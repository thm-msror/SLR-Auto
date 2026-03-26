import argparse
import asyncio
import io
import json
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from atlas.inital_fetch.enrich_openalex import enrich as enrich_openalex
from atlas.inital_fetch.fetch_crossref import fetch_papers as fetch_crossref
from atlas.inital_fetch.fetch_semanticscholar import fetch_papers as fetch_semanticscholar
from atlas.inital_fetch.gpt_research_q import (
    boolean_to_queries,
    build_boolean_query_from_questions,
    parse_boolean_query,
)
from atlas.inital_screen.gpt_criteria import build_criteria_from_question, criteria_to_list
from atlas.inital_screen.gpt_screener_initial import screen_paper
from atlas.read_paper.gpt_categories import build_taxonomy_categories, categories_to_dict
from atlas.read_paper.ieee_client import fetch_ieee_papers as fetch_ieee
from atlas.read_paper.pdf_downloader import SESSION_STATE_PATH, download_pdfs
from atlas.results.generate_full_draft import generate_full_draft
from atlas.results.generate_session_report import export_run_to_excel_bytes
from atlas.results.prisma import build_prisma_svg, has_prisma_data
from atlas.utils.app_helpers import (
    RUN_FILE,
    RUNS_DIR,
    ensure_run_shape,
    new_run,
    paper_id_from,
    save_run,
    select_top_ids,
    update_counts,
    update_prisma,
)
from atlas.utils.utils import deduplicate_papers_by_title_authors, safe_filename


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


APP_PROFILES = {
    "normal": {
        "max_queries": 50,
        "max_per_source": 100,
        "ieee_max_results": 50,
        "s2_max_results": 50,
        "top_n": 50,
    },
    "fast": {
        "max_queries": 5,
        "max_per_source": 5,
        "ieee_max_results": 5,
        "s2_max_results": 5,
        "top_n": 5,
    },
}


def _parse_app_profile(argv: list[str]) -> tuple[str, dict]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=sorted(APP_PROFILES.keys()), default="normal")
    args, _ = parser.parse_known_args(argv)
    return args.mode, APP_PROFILES[args.mode]


APP_MODE, APP_LIMITS = _parse_app_profile(sys.argv[1:])

st.set_page_config(page_title="ATLAS", layout="wide")


UI_DEFAULTS = {
    "started": False,
    "queries_confirmed": False,
    "criteria_confirmed": False,
    "proxy_confirmed": False,
    "themes_confirmed": False,
    "report_generated": False,
    "research_question": "",
    "search_queries": "",
    "screening_criteria": "",
    "research_themes": "",
    "queries_generated": False,
    "criteria_generated": False,
    "themes_generated": False,
    "fetching_done": False,
    "screening_done": False,
    "proxy_upload_ready": False,
    "proxy_authorized": False,
    "full_text_done": False,
    "fetch_log": [],
    "download_log": [],
    "full_report": "",
    "query_error": "",
}


def _ensure_run_session() -> None:
    if "run_path" not in st.session_state:
        run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        run_dir = RUNS_DIR / f".{run_id}"
        run_path = run_dir / RUN_FILE
        run = new_run()
        ensure_run_shape(run)
        save_run(run, run_path)
        st.session_state["run_path"] = str(run_path)
        st.session_state["run_id"] = run_id
        st.session_state["run"] = run

    for key, value in UI_DEFAULTS.items():
        st.session_state.setdefault(key, value)


def _save_run(run: dict) -> None:
    run_path = Path(st.session_state["run_path"])
    save_run(run, run_path)
    st.session_state["run"] = run


def _theme_dict_to_text(categories: dict[str, str]) -> str:
    lines = []
    for name, desc in categories.items():
        line = name.strip()
        if desc:
            line = f"{line}: {desc.strip()}"
        if line:
            lines.append(line)
    return "\n".join(lines)


def _criteria_text_to_list(text: str) -> list[str]:
    return criteria_to_list(text)


def _empty_report_syntheses() -> dict:
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
        "prisma_svg": "",
        "prisma_svg_path": "",
    }


def _reset_generated_report(run: dict) -> None:
    ensure_run_shape(run)
    run["syntheses"] = _empty_report_syntheses()
    run.setdefault("inputs", {}).pop("report_generated", None)
    st.session_state.report_generated = False
    st.session_state.full_report = ""


def _build_initial_results_df(papers_by_id: dict) -> pd.DataFrame:
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
        df = df.sort_values(by="_sort_score", ascending=False).drop(columns=["_sort_score"])
    return df


def _build_top_papers_df(run: dict) -> pd.DataFrame:
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
        df = df.sort_values(by="_sort_score", ascending=False).drop(columns=["_sort_score"])
    return df


def _run_initial_screening_live(run: dict, criteria: list[str], table_placeholder) -> None:
    papers_by_id = run.get("papers_by_id") or {}
    to_screen = [
        (pid, paper)
        for pid, paper in papers_by_id.items()
        if not paper.get("screening")
    ]
    if not to_screen:
        st.session_state.screening_done = True
        return

    run_path = Path(st.session_state["run_path"])
    completed = 0
    max_workers = min(8, os.cpu_count() or 4)

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

            if completed % 5 == 0 or completed == len(to_screen):
                table_placeholder.dataframe(
                    _build_initial_results_df(papers_by_id),
                    use_container_width=True,
                    hide_index=True,
                    column_config={"RS": st.column_config.NumberColumn("RS", width=75)},
                )
            if completed % 20 == 0:
                _save_run(run)

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
    run["stage"] = "screening_done"
    _save_run(run)
    st.session_state.screening_done = True


class _StreamlitLogSink(io.StringIO):
    def __init__(self, placeholder=None):
        super().__init__()
        self.placeholder = placeholder

    def write(self, s: str) -> int:
        written = super().write(s)
        if self.placeholder and s:
            lines = self.getvalue().splitlines()[-8:]
            self.placeholder.code("\n".join(lines), language="text")
        return written


def _fetch_and_enrich(queries: list[str], run: dict, log_placeholder=None) -> tuple[list[dict], list[str]]:
    sink = _StreamlitLogSink(log_placeholder)
    with redirect_stdout(sink):
        max_per_source = APP_LIMITS["max_per_source"]

        ieee_papers = fetch_ieee(queries, max_results=APP_LIMITS["ieee_max_results"])
        if len(ieee_papers) > max_per_source:
            ieee_papers = ieee_papers[:max_per_source]

        crossref_papers = fetch_crossref(queries, max_results=max_per_source)
        if len(crossref_papers) > max_per_source:
            crossref_papers = crossref_papers[:max_per_source]

        s2_papers = fetch_semanticscholar(queries, max_results=APP_LIMITS["s2_max_results"])
        if len(s2_papers) > max_per_source:
            s2_papers = s2_papers[:max_per_source]

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

        combined = ieee_papers + crossref_papers + s2_papers
        deduped = deduplicate_papers_by_title_authors(combined, paper_type="fetched")
        update_prisma(run, after_dedup=len(deduped))
        update_counts(run, deduped_total=len(deduped))

        enriched = enrich_openalex(deduped)
        update_counts(run, enriched_total=len(enriched))

    return enriched, sink.getvalue().splitlines()


def _run_full_text_step(run: dict) -> None:
    sink = _StreamlitLogSink()
    with redirect_stdout(sink):
        papers_by_id = run.get("papers_by_id") or {}
        top_ids = select_top_ids(
            papers_by_id,
            max_n=APP_LIMITS["top_n"],
            min_score=3,
        )

        run["top_paper_ids"] = {
            pid: {"title": papers_by_id[pid].get("title") or pid}
            for pid in top_ids
        }
        update_counts(run, top_selected=len(top_ids))
        _save_run(run)

        pdf_dir = Path(st.session_state["run_path"]).parent / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        download_list = [{"paper": papers_by_id[pid]} for pid in top_ids]
        if download_list:
            download_pdfs(download_list, pdf_dir)

        not_retrieved = 0
        for pid in top_ids:
            title = papers_by_id[pid].get("title") or pid
            pdf_path = pdf_dir / f"{safe_filename(title)}.pdf"
            if pdf_path.exists():
                run["top_paper_ids"][pid]["pdf_path"] = str(pdf_path)
            else:
                not_retrieved += 1

        update_prisma(run, not_retrieved=not_retrieved)
        run["stage"] = "proxy_download_done"
        _save_run(run)

    st.session_state.download_log = sink.getvalue().splitlines()[-10:]
    st.session_state.full_text_done = True


def _build_proxy_helper_zip() -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        with open("scripts/get_session.py", "r", encoding="utf-8") as script_file:
            zf.writestr("get_session.py", script_file.read())
        with open("scripts/proxy_helper_instructions.txt", "r", encoding="utf-8") as instruction_file:
            zf.writestr("proxy_helper_instructions.txt", instruction_file.read())
    return zip_buffer.getvalue()


def _render_prisma_section(run: dict) -> None:
    prisma = run.get("prisma") or {}
    if not has_prisma_data(prisma):
        st.info("PRISMA diagram will appear after papers are fetched.")
        return

    svg_str = build_prisma_svg(prisma)
    components.html(
        f'<div style="overflow-x:auto;">{svg_str}</div>',
        height=720,
        scrolling=True,
    )


def _render_download_buttons(run: dict) -> None:
    prisma = run.get("prisma") or {}
    svg_bytes = build_prisma_svg(prisma).encode("utf-8") if has_prisma_data(prisma) else b""
    excel_bytes = b""
    excel_error = ""
    try:
        excel_bytes = export_run_to_excel_bytes(run)
    except Exception as exc:
        excel_error = str(exc)

    st.download_button(
        "Download PRISMA diagram",
        data=svg_bytes,
        file_name="prisma_diagram.svg",
        mime="image/svg+xml",
        disabled=not has_prisma_data(prisma),
    )

    st.download_button(
        "Download review draft",
        data=st.session_state.full_report.encode("utf-8"),
        file_name="draft_report.md",
        mime="text/markdown",
        disabled=not st.session_state.full_report.strip(),
    )

    run_path = Path(st.session_state["run_path"])
    st.download_button(
        "Download session Excel",
        data=excel_bytes,
        file_name=f"{run_path.parent.name}_session_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        disabled=not excel_bytes,
    )
    if excel_error:
        st.caption(f"Excel export unavailable: {excel_error}")

    st.download_button(
        "Download review data",
        data=run_path.read_bytes(),
        file_name=f"{run_path.parent.name}_{RUN_FILE}",
        mime="application/json",
    )


def _render_report_styles() -> None:
    st.markdown(
        """
<style>
.prisma-flow {
    margin: 1.25rem auto;
    text-align: center;
    width: 100%;
}

.prisma-flow-svg {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
}

.prisma-flow svg {
    display: block;
    margin: 0 auto;
    max-width: min(100%, 860px);
    height: auto;
}

.prisma-flow figcaption {
    margin-top: 0.45rem;
    text-align: center;
    font-size: 0.95rem;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def start_autoslr() -> None:
    research_question = st.session_state.research_question.strip()
    if not research_question:
        return

    run = st.session_state["run"]
    inputs = run.setdefault("inputs", {})
    inputs["research_questions"] = research_question
    _reset_generated_report(run)
    run["stage"] = "research_questions"
    _save_run(run)

    st.session_state.started = True
    st.session_state.queries_confirmed = False
    st.session_state.criteria_confirmed = False
    st.session_state.proxy_confirmed = False
    st.session_state.themes_confirmed = False
    st.session_state.report_generated = False
    st.session_state.queries_generated = False
    st.session_state.criteria_generated = False
    st.session_state.themes_generated = False
    st.session_state.fetching_done = False
    st.session_state.screening_done = False
    st.session_state.proxy_upload_ready = False
    st.session_state.proxy_authorized = False
    st.session_state.full_text_done = False
    st.session_state.fetch_log = []
    st.session_state.download_log = []
    st.session_state.full_report = ""
    st.session_state.query_error = ""


def confirm_queries() -> None:
    query_text = st.session_state.search_queries.strip()
    if not query_text:
        return

    run = st.session_state["run"]
    inputs = run.setdefault("inputs", {})
    try:
        parse_boolean_query(query_text)
        queries = boolean_to_queries(query_text, max_queries=APP_LIMITS["max_queries"])
    except Exception as exc:
        st.session_state.query_error = str(exc)
        return

    inputs["boolean_query_used"] = query_text
    inputs["queries"] = queries
    _reset_generated_report(run)
    run["stage"] = "queries_confirmed"
    _save_run(run)

    st.session_state.query_error = ""
    st.session_state.queries_confirmed = True
    st.session_state.fetching_done = False


def confirm_criteria() -> None:
    criteria_text = st.session_state.screening_criteria.strip()
    if not criteria_text:
        return

    used_criteria = _criteria_text_to_list(criteria_text)
    if not used_criteria:
        return

    run = st.session_state["run"]
    run.setdefault("inputs", {})["criteria_used"] = used_criteria
    _reset_generated_report(run)
    run["stage"] = "criteria_confirmed"
    _save_run(run)

    st.session_state.criteria_confirmed = True
    st.session_state.screening_done = False


def confirm_proxy() -> None:
    if not st.session_state.proxy_upload_ready:
        return

    run = st.session_state["run"]
    run["stage"] = "proxy_confirmed"
    _save_run(run)
    st.session_state.proxy_confirmed = True


def confirm_themes() -> None:
    themes_text = st.session_state.research_themes.strip()
    if not themes_text:
        return

    parsed_themes = categories_to_dict(themes_text)
    if not parsed_themes:
        return

    run = st.session_state["run"]
    run["categories"] = parsed_themes
    run.setdefault("inputs", {})["research_themes_used"] = themes_text
    _reset_generated_report(run)
    run["stage"] = "themes_confirmed"
    _save_run(run)

    st.session_state.themes_confirmed = True
    st.session_state.report_generated = False


def _ensure_playwright_installed() -> None:
    if sys.platform == "win32":
        return

    try:
        import subprocess
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            playwright.chromium.launch(headless=True)
    except Exception:
        with st.spinner("Initializing system dependencies (first run only)..."):
            subprocess.run(["python", "-m", "playwright", "install", "chromium"], check=False)


_ensure_run_session()
run = st.session_state["run"]
inputs = run.setdefault("inputs", {})

col_logo, col_title = st.columns([1, 4])
with col_logo:
    logo_path = Path("assets/logo.png")
    if logo_path.exists():
        st.image(str(logo_path), width=180)
    else:
        st.title("ATLAS")
        
with col_title:
    st.title("ATLAS: Automated Tool for Literature Analysis and Synthesis")
    st.write(
        "Human-guided Automated Systematic Literature Reviews using APIs, LLMs, and PRISMA 2020."
    )


# ---------------- RESEARCH QUESTION ----------------
st.header("Research Question")

with st.expander("Research Question", expanded=False):
    st.text_area(
        "Paste your main research question(s)",
        placeholder="e.g. How can AI systems efficiently retrieve and semantically understand relevant segments from long-form video content?",
        key="research_question",
        disabled=st.session_state.started,
        height=150,
        help="Be specific about topic, method, population, or outcome. Clear questions produce better search strings and screening rules.",
    )

    st.button(
        "Start AutoSLR",
        disabled=st.session_state.started,
        on_click=start_autoslr,
    )


# ---------------- INITIAL SEARCH ----------------
st.header("Initial Paper Search")

fetch_log_placeholder = None

with st.expander("Search Query", expanded=st.session_state.started):
    if not st.session_state.started:
        st.info("Start by entering your research question.")
    else:
        if not st.session_state.queries_generated:
            with st.spinner("Generating a suggestion for your search query..."):
                suggested_query = build_boolean_query_from_questions(
                    inputs.get("research_questions", st.session_state.research_question)
                )
                st.session_state.search_queries = suggested_query
                inputs["boolean_query_suggested"] = suggested_query
                st.session_state.queries_generated = True
                _save_run(run)

        st.text_area(
            "Suggested Boolean query. Edit it before searching.",
            key="search_queries",
            disabled=st.session_state.queries_confirmed,
            height=150,
        )

        st.button(
            "Confirm Queries",
            disabled=st.session_state.queries_confirmed,
            on_click=confirm_queries,
        )

        fetch_log_placeholder = st.empty()

        if st.session_state.query_error:
            st.error(
                f"This query could not be parsed. Check parentheses, quotes, and AND/OR/NOT operators. Details: {st.session_state.query_error}"
            )

        if st.session_state.queries_confirmed:
            if st.session_state.fetch_log:
                fetch_log_placeholder.code("\n".join(st.session_state.fetch_log), language="text")


with st.expander("Screening Criteria", expanded=st.session_state.started):
    if not st.session_state.started:
        st.info("Start by entering your research question.")
    else:
        if not st.session_state.criteria_generated:
            with st.spinner("Generating criteria..."):
                raw_criteria = build_criteria_from_question(inputs.get("research_questions", ""))
                suggested_criteria = criteria_to_list(raw_criteria)
                criteria_text = "\n".join(suggested_criteria)
                st.session_state.screening_criteria = criteria_text
                inputs["criteria_suggested"] = suggested_criteria
                st.session_state.criteria_generated = True
                _save_run(run)

        st.text_area(
            "Suggested inclusion and exclusion rules. Refine them before screening papers.",
            key="screening_criteria",
            disabled=not st.session_state.queries_confirmed or st.session_state.criteria_confirmed,
            height=400,
        )

        st.button(
            "Confirm Criteria",
            disabled=(
                not st.session_state.queries_confirmed
                or st.session_state.criteria_confirmed
                or not st.session_state.fetching_done
            ),
            on_click=confirm_criteria,
        )

        if st.session_state.queries_confirmed and not st.session_state.fetching_done:
            st.info("You can edit the screening criteria while results are loading. You can confirm them once the search finishes.")


if st.session_state.queries_confirmed and not st.session_state.fetching_done:
    with st.spinner("Searching sources and combining results..."):
        enriched_papers, fetch_logs = _fetch_and_enrich(
            inputs.get("queries", []),
            run,
            log_placeholder=fetch_log_placeholder,
        )

        papers_by_id = {}
        for paper in enriched_papers:
            pid = paper_id_from(paper)
            paper["paper_id"] = pid
            if pid not in papers_by_id:
                papers_by_id[pid] = paper

        run["papers_by_id"] = papers_by_id
        run["stage"] = "fetch_complete"
        _save_run(run)

        st.session_state.fetch_log = fetch_logs[-8:]
        st.session_state.fetching_done = True
        st.rerun()


results_table_placeholder = st.empty()
if st.session_state.criteria_confirmed and run.get("papers_by_id"):
    results_table_placeholder.dataframe(
        _build_initial_results_df(run["papers_by_id"]),
        use_container_width=True,
        hide_index=True,
        column_config={"RS": st.column_config.NumberColumn("RS", width=75)},
    )

    if not st.session_state.screening_done:
        with st.spinner("Screening papers against your rules..."):
            _run_initial_screening_live(
                run,
                inputs.get("criteria_used", []),
                results_table_placeholder,
            )

if st.session_state.screening_done and run.get("papers_by_id"):
    st.markdown(
        "*RS (Relevancy Score): A heuristic score based on screening criteria. "
        "Each satisfied inclusion criterion increases the score, while violations "
        "of exclusion criteria decrease it.*"
    )


# ---------------- FULL TEXT READING ----------------
st.header("Full Paper Reading")

with st.expander(
    "Download Papers with a Proxy",
    expanded=(
        st.session_state.screening_done
        or st.session_state.proxy_upload_ready
        or st.session_state.proxy_confirmed
    ),
):
    if not st.session_state.screening_done:
        st.info("Finish screening to continue.")
    else:
        st.write(
            "Download the helper, create your session file, then upload it here to enable full-text retrieval."
        )

        col_helper, col_upload = st.columns(2)
        with col_helper:
            try:
                helper_zip = _build_proxy_helper_zip()
                st.download_button(
                    "Download Access Helper",
                    data=helper_zip,
                    file_name="atlas_helper.zip",
                    mime="application/zip",
                )
            except Exception as exc:
                st.error(f"Could not build helper package: {exc}")

        with col_upload:
            uploaded_session = st.file_uploader(
                "Upload session file (.json)",
                type=["json"],
                key="session_json_upload",
            )
            if uploaded_session:
                try:
                    session_data = json.load(uploaded_session)
                    session_path = Path(SESSION_STATE_PATH)
                    session_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(session_path, "w", encoding="utf-8") as session_file:
                        json.dump(session_data, session_file, indent=2)
                    st.session_state.proxy_upload_ready = True
                    st.session_state.proxy_authorized = True
                    st.success("Session file uploaded. Full-text retrieval is ready.")
                except Exception as exc:
                    st.session_state.proxy_upload_ready = False
                    st.session_state.proxy_authorized = False
                    st.error(f"This session file could not be read. Upload a valid JSON file. Details: {exc}")

        st.button(
            "Confirm Proxy Session File",
            disabled=st.session_state.proxy_confirmed or not st.session_state.proxy_upload_ready,
            on_click=confirm_proxy,
        )


if st.session_state.proxy_confirmed and run.get("top_paper_ids"):
    df = _build_top_papers_df(run)
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={"RS": st.column_config.NumberColumn("RS", width=75)},
    )
    st.markdown(
        "*Showing only the top selected papers after proxy confirmation. "
        "The default extra column is download status because that is the main decision point in this stage.*"
    )

with st.expander("Research Themes", expanded=st.session_state.proxy_confirmed):
    if not st.session_state.proxy_confirmed:
        st.info("Enable full-text retrieval to continue.")
    else:
        if not st.session_state.full_text_done:
            with st.spinner("Reading top papers and identifying themes..."):
                _run_full_text_step(run)

        if st.session_state.full_text_done and not st.session_state.themes_generated:
            with st.spinner("Generating themes..."):
                top_paper_ids = run.get("top_paper_ids") or {}
                papers_by_id = run.get("papers_by_id") or {}
                abstracts = []
                for pid in top_paper_ids:
                    abstract = (papers_by_id.get(pid) or {}).get("abstract", "")
                    if abstract:
                        abstracts.append(abstract)

                generated_themes = build_taxonomy_categories(
                    inputs.get("research_questions", ""),
                    abstracts,
                )

                theme_text = _theme_dict_to_text(generated_themes)
                st.session_state.research_themes = theme_text
                run["categories"] = generated_themes
                run.setdefault("inputs", {})["research_themes_suggested"] = theme_text
                st.session_state.themes_generated = True
                _save_run(run)

        st.text_area(
            "Suggested themes from the top papers. Rename, merge, remove, or add themes.",
            key="research_themes",
            disabled=st.session_state.themes_confirmed,
            height=150,
        )

        st.button(
            "Confirm Themes",
            disabled=st.session_state.themes_confirmed or not st.session_state.research_themes.strip(),
            on_click=confirm_themes,
        )


# ---------------- SYSTEMATIC LITERATURE REVIEW ----------------
st.header("Literature Review Draft")

with st.expander("Generated Draft", expanded=st.session_state.themes_confirmed):
    if not st.session_state.themes_confirmed:
        st.info("Confirm your themes to generate the draft.")
    else:
        if not st.session_state.report_generated:
            with st.spinner("Generating full SLR report..."):
                try:
                    run_dir = Path(st.session_state["run_path"]).parent
                    draft_data = generate_full_draft(run, run_dir / "draft_report.md")
                    st.session_state.full_report = draft_data["draft_report"]
                    run.setdefault("inputs", {})["report_generated"] = True
                    run["stage"] = "report_generated"
                    _save_run(run)
                    st.session_state.report_generated = True
                except Exception as exc:
                    st.error(f"Could not generate the review draft: {exc}")

        _render_report_styles()
        st.markdown(st.session_state.full_report, unsafe_allow_html=True)

        _render_download_buttons(run)


if __name__ == "__main__":
    _ensure_playwright_installed()
