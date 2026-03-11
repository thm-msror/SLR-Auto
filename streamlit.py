# gpt3 professional email generator by stefanrmmr - version June 2022
# https://stefanrmmr-gpt3-email-generator-streamlit-app-ku3fbq.streamlit.app/?ref=streamlit-io-gallery-favorites

import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.app_helpers import (
    RUNS_DIR,
    RUN_FILE,
    ensure_run_shape,
    new_run,
    paper_id_from,
    save_run,
    update_counts,
)
from src.fetch_arxiv import fetch_papers as fetch_arxiv
from src.fetch_crossref import fetch_papers as fetch_crossref
from src.enrich_openalex import enrich as enrich_openalex
from src.gpt_criteria import build_criteria_from_question, criteria_to_list
from src.gpt_research_q import (
    boolean_to_queries,
    build_boolean_query_from_questions,
    parse_boolean_query,
)
from src.gpt_screener_initial import screen_paper
from src.utils import deduplicate_papers_by_title_authors

# DESIGN implement changes to the standard streamlit UI/UX
st.set_page_config(page_title="rephraise", page_icon="img/rephraise_logo.png",)
# Design move app further up and remove top padding
st.markdown('''<style>.css-1egvi7u {margin-top: -4rem;}</style>''',
    unsafe_allow_html=True)
# Design change hyperlink href link color
st.markdown('''<style>.css-znku1x a {color: #9d03fc;}</style>''',
    unsafe_allow_html=True)  # darkmode
st.markdown('''<style>.css-znku1x a {color: #9d03fc;}</style>''',
    unsafe_allow_html=True)  # lightmode
# Design change height of text input fields headers
st.markdown('''<style>.css-qrbaxs {min-height: 0.0rem;}</style>''',
    unsafe_allow_html=True)
# Design change spinner color to primary color
st.markdown('''<style>.stSpinner > div > div {border-top-color: #9d03fc;}</style>''',
    unsafe_allow_html=True)
# Design change min height of text input box
st.markdown('''<style>.css-15tx938{min-height: 0.0rem;}</style>''',
    unsafe_allow_html=True)
# Design hide top header line
hide_decoration_bar_style = '''<style>header {visibility: hidden;}</style>'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)
# Design hide "made with streamlit" footer menu area
hide_streamlit_footer = """<style>#MainMenu {visibility: hidden;}
                        footer {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_footer, unsafe_allow_html=True)

st.markdown(
    """
<style>
.badge-added {
  display: inline-block;
  margin-left: 0.5rem;
  padding: 0.1rem 0.4rem;
  font-size: 0.75rem;
  line-height: 1.2;
  border-radius: 0.4rem;
  background: #e7f7ee;
  color: #0b6b3a;
  border: 1px solid #bfe8cf;
}
.invalid-outline textarea {
  border: 2px solid #ff4b4b !important;
  box-shadow: 0 0 0 1px #ff4b4b !important;
}
</style>
""",
    unsafe_allow_html=True,
)


def _latest_run_path() -> Path | None:
    if not RUNS_DIR.exists():
        return None
    candidates = list(RUNS_DIR.glob(f"*/{RUN_FILE}"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_run(path: Path) -> dict:
    run = json.loads(path.read_text(encoding="utf-8"))
    ensure_run_shape(run)
    return run


def _ensure_session_state() -> None:
    if "run_path" not in st.session_state:
        run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        run_dir = RUNS_DIR / run_id
        run_path = run_dir / RUN_FILE
        run = new_run()
        ensure_run_shape(run)
        save_run(run, run_path)
        st.session_state["run_path"] = str(run_path)
        st.session_state["run_id"] = run_id
        st.session_state["run"] = run

    st.session_state.setdefault("exp_rq", True)
    st.session_state.setdefault("exp_queries", False)
    st.session_state.setdefault("exp_criteria", False)
    st.session_state.setdefault("rq_badge", False)
    st.session_state.setdefault("query_badge", False)
    st.session_state.setdefault("criteria_badge", False)
    st.session_state.setdefault("bool_invalid", False)
    st.session_state.setdefault("fetch_log", [])


def _save_run(run: dict) -> None:
    run_path = Path(st.session_state["run_path"])
    save_run(run, run_path)
    st.session_state["run"] = run


def _badge_html(show: bool) -> str:
    return '<span class="badge-added">Added</span>' if show else ""


def _build_table(papers_by_id: dict) -> pd.DataFrame:
    rows = []
    for i, (pid, paper) in enumerate(papers_by_id.items(), start=1):
        link = paper.get("link")
        if not link and paper.get("doi"):
            link = f"https://doi.org/{paper.get('doi')}"
        score = (paper.get("screening") or {}).get("relevance_score")
        rows.append(
            {
                "#": i,
                "Article Title": paper.get("title") or pid,
                "URL": link or "",
                "✔️": score,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["_sort_score"] = df["Relevance Score"].fillna(-1)
        df = df.sort_values(by="_sort_score", ascending=False).drop(columns=["_sort_score"])
    return df


def _run_initial_screening_live(run: dict, run_path: Path, criteria: list[str], table_placeholder) -> None:
    papers_by_id = run.get("papers_by_id") or {}
    to_screen = [
        (pid, paper)
        for pid, paper in papers_by_id.items()
        if not paper.get("screening")
    ]
    if not to_screen:
        return

    max_workers = min(8, (os.cpu_count() or 4))
    completed = 0

    def task(pid, paper):
        result = screen_paper(paper, criteria)
        return pid, result

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(task, pid, paper): pid for pid, paper in to_screen}
        for fut in as_completed(futures):
            pid = futures[fut]
            paper = papers_by_id.get(pid, {})
            try:
                _, result = fut.result()
                paper["screening"] = result
                completed += 1
            except Exception as exc:
                run.setdefault("errors", []).append(
                    {"stage": "initial_screening", "paper_id": pid, "error": str(exc)}
                )

            if completed % 5 == 0 or completed == len(to_screen):
                table_placeholder.dataframe(_build_table(papers_by_id), use_container_width=True)
            if completed % 20 == 0:
                save_run(run, run_path)

    save_run(run, run_path)


class _StreamlitLogSink(io.StringIO):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        self.lines: list[str] = []

    def write(self, s: str) -> int:
        n = super().write(s)
        if s:
            self.lines = self.getvalue().splitlines()[-5:]
            if self.placeholder:
                self.placeholder.code("\n".join(self.lines), language="text")
        return n


def _fetch_and_enrich(queries: list[str], log_placeholder=None) -> tuple[list[dict], list[str]]:
    sink = _StreamlitLogSink(log_placeholder)
    with redirect_stdout(sink):
        max_total = 100
        arxiv: list[dict] = []
        crossref: list[dict] = []

        for q in queries:
            remaining = max_total - len(arxiv)
            if remaining <= 0:
                break
            arxiv.extend(fetch_arxiv([q], max_results=remaining))

        for q in queries:
            remaining = max_total - (len(arxiv) + len(crossref))
            if remaining <= 0:
                break
            crossref.extend(fetch_crossref([q], max_results=remaining))

        update_counts(st.session_state["run"], fetched_arxiv=len(arxiv), fetched_crossref=len(crossref))
        combined = arxiv + crossref
        if len(combined) > max_total:
            combined = combined[:max_total]
        deduped = deduplicate_papers_by_title_authors(combined, paper_type="fetched")
        update_counts(st.session_state["run"], deduped_total=len(deduped))
        enriched = enrich_openalex(deduped)
        update_counts(st.session_state["run"], enriched_total=len(enriched))
    log_lines = sink.getvalue().splitlines()
    return enriched, log_lines


def main_gpt3emailgen():
    _ensure_session_state()
    run = st.session_state["run"]
    inputs = run.setdefault("inputs", {})

    # st.image('img/image_banner.png')  # TITLE and Creator information
    st.markdown('This project automates large-scale **systematic literature reviews (SLRs)** using APIs, enrichment tools, and LLM-based screening. Instead of manually fetching and filtering hundreds of papers, the pipeline streamlines the process from **query -> fetch -> screen -> summarize**.')
    st.write('\n')  # add spacing


    st.subheader("What is your research?")
    with st.expander("Research Question", expanded=st.session_state["exp_rq"]):
        if st.session_state["rq_badge"]:
            st.markdown(_badge_html(True), unsafe_allow_html=True)

        rq_text = st.text_area(
            "Enter all your research questions here",
            placeholder="How can AI systems efficiently retrieve and semantically understand relevant segments from long-form video content?",
            key="rq_text",
            value=inputs.get("research_questions", ""),
            disabled=bool(inputs.get("research_questions")),
            height=180,
        )

        if st.button("Start AutoSLR"):
            if not rq_text.strip():
                st.error("Please enter at least one research question.")
            else:
                inputs["research_questions"] = rq_text.strip()
                run["stage"] = "research_questions"
                _save_run(run)
                st.session_state["rq_badge"] = True
                st.session_state["exp_rq"] = False
                st.session_state["exp_queries"] = True
                st.session_state["exp_criteria"] = False
                st.rerun()

    st.subheader("Initial Search")
    with st.expander("Search Queries", expanded=st.session_state["exp_queries"]):
        if st.session_state["query_badge"]:
            st.markdown(_badge_html(True), unsafe_allow_html=True)
        if not inputs.get("research_questions"):
            st.info("Enter research questions first.")
        else:
            if not inputs.get("boolean_query_suggested"):
                with st.spinner("Generating suggested boolean query..."):
                    suggested = build_boolean_query_from_questions(inputs["research_questions"])
                    inputs["boolean_query_suggested"] = suggested
                    _save_run(run)

            suggested_bool = inputs.get("boolean_query_suggested", "")
            bool_locked = bool(inputs.get("queries"))

            invalid_class = "invalid-outline" if st.session_state["bool_invalid"] else ""
            st.markdown(f'<div class="{invalid_class}">', unsafe_allow_html=True)
            used_bool = st.text_area(
                "Suggested boolean queries. Edit to fix your search:",
                value=inputs.get("boolean_query_used", suggested_bool),
                key="bool_used",
                disabled=bool(inputs.get("boolean_query_used")),

            )
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("Confirm Queries", disabled=bool_locked):
                try:
                    parse_boolean_query(used_bool)
                except Exception as exc:
                    st.session_state["bool_invalid"] = True
                    st.error(f"Invalid boolean query: {exc}")
                else:
                    st.session_state["bool_invalid"] = False
                    inputs["boolean_query_used"] = used_bool.strip()
                    queries = boolean_to_queries(used_bool, max_queries=50)
                    inputs["queries"] = queries
                    run["stage"] = "boolean_query"
                    _save_run(run)

                    if not inputs.get("criteria_suggested"):
                        with st.spinner("Generating criteria..."):
                            raw = build_criteria_from_question(inputs["research_questions"])
                            suggested_list = criteria_to_list(raw)
                            inputs["criteria_suggested"] = suggested_list
                            _save_run(run)
                    st.session_state["exp_criteria"] = True

                    log_placeholder = st.empty()
                    with st.spinner("Fetching and enriching papers..."):
                        enriched, logs = _fetch_and_enrich(queries, log_placeholder=log_placeholder)
                    papers_by_id = {}
                    for paper in enriched:
                        pid = paper_id_from(paper)
                        paper["paper_id"] = pid
                        if pid not in papers_by_id:
                            papers_by_id[pid] = paper

                    run["papers_by_id"] = papers_by_id
                    run["stage"] = "fetch_enrich"
                    _save_run(run)

                    st.session_state["fetch_log"] = logs[-5:]
                    st.session_state["query_badge"] = True
                    st.rerun()

        if st.session_state.get("fetch_log"):
            st.code("\n".join(st.session_state["fetch_log"]), language="text")

    with st.expander("Initial screening criteria", expanded=st.session_state["exp_criteria"]):
        if st.session_state["criteria_badge"]:
            st.markdown(_badge_html(True), unsafe_allow_html=True)
        if not inputs.get("research_questions"):
            st.info("Enter research questions first.")
        else:
            if not inputs.get("criteria_suggested"):
                with st.spinner("Generating criteria..."):
                    raw = build_criteria_from_question(inputs["research_questions"])
                    suggested_list = criteria_to_list(raw)
                    inputs["criteria_suggested"] = suggested_list
                    _save_run(run)

            suggested_criteria = inputs.get("criteria_suggested", [])
            _ = "\n".join(suggested_criteria)

            criteria_used_text = st.text_area(
                "Suggested criteria. Edit to fix your screening:",
                value="\n".join(inputs.get("criteria_used", suggested_criteria)),
                height=360,
                key="criteria_used_text",
                disabled=bool(inputs.get("criteria_used")),

            )

            if st.button("Confirm Criteria", disabled=bool(inputs.get("criteria_used"))):
                used_list = criteria_to_list(criteria_used_text)
                if not used_list:
                    st.error("Please provide at least one criterion.")
                else:
                    inputs["criteria_used"] = used_list
                    run["stage"] = "criteria"
                    _save_run(run)

                    st.session_state["criteria_badge"] = True
                    st.rerun()

    if run.get("papers_by_id"):
        table_placeholder = st.empty()
        table_placeholder.dataframe(_build_table(run["papers_by_id"]), use_container_width=True)

        if inputs.get("criteria_used") and any(
            not p.get("screening") for p in run["papers_by_id"].values()
        ):
            with st.spinner("Running initial screening..."):
                _run_initial_screening_live(
                    run,
                    Path(st.session_state["run_path"]),
                    inputs["criteria_used"],
                    table_placeholder,
                )


if __name__ == '__main__':
    # call main function
    main_gpt3emailgen()
