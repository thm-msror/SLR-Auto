import io
import zipfile
import json
import os
import sys
import asyncio
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

# Fix for Playwright/Asyncio NotImplementedError on Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from atlas.utils.app_helpers import (
    RUNS_DIR,
    RUN_FILE,
    ensure_run_shape,
    new_run,
    paper_id_from,
    save_run,
    update_counts,
    update_prisma,
    run_full_screening,
    run_category_synthesis,
    resolve_run_path,
)
from atlas.read_paper.pdf_downloader import download_pdfs, SESSION_STATE_PATH
from atlas.read_paper.ieee_client import fetch_ieee_papers as fetch_ieee
from atlas.inital_fetch.fetch_crossref import fetch_papers as fetch_crossref
from atlas.inital_fetch.fetch_semanticscholar import fetch_papers as fetch_semanticscholar
from atlas.inital_fetch.enrich_openalex import enrich as enrich_openalex
from atlas.inital_screen.gpt_criteria import build_criteria_from_question, criteria_to_list
from atlas.inital_fetch.gpt_research_q import (
    boolean_to_queries,
    build_boolean_query_from_questions,
    parse_boolean_query,
)
from atlas.inital_screen.gpt_screener_initial import screen_paper
from atlas.read_paper.gpt_categories import build_taxonomy_categories
from atlas.utils.utils import deduplicate_papers_by_title_authors

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
        "max_per_source": 20,
        "ieee_max_results": 10,
        "s2_max_results": 10,
        "top_n": 5,
    },
}


def _parse_app_profile(argv: list[str]) -> tuple[str, dict]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=sorted(APP_PROFILES.keys()), default="normal")
    args, _ = parser.parse_known_args(argv)
    mode = args.mode
    return mode, APP_PROFILES[mode]


APP_MODE, APP_LIMITS = _parse_app_profile(sys.argv[1:])

# Page configuration
st.set_page_config(page_title="ATLAS - Automated SLR", page_icon="img/rephraise_logo.png", layout="wide")

# ---- CSS ----
st.markdown('''<style>
.css-1egvi7u {margin-top: -4rem;}
.css-znku1x a {color: #9d03fc;}
.css-qrbaxs {min-height: 0.0rem;}
.stSpinner > div > div {border-top-color: #9d03fc;}
.css-15tx938{min-height: 0.0rem;}
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
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
/* PRISMA diagram styles */
.prisma-diagram {
    font-family: Arial, sans-serif;
    font-size: 13px;
    background: #fff;
    padding: 20px;
}
.prisma-box {
    border: 1.5px solid #2c5282;
    border-radius: 4px;
    padding: 8px 12px;
    background: #fff;
    text-align: left;
}
.prisma-box-excluded {
    border: 1.5px solid #c53030;
    border-radius: 4px;
    padding: 8px 12px;
    background: #fff5f5;
    text-align: left;
}
.prisma-section-label {
    writing-mode: vertical-rl;
    transform: rotate(180deg);
    background: #4a90d9;
    color: white;
    padding: 8px 4px;
    border-radius: 4px;
    font-weight: bold;
    text-align: center;
    font-size: 11px;
}
</style>''', unsafe_allow_html=True)


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
                "Publisher": paper.get("publisher") or "",
                "URL": link or "",
                "Relevance Score": score,
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

    # Update PRISMA counts after screening
    total_screened = sum(1 for p in papers_by_id.values() if p.get("screening"))
    excluded_screening = sum(
        1 for p in papers_by_id.values()
        if p.get("screening") and (p["screening"].get("relevance_score") or 0) <= 0
    )
    update_prisma(run, screened=total_screened, excluded_screening=excluded_screening)
    save_run(run, run_path)


class _StreamlitLogSink(io.StringIO):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        self.lines: list[str] = []

    def write(self, s: str) -> int:
        n = super().write(s)
        if s:
            # Filter out error/quota messages but keep "Querying IEEE"
            lines = self.getvalue().splitlines()
            filtered = []
            for line in lines:
                is_ieee_error = "IEEE" in line and any(err in line for err in ["error", "Error", "403", "Exceeded", "already hit", "Skipping", "failed"])
                if not is_ieee_error:
                    filtered.append(line)
            
            self.lines = filtered[-5:]
            if self.placeholder:
                self.placeholder.code("\n".join(self.lines), language="text")
        return n


def _fetch_and_enrich(queries: list[str], run: dict, log_placeholder=None) -> tuple[list[dict], list[str]]:
    sink = _StreamlitLogSink(log_placeholder)
    with redirect_stdout(sink):
        max_per_source = APP_LIMITS["max_per_source"]
        
        # --- IEEE ---
        ieee_papers = fetch_ieee(
            queries,
            max_results=APP_LIMITS["ieee_max_results"],
        )
        if len(ieee_papers) > max_per_source:
            ieee_papers = ieee_papers[:max_per_source]

        # --- Crossref ---
        crossref_papers = fetch_crossref(queries, max_results=max_per_source)
        if len(crossref_papers) > max_per_source:
            crossref_papers = crossref_papers[:max_per_source]

        # --- Semantic Scholar ---
        s2_papers = fetch_semanticscholar(
            queries,
            max_results=APP_LIMITS["s2_max_results"],
        )
        if len(s2_papers) > max_per_source:
            s2_papers = s2_papers[:max_per_source]

        # Record identification counts for PRISMA
        ident = run.get("prisma", {}).get("identification", {})
        ident["ieee"] = len(ieee_papers)
        ident["crossref"] = len(crossref_papers)
        ident["semanticscholar"] = len(s2_papers)
        run.setdefault("prisma", {})["identification"] = ident
        
        update_counts(run, 
                      fetched_ieee=len(ieee_papers), 
                      fetched_crossref=len(crossref_papers),
                      fetched_s2=len(s2_papers))

        combined = ieee_papers + crossref_papers + s2_papers
        deduped = deduplicate_papers_by_title_authors(combined, paper_type="fetched")
        update_prisma(run, after_dedup=len(deduped))
        update_counts(run, deduped_total=len(deduped))

        enriched = enrich_openalex(deduped)
        update_counts(run, enriched_total=len(enriched))

    log_lines = sink.getvalue().splitlines()
    return enriched, log_lines


# ---------------------------------------------------------------------------
# PRISMA diagram
# ---------------------------------------------------------------------------

def _build_prisma_svg(prisma: dict) -> str:
    """
    Build a PRISMA 2020 flow diagram as an SVG string from the prisma counts dict.
    """
    ident = prisma.get("identification", {})
    n_ieee = ident.get("ieee", 0)
    n_crossref = ident.get("crossref", 0)
    n_s2 = ident.get("semanticscholar", 0)
    n_total_id = n_ieee + n_crossref + n_s2
    n_dedup = prisma.get("after_dedup", 0)
    n_removed = n_total_id - n_dedup
    n_screened = prisma.get("screened", n_dedup)
    n_excl_screen = prisma.get("excluded_screening", 0)
    n_sought = prisma.get("sought_retrieval", max(0, n_screened - n_excl_screen))
    n_not_retrieved = prisma.get("not_retrieved", 0)
    n_assessed = prisma.get("assessed_eligibility", max(0, n_sought - n_not_retrieved))
    n_excl_elig = prisma.get("excluded_eligibility", 0)
    n_included = prisma.get("included", max(0, n_assessed - n_excl_elig))

    W, H = 760, 680
    BOX_W, BOX_H = 280, 60
    EXCL_W = 220
    LX = 80   # left column x
    RX = 430  # right/excluded column x
    LABEL_X = 20
    LABEL_W = 50

    def box(x, y, w, h, text, excluded=False, sub=""):
        fill = "#fff5f5" if excluded else "#ffffff"
        stroke = "#c53030" if excluded else "#2c5282"
        lines = []
        for i, t in enumerate([text] + ([sub] if sub else [])):
            lines.append(f'<text x="{x+10}" y="{y+22+i*16}" font-size="12" font-family="Arial" fill="#1a202c">{t}</text>')
        return (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            + "".join(lines)
        )

    def arrow(x1, y1, x2, y2):
        return (
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#4a5568" stroke-width="1.5" marker-end="url(#arr)"/>'
        )

    def section_label(y, h, label):
        return (
            f'<rect x="{LABEL_X}" y="{y}" width="{LABEL_W}" height="{h}" rx="4" fill="#4a90d9"/>'
            f'<text x="{LABEL_X+25}" y="{y+h//2}" font-size="11" font-family="Arial" fill="white" '
            f'text-anchor="middle" dominant-baseline="middle" '
            f'transform="rotate(-90, {LABEL_X+25}, {y+h//2})">{label}</text>'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'style="background:#fff;font-family:Arial,sans-serif;">'
        '<defs><marker id="arr" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">'
        '<polygon points="0 0, 10 3.5, 0 7" fill="#4a5568"/></marker></defs>'
        # Title
        f'<text x="{W//2}" y="25" text-anchor="middle" font-size="15" font-weight="bold" fill="#1a202c">PRISMA 2020 Flow Diagram</text>'
    )

    # --- IDENTIFICATION ---
    ID_Y = 45
    svg += section_label(ID_Y, 100, "Identification")
    svg += box(LX, ID_Y, BOX_W, BOX_H,
               f"Records identified (n = {n_total_id})",
               sub=f"IEEE: {n_ieee}   Crossref: {n_crossref}")
    svg += box(RX, ID_Y, EXCL_W, BOX_H,
               f"Duplicates removed (n = {n_removed})", excluded=True)
    svg += arrow(LX + BOX_W, ID_Y + BOX_H//2, RX, ID_Y + BOX_H//2)
    svg += arrow(LX + BOX_W//2, ID_Y + BOX_H, LX + BOX_W//2, ID_Y + BOX_H + 20)

    # --- SCREENING ---
    SCR_Y = ID_Y + BOX_H + 30
    svg += section_label(SCR_Y, 100, "Screening")
    svg += box(LX, SCR_Y, BOX_W, BOX_H,
               f"Records screened (n = {n_screened})")
    svg += box(RX, SCR_Y, EXCL_W, BOX_H,
               f"Records excluded (n = {n_excl_screen})", excluded=True)
    svg += arrow(LX + BOX_W, SCR_Y + BOX_H//2, RX, SCR_Y + BOX_H//2)
    svg += arrow(LX + BOX_W//2, SCR_Y + BOX_H, LX + BOX_W//2, SCR_Y + BOX_H + 20)

    # --- SOUGHT ---
    SOUGHT_Y = SCR_Y + BOX_H + 30
    svg += box(LX, SOUGHT_Y, BOX_W, BOX_H,
               f"Reports sought for retrieval (n = {n_sought})")
    svg += box(RX, SOUGHT_Y, EXCL_W, BOX_H,
               f"Not retrieved (n = {n_not_retrieved})", excluded=True)
    svg += arrow(LX + BOX_W, SOUGHT_Y + BOX_H//2, RX, SOUGHT_Y + BOX_H//2)
    svg += arrow(LX + BOX_W//2, SOUGHT_Y + BOX_H, LX + BOX_W//2, SOUGHT_Y + BOX_H + 20)

    # --- ELIGIBILITY ---
    ELG_Y = SOUGHT_Y + BOX_H + 30
    svg += section_label(ELG_Y, 100, "Eligibility")
    svg += box(LX, ELG_Y, BOX_W, BOX_H,
               f"Reports assessed for eligibility (n = {n_assessed})")
    svg += box(RX, ELG_Y, EXCL_W, BOX_H,
               f"Reports excluded (n = {n_excl_elig})", excluded=True)
    svg += arrow(LX + BOX_W, ELG_Y + BOX_H//2, RX, ELG_Y + BOX_H//2)
    svg += arrow(LX + BOX_W//2, ELG_Y + BOX_H, LX + BOX_W//2, ELG_Y + BOX_H + 20)

    # --- INCLUDED ---
    INC_Y = ELG_Y + BOX_H + 30
    svg += section_label(INC_Y, 70, "Included")
    svg += box(LX, INC_Y, BOX_W, 55,
               f"Studies included in review (n = {n_included})")

    svg += "</svg>"
    return svg


def _render_prisma_section(run: dict) -> None:
    """Render the PRISMA 2020 flow diagram and SVG download button in Streamlit."""
    prisma = run.get("prisma") or {}
    ident = prisma.get("identification", {})
    total_id = ident.get("ieee", 0) + ident.get("crossref", 0)
    if total_id == 0:
        st.info("PRISMA diagram will appear after papers are fetched.")
        return

    st.subheader("PRISMA 2020 Flow Diagram")
    svg_str = _build_prisma_svg(prisma)
    components.html(
        f'<div style="overflow-x:auto;">{svg_str}</div>',
        height=720,
        scrolling=True,
    )

    # SVG download button
    svg_bytes = svg_str.encode("utf-8")
    st.download_button(
        label="Download PRISMA Diagram as SVG",
        data=svg_bytes,
        file_name="prisma_diagram.svg",
        mime="image/svg+xml",
    )

    # Also show counts summary table
    with st.expander("PRISMA Count Details"):
        prisma_rows = [
            ("Identification - IEEE", ident.get("ieee", 0)),
            ("Identification - Crossref", ident.get("crossref", 0)),
            ("After deduplication", prisma.get("after_dedup", 0)),
            ("Screened", prisma.get("screened", 0)),
            ("Excluded at screening", prisma.get("excluded_screening", 0)),
            ("Sought for retrieval", prisma.get("sought_retrieval", 0)),
            ("Not retrieved", prisma.get("not_retrieved", 0)),
            ("Assessed for eligibility", prisma.get("assessed_eligibility", 0)),
            ("Excluded at eligibility", prisma.get("excluded_eligibility", 0)),
            ("Included", prisma.get("included", 0)),
        ]
        st.table(pd.DataFrame(prisma_rows, columns=["Stage", "Count"]))


def _render_results_table(run: dict):
    papers_by_id = run.get("papers_by_id", {})
    top_ids = run.get("top_paper_ids", {})
    
    if not top_ids:
        return

    st.subheader("Final Synthesis Results")
    
    table_data = []
    for pid, entry in top_ids.items():
        paper = papers_by_id.get(pid, {})
        title = entry.get("title") or paper.get("title") or pid
        
        # Download status
        pdf_path = entry.get("pdf_path")
        if pdf_path and Path(pdf_path).exists():
            dl_status = "Success: Retrieved"
        else:
            dl_status = "Failed: Not Accessible"
            
        # Synthesis status
        full_screen = entry.get("full_screening") or {}
        included = full_screen.get("included")
        if included is True:
            synth_status = "Included in Synthesis"
        elif included is False:
            synth_status = "Excluded (Eligibility)"
        else:
            synth_status = "Pending Analysis"
            
        table_data.append({
            "Paper Title": title,
            "Download Status": dl_status,
            "Synthesis Status": synth_status
        })
        
    df = pd.DataFrame(table_data)
    st.table(df)
    
    # Download button for the table
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results Summary (CSV)",
        data=csv,
        file_name="atlas_results_summary.csv",
        mime="text/csv",
    )
    
    # Synthesis text download
    syntheses = run.get("syntheses", {}).get("categories", {})
    if syntheses:
        synth_text = "# ATLAS Literature Synthesis\n\n"
        for cat, content in syntheses.items():
            synth_text += f"## {cat}\n{content}\n\n"
            
        st.download_button(
            label="Download Full Synthesis Report (Markdown)",
            data=synth_text.encode('utf-8'),
            file_name="atlas_synthesis_report.md",
            mime="text/markdown",
        )


def main_gpt3emailgen():
    _ensure_session_state()
    run = st.session_state["run"]
    inputs = run.setdefault("inputs", {})

    # Top Header with Logo
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

    st.write("\n")

    st.write("\n")

    # -------- STEP 1: Research Question --------
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

    # -------- STEP 2: Boolean Query + Fetch --------
    st.subheader("Initial Search")
    with st.expander("Search Queries", expanded=st.session_state["exp_queries"]):
        if st.session_state["query_badge"]:
            st.markdown(_badge_html(True), unsafe_allow_html=True)
        st.caption(
            f"Run mode: `{APP_MODE}` | max queries: {APP_LIMITS['max_queries']} | "
            f"top papers: {APP_LIMITS['top_n']}"
        )
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
                "Suggested boolean query — edit to refine your search terms:",
                value=inputs.get("boolean_query_used", suggested_bool),
                key="bool_used",
                disabled=bool_locked,
                height=100,
            )
            st.markdown("</div>", unsafe_allow_html=True)
            st.caption(
                "The query will be expanded into multiple search terms. "
                "You can add, remove, or change synonyms above before confirming."
            )

            if st.button("Confirm Queries", disabled=bool_locked):
                try:
                    parse_boolean_query(used_bool)
                except Exception as exc:
                    st.session_state["bool_invalid"] = True
                    st.error(f"Invalid boolean query: {exc}")
                else:
                    st.session_state["bool_invalid"] = False
                    inputs["boolean_query_used"] = used_bool.strip()
                    queries = boolean_to_queries(
                        used_bool,
                        max_queries=APP_LIMITS["max_queries"],
                    )
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
                    with st.spinner("Fetching and enriching papers (IEEE + Crossref)..."):
                        enriched, logs = _fetch_and_enrich(queries, run, log_placeholder=log_placeholder)

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

    # -------- STEP 3: Criteria --------
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

            criteria_used_text = st.text_area(
                "Suggested criteria — edit, remove, or add lines to customise your screening. "
                "Peer-review criteria are included by default (remove if not needed):",
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

    # -------- Papers Table + Screening --------
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


    # -------- STEP 7: PDF Download & Proxy Check (Show only after screening) --------
    is_screening_done = run.get("papers_by_id") and all(
        p.get("screening") for p in run["papers_by_id"].values()
    )
    
    if is_screening_done:
        st.markdown("---")
        st.subheader("Final Pipeline: PDF Download & Analysis")
        
        # --- Proxy Agreement/Popup-style Info ---
        st.warning("### Proxy Authentication Required")
        st.markdown(
            "To download papers from your institutional library (UDST/IEEE), ATLAS needs a valid "
            "login session. Since we are in the cloud, you must 'Hand Over' your session from your laptop."
        )
        
        col_helper, col_upload = st.columns(2)
        
        with col_helper:
            st.write("**1. Get the Helper**")
            st.write("Download the script and the setup guide to your computer.")
            
            try:
                # Create ZIP in memory
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    # Add Script
                    with open("scripts/get_session.py", "r", encoding="utf-8") as f:
                        zf.writestr("get_session.py", f.read())
                    # Add Instructions
                    with open("scripts/proxy_helper_instructions.txt", "r", encoding="utf-8") as f:
                        zf.writestr("proxy_helper_instructions.txt", f.read())
                
                zip_data = zip_buffer.getvalue()
                
                st.download_button(
                    label="Download Helper Package (ZIP)",
                    data=zip_data,
                    file_name="atlas_helper.zip",
                    mime="application/zip",
                    key="dl_helper_zip"
                )
                st.caption("Contains `get_session.py` and instructions.")
            except Exception as e:
                st.error(f"Error loading helper files: {e}")
                
        with col_upload:
            st.write("**2. Upload Session**")
            st.write("Upload the `udst_session.json` generated by the script.")
            uploaded_session = st.file_uploader("Upload Session JSON", type=["json"], key="stage7_session_v2")
            if uploaded_session:
                try:
                    session_data = json.load(uploaded_session)
                    from atlas.read_paper.pdf_downloader import SESSION_STATE_PATH
                    Path(SESSION_STATE_PATH).parent.mkdir(parents=True, exist_ok=True)
                    with open(SESSION_STATE_PATH, "w", encoding="utf-8") as f:
                        json.dump(session_data, f, indent=2)
                    st.session_state["proxy_authorized"] = True
                    st.success("Session Verified! Proxy access enabled.")
                except Exception as e:
                    st.error(f"Invalid session file: {e}")
     
        st.write("\n")
        if not st.session_state.get("proxy_authorized"):
            st.info("Waiting for Proxy Authentication... Stage 7 is locked.")
            st.button("Start Full Pipeline (PDF Download + Analysis)", disabled=True)
        else:
            if st.button("Start Full Pipeline (PDF Download + Analysis)"):
                from atlas.utils.app_helpers import select_top_ids
                
                # 1. Selection
                with st.spinner("Selecting top relevant papers (Minimum Score: 3)..."):
                    p_id_map = run["papers_by_id"]
                    top_ids = select_top_ids(
                        p_id_map,
                        max_n=APP_LIMITS["top_n"],
                        min_score=3,
                    )
                    run["top_paper_ids"] = {
                        pid: {"title": p_id_map[pid]["title"]} 
                        for pid in top_ids
                    }
                    _save_run(run)
                
                # 1.5. Category Generation (Thematic Taxonomy)
                if top_ids and not run.get("categories"):
                    with st.spinner("Generating thematic categories from abstracts..."):
                        abstracts = [
                            p_id_map[pid].get("abstract", "") 
                            for pid in top_ids 
                            if p_id_map[pid].get("abstract")
                        ]
                        if abstracts:
                            categories = build_taxonomy_categories(
                                run["inputs"]["research_questions"], 
                                abstracts
                            )
                            run["categories"] = categories
                            _save_run(run)
                
                # 2. Downloading
                with st.spinner(f"Downloading PDFs for {len(top_ids)} papers..."):
                    pdf_dir = Path(st.session_state["run_path"]).parent / "pdfs"
                    pdf_dir.mkdir(parents=True, exist_ok=True)
                    
                    download_list = [{"paper": p_id_map[pid]} for pid in top_ids]
                    download_pdfs(download_list, pdf_dir)
                    
                    # Update local paths in run['top_paper_ids']
                    from atlas.utils.utils import safe_filename
                    not_retrieved = 0
                    for pid in top_ids:
                        title = p_id_map[pid].get("title", pid)
                        p_file = pdf_dir / f"{safe_filename(title)}.pdf"
                        if p_file.exists():
                            run["top_paper_ids"][pid]["pdf_path"] = str(p_file)
                        else:
                            not_retrieved += 1
                    
                    update_prisma(run, not_retrieved=not_retrieved)
                    _save_run(run)
                
                # 3. Full Analysis (Eligibility)
                if any(entry.get("pdf_path") for entry in run["top_paper_ids"].values()):
                    with st.spinner("Running deep PDF analysis and eligibility check..."):
                        run_full_screening(run, Path(st.session_state["run_path"]))
                
                # 4. Synthesis
                if any(entry.get("full_screening", {}).get("included") for entry in run["top_paper_ids"].values()):
                    with st.spinner("Synthesizing category findings..."):
                        run_category_synthesis(run, Path(st.session_state["run_path"]))
                
                run["stage"] = "finished"
                _save_run(run)
                st.success("Pipeline Complete!")
                st.rerun()

    # -------- FINAL RESULTS: PRISMA + Table (Show only when finished) --------
    if run.get("stage") == "finished" or run.get("top_paper_ids"):
        st.markdown("---")
        _render_prisma_section(run)
        st.markdown("---")
        _render_results_table(run)
    
    st.markdown("---")


def _ensure_playwright_installed():
    """Ensure playwright chromium is installed for Streamlit Cloud."""
    # Do not run on Windows to avoid local loop policy issues
    if sys.platform == 'win32':
        return
        
    try:
        import subprocess
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            p.chromium.launch(headless=True)
    except Exception:
        with st.spinner("Initializing system dependencies (first run only)..."):
            subprocess.run(["python", "-m", "playwright", "install", "chromium"])

if __name__ == '__main__':
    _ensure_playwright_installed()
    main_gpt3emailgen()
