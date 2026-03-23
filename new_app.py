import streamlit as st
import time
import pandas as pd

st.set_page_config(page_title="ATLAS", layout="wide")

# ---------------- SESSION STATE ----------------
defaults = {
    "started": False,
    "queries_confirmed": False,
    "criteria_confirmed": False,
    "proxy_confirmed": False,
    "themes_confirmed": False,

    "research_question": "",
    "search_queries": "",
    "screening_criteria": "",
    "research_themes": "",

    "queries_generated": False,
    "criteria_generated": False,
    "themes_generated": False,

    "fetching_done": False,
    "screening_done": False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------- CALLBACKS ----------------
def start_autoslr():
    if st.session_state.research_question.strip():
        st.session_state.started = True

        # TODO: get_boolean_queries(research_question)

        st.session_state.queries_generated = False
        st.session_state.criteria_generated = False
        st.session_state.fetching_done = False
        st.session_state.screening_done = False


def confirm_queries():
    if st.session_state.search_queries.strip():
        st.session_state.queries_confirmed = True

        # TODO: run search pipeline here


def confirm_criteria():
    if st.session_state.screening_criteria.strip():
        st.session_state.criteria_confirmed = True

        # TODO: apply screening logic here

def confirm_proxy():
    st.session_state.proxy_confirmed = True

        # TODO: apply proxy logic here



def confirm_themes():
    if st.session_state.research_themes.strip():
        st.session_state.themes_confirmed = True

        # TODO: apply themes logic here

# ---------------- HEADER ----------------
col1, col2 = st.columns([1, 6])

with col1:
    st.image("assets/logo.png", width=120)

with col2:
    st.title("ATLAS: Automated Tool for Literature Analysis and Synthesis")
    st.caption(
        "Human-guided Automated Systematic Literature Reviews using APIs, LLMs, and PRISMA 2020."
    )

st.markdown("---")


# ---------------- RESEARCH QUESTION ----------------
st.header("What is your research?")

with st.expander("Research Question", expanded=False):

    st.text_area(
        "Enter all your research questions here",
        placeholder="e.g. How can AI systems efficiently retrieve and semantically understand relevant segments from long-form video content?",
        key="research_question",
        disabled=st.session_state.started,
        height=150
    )

    st.button(
        "Start AutoSLR",
        disabled=st.session_state.started,
        on_click=start_autoslr
    )


# ---------------- INITIAL SEARCH ----------------
st.header("Initial Search")

# -------- SEARCH QUERIES --------
with st.expander(
    "Search Queries",
    expanded=st.session_state.started
):

    if not st.session_state.started:
        st.info("Enter research questions first.")

    else:

        # Generate queries
        if not st.session_state.queries_generated:
            with st.spinner("Generating query suggestion..."):
                time.sleep(2)

                # TODO: replace with real function
                st.session_state.search_queries = (
                    '("artificial intelligence" OR "machine learning") AND '
                    '("video retrieval" OR "multimodal search")'
                )

                st.session_state.queries_generated = True

        st.text_area(
            "Suggested Boolean search query: edit or add your own",
            key="search_queries",
            disabled=st.session_state.queries_confirmed,
            height=150
        )

        st.button(
            "Confirm Queries",
            disabled=st.session_state.queries_confirmed,
            on_click=confirm_queries
        )

        # ✅ Show warning + spinner together
        if st.session_state.queries_confirmed:

            st.warning("⚠️ TODO: Put search + retrieval code here.")

            if not st.session_state.fetching_done:
                with st.spinner("Fetching papers based on query..."):
                    time.sleep(2)
                    st.session_state.fetching_done = True


# -------- SCREENING CRITERIA --------
with st.expander(
    "Initial screening criteria",
    expanded=st.session_state.queries_confirmed
):

    if not st.session_state.started:
        st.info("Enter research questions first.")

    else:

        # Generate criteria
        if not st.session_state.criteria_generated:
            with st.spinner("Generating criteria suggestion..."):
                time.sleep(2)

                # TODO: give_screening_criteria(...)
                st.session_state.screening_criteria = (
                    "- Include peer-reviewed articles\n"
                    "- Include studies from 2015 onwards\n\n"
                    "- Exclude non-English papers\n"
                    "- Exclude abstracts only"
                )

                st.session_state.criteria_generated = True

        st.text_area(
            "Suggested inclusion/exclusion criteria: edit or add your own",
            key="screening_criteria",
            disabled=not st.session_state.queries_confirmed or st.session_state.criteria_confirmed,
            height=150
        )

        st.button(
            "Confirm Criteria",
            disabled=not st.session_state.queries_confirmed or st.session_state.criteria_confirmed,
            on_click=confirm_criteria
        )

        if st.session_state.criteria_confirmed:

            st.warning("⚠️ TODO: Put screening code here.")

            if not st.session_state.screening_done:
                with st.spinner("Screening papers based on criteria..."):
                    time.sleep(2)
                    st.session_state.screening_done = True

# Table of initial papers
if st.session_state.screening_done:

    data = [
        {
            "RS": 0.95,
            "article title": "AI for Video Retrieval: A Survey",
            "publisher": "IEEE",
            "URL": "https://example.com/1",
        },
        {
            "RS": 0.89,
            "article title": "Multimodal Learning in Video Search",
            "publisher": "Springer",
            "URL": "https://example.com/2",
        },
    ]

    df = pd.DataFrame(data)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "RS": st.column_config.NumberColumn("RS", width=75)
        }
    )

    st.markdown(
        "*RS (Relevancy Score): A heuristic score based on screening criteria. "
        "Each satisfied inclusion criterion increases the score, while violations "
        "of exclusion criteria decrease it.*"
    )

# ---------------- FULL TEXT READING ----------------
st.header("Full Text Reading")

# PROXY DOWNLOADER
with st.expander("Download (Proxy Downloader)", expanded=False):

    if not st.session_state.screening_done:
        st.info("Finish initial search first.")
    else:
        st.info("TODO: Put proxy downloader block here.")

        st.button(
            "Confirm Proxy",
            disabled=st.session_state.proxy_confirmed,
            on_click=confirm_proxy
        )


# TABLE AFTER PROXY
if st.session_state.proxy_confirmed:
    df = pd.DataFrame([
        {"RS": 0.92, "article title": "Downloaded Paper", "publisher": "Elsevier", "URL": "link"}
    ])
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.markdown("*This is an example table that will be changed*")

# RESEARCH THEMES
with st.expander("Research Themes", expanded=st.session_state.proxy_confirmed):

    if not st.session_state.proxy_confirmed:
        st.info("Finish initial search first.")
    else:

        if not st.session_state.themes_generated:
            with st.spinner("Generating research themes..."):
                time.sleep(2)

                # TODO: createResearchThemes()
                st.session_state.research_themes = "Theme 1: AI\nTheme 2: Video"
                st.session_state.themes_generated = True

        st.text_area(
            "Suggested themes based on the abstract of the top papers, edit or add your own",
            key="research_themes",
            disabled=st.session_state.themes_confirmed,
            height=150
        )

        st.button(
            "Confirm Themes",
            disabled=st.session_state.themes_confirmed,
            on_click=confirm_themes
        )


# ---------------- SYSTEMATIC LITERATURE REVIEW ----------------
st.header("Systematic Literature Review")

with st.expander(
    "Final Draft",
    expanded=st.session_state.themes_confirmed
):

    if not st.session_state.themes_confirmed:
        st.info("Finish full text reading first.")

    else:

        # Generate report once
        if "report_generated" not in st.session_state:
            st.session_state.report_generated = False

        if not st.session_state.report_generated:
            with st.spinner("Generating full SLR report..."):
                time.sleep(2)

                # TODO: make_full_report(...)
                # Should return STRING (markdown)
                st.session_state.full_report = """
Abstract: Summarizes the background, objectives, methods, main results, and conclusions.

Introduction: Outlines the research topic, its context, the significance of the review, and clearly stated research questions (RQs).

Methodology (The Protocol): The most critical part, detailing how the study was conducted to ensure reproducibility. It includes:
    Inclusion/Exclusion Criteria: Definitions of what studies were selected and why.
    Search Strategy: Databases used, keywords, and search strings applied.
    Study Selection/PRISMA Flow Diagram: A visual representation of how studies were screened and selected.
    Data Extraction & Quality Assessment: How data was collected and how the quality of studies was assessed.
"""
                st.session_state.report_generated = True

        # Render report
        st.subheader(
            "This is a generated SLR paper based on your research question, automated screening, and research themes:"
        )

        st.markdown(st.session_state.full_report)

        st.markdown("<prisma img here>")

        st.button("Download this PRISMA image")

        st.markdown("""
Results/Findings: A systematic presentation of the data extracted, often including charts, tables, and themes, rather than just summaries of papers.
Discussion: Interprets the results, explains the implications of the findings, and discusses trends and contradictions.
Limitations: Acknowledges constraints on the review process, such as search language restrictions or missing studies.
Conclusion & Future Work: Summarizes key findings and suggests areas for future research based on identified gaps.
""")

        st.button("Download Full Paper (Markdown)")
        st.button("Download Detailed SLR Logs")