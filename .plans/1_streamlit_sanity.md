**Title:** Streamlit Workflow for AutoSLR Steps 1–5 + Live Results Table

**Summary**
Implement the Streamlit app flow for research questions, boolean query generation/validation, fetch+enrich, criteria generation, and initial screening. Use persistent run storage in `_runs`, keep UI state in `st.session_state`, and show a live, sorted table that appears after deduplication and updates with relevance scores after screening.

**Implementation Changes**

1. **State + Persistence**

   - Add `st.session_state` keys for: current step, research questions text, boolean suggested/used, expanded queries list, criteria suggested/used, run id/path, papers cache, and UI flags (expanders, badges, errors).
   - Create or resume a run using `src.app_helpers.new_run()` and `save_run()` in `streamlit.py` so refreshes don’t lose progress. Store run JSON under `_runs/<run_id>/_run.json`.
2. **Step 1: Research Questions**

   - “Start AutoSLR” button sets the research questions, disables the text area (`disabled=True`), collapses the Research Question expander, and expands Search Queries.
   - Add a small “Added” badge in the subheader via `st.markdown` + CSS class.
3. **Step 2: Boolean Query**

   - Use `build_boolean_query_from_questions()` to generate a suggested boolean query once.
   - Validate the user edit with `parse_boolean_query()`; on failure:
     - Show `st.error` message.
     - Apply a CSS rule to the text area (keyed by label/aria label or wrapper class) to show a red outline.
   - On success:
     - Expand into individual queries with `boolean_to_queries()`, cap at 50 (match CLI).
     - Save `inputs.boolean_query_suggested`, `inputs.boolean_query_used`, and `inputs.queries` into the run.
     - Collapse Search Queries expander and expand Initial Screening Criteria.
4. **Step 4 (as per Streamlit order): Fetch + Enrich**

   - Run `fetch_arxiv`, `fetch_crossref`, `deduplicate_papers_by_title_authors`, then `enrich_openalex` like `app.py`.
   - Capture stdout during fetch/enrich using `contextlib.redirect_stdout` and show the last 5 lines in `st.code`.
   - Build `papers_by_id` (same shape as `app.py` using `paper_id_from`) and save to run.
5. **Live Results Table**

   - Render a `st.dataframe` immediately after deduplication/enrichment:
     - Columns: `#`, `Title`, `URL`, `Relevance Score`.
     - URL uses `paper.get("link")` or DOI URL when present.
     - Relevance score blank/NaN until screening.
   - After screening completes, update scores and sort descending by score (None/NaN at bottom).
6. **Step 3: Criteria + Step 5 Screening**

   - Generate criteria via `build_criteria_from_question()` and normalize with `criteria_to_list()`.
   - Allow edit; on Confirm Criteria:
     - Save `inputs.criteria_suggested` and `inputs.criteria_used` to run.
     - Run `run_initial_screening()` with used criteria.
     - Update table relevance scores and ordering.
     - Collapse criteria expander and add “Added” badge.
7. **UI Polish**

   - Ensure criteria text area uses `height=` so it’s visibly expanded.
   - Fix the “query â†’ fetch…” string to ASCII (`query -> fetch -> screen -> summarize`) to avoid encoding artifacts.

**Public Interfaces / Data Shape Changes**

- No API changes. Streamlit will persist using existing run JSON schema in `_runs`.

**Test Plan**

1. Enter research questions, click Start:
   - Text area becomes disabled, badge shows, Search Queries expands.
2. Boolean query edit:
   - Invalid query triggers red outline + error.
   - Valid query proceeds, writes to `_runs/<run_id>/_run.json`.
3. Confirm Queries:
   - Fetch + enrich runs, last 5 debug lines shown, table appears with titles/URLs and empty relevance score.
4. Confirm Criteria:
   - Initial screening runs, relevance scores populate and table reorders by score.
5. Refresh page:
   - Run is resumed from `_runs`, state persists.

**Assumptions**

- Network access and API keys are already configured (same as CLI usage).
- If any external API fails, we keep partial results and surface the error via `st.error` while still showing whatever was fetched.
