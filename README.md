# ATLAS: Automated Tool for Literature Analysis and Synthesis

This project automates a systematic literature review workflow using API-based paper retrieval, metadata enrichment, LLM screening, PDF collection, and category-level synthesis.

The current entry point is `app.py`, which runs an interactive, resumable pipeline. The older `main.py` / `config.py` flow is still present in the repository, but the app-driven pipeline is the one documented here.

## Pipeline

![systematic_pipeline](image/README/systematic_pipeline.png)

## Setup

Create and activate an environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you use the provided VM helper, see `vm_setup.sh`.

## Environment Variables

Create a `.env` file in the project root with the credentials required by the GPT-backed steps. This repository currently reads configuration from environment variables used by the codebase, for example:

```env
GPT_ENDPOINT="https://your-endpoint.openai.azure.com/"
GPT_DEPLOYMENT="your-model-deployment"
GPT_KEY="your-key"
GPT_VERSION="2024-12-01-preview"
GPT_MAX_WORKERS="4"
```

You may also need any provider-specific keys required by the fetch / enrichment steps depending on your setup.

## Usage

Start a new run:

```bash
python app.py run
```

Start a new run with a fixed run id:

```bash
python app.py run --run-id my-slr-run
```

Resume a previous run by id:

```bash
python app.py resume my-slr-run
```

Resume using an explicit path:

```bash
python app.py resume .\_runs\my-slr-run\_run.json
```

The app is interactive. It prompts for:

- Research questions
- Whether to accept or edit the generated boolean query
- Whether to accept or replace the suggested criteria
- Whether to accept or replace the suggested categories

## What Gets Saved

Each run is stored under:

```text
_runs/<run_id>/
```

Important files:

- `_runs/<run_id>/_run.json`: full checkpoint state for the run
- `_runs/<run_id>/pdfs/`: downloaded PDFs for top papers
- `_runs/<run_id>/report.md`: final generated report

The run state includes:

- Inputs such as research questions, boolean query, queries, and criteria
- Timing and count statistics per stage
- All fetched and enriched papers keyed by `paper_id`
- Initial screening results
- Top paper selection
- Downloaded PDF paths
- Generated categories
- Full screening output for top papers
- Category syntheses
- Logged errors

Because the pipeline checkpoints its state, you can stop and resume without restarting completed stages.

## Outputs

The generated `report.md` includes:

- Research questions
- Suggested and final boolean queries
- Suggested and final criteria
- Pipeline timings
- Counts for fetched, deduplicated, screened, selected, downloaded, and fully screened papers
- A PRISMA-style flow summary
- A table of top papers
- Category synthesis sections

## Data Flow Details

### Fetch and Enrich

- `src/fetch_arxiv.py` retrieves papers from arXiv.
- `src/fetch_crossref.py` retrieves papers from Crossref.
- `src/utils.py` deduplicates fetched records.
- `src/enrich_openalex.py` enriches the merged set with OpenAlex metadata.

### Initial Screening

Initial screening runs on all fetched papers using the criteria generated earlier in the workflow. Results are stored under each paper's `screening` field and used to rank papers for top selection.

### Top Selection

Top papers are chosen by descending `relevance_score`, capped at 50 papers. If no papers have screening scores, no top set is created.

### PDF Download

PDFs for top papers are downloaded into the run-specific `pdfs/` folder. Paths are attached to each selected paper entry when the file exists on disk.

### Categories, Full Screening, and Synthesis

- Categories are generated from the abstracts of the selected papers.
- Full screening runs only for selected papers that have a downloaded PDF.
- Category synthesis aggregates evidence from the full-screening output and writes a synthesis per category.

## Project Structure

```text
.
|-- app.py
|-- main.py
|-- config.py
|-- prompts/
|   |-- screen_initial.txt
|   |-- screen_full.txt
|   |-- synthesize_category.txt
|-- src/
|   |-- app_helpers.py
|   |-- fetch_arxiv.py
|   |-- fetch_crossref.py
|   |-- enrich_openalex.py
|   |-- pdf_downloader.py
|   |-- gpt_research_q.py
|   |-- gpt_criteria.py
|   |-- gpt_categories.py
|   |-- gpt_screener_initial.py
|   |-- gpt_screener_full.py
|   |-- gpt_synthesis.py
|   |-- report.py
|   `-- utils.py
|-- _runs/
`-- test/
```

## Notes

- `app.py` is the current documented pipeline.
- `main.py` remains in the repository, but it follows an older file-driven workflow.
- Full screening depends on downloaded PDFs. Papers without a PDF are skipped at that stage.
- Category synthesis only runs when categories exist and at least one paper produced usable full-screening evidence.
**ATLAS** is a human-guided web application designed to automate the end-to-end **Systematic Literature Review (SLR)** pipeline. By combining the semantic reasoning of **GPT-4o** with a professional **Streamlit** interface, ATLAS transforms a multi-week manual research process into an efficient, transparent, and reproducible workflow.

---

## 🌟 Key Features

- **Integrated "One-Click" Pipeline**: Automates the journey from paper discovery to final knowledge synthesis in a single operation.
- **Human-in-the-Loop Design**: Keeps researchers in control with editable Boolean query generation and inclusion/exclusion criteria checkpoints.
- **Institutional Proxy Support**: A unique "Local Session Handover" workflow enables cloud-deployed instances to retrieve papers from restricted institutional libraries (e.g., UDST, IEEE) securely.
- **PRISMA 2020 Reporting**: Automatically tracks paper counts at every stage and renders a live, downloadable PRISMA 2020 flow diagram.
- **Thematic Synthesis**: Deep-reads full-text PDFs to extract evidence, categorize findings, and generate coherent research narratives.
- **Exportable Results**: Download your PRISMA diagram (SVG), Results Summary (CSV), and Full Synthesis Report (Markdown) directly from the UI.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- An Azure OpenAI Service endpoint and API Key (GPT-4o).

### 2. Setup
Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

### 3. Configuration
Create a `.env` file in the root directory:

```env
AZURE_OPENAI_ENDPOINT="your_endpoint_here"
AZURE_OPENAI_KEY="your_api_key_here"
AZURE_OPENAI_DEPLOYMENT="your_deployment_name"
AZURE_OPENAI_VERSION="2024-12-01-preview"
IEEE_API_KEY="your_ieee_key_here"
```

### 4. Run the App
Launch the Streamlit interface:

```bash
streamlit run streamlit.py
```

---

## 🛠️ The ATLAS Pipeline

1.  **Objective Definition**: Enter your research questions.
2.  **Boolean Query Generation**: GPT-4o proposes a search string; you review and refine it.
3.  **Paper Identification**: ATLAS fetches records from IEEE Xplore and Crossref, deduplicates them, and enriches metadata via OpenAlex.
4.  **Initial Screening**: Abstracts are semantically screened against your custom inclusion/exclusion criteria.
5.  **Proxy Handover**: Run a local helper script to securely share your library session with the cloud.
6.  **Full Pipeline Execution**: 
    - **Download**: Automatic PDF retrieval.
    - **Eligibility**: Deep analysis of full-text papers.
    - **Synthesis**: Thematic aggregation of findings.
7.  **Final Summary**: Review your results table and PRISMA diagram.

---

## 📂 Project Structure

- `streamlit.py`: Main entry point and UI logic.
- `src/`: Core logic modules (Fetchers, Screeners, Synthesis, PDF Processing).
- `prompts/`: LLM instruction templates for all pipeline stages.
- `reports/`: Documentation and project draft paper.
- `scripts/`: Local helper for secure proxy authentication.
- `assets/`: UI assets and branding.

---

## 📝 License & Citation
Designed and developed for the **DSAI4201 - Selected Topics in Data Science** course at the University of Doha for Science and Technology.

*For detailed methodology, refer to [reports/05_atlas_paper_draft.md](reports/05_atlas_paper_draft.md).*
