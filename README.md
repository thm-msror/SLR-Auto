# ATLAS: Automated Tool for Literature Analysis and Synthesis

ATLAS is a Streamlit application for human-guided systematic literature reviews. It combines API-based paper retrieval, metadata enrichment, GPT-assisted screening, PDF collection, and category-level synthesis in one resumable workflow.

## What It Does

ATLAS supports this pipeline:

1. Enter research questions
2. Generate and edit a Boolean query
3. Fetch papers from IEEE Xplore, Crossref, and Semantic Scholar
4. Deduplicate and enrich metadata
5. Generate and edit screening criteria
6. Run initial abstract screening
7. Select top papers, download PDFs, run full-text analysis, and synthesize findings
8. Export a PRISMA diagram, results table, and synthesis report

## Quick Start

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
python -m playwright install chromium
```

Create a `.env` file in the project root:

```env
GPT_ENDPOINT="https://your-endpoint.openai.azure.com/"
GPT_DEPLOYMENT="your-model-deployment"
GPT_KEY="your-api-key"
GPT_VERSION="2024-12-01-preview"
GPT_RESPONSES_VERSION="2025-03-01-preview"
IEEE_API="your-ieee-key"
```

Run the app:

```bash
streamlit run streamlit.py
```

## Notes on Configuration

- `GPT_KEY` is required for GPT-backed stages.
- `IEEE_API` is optional. If it is missing, ATLAS skips IEEE search and continues with the other sources.
- For deployed Streamlit environments, you can provide the same values through environment variables or Streamlit secrets.

## Outputs

ATLAS stores run state under `._runs/`. A typical run directory contains:

- `_run.json`: checkpointed workflow state
- `pdfs/`: downloaded PDFs
- generated synthesis and result artifacts tied to that run

The UI also lets you download:

- PRISMA diagram as SVG
- Results summary as CSV
- Full synthesis report as Markdown

Important directories:

- `src/`: application logic for retrieval, screening, synthesis, PDF handling, and clients
- `prompts/`: prompt templates used by GPT-backed stages
- `scripts/`: local helper scripts, including proxy session handover support
- `assets/`: UI images and branding
- `reports/`: project documentation and supporting writeups
- `._runs/`: generated run data and artifacts

## Implementation Notes

- Main entrypoint: `streamlit.py`
- GPT client: `src/gpt_client.py`
- IEEE client: `src/ieee_client.py`
- Proxy helper script: `scripts/get_session.py`

For methodology details, see `reports/`.
