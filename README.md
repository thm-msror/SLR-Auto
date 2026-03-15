# ATLAS: Automated Tool for Literature Analysis and Synthesis

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
