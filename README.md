# ATLAS: Automated Tool for Literature Analysis and Synthesis

**ATLAS** is a human-guided, AI-powered system designed to automate the most tedious parts of a Systematic Literature Review (SLR). From defining research questions to generating a PRISMA-compliant report, ATLAS streamlines the entire research pipeline using GPT-4o and scholarly APIs.

> [!TIP]
> **Don't have API keys?** If you want to try the tool without configuring your own environment or keys, use our **[Deployed Interface](https://slr-auto.streamlit.app/)**.

---

## 🚀 Key Features
-   **Boolean Query Generator:** Translates research questions into complex search strings.
-   **Multi-Source Fetching:** Integrates with IEEE Xplore, Crossref, and Semantic Scholar.
-   **Intelligent Screening:** LLM-based abstract grading with a custom **Relevancy Score (RS)**.
-   **Human-in-the-Loop:** Every major decision (queries, criteria, themes) is editable and requires user approval.
-   **Automatic Synthesis:** Generates a full SLR markdown draft, an Excel audit trail, and a **PRISMA 2020 Flow Diagram**.

---

## Local Installation & Setup

### Prerequisites
-   Python 3.10 or higher.
-   Azure OpenAI API Key (Required).
-   IEEE Xplore API Key (Optional).

### Steps
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/joey-en/SLR-Auto.git
    cd SLR-Auto
    ```
2.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\Activate.ps1
    # Linux/Mac
    source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -e .
    python -m playwright install chromium
    ```
4.  **Configure Environment Variables:**
    Create a `.env` file in the root directory and add your keys:
    ```env
    GPT_ENDPOINT="https://your-endpoint.openai.azure.com/"
    GPT_DEPLOYMENT="your-model-deployment"
    GPT_KEY="your-api-key"
    GPT_VERSION="2024-12-01-preview"
    GPT_RESPONSES_VERSION="2025-03-01-preview"
    IEEE_API="your-ieee-key"
    ```

---

## Running the Application
To launch the Streamlit interface:
```bash
streamlit run streamlit.py
```
For a quick test with limited results, use:
```bash
streamlit run streamlit.py -- --mode fast
```

---

## Project Documentation & Reports
For detailed methodology, technical design, and presentation materials, see the `reports/` directory:
-   **[Technical Repository Guide](reports/repo_technical_guide.md):** Deep-dive into the code and system flow.
-   **[Frontend & UI Features](reports/frontend_features.md):** Explanation of the Streamlit interface.

---

## Outputs & File Structure
ATLAS stores run state under `data/runs/`. A typical run directory contains:
-   `log.json`: checkpointed workflow state.
-   `pdfs/`: all downloaded PDF files.
-   `SLR_draft.md`: The generated synthesis report.

---

## Authors
-   **Joy Anne P. Dela Cruz** (60301959@udst.edu.qa)
-   **Tehreem Masroor** (60302531@udst.edu.qa)
-   *DSAI4201 - Selected Topics in Data Science, University of Doha for Science and Technology.*
