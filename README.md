# AI-Backed Systematic Literature Review (SLR) Automation

This project automates large-scale **systematic literature reviews (SLRs)** using APIs, enrichment tools, and LLM-based screening. Instead of manually fetching and filtering hundreds of papers, the pipeline streamlines the process from **query → fetch → screen → summarize**.

## Quick Setup (Bash)

Use the included setup script to create a Conda environment named `autoslr` (if missing), install dependencies, install Playwright browsers, and create `.env` if it does not exist.

```bash
chmod +x setup.sh
./setup.sh
conda activate autoslr
python main.py
```

## Overview

### Query Input

Specify search queries in `config.py`.
READ_PAPERS_FOLDER

```python
QUERIES = [
    "semantic video retrieval AND action recognition AND natural language query",
    "LLM-assisted video understanding",
    ...
]
```

### Fetch Papers

- **arXiv API** (`https://arxiv.org/help/api/`):
  - returns title, authors, abstract, publish/update date, DOI, and link.
- **Crossref API** (`https://api.crossref.org/`):

  - returns title, authors, publisher, DOI, publish date (often _without abstract_).

- Papers from both sources are merged. Duplicate detection is done based on title/DOI.

### Enrich Metadata

- **OpenAlex API** (`https://api.openalex.org/works/doi:{doi}`):

  - Retrieves missing abstracts (esp. for Crossref papers).
  - Adds citation counts, reference counts, fields of study, and OpenAlex link.

### LLM Screening

- Each paper’s abstract is passed through an inital **LLM screener** that gives a relevance score based on the inclusion–exclusion criteria.
- Extracts structured metadata:

  - **Test type**
  - **Modalities**
  - **Key technologies**
  - **Datasets**
  - **Repositories**
  - **Applications**
  - **Limitations**
  - **General notes**

### Reading Top Papers

- **Automatic Paper Collection**
  The pipeline now **automatically downloads** PDFs for the most relevant papers using `src/pdf_downloader.py`. It attempts to fetch from:
  - **Open Access**: ArXiv, CVF, OpenAlex, Unpaywall.
  - **University Proxies**: IEEE, ACM, Springer, ScienceDirect (requires UDST login).
  - **Fallbacks**: Sci-Hub, ResearchGate.

  PDFs are saved in:
  ```
  data\3_top_papers\pdf_papers\
  ```

- **Automatic PDF-to-Markdown Conversion**
  Each PDF is processed and converted into a **clean, machine-readable Markdown file.**
  This step ensures that the paper content (titles, sections, and paragraphs) is properly structured for use by the LLM reader.

- **LLM-Powered Paper Analysis**
  The converted Markdown files are then read by an **LLM (Large Language Model) reader**, which extracts detailed notes on **how each paper addresses the user-defined research gaps** (for example, video segmentation, dataset usage, model architecture, etc.).
  The output is a structured summary for each paper, mapping findings to specific research gaps.

### Summarizing Findings

- **Grouping by Research Gaps**
  The notes generated from all papers are automatically grouped according to the predefined categories or research gaps (e.g., _video analysis_, _dataset design_, _retrieval methods_, etc.).

- **LLM Summarization and Thematic Analysis**
  These grouped notes are then processed by a **second-stage summarization LLM**, which synthesizes the collective insights into:

  - Common **themes and trends** across papers
  - Key **differences or contrasting findings**
  - Remaining **research gaps or open problems**

- **Output and Storage**
  The final literature review results are saved as a Markdown report at:

  ```
  data\5_summaries\paper_reviews.md
  ```

  Example:

  > The approach to video segmentation in multimodal retrieval literature is highly varied, with a significant portion of studies bypassing the challenge altogether by using pre-segmented datasets like MSR-VTT or Moments in Time (`Bridging_the_Semantic_Gap...`, `Condensed Movies...`, `Spoken Moments...`). For papers that do process untrimmed videos, methods range from simple, content-agnostic strategies to more sophisticated, content-aware techniques. Common simple approaches include uniformly sampling a fixed number of clips (`Conditional Cross Correlation Network...`, `Unified Static and Dynamic Network...`) or dividing the video into fixed-length segments with or without overlap (`Local-Global Video-Text Interactions...`, `Towards Fast Adaptation...`). More advanced methods leverage content by using scene detection tools like PySceneDetect to create semantically coherent shots (`ContextIQ...`, `HumanOmni...`). A smaller but notable set of papers employs other modalities to guide segmentation, using ASR timestamps (`Multi-granularity Correspondence Learning...`), slide changes in presentations (`PreMind...`), or even word boundaries from transcripts (`Understanding Co-speech Gestures...`) to define clip boundaries. Conversely, some models are explicitly designed to be proposal-free, processing entire long-form videos without any pre-segmentation (`ECLIPSE...`, `Audio Does Matter...`). A large number of papers, however, do not mention their segmentation method, treating it as an assumed preprocessing step.

## Project Structure

```
.
├── data/
│   ├── 1_fetched_papers/         # raw + enriched papers
│   ├── 2_screened_papers/        # LLM-based screened papers
│   ├── 3_top_papers/             # top papers and their PDF and Markdowns Files
│   ├── 4_read_papers/            # LLM full reading notes each paper
│   └── 5_summaries/              # LLM summary of literature gaps
├── prompts/
│   ├── gpt_new_prompt.txt        # helpful for creating prompts on new topics
│   ├── screening_prompt.txt      # initial screening prompt
│   ├── pdf_reading_prompt.txt    # full paper reading prompt
├── src/
│   ├── fetch_arxiv.py            # arXiv fetcher
│   ├── fetch_crossref.py         # Crossref fetcher
│   ├── enrich_openalex.py        # OpenAlex enrichment
│   ├── llm_screener_bullets.py   # LLM-based screening
│   ├── filter_papers.py          # Filtering papers based on relevance scores
│   ├── pdf_downloader.py         # 🔹 Automatic PDF downloader (Proxies + OA)
│   ├── marker_convert.py         # PDF→Markdown pure helpers (no writes)
│   ├── llm_reader.py             # LLM-based full paper reader
│   ├── paper_reviews.py          # LLM-based literature summarizer
│   └── utils.py                  # helpers (JSON save/load, dedup)
├── config.py                     # search queries & other options
└── main.py                       # orchestrates full pipeline using files in src/
```

## Usage

### 1. Set up environment

Create a `.env` file in the project root with a **FANAR API key** for LLM screening:

```bash
FANAR_API_KEY = "your_key_here"
GEMINI_API_KEY= "your_key_here"
```

### 2. Configure your search and screening

Open `config.py`. Add the queries to run and LLM prompt text files:

```python
# ---------------------- Search Query Terms ----------------------
QUERIES = [
    "semantic video retrieval AND action recognition",
    "multimodal video question answering"
]
MAX_QUERIES = 10

# ---------------------- Reading Guides ----------------------

CRITERIA = [
    "Task relevant (video retrieval / QA / semantic search)",
    "Uses CV (detection, action recognition, scene understanding)",
    "Uses Audio/ASR",
    ...
]

GAPS = [
    "video_segmentation",
    "frame_sampling_method",
    "input_video_length",
    ...]

# ---------------------- LLM Prompts ----------------------
LLM_SCREENING_PROMPT_TXT     = r"prompts\screening_prompt.txt"
LLM_FULL_READ_PROMPT_TXT     = r"prompts\pdf_reading_prompt.txt"
```

During screening, each paper’s abstract and metadata are passed into an LLM prompt that checks each criterion scored as YES / NO / INSUFFICIENT INFO:
Higher scores mean stronger alignment with the SLR goals.

### 3. Run the pipeline

Fetch, enrich, and screen papers with main:

```bash
python main.py
```

> **Note:** The first time it runs, a browser window may open asking you to **log in to the UDST Library**. Once logged in, press ENTER in the terminal to save the session. Future runs will use the saved session.

### 4. Understand outputs

Each stage saves its output so you can **resume without re-running** everything.
If a `saved_*` variable has a file path → that step **loads** it.
If it’s empty → the step **runs** and writes a new file. Example:
To only re-summarize findings, keep everything else saved and set
`saved_gap_reviews = ""`.

#### Saved paths In `config.py`

```python
# ---------------------- Checkpoints ------------------------
# Uncomment to skip that pipeline. Comment out to run normally
# To skip a step all other steps before it must be uncommented
# -----------------------------------------------------------
saved_enriched_papers  = r"data\1_fetched_papers\enriched_6325_papers_2025-10-15T05-06-28.json"
saved_screened_papers  = r"data\2_screened_papers\screened_6325_papers_2025-10-15T05-06-28.json"
saved_top_papers       = r"data\3_top_papers\top_53_papers_20251016_212754.json"
skip_md_conversion     = True
saved_read_papers      = r"data\4_read_papers\full_read.json"
saved_gap_reviews      = r"data\5_summaries\paper_reviews.json"
```
