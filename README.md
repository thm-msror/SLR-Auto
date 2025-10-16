# AI-Backed Systematic Literature Review (SLR) Automation

This project automates large-scale **systematic literature reviews (SLRs)** using APIs, enrichment tools, and LLM-based screening. Instead of manually fetching and filtering hundreds of papers, the pipeline streamlines the process from **query → fetch → screen → summarize**.

#### Run the command to install required packages: `pip install -r requirements.txt`

## Overview

### Query Input

Specify search queries in `config.py`.

  ```python
  QUERIES = [
      "semantic video retrieval AND action recognition AND natural language query",
      "LLM-assisted video understanding",
      ...
  ]
  ```

### Fetch Papers

  * **arXiv API** (`https://arxiv.org/help/api/`):
      * returns title, authors, abstract, publish/update date, DOI, and link.
  * **Crossref API** (`https://api.crossref.org/`):
      * returns title, authors, publisher, DOI, publish date (often *without abstract*).

* Papers from both sources are merged. Duplicate detection is done based on title/DOI.

### Enrich Metadata

   * **OpenAlex API** (`https://api.openalex.org/works/doi:{doi}`):

     * Retrieves missing abstracts (esp. for Crossref papers).
     * Adds citation counts, reference counts, fields of study, and OpenAlex link.

### LLM Screening

   * Each paper’s abstract is passed through an **LLM screener** that gives a relevance score based on the inclusion–exclusion criteria.
   * Extracts structured metadata:

     * **Test type**
     * **Modalities**
     * **Key technologies**
     * **Datasets**
     * **Applications**
     * **Limitations**
     * **General notes**

### Filter & Summarize

   * The most relevant papers are kept.
   * Summarization aggregates:

     * Key datasets
     * Methods and models
     * Notable papers & authors
     * Emerging trends


## Project Structure

```
.
├── data/
│   ├── fetched_articles/         # raw + enriched papers
│   ├── screened_articles/        # LLM-based screened papers
│   └── summaries/                # LLM-based & table summary markdowns
├── src/
│   ├── fetch_arxiv.py            # arXiv fetcher
│   ├── fetch_crossref.py         # Crossref fetcher
│   ├── enrich_openalex.py        # OpenAlex enrichment
│   ├── llm_screener.py           # LLM-based screening (legacy wrapper)
|   ├── llm_screener_bullets.py   # LLM screening (pure helpers; main saves files)
│   ├── marker_convert.py         # PDF→Markdown pure helpers (no writes)
│   └── utils.py                  # helpers (JSON save/load, dedup)
├── config.py                     # search queries & other options
└── main.py                       # orchestrates fetch pipeline
```

## Usage

### 1. Set up environment

Create a `.env` file in the project root with a **FANAR API key** for LLM screening:

```bash
FANAR_API_KEY=your_fanar_api_key_here
```

### 2. Configure your search and screening

Open `config.py`. Add the queries to run and LLM prompt text files:

```python
QUERIES = [
    "semantic video retrieval AND action recognition",
    "multimodal video question answering"
]
MAX_RESULTS = 250

SUMMARY_PROMPT_TXT = "data/summarization_prompt.txt"
LLM_SCREENING_PROMPT_TXT = "data/screening_prompt.txt"
```

### 3. Run the pipeline

Fetch, enrich, and screen papers with main:

```bash
python main.py
```

### 4. View Output

By default, the pipeline saves results into three main folders:

* **Raw fetched papers** (arXiv, Crossref, enriched metadata) →
  `data/fetched_articles/`

* **LLM-screened papers** →
  `data/screened_articles/`

* **Summaries** →
  `data/summaries/`

### 5. Bullet-Style Screening (JSON + TXT)

If you want structured **JSON** + human-readable **TXT** outputs, use the bullet-based screener:

Note: The screening module now exposes pure helpers; `main.py` orchestrates all saving. Running `main.py` will produce:

- `data/screened_articles/all_screened_papers.json` → full JSON with LLM screening results
- `data/screened_articles/all_screened_bullets.txt` → bullets for quick review
- `data/screened_articles/checkpoints/` → incremental checkpoints (managed by main)

### 6. Relevance Scoring & Inclusion/Exclusion Criteria

During screening, each paper’s abstract and metadata are passed into an LLM prompt that checks **eight strict criteria**.  
Each criterion is scored as **YES / NO / INSUFFICIENT INFO**:

1. Task relevant (video retrieval / QA / semantic search)  
2. Uses CV (detection, action recognition, scene understanding)  
3. Uses Audio/ASR (speech, audio-visual events)  
4. Uses NLP/LLM for query or answers  
5. Multimodal fusion (vision+audio+text)  
6. Has experiment on real video data  
7. Supports natural-language/semantic queries (query-by-meaning)  
8. Mentions retrieval metrics (Recall@K, mAP, R@1, etc.)

#### Relevance Score
- Calculated as the **number of YES answers** across these 8 criteria.  
- Higher scores mean stronger alignment with the SLR goals.

#### Inclusion/Exclusion Decision
- **INCLUDE** if:
  - `Task relevant = YES` **AND**  
  - At least one of the following:  
    - `Has experiment = YES`  
    - `Mentions retrieval metrics = YES`  
    - `Supports natural-language queries = YES`
- **EXCLUDE** otherwise.  

The output for each paper includes:
- A structured JSON object (`all_screened_papers.json`)  
- A plain-text bullet summary (`all_screened_bullets.txt`)  
- Incremental checkpoints for recovery (`/data/screened_articles/checkpoints/`)  

### 7. Summarization of Highly Relevant Papers

From the **6,702 screened papers**, we identified **41 highly relevant papers** by selecting those with a **relevance score of 7 or 8** in `all_screened_papers.json` and got the same 41 papers from `all_screened_bullets.txt` to get them in bullet form as well.

We then performed **non-LLM summarization** in two ways:  

1. **Structured Table Summary** – Using `ijson`, we parsed the 41 papers to generate a **Markdown table** capturing titles, authors, key technologies, datasets, and applications.  

2. **Dataset & Method Counts** – We processed the same 41 papers to **count the occurrence of datasets, methods, and technologies**, clean dataset names (e.g., unify variations like `MSR-VTT` / `MSRVTT`), and produced a **concise Markdown summary (`highly_relevant_summary.md`)** showing exact counts and trends.  

These steps give both a **detailed review-ready table** and a **quantitative overview** of the core papers in the SLR.


### Advanced configuration

To re-use checkpoints or skipping steps, specify controls in `config.py`:

```python
arvix_fetch_path    = None     # set to a fetched paper json to skip arVix fetching
crossref_fetch_path = None     # set to a fetched paper json to skip crossref fetching
older_fetch_pathes  = []       # optional older paper json checkpoints 
all_fetched_path    = None     # set to a merged paper json path to skip fetching and enriching
all_screened_path   = None     # set to a merged file to skip LLM screening
```

## New: Single-source path control via config.py

All paths are configured in `config.py`. `src/*` modules provide pure functions without performing primary file writes; `main.py` orchestrates the full pipeline and handles saving.

Key config entries:

- `FETCHED_PAPERS_FOLDER`: where fetched and enriched JSON is stored
- `SCREENED_PAPERS_FOLDER`: where screened JSON/TXT and checkpoints are saved
- `SUMMARY_FOLDER`: where non-LLM summaries are saved
- `PDF_PAPERS_FOLDER`: drop any PDFs you want converted here
- `MARKDOWN_PAPERS_FOLDER`: converted Markdown output goes here
- `all_screened_papers_path`, `all_screened_bullets_path`, `SCREENING_CHECKPOINT_DIR`: standard screened outputs

## PDF → Markdown conversion (generalized)

- Place PDFs in `data/pdf_papers/` (configurable via `PDF_PAPERS_FOLDER`).
- `main.py` uses `src/marker_convert.py` pure helpers to convert them and writes the Markdown files to `data/markdown_papers/` (configurable via `MARKDOWN_PAPERS_FOLDER`).

## Developer notes (design rules)

- Do not embed hardcoded paths in `src/*` modules.
- `src/*` should expose pure, testable functions (operate on Python objects/strings), not save files.
- `main.py` should perform: load → process (call `src/*`) → save. Backups/checkpoints are allowed as optional behavior but are still orchestrated from `main.py`.

