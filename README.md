# AI-Backed Systematic Literature Review (SLR) Automation

This project automates large-scale **systematic literature reviews (SLRs)** using APIs, enrichment tools, and LLM-based screening. Instead of manually fetching and filtering hundreds of papers, the pipeline streamlines the process from **query → fetch → screen → summarize**.


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
│   ├── fetched_articles/  # raw + enriched papers
│   ├── screened_articles/ # LLM-based screened papers
│   └── summaries/         # LLM-based & table summary markdowns
├── src/
│   ├── fetch_arxiv.py     # arXiv fetcher
│   ├── fetch_crossref.py  # Crossref fetcher
│   ├── enrich_openalex.py # OpenAlex enrichment
│   ├── llm_screener.py    # LLM-based screening
│   └── utils.py           # helpers (JSON save/load, dedup)
├── config.py              # search queries & other options
└── main.py                # orchestrates fetch pipeline
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


### Advanced configuration

To re-use checkpoints or skipping steps, specify controls in `config.py`:

```python
arvix_fetch_path    = None     # set to a fetched paper json to skip arVix fetching
crossref_fetch_path = None     # set to a fetched paper json to skip crossref fetching
older_fetch_pathes  = []       # optional older paper json checkpoints 
all_fetched_path    = None     # set to a merged paper json path to skip fetching and enriching
all_screened_path   = None     # set to a merged file to skip LLM screening
```

