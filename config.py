# ---------------- Search Query Terms ----------------
video_terms = [
    "semantic video retrieval",
    "video clip retrieval",
    "video QA",
    "video question answering",
    "multimodal video",
    "long video retrieval",
    "video search"
]

method_terms = [
    "action recognition",
    "context-aware",
    "object detection",
    "audio-visual",
    "speech recognition"
]

language_terms = [
    "natural language query",
    "semantic query",
    "language model",
    "LLM",
    "NLP"
]

# Generate query triplets (video + method + language)
QUERIES = [f"{v} AND {m} AND {lang}" 
           for v in video_terms 
           for m in method_terms 
           for lang in language_terms]

MAX_QUERIES = 250

# ---------------- Prompt Files ----------------
LLM_SUMMARIZATION_PROMPT_TXT = "data/summarization_prompt.txt"
LLM_SCREENING_PROMPT_TXT = "data/screening_prompt.txt"

# ---------------- Folders ----------------
FETCHED_PAPERS_FOLDER = "data/fetched_articles"
SCREENED_PAPERS_FOLDER = "data/screened_articles"
SUMMARY_FOLDER = "data/summaries"

# Generalized PDF/Markdown folders for user-provided papers
PDF_PAPERS_FOLDER = "data/pdf_papers"
MARKDOWN_PAPERS_FOLDER = "data/markdown_papers"

# Conditional skips to speed up reruns
SKIP_PDF_CONVERSION_IF_UP_TO_DATE = True  # if all PDFs already have .md, skip
SKIP_SUMMARIZATION_IF_EXISTS = True  # if a summary for MAX_QUERIES exists, skip

# ---------------- Optional Checkpoints / Skip Paths ----------------
arvix_fetch_path = "data/fetched_articles/raw_fetch/checkpoints/.arxiv_backup_2025-10-14T23-10-16.json" # Set to a fetched arXiv papers path to skip arXiv fetching
crossref_fetch_path = "data/fetched_articles/raw_fetch/checkpoints/.crossref_backup_2025-10-15T01-24-31.json" # Set to a fetched Crossref papers path to skip Crossref fetching
older_fetch_paths = []     # Optional older checkpoints

# Skip fetching if exists
all_fetched_path = "data/fetched_articles/enriched_6325_2025-10-15T05-06-28.json"
# Skip screening if exists
all_screened_papers_path = "data/screened_articles/all_screened_papers.json"
# Deduplicated bullet text
all_screened_bullets_path = "data/screened_articles/all_screened_bullets.txt"

# Screening checkpoints directory (created by main)
SCREENING_CHECKPOINT_DIR = f"{SCREENED_PAPERS_FOLDER}/checkpoints"

# ---------------- Enhanced Enrichment Options (No API Keys Required) ----------------
# Enable free fallback enrichment for papers without OpenAlex data
ENABLE_CROSSREF_FALLBACK = True      # Try CrossRef API (free, no key required)
ENABLE_IEEE_WEB_FALLBACK = True      # Try IEEE web scraping for IEEE papers
ENABLE_ELSEVIER_WEB_FALLBACK = True  # Try Elsevier web scraping for Elsevier papers

# Note: All fallbacks use free methods - no API keys required!
# - CrossRef: Free academic API
# - IEEE Web: Scrapes public IEEE Xplore pages
# - Elsevier Web: Scrapes public ScienceDirect pages