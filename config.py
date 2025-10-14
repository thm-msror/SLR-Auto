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
QUERIES = [f"{v} AND {m} AND {l}" 
           for v in video_terms 
           for m in method_terms 
           for l in language_terms]

MAX_QUERIES = 250

# ---------------- Prompt Files ----------------
LLM_SUMMARIZATION_PROMPT_TXT = "data/summarization_prompt.txt"
LLM_SCREENING_PROMPT_TXT = "data/screening_prompt.txt"

# ---------------- Folders ----------------
FETCHED_PAPERS_FOLDER   = "data/fetched_articles"
SCREENED_PAPERS_FOLDER  = "data/screened_articles"
SUMMARY_FOLDER          = "data/summaries"

# Generalized PDF/Markdown folders for user-provided papers
PDF_PAPERS_FOLDER       = "data/pdf_papers"
MARKDOWN_PAPERS_FOLDER  = "data/markdown_papers"

# Conditional skips to speed up reruns
SKIP_PDF_CONVERSION_IF_UP_TO_DATE = True  # if all PDFs already have .md, skip
SKIP_SUMMARIZATION_IF_EXISTS      = True  # if a summary for MAX_QUERIES exists, skip

# ---------------- Optional Checkpoints / Skip Paths ----------------
arvix_fetch_path       = None   # Set to a fetched arXiv papers path to skip arXiv fetching
crossref_fetch_path    = None   # Set to a fetched Crossref papers path to skip Crossref fetching
older_fetch_paths      = []     # Optional older checkpoints

all_fetched_path        = "data/fetched_articles/fetched_6702_2025-09-19T10-10-59.json"  # Skip fetching if exists
all_screened_papers_path  = "data/screened_articles/all_screened_papers.json"            # Skip screening if exists
all_screened_bullets_path = "data/screened_articles/all_screened_bullets.txt"           # Deduplicated bullet text  

# Screening checkpoints directory (created by main)
SCREENING_CHECKPOINT_DIR = f"{SCREENED_PAPERS_FOLDER}/checkpoints"