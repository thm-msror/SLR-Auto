# src/config.py
video_terms = ["semantic video retrieval", "video clip retrieval", "video QA", "video question answering", "multimodal video", "long video retrieval", "video search"]
method_temrms = ["action recognition", "context-aware", "object detection", "audio-visual", "speech recognition"]
laguage_terms = ["natural language query", "semantic query", "language model", "LLM", "NLP"]

QUERIES = [ f"{v} AND {m} AND {l}" # Generate query triplets (video + method + language)
           for v in video_terms for m in method_temrms for l in laguage_terms ]
MAX_QUERIES = 250

# ----------------------------------------------------------------------

SUMMARY_PROMPT_TXT = "data/summarization_prompt.txt"
LLM_SCREENING_PROMPT_TXT = "data/screening_prompt.txt"

# ----------------------------------------------------------------------

FETCHED_PAPERS_FOLDER = "data/fetched_articles"
SCREENED_PAPERS_FOLDER = "data/screened_articles"
SUMMARY_FOLDER = "data/summaries"

# ----------------------------------------------------------------------
# Optional file paths to skip parts of the program and/or use checkpoints

arvix_fetch_path    = None     # set to a fetched arVix papers path to skip arVix fetching
crossref_fetch_path = None     # set to a fetched crossref papers path to skip crossref fetching
older_fetch_pathes  = []       # optional older checkpoints
all_fetched_path    = "data/fetched_articles/fetched_6702_2025-09-19T10-10-59.json"     # set to a merged file path to skip fetching
all_screened_path   = None     # set to a merged file to skip screening

