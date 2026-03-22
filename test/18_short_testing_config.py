VIDEO_TERMS = ["semantic video retrieval", "video clip retrieval", "video QA", "video question answering", "multimodal video", "long video retrieval", "video search"]
# METHOD_TERMS = ["action recognition", "context-aware", "object detection", "audio-visual", "speech recognition"]
# LANGUAGE_TERMS = ["natural language query", "semantic query", "language model", "LLM", "NLP"]

# # Generate query triplets (video + method + language)
# QUERIES = [
#     f"{v} AND {m} AND {l}"
#     for v in VIDEO_TERMS
#     for m in METHOD_TERMS
#     for l in LANGUAGE_TERMS
# ]

# MAX_QUERIES = 250

QUERIES = VIDEO_TERMS
MAX_QUERIES = 10

FETCHED_PAPERS_FOLDER = "data_TRIAL/fetched_articles"
arvix_fetch_path = None
crossref_fetch_path = None
older_fetch_pathes = [] # if there are older fetches you want to add
all_fetched_path = "data_TRIAL/fetched_articles/fetched_76_2025-09-17T22-56-13.json" 
# - will skip the fetching process

SCREENED_PAPERS_FOLDER = "data_TRIAL/screened_articles"
LLM_SCREENING_PROMPT_TXT = "data/screening_prompt.txt"
all_screened_path = None
# - will skip the screening process

SUMMARY_FOLDER = "data_TRIAL/summaries"