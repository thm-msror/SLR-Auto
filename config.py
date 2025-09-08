VIDEO_TERMS = ["semantic video retrieval", "video clip retrieval", "video QA", "video question answering", "multimodal video", "long video retrieval", "video search"]
METHOD_TERMS = ["action recognition", "context-aware", "object detection", "audio-visual", "speech recognition"]
LANGUAGE_TERMS = ["natural language query", "semantic query", "language model", "LLM", "NLP"]

# Generate query triplets (video + method + language)
QUERIES = [
    f"{v} AND {m} AND {l}"
    for v in VIDEO_TERMS
    for m in METHOD_TERMS
    for l in LANGUAGE_TERMS
]

MAX_QUERIES = 250

FETCHED_PAPERS_FOLDER = "data/fetched_articles"
SCREENED_PAPERS_FOLDER = "data/screened_articles"
SUMMARY_FOLDER = "data/summaries"

LLM_SCREENING_PROMPT_TXT = "data/screening_prompt.txt"

# OPTIONAL: 
screened_arvix_path = "data/screened_articles/arXiv_200_2025-09-07T06-55-45.json"
screened_crossref_path = "data/screened_articles/crossref_200_2025-09-07T06-55-45.json"