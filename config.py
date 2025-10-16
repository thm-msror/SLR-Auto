# ---------------------- Search Query Terms ----------------------
video_terms = [ "semantic video retrieval", "video clip retrieval", "video QA", "video question answering", "multimodal video", "long video retrieval", "video search"]
method_terms = [ "action recognition", "context-aware", "object detection", "audio-visual", "speech recognition"]
language_terms = [ "natural language query", "semantic query", "language model", "LLM", "NLP"]

# Generate query triplets (video + method + language)
QUERIES = [f"{v} AND {m} AND {lang}" 
           for v in video_terms 
           for m in method_terms 
           for lang in language_terms]

MAX_QUERIES = 10

# ---------------------- LLM Prompts ----------------------
LLM_SUMMARIZATION_PROMPT_TXT = r"prompts\summarization_prompt.txt"
LLM_SCREENING_PROMPT_TXT     = r"prompts\screening_prompt.txt"

# ---------------------- Folders ----------------------
FETCHED_PAPERS_FOLDER  = r"data\1_fetched_papers"
SCREENED_PAPERS_FOLDER = r"data\2_screened_papers"
TOP_PAPERS_FOLDER      = r"data\3_top_papers"

# ---------------------- Checkpoints ------------------------
# Put paths of files to skip that pipeline. Comment out to run normally
# -----------------------------------------------------------
# saved_arvix_fetch      = 
# saved_crossref_fetch   = 
# saved_enriched_papers  = r"data\1_fetched_papers\enriched_6325_papers_2025-10-15T05-06-28.json"
# saved_screened_papers  = r"data\2_screened_papers\screened_6325_papers_2025-10-15T05-06-28.json"
# saved_top_papers       = r"data\3_top_papers\top_53_papers_20251016_212754.json"