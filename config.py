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

# ---------------------- Reading Guides ----------------------

CRITERIA = [
    "Task relevant (video retrieval / QA / semantic search)",
    "Uses CV (detection, action recognition, scene understanding)",
    "Uses Audio/ASR",
    "Uses NLP/LLM",
    "Multimodal fusion (vision+audio+text)",
    "Has experiment on real video data",
    "Supports natural-language/semantic queries (query-by-meaning)",
    "Mentions retrieval metrics (Recall@K, mAP, R@1, etc.)",
]

GAPS = ["video_segmentation", "frame_sampling_method", "input_video_length", "spatiotemporal_analysis", "visual_analysis", 
"speech_audio_analysis", "sound_audio_analysis", "qa_interaction", "retrieval_level", "inferring_method", 
"video_representation", "model_pipeline", "dataset", "comparisons", "hyperparameters", "environment", 
"repository", "authors"]

# ---------------------- LLM Prompts ----------------------
LLM_SUMMARIZATION_PROMPT_TXT = r"prompts\summarization_prompt.txt"
LLM_SCREENING_PROMPT_TXT     = r"prompts\screening_prompt.txt"
LLM_FULL_READ_PROMPT_TXT     = r"prompts\pdf_reading_prompt.txt"  

# ---------------------- Folders ----------------------
FETCHED_PAPERS_FOLDER  = r"data\1_fetched_papers"
SCREENED_PAPERS_FOLDER = r"data\2_screened_papers"
TOP_PAPERS_FOLDER      = r"data\3_top_papers"
READ_PAPERS_FOLDER     = r"data\4_read_papers"
SUMMARY_FOLDER         = r"data\5_summaries"


# ---------------------- Checkpoints ------------------------
# Uncomment to skip that pipeline. Comment out to run normally
# To skip a step all other steps before it must be uncommented
# -----------------------------------------------------------
# saved_arvix_fetch      = None
# saved_crossref_fetch   = None
saved_enriched_papers  = r"data\1_fetched_papers\enriched_6325_papers_2025-10-15T05-06-28.json"
saved_screened_papers  = r"data\2_screened_papers\screened_6325_papers_2025-10-15T05-06-28.json"
saved_top_papers       = r"data\3_top_papers\top_53_papers_20251016_212754.json"
skip_md_conversion     = True
saved_read_papers      = r"data\4_read_papers\full_read.json"
saved_gap_reviews      = r"data\5_summaries\paper_reviews.json"