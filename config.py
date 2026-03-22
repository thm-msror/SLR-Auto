import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
IEEE_API = os.getenv("IEEE_API")

# --- Default Prompt Paths ---
# These are the default instruction sets used by ATLAS.
LLM_SCREENING_PROMPT_TXT = r"prompts\screening_prompt.txt"
LLM_FULL_READ_PROMPT_TXT = r"prompts\pdf_reading_prompt.txt"
LLM_CATEGORIES_PROMPT_TXT = r"prompts\make_categories.txt"
LLM_SYNTHESIS_PROMPT_TXT = r"prompts\synthesize_category.txt"

# --- Storage Settings ---
# Note: In the Streamlit app, most data is stored in the '_runs' directory.
FETCHED_PAPERS_FOLDER  = r"data\1_fetched_papers"
SCREENED_PAPERS_FOLDER = r"data\2_screened_papers"
TOP_PAPERS_FOLDER      = r"data\3_top_papers"
READ_PAPERS_FOLDER     = r"data\4_read_papers"
SUMMARY_FOLDER         = r"data\5_summaries"