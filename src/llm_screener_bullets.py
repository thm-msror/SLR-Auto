# src/llm_screener_bullets.py
import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from src.utils import append_to_json, save_checkpoint

load_dotenv(".env")

client = OpenAI(
    base_url=os.getenv("FANAR_BASE_URL", "https://api.fanar.qa/v1"),
    api_key=os.getenv("FANAR_API_KEY"),
)

# Default paths are deprecated here; the caller (main) should supply them.
# Keeping optional defaults for backwards compatibility via config if imported directly.
try:
    import config as _config
    SCREENED_ARTICLES_DIR = Path(_config.SCREENED_PAPERS_FOLDER)
    ALL_JSON_FILE = Path(_config.all_screened_papers_path)
    ALL_BULLETS_FILE = Path(_config.all_screened_bullets_path)
    CHECKPOINT_DIR = Path(_config.SCREENING_CHECKPOINT_DIR)
except Exception:
    SCREENED_ARTICLES_DIR = Path("data/screened_articles")
    ALL_JSON_FILE = SCREENED_ARTICLES_DIR / "all_screened_papers.json"
    ALL_BULLETS_FILE = SCREENED_ARTICLES_DIR / "all_screened_bullets.txt"
    CHECKPOINT_DIR = SCREENED_ARTICLES_DIR / "checkpoints"

# ---------------- Prompt builder ----------------
def build_prompt(paper, prompt_txt_path):
    """Build the LLM prompt from a paper dict and a prompt template file.

    Args:
        paper: Dict with at least title, abstract, authors, published, doi, link.
        prompt_txt_path: Path to a text prompt template.

    Returns:
        Full prompt string ready to send to the model.
    """
    with open(prompt_txt_path, "r", encoding="utf-8") as f:
        base_prompt = f.read()
    meta = f"""
Input Paper Metadata:
- Title: {paper.get("title")}
- Abstract: {paper.get("abstract")}
- Authors: {paper.get("authors")}
- Published: {paper.get("published")}
- DOI: {paper.get("doi")}
- Link: {paper.get("link")}
"""
    return base_prompt + "\n" + meta

# ---------------- LLM call ----------------
def call_llm(paper, prompt_txt_path):
    """Call the FANAR chat completion API for a single paper.

    Args:
        paper: Paper metadata dict.
        prompt_txt_path: Path to a text prompt template.

    Returns:
        The assistant message content (bullet-style screening text).
    """
    prompt = build_prompt(paper, prompt_txt_path)
    response = client.chat.completions.create(
        model="Fanar",
        messages=[
            {"role": "system", "content": "You are an expert SLR screener assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()

# ---------------- Bullet parser ----------------
def parse_bullets_to_json(bullet_text, criteria):
    """Parse bullet-style screening output into a structured dict.

    Args:
        bullet_text: The LLM-produced bullets string.

    Returns:
        Dict with normalized fields and YES/NO/INSUFFICIENT INFO flags.
    """
    # Mapping for metadata fields
    keys_map = {
        "Notes": "notes",
        "Reason of relevance": "reason_of_relevance",
        "Repository Link": "repository",
        "Key technologies / methods": "key_technologies",
        "Datasets": "datasets",
        "Application": "application",
        "Limitations": "limitations",
        "Decision": "decision",
        "Top evidence": "top_evidence",
    }

    parsed = {v: "" for v in keys_map.values()}
    parsed["top_evidence"] = []
    for k in criteria:
        parsed[k] = "INSUFFICIENT INFO"  # default value

    current_key = None
    for line in bullet_text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Handle multi-line continuation
        if not line.startswith("-") and current_key:
            if current_key == "top_evidence":
                parsed[current_key].append(line)
            else:
                parsed[current_key] += " " + line
            continue

        # Handle new bullet
        if line.startswith("-"):
            try:
                key, value = line[1:].split(":", 1)
                key = key.strip()
                value = value.strip()

                if key in keys_map:
                    target_key = keys_map[key]
                    if target_key == "top_evidence":
                        parsed[target_key] = [v.strip() for v in value.split(";") if v.strip()]
                    else:
                        parsed[target_key] = value
                    current_key = target_key

                elif key in criteria:
                    val_upper = value.split("(")[0].strip().upper()
                    if "YES" in val_upper:
                        parsed[key] = "YES"
                    elif "INSUFFICIENT INFO" in val_upper:
                        parsed[key] = "INSUFFICIENT INFO"
                    elif "NO" in val_upper:
                        parsed[key] = "NO"
                    else:
                        parsed[key] = "INSUFFICIENT INFO"
                    current_key = key

                else:
                    current_key = None

            except ValueError:
                current_key = None

    # Add raw bullet text
    parsed["llm_screening_raw"] = bullet_text
    return parsed

# ---------------- Relevance scoring ----------------
def relevance_score(parsed_json, criteria):
    """Compute relevance score as count of YES among predefined criteria."""
    return sum(1 for k in criteria if parsed_json.get(k, "NO").upper() == "YES")

# ---------------- Main screening function ----------------
def screen_papers(papers, prompt_txt_path, criteria, batch=50, track=False):
    """
    Fetches papers from arXiv API given a list of queries.

    Args:
        queries (list[str]): Search terms (keywords).
        max_results (int): Max total papers per query.
        start (int): Start index for pagination.
        per_query (int): How many results to fetch per API call (max=2000).
        delay (int): Seconds to wait between API calls (to respect rate limits).
        track (bool|str): Directory path to store raw fetches if True/str.

    Returns:
        list[dict]: List of papers with metadata.
    """
    all_screened = []
    errors = []

    # Handle track as a directory
    track_dir = None
    if track:
        track_dir = "track" if track is True else str(track)
        os.makedirs(track_dir, exist_ok=True)
        # Save a timestamped checkpoint of the input queries for traceability
        save_checkpoint(papers, track_dir, ".all_papers_to_be_screened") 
    totalq = len(papers)

    for i, paper in enumerate(papers):
        try:
            print(f"  > ({i} of {totalq}) {paper.get('title','N/A')}")
            bullet_text = call_llm(paper, prompt_txt_path)

            parsed_json = parse_bullets_to_json(bullet_text, criteria)
            parsed_json["relevance_score"] = relevance_score(parsed_json, criteria)

            entry = {
                "paper": paper,
                "llm_screening": parsed_json
            }
            

            all_screened.append(entry)

        except Exception as e:
            error_message = f"ERROR screening paper '{paper.get('title', 'unknown')}': {e}"
            print(f"!!! {error_message}")
            error = {
                "paper": paper,
                "llm_screening": error_message
            }
            errors.append(error)

            continue

        # Save backup per query
        if track_dir and i%batch == 0:
            backup_path = save_checkpoint(all_screened, track_dir, ".screening_backup")
            errors_path = save_checkpoint(errors, track_dir, ".screening_errors")
            queries_dir = save_checkpoint(papers[i:], track_dir, ".screening_queries_remaining")
            
        
    return all_screened #, errors