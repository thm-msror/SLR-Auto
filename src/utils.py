# src/utils.py
import os
import glob
import json
import re
import time
from datetime import datetime
from pathlib import Path

# ---------------- Filename Sanitization ----------------
def safe_filename(text, max_len=150):
    text = re.sub(r'[\\/*?:"<>|\n\r\t]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_len]


# ---------------- JSON / Checkpoint ----------------
def save_json(content: str, folder: str = "data/saved", filename: str = "json"):
    """Save JSON content with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename += f"_{timestamp}.json"
    filepath = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)
    print(f" Saved JSON: {filepath}")
    return filepath


def save_checkpoint(data, folder, prefix):
    """
    Save JSON checkpoint with timestamp, keep only last 3 files.
    """
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"{prefix}_{timestamp}.json"
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Keep only last 3 files
    files = sorted(glob.glob(os.path.join(folder, f"{prefix}_*.json")))
    if len(files) > 3:
        for old_file in files[:-3]:
            os.remove(old_file)
            print(f" Deleted old checkpoint: {old_file}")

    return filepath

def load_json(filepath):
    """Load JSON file into Python object."""
    print(f" Loading {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------- PROMPT / INPUT ----------------
def load_prompt(path: str, default: str = "") -> str:
    prompt_path = Path(path)
    if not prompt_path.exists():
        return default
    content = prompt_path.read_text(encoding="utf-8")
    if default:
        stripped = content.strip()
        return stripped if stripped else default
    return content


def read_multiline_input(prompt: str) -> str:
    print(prompt)
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()

# ---------------- TEXT / CLEANING ----------------
def strip_json_comments(json_with_comments: str) -> str:
    """Remove # comments and trailing commas from JSON string."""
    cleaned_lines = []
    for line in json_with_comments.splitlines():
        quote_open = False
        clean_line = ""
        for i, char in enumerate(line):
            if char == '"' and (i == 0 or line[i - 1] != '\\'):
                quote_open = not quote_open
            if char == '#' and not quote_open:
                break
            clean_line += char
        cleaned_lines.append(clean_line.rstrip())
    cleaned_lines = "\n".join(cleaned_lines)
    return re.sub(r',(\s*[}\]])', r'\1', cleaned_lines)  # remove trailing commas

def trim_spaces(text):
    try:
        return " ".join(text.replace("\n", " ").split())
    except:
        return text

# ---------------- PAPER CLEANING / DEDUP ----------------
def normalize_text(text):
    """Normalize text for comparison: lowercase, strip, remove extra spaces."""
    if not text:
        return ""
    return " ".join(str(text).lower().strip().split())

def get_paper_identifier(paper_data):
    """Get normalized identifier (title + authors) for deduplication."""
    title = normalize_text(paper_data.get("title", ""))
    authors = paper_data.get("authors", "")
    
    # Handle authors - could be string or list
    if isinstance(authors, list):
        authors = ", ".join(authors)
    authors = normalize_text(authors)
    
    # Combine title and authors for unique identification
    return f"{title}|{authors}"

def deduplicate_papers_by_title_authors(papers, paper_type="fetched"):
    """
    Universal deduplication function for both fetched and screened papers.
    Deduplicates by title + authors combination.
    
    Args:
        papers: List of paper objects (fetched or screened format)
        paper_type: "fetched" or "screened" to handle different data structures
    
    Returns:
        List of deduplicated papers
    """
    seen_identifiers = set()
    deduped = []
    duplicates_removed = 0

    for paper in papers:
        # Handle different paper structures
        if paper_type == "screened":
            # Screened papers: {"paper": {...}, "llm_screening": {...}}
            paper_data = paper.get("paper", {})
        else:
            # Fetched papers: direct paper object
            paper_data = paper

        # Get normalized identifier
        identifier = get_paper_identifier(paper_data)
        
        # Skip if no title (invalid paper)
        if not identifier.split("|")[0]:  # No title
            continue
            
        # Check for duplicates
        if identifier in seen_identifiers:
            duplicates_removed += 1
            print(f"  Removed duplicate: {paper_data.get('title', 'N/A')[:50]}...")
            continue
            
        seen_identifiers.add(identifier)
        deduped.append(paper)

    print(f"  Deduplicated {len(papers)} -> {len(deduped)} {paper_type} papers by title+authors.")
    print(f"  Removed {duplicates_removed} duplicates.")
    return deduped

def clean_bullets(screened_papers):
    """
    Generate bullet text from screened papers.
    Deduplicates by title.
    """
    seen_titles = set()
    bullets = []

    for entry in screened_papers:
        paper = entry.get("paper", {})
        title = paper.get("title", "N/A")
        if title in seen_titles:
            continue
        seen_titles.add(title)

        llm_screening = entry.get("llm_screening", {})
        bullet_text = llm_screening.get("llm_screening_raw") or ""

        if not bullet_text:
            bullet_lines = []
            for key in [
                "notes", "reason_of_relevance", "key_technologies",
                "datasets", "application", "limitations", "decision", "top_evidence"
            ]:
                val = llm_screening.get(key)
                if isinstance(val, list):
                    val_text = "; ".join(map(str, val))
                else:
                    val_text = str(val or "")
                if val_text:
                    bullet_lines.append(f"- {key.replace('_', ' ').title()}: {val_text}")
            bullet_text = "\n".join(bullet_lines)

        bullets.append(f"Title: {title}\n{bullet_text.strip()}\n")

    return "\n".join(bullets)

# ---------------- MARKDOWN ----------------
def save_md(content: str, folder: str = "data/saved", filename: str = "summary"):
    """Save Markdown content with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename += f"_{timestamp}.md"
    filepath = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f" Saved Markdown: {filepath}")
    return filepath

# ---------------- APPEND TO JSON ----------------
def append_to_json(new_data, filepath):
    """Append to JSON (list or dict), create if not exists."""
    existing = None
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            print(f" Could not read {filepath}, starting fresh.")
            existing = None

    if existing is None:
        merged = new_data
    elif isinstance(existing, list):
        merged = existing + new_data if isinstance(new_data, list) else existing + [new_data]
    elif isinstance(existing, dict):
        if isinstance(new_data, dict):
            merged = {**existing, **new_data}
        else:
            raise ValueError(" Cannot append list to dict JSON file")
    else:
        raise ValueError(" Unsupported JSON structure")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f" Appended JSON: {filepath}")
    return filepath

# ---------------- STRING FORMATING ----------------
def print_time(t0, action_name) -> str:
    seconds = time.time() - t0
    if seconds < 60:
        total_time = f"{seconds:.2f} second{'s' if seconds != 1 else ''}"
    elif seconds < 3600:  # less than 1 hour
        minutes = seconds / 60
        total_time = f"{minutes:.2f} minute{'s' if minutes != 1 else ''}"
    elif seconds < 86400:  # less than 1 day
        hours = seconds / 3600
        total_time = f"{hours:.2f} hour{'s' if hours != 1 else ''}"
    else:
        days = seconds / 86400
        total_time = f"{days:.2f} day{'s' if days != 1 else ''}"
    
    print("="*80)
    print(f" >> {action_name} fetch took {total_time}")
    print("="*80)

    return total_time
