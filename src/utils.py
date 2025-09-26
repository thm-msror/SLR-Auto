# src/utils.py
import os
import glob
import json
import re
from datetime import datetime

# ---------------- JSON / Checkpoint ----------------
def save_json(data, filepath):
    """
    Save Python object to JSON file.
    Overwrites existing file. No timestamp by default.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
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

    print(f" Saved JSON: {filepath}")

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
def clean_papers(raw_papers, remove_duplicates_only=False):
    """
    Clean papers and deduplicate based on DOI -> link -> title.
    """
    cleaned = []
    seen_ids = set()

    for p in raw_papers:
        paper = p.copy()
        # Use DOI first, then link, then title
        identifier = paper.get("paper", {}).get("doi") \
                     or paper.get("paper", {}).get("link") \
                     or paper.get("paper", {}).get("title", "").lower()
        if not identifier:
            continue
        if identifier.lower() in seen_ids:
            continue
        seen_ids.add(identifier.lower())

        if remove_duplicates_only:
            cleaned.append(paper)
            continue

        # Clean fields
        paper_data = paper.get("paper", {})
        paper_data["title"] = paper_data.get("title", "").strip()
        if "authors" in paper_data and isinstance(paper_data["authors"], list):
            paper_data["authors"] = ", ".join(paper_data["authors"])
        abstract = paper_data.get("abstract", "")
        if abstract:
            paper_data["abstract"] = abstract.strip()

        cleaned.append(paper)

    print(f" Cleaned {len(cleaned)} unique papers from {len(raw_papers)} objects.")
    return cleaned

def deduplicate_papers_by_link(papers):
    """
    Deduplicate papers based on 'link' field.
    Keeps first occurrence.
    """
    seen_links = set()
    deduped = []

    for entry in papers:
        paper = entry.get("paper", {})
        link = paper.get("link", "").strip().lower()
        if not link:
            continue
        if link in seen_links:
            continue
        seen_links.add(link)
        deduped.append(entry)

    print(f" Deduplicated {len(papers)} -> {len(deduped)} papers based on link.")
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
