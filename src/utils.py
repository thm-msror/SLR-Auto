# src/utils.py
import os
import glob
import json
import re
from datetime import datetime

def save_json(data, folder = "data/saved", filename = "articles" ):
    timestamp = datetime.now().isoformat(timespec='seconds').replace(":", "-")
    """
    Save Python object to JSON file.
    """
    filename += f"{timestamp}.json"
    filepath = f"{folder}/{filename}"

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved JSON: {filepath}")

    return filepath

def save_checkpoint(data, folder, prefix):
    """Save JSON and keep only the last 3 files."""
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"{prefix}_{timestamp}.json"
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved JSON: {filepath}")

    # --- Keep only last 3 files ---
    files = sorted(glob.glob(os.path.join(folder, f"{prefix}_*.json")))
    if len(files) > 3:
        for old_file in files[:-3]:
            os.remove(old_file)
            print(f"🗑️ Deleted old checkpoint: {old_file}")

    return filepath

def load_json(filepath):
    """
    Load JSON file into Python object.
    """
    print(f"💾 Loading {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
    
def strip_json_comments(json_with_comments: str) -> str:
    cleaned_lines = []
    for line in json_with_comments.splitlines():
        # Remove anything after '#' unless it's inside a string (basic check)
        quote_open = False
        clean_line = ""
        for i, char in enumerate(line):
            if char == '"' and (i == 0 or line[i - 1] != '\\'):  # Toggle on unescaped "
                quote_open = not quote_open
            if char == '#' and not quote_open:
                break  # Comment found outside string
            clean_line += char
        cleaned_lines.append(clean_line.rstrip())
    cleaned_lines = "\n".join(cleaned_lines)
    
    return re.sub(r',(\s*[}\]])', r'\1', cleaned_lines) #remove trailing commas
    
def trim_spaces(text):
    try: 
        return " ".join(text.replace("\n", " ").split())
    except:
        return text

def clean_papers(raw_papers, remove_duplicates_only = False):
    """
    Clean and normalize raw arXiv metadata.
      - Strips title and abstract
      - Removes Duplicates

    Args:
        raw_papers (list[dict]): List of raw paper dicts.

    Returns:
        list[dict]: Cleaned metadata.
    """
    cleaned = []
    seen_links = set()

    for p in raw_papers:
        paper = p.copy()  # keep all existing keys

        link = paper.get("link", "")
        if link.lower() in seen_links: continue
        seen_links.add(link.lower())

        if remove_duplicates_only: continue


        # Clean title
        paper["title"] = trim_spaces(paper.get("title", ""))

        # Clean authors
        if "authors" in paper:
            if isinstance(paper["authors"], list):
                paper["authors"] = ", ".join(paper["authors"])

        # Clean abstract
        abstract = paper.get("abstract", "")
        if abstract: paper["abstract"] = trim_spaces(abstract)

        # Normalize publisher, dates, etc.
        for key, value in paper.items():
            try: 
                paper[key] = value.strip()
            except: continue

        cleaned.append(paper)

    print(f"🧹 Cleaned {len(cleaned)} unique papers from {len(raw_papers)} objects.")
    return cleaned

def save_md(content: str, folder: str = "data/saved", filename: str = "summary"):
    """
    Save Markdown content to a .md file with a timestamped filename.

    Args:
        content (str): Markdown text to save.
        folder (str): Target directory to save the file. Default is 'data/saved'.
        filename (str): Base filename (without extension). Default is 'summary'.

    Returns:
        str: The full file path of the saved Markdown file.
    """
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create full path
    filename += f"_{timestamp}.md"
    filepath = os.path.join(folder, filename)

    # Ensure folder exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"💾 Saved Markdown: {filepath}")
    return filepath

def append_to_json(new_data, filepath):
    """
    Append new data to an existing JSON file (list or dict).
    - If file doesn't exist, create it.
    - If existing is a list, extend with list or append a single dict.
    - If existing is a dict, update with new dict.
    """
    existing = None

    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            print(f"⚠️ Warning: Could not read {filepath}, starting fresh.")
            existing = None

    # Merge logic
    if existing is None:
        merged = new_data
    elif isinstance(existing, list):
        if isinstance(new_data, list):
            merged = existing + new_data
        else:
            merged = existing + [new_data]
    elif isinstance(existing, dict):
        if isinstance(new_data, dict):
            merged = {**existing, **new_data}
        else:
            raise ValueError("❌ Cannot append list to dict JSON file")
    else:
        raise ValueError("❌ Unsupported JSON structure")

    # Save back
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"💾 Appended JSON: {filepath}")
    return filepath
