import os
import json
from datetime import datetime

timestamp = datetime.now().isoformat(timespec='seconds').replace(":", "-")

def save_json(data, filepath = f"data/save_articles/articles_{timestamp}.json", filename = None ):
    """
    Save Python object to JSON file.
    """
    if filename and not filepath: filepath = f"data/save_articles/{filename}_{timestamp}.json"

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved JSON: {filepath}")


def load_json(filepath):
    """
    Load JSON file into Python object.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_papers(raw_papers):
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
    seen_titles = set()

    for paper in raw_papers:
        title = paper.get("title", "").strip()
        abstract = paper.get("summary", "").strip()

        # Skip duplicates
        if title.lower() in seen_titles:
            continue
        seen_titles.add(title.lower())

        cleaned.append({
            "title": title,
            "authors": ", ".join(paper.get("authors", [])),
            "abstract": abstract,
            "published": paper.get("published", ""),
            "updated": paper.get("updated", ""),
            "arxiv_id": paper.get("id", ""),
            "url": paper.get("link", "")
        })

    print(f"🧹 Cleaned {len(cleaned)} unique papers.")
    return cleaned
