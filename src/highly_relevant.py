"""Helpers for selecting highly relevant papers and extracting their bullets.

This module provides pure helpers and file-based wrappers. Prefer the pure
helpers from `main.py` so that saving is orchestrated in one place.
"""
# src/highly_relevant.py
import json
from pathlib import Path
from typing import List

def filter_highly_relevant_papers_from_list(all_papers: List[dict]):
    """Select in-memory the papers with INCLUDE decision and score 7 or 8.

    Args:
        all_papers: List of screened paper entries of the shape
            {"paper": {...}, "llm_screening": {...}}.

    Returns:
        List of entries meeting the highly-relevant criteria.
    """
    highly_relevant = []
    seen_links = set()
    for entry in all_papers:
        paper = entry.get("paper", {})
        llm_screening = entry.get("llm_screening", {})
        score = llm_screening.get("relevance_score", 0)
        decision = llm_screening.get("decision", "").upper()
        link = paper.get("link", "").strip().lower()
        if not link or link in seen_links:
            continue
        if score in [7, 8]:
            highly_relevant.append(entry)
            seen_links.add(link)
    return highly_relevant

def filter_highly_relevant_papers(json_path, output_json_path):
    """File-based wrapper: load JSON, filter, and save the result.

    Prefer using `filter_highly_relevant_papers_from_list` in the caller and
    save from there to keep I/O centralized in `main.py`.
    """
    """
    Filter highly relevant papers (relevance_score 7 or 8 and decision == 'INCLUDE') from JSON.

    Args:
        json_path (str or Path): Path to the full screened papers JSON.
        output_json_path (str or Path): Path to save the filtered highly relevant JSON.

    Returns:
        list: List of highly relevant paper entries.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        all_papers = json.load(f)
    highly_relevant = filter_highly_relevant_papers_from_list(all_papers)

    # Save filtered JSON
    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(highly_relevant, f, indent=2, ensure_ascii=False)

    print(f" Saved {len(highly_relevant)} highly relevant papers to {output_json_path}")
    return highly_relevant


def extract_bullets_for_highly_relevant_text(high_json: List[dict], full_bullets_text: str) -> str:
    """Pure helper: return bullets text for the provided highly relevant entries.

    Args:
        high_json: List of highly relevant paper entries.
        full_bullets_text: Contents of the full bullets TXT file.

    Returns:
        Concatenated bullets text for just the highly relevant papers.
    """
    # Create a set of titles of highly relevant papers
    high_titles = {entry.get("paper", {}).get("title", "").strip() for entry in high_json}

    # Split bullets by "Title:" and keep only relevant ones
    bullets_split = full_bullets_text.split("\nTitle: ")
    relevant_bullets = []
    for chunk in bullets_split:
        if not chunk.strip():
            continue
        lines = chunk.splitlines()
        title_line = lines[0].strip()
        title = title_line if "Title:" not in title_line else title_line.replace("Title:", "").strip()
        if title in high_titles:
            relevant_bullets.append(f"Title: {title}\n" + "\n".join(lines[1:]).strip())
    return "\n\n".join(relevant_bullets)


def extract_bullets_for_highly_relevant(high_json, bullets_txt_path, output_txt_path):
    """File-based wrapper: read full bullets, filter, and save the result.

    Prefer using `extract_bullets_for_highly_relevant_text` in the caller and
    save from there to keep I/O centralized in `main.py`.
    """
    """
    Extract bullets corresponding to highly relevant papers from full bullets TXT.

    Args:
        high_json (list): List of highly relevant papers (from filter_highly_relevant_papers).
        bullets_txt_path (str or Path): Path to the full bullets TXT.
        output_txt_path (str or Path): Path to save bullets of highly relevant papers.

    Returns:
        str: Concatenated bullets text of highly relevant papers.
    """
    # Read full bullets
    with open(bullets_txt_path, "r", encoding="utf-8") as f:
        full_bullets = f.read()

    # Build content using pure function
    content = extract_bullets_for_highly_relevant_text(high_json, full_bullets)

    # Save bullets
    Path(output_txt_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f" Saved bullets of {content.count('Title: ')} highly relevant papers to {output_txt_path}")
    return content
