# src/summarizer.py
import os
import json
from pathlib import Path
from collections import Counter

import ijson
from openai import OpenAI

# Initialize client lazily to avoid import-time errors
client = None

def get_client():
    global client
    if client is None:
        client = OpenAI(api_key=os.getenv("FANAR_API_KEY"), base_url="https://api.fanar.qa/v1")
    return client

# ---------------- Helper Functions ----------------
def first_sentence(text):
    """Return the first sentence from a string."""
    if not text:
        return "N/A"
    for sep in (".", "?", "!"):
        if sep in text:
            return text.split(sep)[0].strip() + sep
    return text.strip()


def list_to_text(val):
    """Convert list or string to comma-separated text."""
    if isinstance(val, list):
        return ", ".join(map(str, val))
    if isinstance(val, str):
        return val
    return ""


def sanitize_cell(text):
    """
    Clean text for safe one-line Markdown table cells.
    - Removes newlines and excessive spaces.
    - Escapes pipe characters.
    """
    if not text:
        return "N/A"
    text = str(text).replace("\n", " ").replace("\r", " ").strip()
    text = " ".join(text.split())  # collapse multiple spaces
    text = text.replace("|", "\\|")  # escape table pipes
    return text


def paper_table(papers):
    """
    Build a Markdown-safe, GitHub-renderable table of papers.
    Each row is a single line (no embedded newlines) to ensure
    proper rendering in GitHub Markdown preview.
    """
    header = (
        "| Title | Publisher | Link | Notes | Reason | Tech | Datasets | App | Limits | Evidence | Score |\n"
        "|-------|-----------|------|-------|--------|------|----------|-----|--------|----------|-------|"
    )
    rows = []
    publisher_counter = Counter()
    seen = set()

    for item in papers:
        paper = item.get("paper", {})
        screen = item.get("llm_screening", {})

        title = sanitize_cell(paper.get("title", "N/A"))
        publisher = paper.get("publisher")
        if not publisher:
            doi = paper.get("doi", "")
            link = paper.get("link", "")
            if doi.startswith("10.48550/arXiv") or "arxiv.org" in link:
                publisher = "arXiv"
            else:
                publisher = "N/A"
        publisher = sanitize_cell(publisher)

        # Link resolution
        if paper.get("datacite_url"):
            link = paper["datacite_url"]
        elif paper.get("openalex_id"):
            link = paper["openalex_id"]
        elif paper.get("doi"):
            link = f"https://doi.org/{paper['doi']}"
        else:
            link = "N/A"
        link = sanitize_cell(link)

        dedup_key = (title.strip(), publisher, link)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        publisher_counter[publisher] += 1

        notes = sanitize_cell(first_sentence(screen.get("notes", "")))
        reason = sanitize_cell(first_sentence(screen.get("reason_of_relevance", "")))
        key_tech = sanitize_cell(screen.get("key_technologies", "N/A"))
        datasets = sanitize_cell(screen.get("datasets", "N/A"))
        application = sanitize_cell(screen.get("application", "N/A"))
        limitations = sanitize_cell(screen.get("limitations", "N/A"))
        top_evidence = sanitize_cell(" ".join(screen.get("top_evidence", [])) or "N/A")
        relevance_score = sanitize_cell(screen.get("relevance_score", "N/A"))

        row = (
            f"| {title} | {publisher} | {link} | {notes} | {reason} | "
            f"{key_tech} | {datasets} | {application} | {limitations} | "
            f"{top_evidence} | {relevance_score} |"
        )
        rows.append(row)

    publisher_table_header = "\n\n| Publisher | Count |\n|-----------|-------|"
    publisher_table_rows = [f"| {pub} | {cnt} |" for pub, cnt in publisher_counter.items()]
    publisher_table_md = publisher_table_header + "\n" + "\n".join(publisher_table_rows)

    return header + "\n" + "\n".join(rows) + publisher_table_md

# ---------------- Non-LLM Table Summary ----------------
def summarize_no_llm(json_file):
    """
    Perform a non-LLM summary (table-based) using ijson to iterate through highly relevant JSON.
    Args:
        json_file (str or Path): Path to highly relevant papers JSON.
    Returns:
        str: Markdown table summarizing papers.
    """
    papers = []

    with open(json_file, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        while first_char.isspace():
            first_char = f.read(1)

    with open(json_file, "r", encoding="utf-8") as f:
        if first_char == "[":  # Root is array
            # FIX: "item" is correct when root is an array
            for entry in ijson.items(f, "item"):
                papers.append(entry)
        elif first_char == "{":  # Root is object with "papers"
            for entry in ijson.items(f, "papers.item"):
                papers.append(entry)
        else:
            raise ValueError("Unexpected JSON root type")

    # Debugging
    print(f"DEBUG: Parsed {len(papers)} papers from {json_file}")
    titles = [p.get("paper", {}).get("title", "N/A") for p in papers if isinstance(p, dict)]
    print("DEBUG Titles:", titles)

    return paper_table(papers)
