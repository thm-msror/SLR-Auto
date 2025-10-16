# src/summarizer.py
from collections import Counter
from typing import List

def first_sentence(text):
    """Return the first sentence from a string."""
    if not text:
        return "N/A"
    for sep in (".", "?", "!"):
        if sep in text:
            return text.split(sep)[0].strip() + sep
    return text.strip()

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

# ============

def filter_top_papers(all_papers: List[dict], scoreList):
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
        if score in scoreList:
            highly_relevant.append(entry)
            seen_links.add(link)
    return highly_relevant

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
