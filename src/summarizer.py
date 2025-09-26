# src/summarizer.py
import ijson
import json
from openai import OpenAI
import os
from pathlib import Path

client = OpenAI(api_key=os.getenv("FANAR_API_KEY"), base_url="https://api.fanar.qa/v1")

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


# ---------------- Paper Table ----------------
def paper_table(papers):
    """
    Generate a Markdown table summarizing papers with specific fields.
    Limits notes, reason_of_relevance, and top evidence to first sentence.
    """
    header = (
        "| Title | OpenAlex ID | Link | Publisher | Notes | Reason of Relevance | "
        "Key Technologies | Datasets | Application | Limitations | Top Evidence | Relevance Score |\n"
        "|-------|------------|------|-----------|-------|------------------|-----------------|---------|------------|------------|--------------|----------------|"
    )
    rows = []

    for item in papers:
        paper = item.get("paper", {})
        screen = item.get("llm_screening", {})

        title = paper.get("title", "N/A")
        openalex_id = paper.get("openalex_id", "N/A")
        link = paper.get("link", "N/A")
        publisher = paper.get("publisher", "N/A")
        notes = first_sentence(screen.get("notes", ""))
        reason = first_sentence(screen.get("reason_of_relevance", ""))
        key_tech = screen.get("key_technologies", "N/A")
        datasets = screen.get("datasets", "N/A")
        application = screen.get("application", "N/A")
        limitations = screen.get("limitations", "N/A")
        top_evidence = first_sentence(" ".join(screen.get("top_evidence", [])))
        relevance_score = screen.get("relevance_score", "N/A")  # <-- fix here

        rows.append(
            f"| {title} | {openalex_id} | {link} | {publisher} | {notes} | {reason} | "
            f"{key_tech} | {datasets} | {application} | {limitations} | {top_evidence} | {relevance_score} |"
        )


    return header + "\n" + "\n".join(rows)

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
        for entry in ijson.items(f, "item"):
            papers.append(entry)

    return paper_table(papers)