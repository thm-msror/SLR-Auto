# src/summarize_crossref_md.py
import ijson
import json
import os
import textwrap

# default paths (can be overridden)
DEFAULT_INPUT_FILE = "data/screened_articles/Crossref_screened.json"
DEFAULT_TEMP_FILE = "data/summaries/Crossref_summary_tmp.json"
DEFAULT_OUTPUT_FILE = "data/summaries/Crossref_summary.md"

BATCH_SIZE = 500  # how often to save progress
WRAP_WIDTH = 80   # wrap long text at 80 characters

def wrap_text(text, width=WRAP_WIDTH):
    """Wrap text for Markdown display, preserving line breaks"""
    if not text:
        return ""
    return "<br>".join(textwrap.wrap(text, width=width))

def extract_summary(entry):
    paper = entry.get("paper", {})
    screening = entry.get("llm_screening", {})

    # Only include papers that are marked included AND have relevance above 5
    if not screening.get("included", False):
        return None
    if screening.get("relevance", 0) <= 5:
        return None

    title = paper.get("title", "N/A")
    link = paper.get("link") or (f"https://doi.org/{paper['doi']}" if paper.get("doi") else "")
    title_md = f"[{title}]({link})" if link else title

    authors = ", ".join(paper.get("authors", []))
    year = paper.get("published", "")[:4]
    publisher = paper.get("publisher", "N/A")

    datasets = screening.get("datasets") or []
    if not isinstance(datasets, list):
        datasets = [str(datasets)]
    datasets_str = ", ".join(datasets)

    methods = screening.get("key_technologies", []) + ([screening.get("task_type")] if screening.get("task_type") else [])
    methods_str = ", ".join(filter(None, methods))

    abstract = wrap_text(paper.get("abstract") or "")
    reason = wrap_text(screening.get("reason_of_relevance") or "")

    return {
        "title": title_md,
        "authors": authors,
        "year": year,
        "publisher": publisher,
        "datasets": datasets_str,
        "methods": methods_str,
        "abstract": abstract,
        "reason": reason,
        "relevance": screening.get("relevance", 0)
    }

def append_to_temp(batch, temp_file):
    """Append batch of results to temp file incrementally"""
    existing = []
    if os.path.exists(temp_file):
        with open(temp_file, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []

    existing.extend(batch)
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

def write_markdown(summaries, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Crossref Highly Relevant Papers (Included, Relevance > 5)\n\n")
        f.write("| Title | Authors | Year | Publisher | Datasets | Methods | Abstract | Reason of Relevance |\n")
        f.write("|-------|---------|------|-----------|----------|---------|---------|-------------------|\n")
        for r in summaries:
            f.write(
                f"| {r['title']} | {r['authors']} | {r['year']} | {r['publisher']} | {r['datasets']} | {r['methods']} | {r['abstract']} | {r['reason']} |\n"
            )

def summarize_crossref(input_file=None, temp_file=None, output_file=None):
    """Summarize Crossref screened papers without LLM"""

    input_file = input_file or DEFAULT_INPUT_FILE
    temp_file = temp_file or DEFAULT_TEMP_FILE
    output_file = output_file or DEFAULT_OUTPUT_FILE

    batch = []
    count = 0

    # Phase 1: Extract & save incrementally
    with open(input_file, "r", encoding="utf-8") as f:
        objects = ijson.items(f, "item")
        for entry in objects:
            summary = extract_summary(entry)
            if summary:
                batch.append(summary)
                count += 1

                if len(batch) >= BATCH_SIZE:
                    append_to_temp(batch, temp_file)
                    print(f"✅ Saved {len(batch)} more papers (total so far: {count})")
                    batch.clear()

    if batch:
        append_to_temp(batch, temp_file)
        print(f"✅ Saved final {len(batch)} papers (total: {count})")

    # Phase 2: Sort & write Markdown
    with open(temp_file, "r", encoding="utf-8") as f:
        summaries = json.load(f)

    summaries.sort(key=lambda x: (x['relevance'], x['year']), reverse=True)
    write_markdown(summaries, output_file)

    print(f"🎉 Done! Wrote {len(summaries)} highly relevant papers to {output_file}")