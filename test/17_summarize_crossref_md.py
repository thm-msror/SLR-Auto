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

def _flatten_and_stringify(seq):
    """Flatten one-level nested lists and turn all items into strings."""
    out = []
    for item in seq or []:
        if item is None:
            continue
        if isinstance(item, list) or isinstance(item, tuple):
            for sub in item:
                if sub is None:
                    continue
                out.append(str(sub))
        else:
            out.append(str(item))
    return out

def extract_summary(entry):
    paper = entry.get("paper", {}) or {}
    screening = entry.get("llm_screening", {}) or {}

    # Only include papers that are marked included AND have relevance above 5
    if not screening.get("included", False):
        return None
    if screening.get("relevance", 0) <= 8:
        return None

    title = paper.get("title", "N/A")
    link = paper.get("link") or (f"https://doi.org/{paper['doi']}" if paper.get("doi") else "")
    title_md = f"[{title}]({link})" if link else title

    authors = ", ".join(paper.get("authors", []))
    year = (paper.get("published") or "")[:4]
    publisher = paper.get("publisher") or "N/A"

    datasets = screening.get("datasets") or []
    if not isinstance(datasets, list):
        datasets = [str(datasets)]
    datasets_str = ", ".join(_flatten_and_stringify(datasets))

    # flatten key_technologies and task_type (task_type may be a string)
    key_tech = screening.get("key_technologies") or []
    task_type = screening.get("task_type")
    methods_list = []
    methods_list.extend(_flatten_and_stringify(key_tech))
    if task_type:
        if isinstance(task_type, list):
            methods_list.extend(_flatten_and_stringify(task_type))
        else:
            methods_list.append(str(task_type))
    # remove empty strings and duplicates while preserving order
    seen = set()
    methods_filtered = []
    for m in methods_list:
        m_str = m.strip()
        if not m_str:
            continue
        if m_str not in seen:
            seen.add(m_str)
            methods_filtered.append(m_str)
    methods_str = ", ".join(methods_filtered)

    abstract = wrap_text(paper.get("abstract") or "")
    reason = wrap_text(screening.get("reason_of_relevance") or "")

    # Semantic Scholar citation/reference counts, default to null if missing
    citations = paper.get("semanticScholar_citations")
    references = paper.get("semanticScholar_refs")
    citations = citations if citations is not None else None
    references = references if references is not None else None

    return {
        "title": title_md,
        "authors": authors,
        "year": year,
        "publisher": publisher,
        "datasets": datasets_str,
        "methods": methods_str,
        "citations": citations,
        "references": references,
        "abstract": abstract,
        "reason": reason,
        "relevance": int(screening.get("relevance", 0) or 0)
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
        f.write("| Title | Authors | Year | Publisher | Datasets | Methods | Citations | References | Abstract | Reason of Relevance |\n")
        f.write("|-------|---------|------|-----------|----------|---------|-----------|------------|---------|-------------------|\n")
        for r in summaries:
            f.write(
                f"| {r['title']} | {r['authors']} | {r['year']} | {r['publisher']} | {r['datasets']} | {r['methods']} | "
                f"{r['citations'] if r['citations'] is not None else ''} | "
                f"{r['references'] if r['references'] is not None else ''} | "
                f"{r['abstract']} | {r['reason']} |\n"
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
                    print(f"Saved {len(batch)} more papers (total so far: {count})")
                    batch.clear()

    if batch:
        append_to_temp(batch, temp_file)
        print(f"Saved final {len(batch)} papers (total: {count})")

    # Ensure temp file exists (safety)
    if not os.path.exists(temp_file):
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump([], f)

    # Phase 2: Sort & write Markdown
    with open(temp_file, "r", encoding="utf-8") as f:
        try:
            summaries = json.load(f)
        except json.JSONDecodeError:
            summaries = []

    # guard: all summaries should have 'relevance' and 'year' keys
    for s in summaries:
        s.setdefault("relevance", 0)
        s.setdefault("year", "")

    summaries.sort(key=lambda x: (int(x.get('relevance', 0)), x.get('year', "")), reverse=True)
    write_markdown(summaries, output_file)

    print(f"Done! Wrote {len(summaries)} highly relevant papers to {output_file}")
