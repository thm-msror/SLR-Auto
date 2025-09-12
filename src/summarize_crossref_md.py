# src/summarize_crossref_md.py
import ijson
import json
import os

# default paths (can be overridden)
DEFAULT_INPUT_FILE = "data/screened_articles/Crossref_screened.json"
DEFAULT_TEMP_FILE = "data/summaries/Crossref_summary_tmp.json"
DEFAULT_OUTPUT_FILE = "data/summaries/Crossref_summary.md"

BATCH_SIZE = 500  # how often to save progress

def extract_summary(entry):
    paper = entry.get("paper", {})
    screening = entry.get("llm_screening", {})

    if not screening.get("included", False):
        return None
    if screening.get("relevance", 0) <= 5:
        return None

    title = paper.get("title", "N/A")
    link = paper.get("link") or (f"https://doi.org/{paper['doi']}" if paper.get("doi") else "")
    title_md = f"[{title}]({link})" if link else title

    datasets = screening.get("datasets") or []
    if not isinstance(datasets, list):
        datasets = [str(datasets)]

    citations = screening.get("citations") or []
    if not isinstance(citations, list):
        citations = [str(citations)]

    references = screening.get("references") or []
    if not isinstance(references, list):
        references = [str(references)]

    return {
        "title": title_md,
        "authors": ", ".join(paper.get("authors", [])),
        "year": paper.get("published", "")[:4],
        "datasets": ", ".join(datasets),
        "methods": ", ".join(filter(None, screening.get("key_technologies", []) + [screening.get("task_type")])),
        "citations": ", ".join(citations),
        "references": ", ".join(references),
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
        f.write("# Crossref Highly Relevant Papers (Sorted by Relevance)\n\n")
        f.write("| Title | Authors | Year | Datasets | Methodologies | Citations | References |\n")
        f.write("|-------|---------|------|----------|---------------|-----------|------------|\n")
        for r in summaries:
            f.write(f"| {r['title']} | {r['authors']} | {r['year']} | {r['datasets']} | {r['methods']} | {r['citations']} | {r['references']} |\n")

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


if __name__ == "__main__":
    main()
