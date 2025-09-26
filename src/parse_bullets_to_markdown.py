# parse_bullets_to_markdown.py
import re
from collections import Counter, defaultdict
from pathlib import Path
import string

BULLETS_FILE = Path("data/screened_articles/highly_relevant_bullets.txt")
OUTPUT_MD = Path("data/screened_articles/highly_relevant_summary.md")

def parse_bullets_file(file_path):
    """
    Parses the bullets file into a list of paper dictionaries.
    """
    papers = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split by "Title:" markers
    entries = content.split("Title:")
    for entry in entries[1:]:  # skip anything before first title
        lines = entry.strip().split("\n")
        paper = {"Title": lines[0].strip()}
        for line in lines[1:]:
            if "-" not in line:
                continue
            key_value = line.strip().lstrip("- ").split(":", 1)
            if len(key_value) == 2:
                key, value = key_value
                paper[key.strip()] = value.strip()
        papers.append(paper)
    return papers

def clean_text(text):
    """
    Lowercase and remove punctuation for uniform counting.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

def extract_top_datasets(papers, top_n=10):
    """
    Count datasets individually and normalize names (merge variations like MSR-VTT/MSRVTT).
    """
    dataset_counter = Counter()
    dataset_domains = defaultdict(list)

    for p in papers:
        ds = p.get("Datasets", "N/A")
        app = p.get("Application", "N/A")
        if ds != "N/A":
            for d in ds.split(","):
                d_clean = clean_text(d)
                if d_clean:
                    dataset_counter[d_clean] += 1
                    dataset_domains[d_clean].append(app)

    top_datasets = dataset_counter.most_common(top_n)
    md_rows = []
    for rank, (ds, count) in enumerate(top_datasets, 1):
        domain = Counter(dataset_domains[ds]).most_common(1)[0][0] if dataset_domains[ds] else "N/A"
        md_rows.append(f"| {rank} | {ds} | {domain} | {count} | N/A |")
    return "| Rank | Dataset Name | Domain | Count | Most Common Year |\n" + "|------|-------------|--------|:-----:|----------------|\n" + "\n".join(md_rows)

def extract_top_methods(papers, top_n=10):
    """
    Count key technologies / methods individually, cleaning names.
    """
    method_counter = Counter()

    for p in papers:
        methods = p.get("Key technologies / methods", "N/A")
        if methods != "N/A":
            for m in methods.split(","):
                m_clean = clean_text(m)
                if m_clean:
                    method_counter[m_clean] += 1

    top_methods = method_counter.most_common(top_n)
    md_rows = []
    for rank, (method, count) in enumerate(top_methods, 1):
        md_rows.append(f"| {rank} | {method} | {count} |")
    return "| Rank | Key Technology / Method | Count |\n" + "|------|-----------------------|:-----:|\n" + "\n".join(md_rows)

def extract_notable_papers(papers):
    """
    Selects notable papers based on recency; here we take the last paper as 'most recent'.
    """
    if not papers:
        return "N/A"
    p = papers[-1]
    md = f"### Most Recent Relevant Paper\n\n"
    for k in ["Title", "Task relevant (video retrieval / QA / semantic search)", 
              "Uses CV (detection, action recognition, scene understanding)", 
              "Uses Audio/ASR", "Uses NLP/LLM", "Multimodal fusion (vision+audio+text)",
              "Has experiment on real video data", "Supports natural-language/semantic queries (query-by-meaning)",
              "Mentions retrieval metrics (Recall@K, mAP, R@1, etc.)", "Modalities",
              "Key technologies / methods", "Datasets", "Application", "Limitations", "Top evidence"]:
        value = p.get(k, "N/A")
        md += f"- **{k}:** {value}\n"
    return md

def generate_markdown(papers):
    md = "# Highly Relevant Papers Summary\n\n"
    md += "## 1. Datasets\n\n"
    md += extract_top_datasets(papers) + "\n\n"
    md += "## 2. Top Key Technologies / Methods\n\n"
    md += extract_top_methods(papers) + "\n\n"
    md += "## 3. Notable Papers\n\n"
    md += extract_notable_papers(papers) + "\n\n"
    return md

if __name__ == "__main__":
    papers = parse_bullets_file(BULLETS_FILE)
    markdown_text = generate_markdown(papers)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    print(f"Markdown summary saved to {OUTPUT_MD}")
