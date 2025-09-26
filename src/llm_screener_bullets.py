# src/llm_screener_bullets.py
import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from src.utils import append_to_json, save_checkpoint

load_dotenv(".env")

client = OpenAI(
    base_url=os.getenv("FANAR_BASE_URL", "https://api.fanar.qa/v1"),
    api_key=os.getenv("FANAR_API_KEY"),
)

SCREENED_ARTICLES_DIR = Path("data/screened_articles")
SCREENED_ARTICLES_DIR.mkdir(parents=True, exist_ok=True)

ALL_JSON_FILE = SCREENED_ARTICLES_DIR / "all_screened_papers.json"
ALL_BULLETS_FILE = SCREENED_ARTICLES_DIR / "all_screened_bullets.txt"
CHECKPOINT_DIR = SCREENED_ARTICLES_DIR / "checkpoints"

# ---------------- Prompt builder ----------------
def build_prompt(paper, prompt_txt_path):
    with open(prompt_txt_path, "r", encoding="utf-8") as f:
        base_prompt = f.read()
    meta = f"""
Input Paper Metadata:
- Title: {paper.get("title")}
- Abstract: {paper.get("abstract")}
- Authors: {paper.get("authors")}
- Published: {paper.get("published")}
- DOI: {paper.get("doi")}
- Link: {paper.get("link")}
"""
    return base_prompt + "\n" + meta

# ---------------- LLM call ----------------
def call_llm(paper, prompt_txt_path):
    prompt = build_prompt(paper, prompt_txt_path)
    response = client.chat.completions.create(
        model="Fanar",
        messages=[
            {"role": "system", "content": "You are an expert SLR screener assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()

# ---------------- Bullet parser ----------------
def parse_bullets_to_json(bullet_text):
    # Mapping for metadata fields
    keys_map = {
        "Notes": "notes",
        "Reason of relevance": "reason_of_relevance",
        "Key technologies / methods": "key_technologies",
        "Datasets": "datasets",
        "Application": "application",
        "Limitations": "limitations",
        "Decision": "decision",
        "Top evidence": "top_evidence",
    }

    # Yes/No/INSUFFICIENT INFO fields used for relevance scoring
    yes_no_keys = [
        "Task relevant (video retrieval / QA / semantic search)",
        "Uses CV (detection, action recognition, scene understanding)",
        "Uses Audio/ASR",
        "Uses NLP/LLM",
        "Multimodal fusion (vision+audio+text)",
        "Has experiment on real video data",
        "Supports natural-language/semantic queries (query-by-meaning)",
        "Mentions retrieval metrics (Recall@K, mAP, R@1, etc.)",
    ]

    parsed = {v: "" for v in keys_map.values()}
    parsed["top_evidence"] = []
    for k in yes_no_keys:
        parsed[k] = "INSUFFICIENT INFO"  # default value

    current_key = None
    for line in bullet_text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Handle multi-line continuation
        if not line.startswith("-") and current_key:
            if current_key == "top_evidence":
                parsed[current_key].append(line)
            else:
                parsed[current_key] += " " + line
            continue

        # Handle new bullet
        if line.startswith("-"):
            try:
                key, value = line[1:].split(":", 1)
                key = key.strip()
                value = value.strip()

                if key in keys_map:
                    target_key = keys_map[key]
                    if target_key == "top_evidence":
                        parsed[target_key] = [v.strip() for v in value.split(";") if v.strip()]
                    else:
                        parsed[target_key] = value
                    current_key = target_key

                elif key in yes_no_keys:
                    val_upper = value.split("(")[0].strip().upper()
                    parsed[key] = val_upper if val_upper in ["YES", "NO", "INSUFFICIENT INFO"] else "INSUFFICIENT INFO"
                    current_key = key

                else:
                    current_key = None

            except ValueError:
                current_key = None

    # Add raw bullet text
    parsed["llm_screening_raw"] = bullet_text
    return parsed

# ---------------- Relevance scoring ----------------
def relevance_score(parsed_json):
    criteria = [
        "Task relevant (video retrieval / QA / semantic search)",
        "Uses CV (detection, action recognition, scene understanding)",
        "Uses Audio/ASR",
        "Uses NLP/LLM",
        "Multimodal fusion (vision+audio+text)",
        "Has experiment on real video data",
        "Supports natural-language/semantic queries (query-by-meaning)",
        "Mentions retrieval metrics (Recall@K, mAP, R@1, etc.)",
    ]
    return sum(1 for k in criteria if parsed_json.get(k, "NO").upper() == "YES")

# ---------------- Main screening function ----------------
def screen_papers(papers, batch_size=50, model="Fanar", prompt_txt_path="screening_prompt.txt"):
    import time

    screened_so_far = []
    if ALL_JSON_FILE.exists():
        with open(ALL_JSON_FILE, "r", encoding="utf-8") as f:
            screened_so_far = json.load(f)
        print(f" Resuming: {len(screened_so_far)} papers already screened.")
    else:
        print("  Starting fresh screening run...")

    seen_ids = {p["paper"].get("link") or p["paper"].get("title") for p in screened_so_far}
    results = screened_so_far.copy()

    for i in range(0, len(papers), batch_size):
        batch = [p for p in papers[i:i+batch_size] if (p.get("link") or p.get("title")) not in seen_ids]
        if not batch:
            continue

        print(f"\n📝 Screening batch {i//batch_size+1} ({len(batch)} papers)...")
        bullets = []
        batch_results = []

        for paper in batch:
            try:
                print(f"> {paper.get('title','N/A')}")
                bullet_text = call_llm(paper, prompt_txt_path)
                bullets.append(bullet_text)

                parsed_json = parse_bullets_to_json(bullet_text)
                parsed_json["relevance_score"] = relevance_score(parsed_json)

                entry = {
                    "paper": paper,
                    "llm_screening": parsed_json
                }

                batch_results.append(entry)

            except Exception as e:
                print(f" Error screening paper '{paper.get('title', 'unknown')}': {e}")
                continue

        # Append bullets to TXT
        with open(ALL_BULLETS_FILE, "a", encoding="utf-8") as f:
            for paper, bullet_text in zip(batch, bullets):
                f.write(f"Title: {paper.get('title','N/A')}\n")
                f.write(bullet_text.strip() + "\n\n")

        results.extend(batch_results)
        for p in batch:
            seen_ids.add(p.get("link") or p.get("title"))

        append_to_json(batch_results, ALL_JSON_FILE)
        save_checkpoint(results, CHECKPOINT_DIR, prefix="screened")

        time.sleep(1)

    print(f"\n Finished screening. Total papers: {len(results)}")
    return results
