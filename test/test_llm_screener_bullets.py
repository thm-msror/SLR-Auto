# test/test_llm_screener_bullets.py
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env")

client = OpenAI(
    base_url="https://api.fanar.qa/v1",
    api_key=os.getenv("FANAR_API_KEY"),
)

RESULTS_DIR = "test/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
FETCHED_JSON_PATH = "test/fetched_articles.json"



def load_fetched_articles(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(paper):
    """
    Builds a bullet-point style LLM prompt for strict SLR screening.
    Extracts all fields from old JSON plus YES/NO inclusion criteria.
    """
    base_prompt = f"""
You are an expert SLR screener assistant. Be literal, conservative, and concise.
Output ONLY the bullet points exactly as requested. If you cannot answer exactly, output "FORMAT_ERROR".

Goal: Decide whether to INCLUDE or EXCLUDE this paper from the systematic literature review (SLR)
based on strict inclusion/exclusion criteria. Do NOT hallucinate; only use information in the title, abstract, or metadata.

Input Paper Metadata:
- Title: {paper.get("title")}
- Abstract: {paper.get("abstract")}
- Authors: {paper.get("authors")}
- Published: {paper.get("published")}
- DOI: {paper.get("doi")}
- Link: {paper.get("link")}

Instructions:

1. Evaluate the following inclusion/exclusion criteria. Mark each as YES / NO / INSUFFICIENT INFO.
   Provide brief explanation in parentheses if helpful.

   - Task relevant (video retrieval / QA / semantic search): YES if explicitly about video retrieval, QA, or semantic search tasks; NO if general action recognition/classification; INSUFFICIENT INFO if unclear.
   - Uses CV (detection, action recognition, scene understanding): YES if any computer vision methods used; NO if none.
   - Uses Audio/ASR (speech, audio-visual events): YES if audio, speech, or audio-visual events used; NO if none.
   - Uses NLP/LLM for query or answers: YES if natural language processing or LLMs are used; NO if not.
   - Multimodal fusion (vision+audio+text): YES if multiple modalities fused; NO if single modality.
   - Has experiment on real video data: YES if tested on real video datasets; NO if purely theoretical; INSUFFICIENT INFO if unclear.
   - Supports natural-language/semantic queries (query-by-meaning): YES if human-language queries supported; NO if not.
   - Mentions retrieval metrics (Recall@K, mAP, R@1, etc.): YES if evaluation reports standard retrieval metrics; NO if not.

2. Extract additional metadata fields exactly as listed:

   - Modalities: video, audio, text, multimodal (list all that apply)
   - Key technologies / methods: list any methods, models, or techniques mentioned
   - Datasets: list any datasets explicitly used or tested on
   - Application: domain or type of video/application
   - Limitations: list any limitations mentioned
   - Notes: 1–2 sentence highlight of the paper
   - Reason of relevance: short justification for INCLUDE/EXCLUDE decision

3. Make a strict INCLUDE/EXCLUDE decision using this rule:
   - INCLUDE if Task relevant = YES AND (Has experiment = YES OR Mentions retrieval metrics = YES OR Supports natural-language queries = YES)
   - Otherwise, EXCLUDE

Micro Examples (showing strict decisions):

Example 1 (EXCLUDE)
Paper metadata:
Title: "Action recognition on small datasets"
Abstract: "We propose a CNN classifier for action classification on UCF101."
-- Model output:
- Task relevant (video retrieval / QA / semantic search): NO
- Uses CV (detection, action recognition, scene understanding): YES
- Uses Audio/ASR: NO
- Uses NLP/LLM: NO
- Multimodal fusion (vision+audio+text): NO
- Has experiment on real video data: YES
- Supports natural-language/semantic queries (query-by-meaning): NO
- Mentions retrieval metrics (Recall@K, mAP, R@1, etc.): NO
- Modalities: video
- Key technologies / methods: CNN classifier
- Datasets: UCF101
- Application: action recognition
- Limitations: small dataset
- Notes: CNN-based action classification on UCF101.
- Reason of relevance: Only action classification; does not perform retrieval or semantic query.
- Decision: EXCLUDE
- Top evidence:
  - "CNN classifier for action classification on UCF101"

Example 2 (INCLUDE)
Paper metadata:
Title: "Semantic video retrieval with text queries"
Abstract: "We present a retrieval system mapping text queries to video embeddings. Recall@1 and mAP evaluated on MSR-VTT."
-- Model output:
- Task relevant (video retrieval / QA / semantic search): YES
- Uses CV (detection, action recognition, scene understanding): YES
- Uses Audio/ASR: NO
- Uses NLP/LLM: YES
- Multimodal fusion (vision+audio+text): YES
- Has experiment on real video data: YES
- Supports natural-language/semantic queries (query-by-meaning): YES
- Mentions retrieval metrics (Recall@K, mAP, R@1, etc.): YES
- Modalities: video, text
- Key technologies / methods: video-text embedding, retrieval model
- Datasets: MSR-VTT
- Application: video retrieval
- Limitations: none explicitly mentioned
- Notes: Maps text queries to video embeddings and evaluates with standard retrieval metrics.
- Reason of relevance: Explicit retrieval system, uses real video data, standard metrics evaluated.
- Decision: INCLUDE
- Top evidence:
  - "Recall@1 and mAP evaluated on MSR-VTT"

Output format (bullet points exactly, do not change the keys):
- Task relevant (video retrieval / QA / semantic search):
- Uses CV (detection, action recognition, scene understanding):
- Uses Audio/ASR:
- Uses NLP/LLM:
- Multimodal fusion (vision+audio+text):
- Has experiment on real video data:
- Supports natural-language/semantic queries (query-by-meaning):
- Mentions retrieval metrics (Recall@K, mAP, R@1, etc.):
- Modalities:
- Key technologies / methods:
- Datasets:
- Application:
- Limitations:
- Notes:
- Reason of relevance:
- Decision:
- Top evidence:
"""
    return base_prompt



def call_llm(paper):
    """
    Calls the LLM with the strict SLR prompt and returns the raw bullet-point output.
    """
    prompt = build_prompt(paper)
    response = client.chat.completions.create(
        model="Fanar",
        messages=[
            {"role": "system", "content": "You are an expert SLR screener assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


def parse_bullets_to_json(bullet_text):
    """
    Parses LLM bullet output into structured JSON for SLR screening.
    Handles:
      - YES/NO/INSUFFICIENT INFO fields with explanations in parentheses
      - repeated lines or messy formatting
      - Top evidence as a list
    Returns a dict with all required keys.
    """
    required_keys = [
        "Task relevant (video retrieval / QA / semantic search)",
        "Uses CV (detection, action recognition, scene understanding)",
        "Uses Audio/ASR",
        "Uses NLP/LLM",
        "Multimodal fusion (vision+audio+text)",
        "Has experiment on real video data",
        "Supports natural-language/semantic queries (query-by-meaning)",
        "Mentions retrieval metrics (Recall@K, mAP, R@1, etc.)",
        "Modalities",
        "Key technologies / methods",
        "Datasets",
        "Application",
        "Limitations",
        "Notes",
        "Reason of relevance",
        "Decision",
        "Top evidence"
    ]

    parsed = {k: "" for k in required_keys}  # initialize all keys

    current_key = None
    lines = bullet_text.splitlines()
    for line in lines:
        line = line.strip()
        if not line.startswith("-"):
            # continuation lines for Notes, Reason, or Top evidence
            if current_key and parsed[current_key]:
                if current_key == "Top evidence":
                    parsed[current_key].append(line)
                else:
                    parsed[current_key] += " " + line
            continue

        # split key/value
        try:
            key, value = line[1:].split(":", 1)
            key = key.strip()
            value = value.strip()
            if key not in required_keys:
                current_key = None
                continue

            # Handle Top evidence as list, splitting on semicolons or keeping as single entry
            if key == "Top evidence":
                parsed[key] = [v.strip() for v in value.split(";") if v.strip()]
            else:
                parsed[key] = value
            current_key = key
        except ValueError:
            current_key = None  # skip malformed lines

    # Normalize first 8 boolean fields to YES/NO/INSUFFICIENT INFO
    for k in required_keys[:8]:
        val = parsed[k].split("(")[0].strip().upper()  # remove any parentheses/explanations
        if val not in ["YES", "NO", "INSUFFICIENT INFO"]:
            val = "INSUFFICIENT INFO"
        parsed[k] = val

    # Ensure Top evidence is a list
    if not isinstance(parsed["Top evidence"], list):
        parsed["Top evidence"] = []

    # Deduplicate repeated lines in string fields
    for k in required_keys[8:-1]:  # skip first 8 and Top evidence
        if parsed[k]:
            # remove duplicate sentences (split by periods)
            sentences = [s.strip() for s in parsed[k].split(".") if s.strip()]
            seen = set()
            deduped = []
            for s in sentences:
                if s not in seen:
                    seen.add(s)
                    deduped.append(s)
            parsed[k] = ". ".join(deduped)

    return parsed


def relevance_score(parsed_json):
    """
    Computes a relevance score for an SLR paper based on strict inclusion criteria.

    Method:
    - We consider the first 8 fields of the LLM bullet output:
        1. Task relevant (video retrieval / QA / semantic search)
        2. Uses CV (detection, action recognition, scene understanding)
        3. Uses Audio/ASR
        4. Uses NLP/LLM
        5. Multimodal fusion (vision+audio+text)
        6. Has experiment on real video data
        7. Supports natural-language/semantic queries (query-by-meaning)
        8. Mentions retrieval metrics (Recall@K, mAP, R@1, etc.)
    - Each field is expected to have value YES / NO / INSUFFICIENT INFO.
    - Count 1 point for each field marked YES.
    - Fields marked NO or INSUFFICIENT INFO count as 0.
    - The final relevance score is the sum of YES values, ranging from 0 (no inclusion evidence) to 8 (strong evidence of inclusion).

    Example:
        parsed_json = {
            "Task relevant (video retrieval / QA / semantic search)": "YES",
            "Uses CV (detection, action recognition, scene understanding)": "NO",
            ...
        }
        relevance_score(parsed_json)  # returns the number of YES fields

    Returns:
        int: relevance score (0 to 8)
    """
    criteria_keys = [
        "Task relevant (video retrieval / QA / semantic search)",
        "Uses CV (detection, action recognition, scene understanding)",
        "Uses Audio/ASR",
        "Uses NLP/LLM",
        "Multimodal fusion (vision+audio+text)",
        "Has experiment on real video data",
        "Supports natural-language/semantic queries (query-by-meaning)",
        "Mentions retrieval metrics (Recall@K, mAP, R@1, etc.)",
    ]

    score = 0
    for k in criteria_keys:
        v = parsed_json.get(k, "NO").upper()
        if v == "YES":
            score += 1
    return score


def screen_test_papers(fetched_json_path):
    papers = load_fetched_articles(fetched_json_path)
    results = []
    for i, paper in enumerate(papers, 1):
        print(f"> Screening paper {i}: {paper['title']}")
        try:
            bullets = call_llm(paper)
            print("LLM bullet output:\n", bullets, "\n")

            parsed = parse_bullets_to_json(bullets)
            score = relevance_score(parsed)

            result = {
                "paper": paper,
                "llm_screening": parsed,
                "relevance_score": score,
            }
            results.append(result)

            # Save each result individually
            out_path = Path(RESULTS_DIR) / f"screened_paper_{i}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            print(f"Error screening paper {i}: {e}")

    # Save combined results
    combined_path = Path(RESULTS_DIR) / "screened_all.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    screen_test_papers(FETCHED_JSON_PATH)
