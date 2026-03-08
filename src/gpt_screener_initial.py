from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from gpt_client import call_gpt_chat


DEFAULT_PROMPT = (
    "You are screening papers for a systematic literature review (SLR).\n"
    "Use ONLY the provided title, abstract, year, and publisher.\n"
    "For each criterion, answer YES, NO, or INSUFFICIENT based on the paper metadata.\n"
    "If the information is not stated, answer INSUFFICIENT.\n\n"
    "Output format (strict):\n"
    "C1: YES|NO|INSUFFICIENT\n"
    "C2: YES|NO|INSUFFICIENT\n"
    "...\n"
    "No explanations, no extra text."
)

def load_prompt(path: str) -> str:
    prompt_path = Path(path)
    if not prompt_path.exists():
        return DEFAULT_PROMPT
    content = prompt_path.read_text(encoding="utf-8").strip()
    return content or DEFAULT_PROMPT


def _extract_year(paper: Dict[str, Any]) -> str:
    for key in ("year", "published_year"):
        val = paper.get(key)
        if isinstance(val, int):
            return str(val)
        if isinstance(val, str) and val.strip():
            return val.strip()
    published = paper.get("published")
    if isinstance(published, str) and len(published) >= 4:
        return published[:4]
    return ""


def build_prompt(paper: Dict[str, Any], criteria: List[str], prompt_txt_path: str) -> str:
    base = load_prompt(prompt_txt_path)

    title = paper.get("title", "") or ""
    abstract = paper.get("abstract", "") or ""
    year = _extract_year(paper)
    publisher = paper.get("publisher", "") or ""
    period = paper.get("period", "") or ""

    meta = [
        "Input Paper Metadata:",
        f"- Title: {title}",
        f"- Abstract: {abstract}",
    ]
    if year:
        meta.append(f"- Year: {year}")
    if publisher:
        meta.append(f"- Publisher: {publisher}")
    if period:
        meta.append(f"- Period: {period}")

    criteria_lines = [
        f"C{i+1}. {_strip_criterion_prefix(c)}" for i, c in enumerate(criteria)
    ]

    return (
        base
        + "\n\n"
        + "\n".join(meta)
        + "\n\nCriteria:\n"
        + "\n".join(criteria_lines)
    )


def _strip_criterion_prefix(text: str) -> str:
    stripped = text.strip()
    upper = stripped.upper()
    if upper.startswith("INCLUDE:"):
        return stripped.split(":", 1)[1].strip()
    if upper.startswith("EXCLUDE:"):
        return stripped.split(":", 1)[1].strip()
    return stripped


def read_multiline_input(prompt: str) -> str:
    print(prompt)
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def call_llm_screen(
    paper: Dict[str, Any],
    criteria: List[str],
    prompt_txt_path: str = "prompts/screen_initial.txt",
) -> str:
    prompt = build_prompt(paper, criteria, prompt_txt_path)
    return call_gpt_chat(
        messages=[
            {"role": "system", "content": "You are an expert SLR screener assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=800,
    )


def parse_screening_answers(raw: str, criteria: List[str]) -> List[Dict[str, str]]:
    answers = ["INSUFFICIENT" for _ in criteria]

    pattern = re.compile(
        r"^C(\d+)\s*[:\-]\s*(YES|NO|INSUFFICIENT|INCLUDED|EXCLUDED)\b",
        re.IGNORECASE,
    )
    for line in raw.splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(answers):
            token = m.group(2).upper()
            criterion = (criteria[idx] or "").strip().upper()
            if token in {"INCLUDED", "EXCLUDED"}:
                if criterion.startswith("EXCLUDE:"):
                    answers[idx] = "YES" if token == "EXCLUDED" else "NO"
                else:
                    answers[idx] = "YES" if token == "INCLUDED" else "NO"
            else:
                answers[idx] = token

    return [
        {"criterion": criteria[i], "answer": answers[i]}
        for i in range(len(criteria))
    ]


def relevance_score(parsed: List[Dict[str, str]]) -> int:
    score = 0
    for item in parsed:
        criterion = (item.get("criterion") or "").strip()
        answer = (item.get("answer") or "").strip().upper()
        if criterion.upper().startswith("INCLUDE"):
            if answer == "YES":
                score += 1
            elif answer == "NO":
                score -= 1
        elif criterion.upper().startswith("EXCLUDE"):
            if answer == "NO":
                score += 1
            elif answer == "YES":
                score -= 1
    return score


def count_answers(parsed: List[Dict[str, str]]) -> Dict[str, Dict[str, int]]:
    counts = {
        "include": {"yes": 0, "no": 0, "insufficient": 0},
        "exclude": {"yes": 0, "no": 0, "insufficient": 0},
    }
    for item in parsed:
        answer = (item.get("answer") or "").strip().upper()
        criterion = (item.get("criterion") or "").strip().upper()
        if criterion.startswith("INCLUDE"):
            group = "include"
        elif criterion.startswith("EXCLUDE"):
            group = "exclude"
        else:
            continue

        if answer == "YES":
            counts[group]["yes"] += 1
        elif answer == "NO":
            counts[group]["no"] += 1
        else:
            counts[group]["insufficient"] += 1
    return counts


def screen_paper(
    paper: Dict[str, Any],
    criteria: List[str],
    prompt_txt_path: str = "prompts/screen_initial.txt",
) -> Dict[str, Any]:
    raw = call_llm_screen(paper, criteria, prompt_txt_path)
    parsed = parse_screening_answers(raw, criteria)
    score = relevance_score(parsed)
    counts = count_answers(parsed)
    return {
        "relevance_score": score,
        "counts": counts,
        "answers": parsed,
        "raw": raw,
    }


DEFAULT_PAPER = {
    "title": (
        "ViTA: An Efficient Video-to-Text Algorithm using VLM for RAG-based "
        "Video Analysis System"
    ),
    "abstract": (
        "Retrieval-augmented generation (RAG) is used in natural language processing "
        "(NLP) to provide query-relevant information in enterprise documents to "
        "large language models (LLMs). When enterprise data is primarily videos, "
        "vision-language models (VLMs) are necessary to convert information in "
        "videos into text. While essential, this conversion is a bottleneck, "
        "especially for large corpora of videos, delaying timely use of enterprise "
        "videos to generate useful responses. We propose ViTA, a method that "
        "leverages two characteristics of VLMs to expedite conversion. As VLMs "
        "output more text tokens, they incur higher latency, and large VLMs extract "
        "more details but are slower per token than lightweight VLMs. ViTA first "
        "uses a lightweight VLM to capture the gist of an image or video clip, then "
        "guides a heavyweight VLM to extract additional details using a preset "
        "small number of output tokens. Experiments show ViTA reduces conversion "
        "time by up to 43% without compromising response accuracy compared to a "
        "baseline using only a heavyweight VLM."
    ),
    "year": "2024",
    "publisher": "CVPRW 2024 (Computer Vision Foundation)",
}

DEFAULT_PAPER_EXCLUDE = {
    "title": (
        "T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video "
        "Generation"
    ),
    "abstract": (
        "Text-to-video (T2V) generation models have advanced significantly, yet their "
        "ability to compose different objects, attributes, actions, and motions into "
        "a video remains unexplored. Previous text-to-video benchmarks also neglect "
        "this important ability for evaluation. In this work, we conduct the first "
        "systematic study on compositional text-to-video generation. We propose "
        "T2V-CompBench, the first benchmark tailored for compositional text-to-video "
        "generation. T2V-CompBench encompasses diverse aspects of compositionality, "
        "including consistent attribute binding, dynamic attribute binding, spatial "
        "relationships, motion binding, action binding, object interactions, and "
        "generative numeracy. We further carefully design evaluation metrics of "
        "MLLM-based metrics, detection-based metrics, and tracking-based metrics, "
        "which can better reflect the compositional text-to-video generation quality "
        "of seven proposed categories with 700 text prompts. The effectiveness of "
        "the proposed metrics is verified by correlation with human evaluations. We "
        "also benchmark various text-to-video generative models and conduct in-depth "
        "analysis across different models and different compositional categories. We "
        "find that compositional text-to-video generation is highly challenging for "
        "current models, and we hope that our attempt will shed light on future "
        "research in this direction."
    ),
    "year": "",
    "publisher": "",
}

DEFAULT_CRITERIA = [
    'INCLUDE: Does the study focus on AI systems for indexing long-form video content?', 
    'INCLUDE: Does the study address techniques for retrieving specific clips from long videos?', 
    'INCLUDE: Does the study discuss semantic understanding of video content using AI?', 
    'INCLUDE: Does the study involve scalable methods for video indexing or retrieval?', 
    'INCLUDE: Does the study propose or evaluate algorithms for video content analysis?', 
    'INCLUDE: Does the study include experimental results related to video retrieval or indexing?', 
    'INCLUDE: Does the study focus on long-form video content rather than short clips?', 
    'INCLUDE: Does the study utilize machine learning or deep learning techniques for video analysis?', 
    'INCLUDE: Does the study involve natural language processing (NLP) for video understanding?', 
    'INCLUDE: Does the study address challenges specific to large-scale video datasets?', 
    'INCLUDE: Does the study compare multiple AI-based methods for video retrieval or indexing?', 
    'INCLUDE: Does the study provide insights into improving efficiency in video content processing?', 
    'INCLUDE: Does the study discuss metadata or annotation techniques for video indexing?', 
    'INCLUDE: Does the study focus on user queries or search mechanisms for video retrieval?', 
    'INCLUDE: Does the study explore multimodal approaches (e.g., audio, text, visual) for video understanding?', 
    'EXCLUDE: Does the study focus exclusively on non-AI methods for video indexing or retrieval?', 
    'EXCLUDE: Does the study address only short-form video content?', 
    'EXCLUDE: Does the study focus solely on hardware or infrastructure without discussing AI techniques?', 
    'EXCLUDE: Does the study lack experimental validation or case studies?', 
    'EXCLUDE: Does the study focus on live-streaming or real-time video processing without indexing or retrieval?', 
    'EXCLUDE: Does the study primarily address image processing rather than video content?', 
    'EXCLUDE: Does the study focus on entertainment or social media trends without technical AI methods?', 
    'EXCLUDE: Does the study lack relevance to scalable solutions for long-form video content?', 
    'EXCLUDE: Does the study focus exclusively on video compression or storage techniques?', 
    'EXCLUDE: Does the study address only theoretical concepts without practical implementation?', 
    'EXCLUDE: Does the study focus solely on human annotation without AI involvement?', 
    'EXCLUDE: Does the study address only video editing or production workflows?', 
    'EXCLUDE: Does the study focus on unrelated AI applications outside video indexing or retrieval?', 
    'EXCLUDE: Does the study lack discussion of semantic understanding or retrieval mechanisms?', 
    'EXCLUDE: Does the study focus exclusively on audio or text analysis without video content?'
    ]


def testCLI() -> None:
    first = input(
        "Enter '1' for good example, '2' for bad example or...\nEnter your own title: "
    ).strip()

    if first in {"1", ""}:
        paper = dict(DEFAULT_PAPER)
        criteria = list(DEFAULT_CRITERIA)
        print("\nUsing default included paper and criteria.")
    elif first in {"2", "exclude"}:
        paper = dict(DEFAULT_PAPER_EXCLUDE)
        criteria = list(DEFAULT_CRITERIA)
        print("\nUsing default excluded paper and default criteria.")
    else:
        title = first
        abstract = read_multiline_input("Abstract (end with empty line):")
        year = input("Year: ").strip()
        publisher = input("Publisher: ").strip()
        period = input("Period: ").strip()
        criteria_text = read_multiline_input(
            "Criteria (one per line, end with empty line):"
        )
        paper = {
            "title": title or DEFAULT_PAPER["title"],
            "abstract": abstract or DEFAULT_PAPER["abstract"],
            "year": year or DEFAULT_PAPER["year"],
            "publisher": publisher or DEFAULT_PAPER["publisher"],
            "period": period,
        }
        criteria = (
            [line.strip() for line in criteria_text.splitlines() if line.strip()]
            if criteria_text
            else list(DEFAULT_CRITERIA)
        )
        if not criteria_text:
            print("\nUsing default criteria.")

    result = screen_paper(paper, criteria)
    print("\nResult (JSON):")
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    testCLI()
