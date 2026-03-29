from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from atlas.inital_screen.prompts import RQ_CRITERIA_PROMPT
from atlas.utils.gpt_client import GUIDED_IDEATION_TEMPERATURE, call_gpt_chat
from atlas.utils.utils import read_multiline_input

def build_criteria_from_question(question_text: str) -> str:
    system = RQ_CRITERIA_PROMPT
    if not system:
        system = (
            "You are an SLR methodologist. Convert the research question(s) into "
            "clear inclusion and exclusion criteria that can be answered yes/no. "
            "Return ONLY line-based criteria, one per line, each prefixed with "
            "'INCLUDE:' or 'EXCLUDE:'."
        )
    user = f"Research question(s):\n{question_text}\n\nCriteria:"
    return call_gpt_chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=GUIDED_IDEATION_TEMPERATURE,
        max_tokens=800,
    )


def _normalize_item(text: str) -> str:
    item = text.strip()
    if not item:
        return ""
    lowered = item.lower()
    if lowered.startswith("include:") or lowered.startswith("inclusion:"):
        return "INCLUDE: " + item.split(":", 1)[1].strip()
    if lowered.startswith("exclude:") or lowered.startswith("exclusion:"):
        return "EXCLUDE: " + item.split(":", 1)[1].strip()
    if lowered.startswith("[include]"):
        return "INCLUDE: " + item[len("[include]"):].strip()
    if lowered.startswith("[exclude]"):
        return "EXCLUDE: " + item[len("[exclude]"):].strip()
    return item


def criteria_to_list(raw: str) -> List[str]:
    text = raw.strip()
    if not text:
        return []

    # Try JSON first
    try:
        data: Any = json.loads(text)
        if isinstance(data, list):
            items = [str(x).strip() for x in data if str(x).strip()]
            return _dedupe_preserve(items)
        if isinstance(data, dict):
            items: List[str] = []
            inclusion = data.get("inclusion") or data.get("include") or []
            exclusion = data.get("exclusion") or data.get("exclude") or []
            if isinstance(inclusion, list):
                items.extend([f"INCLUDE: {str(x).strip()}" for x in inclusion if str(x).strip()])
            if isinstance(exclusion, list):
                items.extend([f"EXCLUDE: {str(x).strip()}" for x in exclusion if str(x).strip()])
            if not items and "criteria" in data and isinstance(data["criteria"], list):
                items = [str(x).strip() for x in data["criteria"] if str(x).strip()]
            return _dedupe_preserve(items)
    except json.JSONDecodeError:
        pass

    # Fallback: parse bullet/line list
    lines = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if not cleaned:
            continue
        lines.append(_normalize_item(cleaned))
    return _dedupe_preserve(lines)


def _dedupe_preserve(items: List[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def testCLI() -> None:
    questions = read_multiline_input(
        "Paste research question(s) (end with empty line):\n(Press enter to use default)"
    )
    if not questions:
        questions = DEFAULT_QUESTION
        print(f"...Using default question: {questions}")

    raw = build_criteria_from_question(questions)
    criteria = criteria_to_list(raw)

    print("\nCriteria list:")
    for i, item in enumerate(criteria, start=1):
        print(f"{str(i).rjust(3)}. {item}")

    print("\nPython list:")
    print(criteria)

DEFAULT_QUESTION = (
    "How can AI systems efficiently retrieve and semantically understand relevant segments from long-form video content?\n"
    "How can long-form videos be segmented or represented to enable efficient querying?\n"
    "How can natural language queries be used to retrieve relevant segments from long videos?\n"
    "How can multimodal AI models maintain contextual understanding across long video sequences?\n"
    "What methods enable scalable querying and analysis of long-form video datasets?"
)
if __name__ == "__main__":
    testCLI()
