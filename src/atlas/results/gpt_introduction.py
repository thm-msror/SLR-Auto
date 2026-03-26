from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas.utils.gpt_client import call_gpt_chat
from atlas.utils.utils import read_multiline_input


DEFAULT_INTRODUCTION_PROMPT = (
    "You are an expert research writing assistant. "
    "Given one or more research questions, write a concise academic introduction "
    "paragraph that frames the problem, motivates its importance, and reflects the "
    "themes in the questions. Use exactly one paragraph. Do not use headings, "
    "bullets, numbering, or quotation marks. Output only the paragraph text."
)

DEFAULT_QUESTION = (
    "How can AI systems efficiently retrieve and semantically understand relevant segments from long-form video content?\n"
    "How can long-form videos be segmented or represented to enable efficient querying?\n"
    "How can natural language queries be used to retrieve relevant segments from long videos?\n"
    "How can multimodal AI models maintain contextual understanding across long video sequences?\n"
    "What methods enable scalable querying and analysis of long-form video datasets?"
)


def build_introduction_from_questions(
    questions_text: str,
    prompt_text: str = DEFAULT_INTRODUCTION_PROMPT,
    model_name: Optional[str] = None,
    temperature: float = 0.3,
    max_output_tokens: int = 350,
) -> str:
    cleaned_questions = (questions_text or "").strip() or DEFAULT_QUESTION
    system = (prompt_text or "").strip() or DEFAULT_INTRODUCTION_PROMPT
    user = f"Research questions:\n{cleaned_questions}\n\nIntroduction paragraph:"

    return call_gpt_chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_output_tokens,
    )


def testCLI() -> None:
    questions = read_multiline_input(
        "Paste research question(s) and press Enter on an empty line to submit."
    )
    introduction = build_introduction_from_questions(questions or DEFAULT_QUESTION)
    print(introduction)


if __name__ == "__main__":
    testCLI()
