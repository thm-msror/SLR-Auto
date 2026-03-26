from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas.results.prompts import ATLAST_METHODOLOGY_CONTEXT
from atlas.utils.gpt_client import call_gpt_chat


DEFAULT_METHODLOGY_PROMPT = (
    "You are an expert academic writing assistant for systematic literature reviews. "
    "Write a methodology section in plain academic prose using the provided review inputs "
    "and the ATLAS workflow context. "
    "Output only paragraphs with no headings, bullets, numbering, markdown, or code fences. "
    "The final paragraph must explain the PRISMA flow using the exact counts provided. "
    "Describe only what is supported by the supplied inputs and context. "
    "Do not invent databases, dates, screening steps, or manual procedures that are not present."
)

DEFAULT_INPUTS: Dict[str, Any] = {
    "research_questions": (
        "How can AI systems efficiently retrieve and semantically understand relevant "
        "segments from long-form video content?"
    ),
    "boolean_query_used": (
        "(\"artificial intelligence\" OR \"AI systems\" OR \"AI-based\") AND "
        "(\"video retrieval\" OR \"content retrieval\" OR \"video indexing\") AND "
        "(\"semantic understanding\" OR \"semantic analysis\" OR \"semantic comprehension\") AND "
        "(\"long-form video\" OR \"long video\" OR \"extended video content\")"
    ),
    "queries": [
        "\"artificial intelligence\" \"semantic understanding\" \"video retrieval\" \"long-form video\"",
        "\"artificial intelligence\" \"semantic understanding\" \"video retrieval\" \"long video\"",
        "\"artificial intelligence\" \"semantic understanding\" \"extended video content\" \"video retrieval\"",
        "\"artificial intelligence\" \"video retrieval\" \"semantic analysis\" \"long-form video\"",
        "\"artificial intelligence\" \"video retrieval\" \"semantic analysis\" \"long video\"",
    ],
    "criteria_used": [
        "INCLUDE: Was the study published in a peer-reviewed venue such as a journal, conference proceedings, or workshop co-located with a major conference?",
        "EXCLUDE: Was the study published as a non-peer-reviewed document, such as a preprint, blog post, or technical report without peer review?",
        "INCLUDE: Does the title or abstract explicitly mention AI or machine learning methods applied to video content analysis?",
        "INCLUDE: Does the title or abstractsd reference techniques for semantic understanding or retrieval of video segments?",
        "INCLUDE: Does the title or abstract mention long-form video content or extended-duration video analysis?",
        "INCLUDE: Does the title or abstract describe methods for segmenting video content based on semantic or contextual understanding?",
        "INCLUDE: Does the title or abstract discuss efficiency or optimization in video retrieval or analysis processes?",
        "INCLUDE: Does the title or abstract reference natural language processing (NLP) or multimodal approaches in the context of video understanding?",
        "INCLUDE: Does the title or abstract mention the use of deep learning architectures for video content analysis?",
        "INCLUDE: Does the title or abstract describe applications or systems for video retrieval in domains such as education, entertainment, or surveillance?",
        "INCLUDE: Does the title or abstract reference benchmarks, datasets, or evaluation metrics specific to video retrieval or semantic understanding?",
        "EXCLUDE: Does the title or abstract focus exclusively on image-based analysis without addressing video content?",
        "EXCLUDE: Does the title or abstract describe methods unrelated to AI, such as manual video annotation or purely statistical approaches?",
        "EXCLUDE: Does the title or abstract focus solely on short-form video content without addressing long-form video?",
        "EXCLUDE: Does the title or abstract describe techniques for video compression or storage without addressing retrieval or semantic understanding?",
        "EXCLUDE: Does the title or abstract focus exclusively on hardware or infrastructure optimization without addressing AI methods for video analysis?",
        "EXCLUDE: Does the title or abstract describe methods for audio-only analysis without addressing video content?",
        "EXCLUDE: Does the title or abstract focus solely on entertainment or streaming platforms without discussing retrieval or semantic understanding?",
        "EXCLUDE: Does the title or abstract describe methods for real-time video processing without addressing semantic understanding or retrieval?",
        "EXCLUDE: Does the title or abstract focus exclusively on surveillance or security applications without addressing semantic understanding or retrieval?",
        "EXCLUDE: Does the title or abstract describe methods for video editing or production without addressing retrieval or semantic understanding?",
        "EXCLUDE: Does the title or abstract focus exclusively on theoretical or conceptual frameworks without presenting empirical results or applications?",
        "EXCLUDE: Does the title or abstract describe methods for social media analysis without addressing long-form video content?",
        "EXCLUDE: Does the title or abstract focus exclusively on user interaction or recommendation systems without addressing video retrieval or semantic understanding?",
        "EXCLUDE: Does the title or abstract describe methods for gaming or virtual reality applications without addressing long-form video retrieval or semantic understanding?",
        "EXCLUDE: Does the title or abstract focus exclusively on historical or archival video without addressing AI-based retrieval or semantic understanding?",
        "EXCLUDE: Does the title or abstract describe methods for video summarization without addressing retrieval or semantic understanding?",
    ],
    "prisma": {
        "identification": {
            "ieee": 0,
            "crossref": 5,
            "semanticscholar": 0,
        },
        "after_dedup": 3,
        "screened": 3,
        "excluded_screening": 1,
        "sought_retrieval": 2,
        "not_retrieved": 2,
        "assessed_eligibility": 0,
        "excluded_eligibility": 0,
        "included": 0,
    },
}


def build_methodology_section(
    research_questions: str,
    boolean_query_used: str,
    criteria_used: List[str],
    prisma: Dict[str, Any],
    atlas_context: str = ATLAST_METHODOLOGY_CONTEXT,
    queries: Optional[List[str]] = None,
    prompt_text: str = DEFAULT_METHODLOGY_PROMPT,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 1200,
) -> str:
    cleaned_criteria = [item.strip() for item in criteria_used if item and item.strip()]
    if not research_questions.strip():
        raise ValueError("research_questions is required.")
    if not boolean_query_used.strip():
        raise ValueError("boolean_query_used is required.")
    if not cleaned_criteria:
        raise ValueError("At least one inclusion/exclusion criterion is required.")
    if not prisma:
        raise ValueError("prisma is required.")

    query_lines = queries or []
    user = _build_methodology_user_content(
        research_questions=research_questions.strip(),
        boolean_query_used=boolean_query_used.strip(),
        queries=query_lines,
        criteria_used=cleaned_criteria,
        prisma=prisma,
        atlas_context=(atlas_context or "").strip(),
    )

    system = (prompt_text or "").strip() or DEFAULT_METHODLOGY_PROMPT

    return call_gpt_chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_output_tokens,
    )


def _build_methodology_user_content(
    research_questions: str,
    boolean_query_used: str,
    queries: List[str],
    criteria_used: List[str],
    prisma: Dict[str, Any],
    atlas_context: str,
) -> str:
    lines = [
        "Write a methodology section for a paper that used ATLAS.",
        "",
        "ATLAS methodology context:",
        atlas_context,
        "",
        "Study inputs:",
        f"Research questions: {research_questions}",
        f"Boolean query used: {boolean_query_used}",
        "",
        "Expanded executable queries:",
    ]

    if queries:
        for idx, query in enumerate(queries, start=1):
            lines.append(f"{idx}. {query}")
    else:
        lines.append("None provided.")

    lines.extend(
        [
            "",
            "Inclusion and exclusion criteria:",
        ]
    )
    for idx, criterion in enumerate(criteria_used, start=1):
        lines.append(f"{idx}. {criterion}")

    lines.extend(
        [
            "",
            "PRISMA counts:",
            json.dumps(prisma, ensure_ascii=False, indent=2),
            "",
            "Requirements:",
            "1. Write only plain paragraphs.",
            "2. Do not use headings, bullets, numbering, or labels.",
            "3. Explain the ATLAS-supported search strategy, database selection, and screening process using the provided context.",
            "4. Use the exact boolean query and criteria as study-specific inputs where helpful.",
            "5. Make the final paragraph explain the PRISMA flow using the exact numbers supplied.",
        ]
    )
    return "\n".join(lines)


def testCLI() -> None:
    methodology = build_methodology_section(
        research_questions=DEFAULT_INPUTS["research_questions"],
        boolean_query_used=DEFAULT_INPUTS["boolean_query_used"],
        queries=DEFAULT_INPUTS["queries"],
        criteria_used=DEFAULT_INPUTS["criteria_used"],
        prisma=DEFAULT_INPUTS["prisma"],
    )
    print(methodology)


if __name__ == "__main__":
    testCLI()
