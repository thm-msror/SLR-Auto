from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from gpt_client import call_gpt_chat


DEFAULT_RESEARCH_QUESTION = (
    "How can AI systems efficiently index, retrieve, and semantically understand "
    "long-form video content at scale?"
)

DEFAULT_CATEGORY_PROMPT = (
    "You are an expert SLR taxonomy builder. Given a research question and a set of "
    "paper abstracts, propose a concise taxonomy of categories for organizing the "
    "literature. Use the abstracts as guidance, but keep categories generalizable. "
    "Return ONLY plain text, one category per line, formatted as "
    "'<category name>: short explanation of what belongs in the category'. "
    "Output exactly 6 categories. "
    "Do NOT use labels like 'Category:' or 'Explanation:'. "
    "No JSON, no extra commentary."
)

MERGE_CATEGORY_PROMPT = (
    "You are merging candidate taxonomy categories for a literature review. "
    "Combine duplicates, normalize wording, and output ONLY plain text with one "
    "category per line in the format "
    "'<category name>: short explanation of what belongs in the category'. "
    "Output exactly 6 categories. "
    "Do NOT use labels like 'Category:' or 'Explanation:'. "
    "No JSON, no extra commentary."
)

SUMMARY_GROUP_PROMPT = (
    "You are summarizing a group of paper abstracts into a single concise, coherent "
    "mega-abstract focused on the research question. Capture key themes, methods, "
    "datasets, and evaluation signals. Return ONLY the summary text as one paragraph "
    "(no bullets, no headings). Keep it short and information-dense."
)


def load_prompt(path: str) -> str:
    prompt_path = Path(path)
    if not prompt_path.exists():
        return ""
    return prompt_path.read_text(encoding="utf-8")


def build_taxonomy_categories(
    research_question: str,
    abstracts: List[str],
    prompt_path: str = "prompts/make_categories.txt",
    model_name: Optional[str] = None,
    max_output_tokens: int = 300,
    max_input_tokens: int = 60000,
    safety_margin: float = 0.9,
    allow_map_reduce: bool = True,
    keep_first_n: int = 20,
    summary_group_size: int = 20,
    summary_max_output_tokens: int = 500,
    max_summary_rounds: int = 3,
) -> Dict[str, str]:
    question = (research_question or "").strip()
    if not question:
        raise ValueError("Research question is required.")

    cleaned_abstracts = [a.strip() for a in abstracts if a and a.strip()]

    system = load_prompt(prompt_path).strip() or DEFAULT_CATEGORY_PROMPT

    user = _build_categories_user_content(question, cleaned_abstracts)
    if _fits_context(system, user, max_input_tokens, max_output_tokens, safety_margin):
        raw = call_gpt_chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            model_name=model_name,
            temperature=0.2,
            max_tokens=max_output_tokens,
        )
        return categories_to_dict(raw)

    if not allow_map_reduce:
        raise ValueError(
            "Input exceeds token budget. Enable map-reduce or increase max_input_tokens."
        )

    # --- Compression strategy: keep first N abstracts, summarize the rest ---
    compressed = _compress_abstracts_with_summaries(
        question=question,
        abstracts=cleaned_abstracts,
        keep_first_n=keep_first_n,
        group_size=summary_group_size,
        model_name=model_name,
        system_prompt=system,
        max_input_tokens=max_input_tokens,
        max_output_tokens=summary_max_output_tokens,
        safety_margin=safety_margin,
        max_rounds=max_summary_rounds,
    )
    if compressed:
        user = _build_categories_user_content(question, compressed)
        if _fits_context(system, user, max_input_tokens, max_output_tokens, safety_margin):
            raw = call_gpt_chat(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                model_name=model_name,
                temperature=0.2,
                max_tokens=max_output_tokens,
            )
            return categories_to_dict(raw)

    # --- Fallback Map: chunk abstracts and build candidate categories per chunk ---
    budget_tokens = _abstract_budget_tokens(
        question=question,
        system=system,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        safety_margin=safety_margin,
    )
    if budget_tokens <= 0:
        raise ValueError(
            "Input budget too small for the current prompt/output settings. "
            "Increase max_input_tokens or lower max_output_tokens."
        )

    chunks = _split_abstracts_by_tokens(cleaned_abstracts, budget_tokens)
    candidate_categories: Dict[str, str] = {}
    for chunk in chunks:
        user_chunk = _build_categories_user_content(question, chunk)
        raw = call_gpt_chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_chunk},
            ],
            model_name=model_name,
            temperature=0.2,
            max_tokens=max_output_tokens,
        )
        chunk_categories = categories_to_dict(raw)
        for category, desc in chunk_categories.items():
            if category not in candidate_categories:
                candidate_categories[category] = desc
            elif not candidate_categories[category] and desc:
                candidate_categories[category] = desc

    if not candidate_categories:
        return {}

    # --- Reduce: merge candidate categories into a final taxonomy ---
    reduce_user = _build_merge_user_content(question, candidate_categories)
    if _fits_context(
        MERGE_CATEGORY_PROMPT,
        reduce_user,
        max_input_tokens,
        max_output_tokens,
        safety_margin,
    ):
        merged_raw = call_gpt_chat(
            messages=[
                {"role": "system", "content": MERGE_CATEGORY_PROMPT},
                {"role": "user", "content": reduce_user},
            ],
            model_name=model_name,
            temperature=0.2,
            max_tokens=max_output_tokens,
        )
        merged = categories_to_dict(merged_raw)
        return merged or candidate_categories

    return candidate_categories


def categories_to_dict(raw: str) -> Dict[str, str]:
    text = raw.strip()
    if not text:
        return {}

    # Normalize Category/Explanation blocks into "CategoryName: Explanation"
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    normalized_lines: List[str] = []
    i = 0
    while i < len(raw_lines):
        line = re.sub("^\\s*(?:[-*\\u2022]|\\d+[.)])\\s*", "", raw_lines[i]).strip()
        m_cat = re.match(r"^category\\s*:\\s*(.+)$", line, re.IGNORECASE)
        if m_cat:
            cat = m_cat.group(1).strip()
            desc = ""
            if re.search(r"explanation\\s*:", line, re.IGNORECASE):
                parts = re.split(r"explanation\\s*:", line, flags=re.IGNORECASE)
                if len(parts) > 1:
                    cat = parts[0].replace("Category:", "").replace("category:", "").strip()
                    desc = parts[1].strip()
                    normalized_lines.append(f"{cat}: {desc}")
                    i += 1
                    continue
            if i + 1 < len(raw_lines):
                nxt = re.sub("^\\s*(?:[-*\\u2022]|\\d+[.)])\\s*", "", raw_lines[i + 1]).strip()
                m_exp = re.match(r"^explanation\\s*:\\s*(.+)$", nxt, re.IGNORECASE)
                if m_exp:
                    desc = m_exp.group(1).strip()
                    normalized_lines.append(f"{cat}: {desc}")
                    i += 2
                    continue
            normalized_lines.append(cat)
            i += 1
            continue
        normalized_lines.append(line)
        i += 1

    text = "\n".join(normalized_lines)

    # Try JSON
    try:
        data: Any = json.loads(text)
        if isinstance(data, dict):
            return _dedupe_preserve_mapping(
                {str(k).strip(): str(v).strip() for k, v in data.items() if str(k).strip()}
            )
        if isinstance(data, list):
            items = [str(x).strip() for x in data if str(x).strip()]
            return _dedupe_preserve_mapping({item: "" for item in items})
    except json.JSONDecodeError:
        pass

    # Fallback: parse bullets/lines
    lines: List[str] = []
    for line in text.splitlines():
        cleaned = re.sub("^\\s*(?:[-*\\u2022]|\\d+[.)])\\s*", "", line).strip()
        if not cleaned:
            continue
        lines.append(cleaned)

    if not lines and text:
        lines = [text]

    if len(lines) == 1:
        line = lines[0]
        # Split numbered inline lists like "1) ... 2) ... 3) ..."
        if re.search(r"\d+[.)]\s+", line):
            parts = [p.strip() for p in re.split(r"\s*\d+[.)]\s+", line) if p.strip()]
            if parts:
                lines = parts
        # Split common delimiters if still single
        if len(lines) == 1 and re.search(r"[;|]", line):
            parts = [p.strip() for p in re.split(r"[;|]", line) if p.strip()]
            if parts:
                lines = parts
        if len(lines) == 1 and re.search(r"[;,]", line):
            parts = [p.strip() for p in re.split(r"[;,]", line) if p.strip()]
            if parts:
                lines = parts

    mapping: Dict[str, str] = {}
    for line in lines:
        if ":" in line:
            category, desc = line.split(":", 1)
            category = category.strip()
            desc = desc.strip()
        else:
            category = line.strip()
            desc = ""
        if not category:
            continue
        if category not in mapping:
            mapping[category] = desc
        elif not mapping[category] and desc:
            mapping[category] = desc

    return _dedupe_preserve_mapping(mapping)


def _build_categories_user_content(question: str, abstracts: List[str]) -> str:
    lines = ["Research Question:", question.strip(), "", "Abstracts:"]
    if not abstracts:
        lines.append("- None provided.")
    else:
        for i, abstract in enumerate(abstracts, start=1):
            lines.append(f"{i}. {abstract}")
    return "\n".join(lines)


def _build_merge_user_content(question: str, categories: Dict[str, str]) -> str:
    lines = ["Research Question:", question.strip(), "", "Candidate Categories:"]
    for cat, desc in categories.items():
        if desc:
            lines.append(f"- {cat}: {desc}")
        else:
            lines.append(f"- {cat}")
    return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def _fits_context(
    system: str,
    user: str,
    max_input_tokens: int,
    max_output_tokens: int,
    safety_margin: float,
) -> bool:
    input_tokens = _estimate_tokens(system) + _estimate_tokens(user)
    total = input_tokens + max_output_tokens
    return total <= int(max_input_tokens * safety_margin)


def _abstract_budget_tokens(
    question: str,
    system: str,
    max_input_tokens: int,
    max_output_tokens: int,
    safety_margin: float,
) -> int:
    base_user = _build_categories_user_content(question, [])
    base_tokens = _estimate_tokens(system) + _estimate_tokens(base_user)
    return int(max_input_tokens * safety_margin) - base_tokens - max_output_tokens


def _summary_budget_tokens(
    question: str,
    max_input_tokens: int,
    max_output_tokens: int,
    safety_margin: float,
) -> int:
    base_user = _build_summary_user_content(question, [])
    base_tokens = _estimate_tokens(SUMMARY_GROUP_PROMPT) + _estimate_tokens(base_user)
    return int(max_input_tokens * safety_margin) - base_tokens - max_output_tokens


def _build_summary_user_content(question: str, abstracts: List[str]) -> str:
    lines = ["Research Question:", question.strip(), "", "Abstracts:"]
    if not abstracts:
        lines.append("- None provided.")
    else:
        for i, abstract in enumerate(abstracts, start=1):
            lines.append(f"{i}. {abstract}")
    return "\n".join(lines)


def _summarize_group(
    question: str,
    abstracts: List[str],
    model_name: Optional[str],
    max_input_tokens: int,
    max_output_tokens: int,
    safety_margin: float,
) -> str:
    user = _build_summary_user_content(question, abstracts)
    if not _fits_context(
        SUMMARY_GROUP_PROMPT, user, max_input_tokens, max_output_tokens, safety_margin
    ):
        budget_tokens = _summary_budget_tokens(
            question=question,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            safety_margin=safety_margin,
        )
        if budget_tokens > 0 and abstracts:
            per_abs_budget = max(40, budget_tokens // len(abstracts))
            trimmed: List[str] = []
            for abstract in abstracts:
                trimmed.append(_truncate_to_tokens(abstract, per_abs_budget))
            user = _build_summary_user_content(question, trimmed)
            if not _fits_context(
                SUMMARY_GROUP_PROMPT,
                user,
                max_input_tokens,
                max_output_tokens,
                safety_margin,
            ):
                merged = " ".join(trimmed).strip()
                merged = _truncate_to_tokens(merged, budget_tokens)
                user = _build_summary_user_content(question, [merged])

    return call_gpt_chat(
        messages=[
            {"role": "system", "content": SUMMARY_GROUP_PROMPT},
            {"role": "user", "content": user},
        ],
        model_name=model_name,
        temperature=0.2,
        max_tokens=max_output_tokens,
    )


def _summarize_remaining_abstracts(
    question: str,
    abstracts: List[str],
    group_size: int,
    model_name: Optional[str],
    max_input_tokens: int,
    max_output_tokens: int,
    safety_margin: float,
) -> str:
    if not abstracts:
        return ""

    summaries: List[str] = []
    group_size = max(1, group_size)
    for i in range(0, len(abstracts), group_size):
        group = abstracts[i : i + group_size]
        summary = _summarize_group(
            question=question,
            abstracts=group,
            model_name=model_name,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            safety_margin=safety_margin,
        )
        if summary.strip():
            summaries.append(summary.strip())

    return " ".join(summaries).strip()


def _compress_abstracts_with_summaries(
    question: str,
    abstracts: List[str],
    keep_first_n: int,
    group_size: int,
    model_name: Optional[str],
    system_prompt: str,
    max_input_tokens: int,
    max_output_tokens: int,
    safety_margin: float,
    max_rounds: int,
) -> List[str]:
    if not abstracts:
        return []

    keep_first_n = max(0, keep_first_n)
    kept = abstracts[:keep_first_n] if keep_first_n else []
    remainder = abstracts[keep_first_n:] if keep_first_n else list(abstracts)

    if not remainder:
        return kept

    rounds = 0
    while remainder and rounds < max_rounds:
        gigantic = _summarize_remaining_abstracts(
            question=question,
            abstracts=remainder,
            group_size=group_size,
            model_name=model_name,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            safety_margin=safety_margin,
        )
        combined = kept + ([gigantic] if gigantic else [])
        user = _build_categories_user_content(question, combined)
        if _fits_context(
            system_prompt or DEFAULT_CATEGORY_PROMPT,
            user,
            max_input_tokens,
            max_output_tokens,
            safety_margin,
        ):
            return combined

        remainder = [gigantic] if gigantic else []
        rounds += 1

    return kept + (remainder if remainder else [])


def _split_abstracts_by_tokens(abstracts: List[str], budget_tokens: int) -> List[List[str]]:
    if budget_tokens <= 0:
        return []

    chunks: List[List[str]] = []
    current: List[str] = []
    current_tokens = 0

    for abstract in abstracts:
        candidate = abstract
        tokens = _estimate_tokens(candidate)

        if tokens > budget_tokens:
            candidate = _truncate_to_tokens(candidate, budget_tokens)
            tokens = _estimate_tokens(candidate)

        if current and current_tokens + tokens > budget_tokens:
            chunks.append(current)
            current = [candidate]
            current_tokens = tokens
        else:
            current.append(candidate)
            current_tokens += tokens

    if current:
        chunks.append(current)

    return chunks


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if _estimate_tokens(text) <= max_tokens:
        return text
    max_chars = max_tokens * 4
    trimmed = text[:max_chars].rstrip()
    return trimmed + " ..."


def _dedupe_preserve_mapping(items: Dict[str, str]) -> Dict[str, str]:
    deduped: Dict[str, str] = {}
    for key, value in items.items():
        cleaned = key.strip()
        if not cleaned or cleaned in deduped:
            continue
        deduped[cleaned] = value.strip() if isinstance(value, str) else str(value)
    return deduped


def testCLI() -> None:
    data_path = Path(r"data\3_top_papers\top_53_papers_20251016_212754.json")
    print(f"Reading abstracts from: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    raw = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Expected a list of papers in the data file.")

    abstracts: List[str] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        paper = item.get("paper") or {}
        if isinstance(paper, dict):
            abstract = paper.get("abstract") or ""
            if isinstance(abstract, str) and abstract.strip():
                abstracts.append(abstract.strip())

    if not abstracts:
        raise ValueError("No abstracts found in the data file.")

    categories = build_taxonomy_categories(
        research_question=DEFAULT_RESEARCH_QUESTION,
        abstracts=abstracts,
    )

    print("\nCategories (JSON):")
    print(json.dumps(categories, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    testCLI()
