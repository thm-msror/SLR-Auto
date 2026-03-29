from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas.results.prompts import RESULTS_FINDINGS_PROMPT
from atlas.utils.gpt_client import call_gpt_chat


DEFAULT_JSON_PATH = Path("src/atlas/results/example.json")

def load_results_json(json_path: str | Path) -> Dict[str, Any]:
    path = Path(json_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_ieee_references_from_top_papers(data: Dict[str, Any]) -> List[str]:
    top_paper_ids = data.get("top_paper_ids") or {}
    papers_by_id = data.get("papers_by_id") or {}
    references: List[str] = []

    for paper_id in top_paper_ids.keys():
        top_entry = top_paper_ids.get(paper_id) or {}
        if not _should_include_reference(top_entry):
            continue
        paper = papers_by_id.get(paper_id) or {}
        merged = {**paper, **top_entry}
        references.append(_format_ieee_reference(merged, paper_id))

    return [f"[{index}] {reference}" for index, reference in enumerate(references, start=1)]


def build_ieee_references_text(data: Dict[str, Any]) -> str:
    return "\n".join(build_ieee_references_from_top_papers(data))


def rewrite_results_findings(
    theme_drafts: str | Mapping[str, str],
    references: str | List[str],
    prompt_text: str = RESULTS_FINDINGS_PROMPT,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 1800,
) -> str:
    theme_block = _normalize_theme_drafts(theme_drafts)
    reference_block = _normalize_references(references)

    if not theme_block:
        raise ValueError("theme_drafts is required.")
    if not reference_block:
        raise ValueError("references is required.")

    system = (prompt_text or "").strip() or RESULTS_FINDINGS_PROMPT
    user = _build_results_user_content(theme_block, reference_block)

    return call_gpt_chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_output_tokens,
    )


def load_default_results_inputs() -> tuple[Dict[str, Any], str, Mapping[str, str]]:
    data = load_results_json(DEFAULT_JSON_PATH)
    references = build_ieee_references_text(data)
    theme_drafts = data.get("categories") or {}
    if not theme_drafts:
        raise ValueError(f"No categories found in default JSON: {DEFAULT_JSON_PATH}")
    return data, references, theme_drafts


def _build_results_user_content(theme_block: str, reference_block: str) -> str:
    return "\n".join(
        [
            "Rewrite the following draft Results and Findings section.",
            "Keep the same theme order.",
            "Add clear transition or linking phrases between themes so the section flows as one narrative.",
            "Use IEEE bracket citations from the provided references wherever claims are tied to specific papers.",
            "Output only paragraphs.",
            "",
            "Reference list:",
            reference_block,
            "",
            "Draft theme findings:",
            theme_block,
        ]
    )


def _normalize_theme_drafts(theme_drafts: str | Mapping[str, str]) -> str:
    if isinstance(theme_drafts, str):
        return theme_drafts.strip()

    lines: List[str] = []
    for theme, draft in theme_drafts.items():
        theme_name = str(theme).strip()
        draft_text = str(draft).strip()
        if not theme_name or not draft_text:
            continue
        lines.append(f"Theme: {theme_name}")
        lines.append(draft_text)
        lines.append("")
    return "\n".join(lines).strip()


def _normalize_references(references: str | List[str]) -> str:
    if isinstance(references, str):
        return references.strip()
    return "\n".join(str(item).strip() for item in references if str(item).strip()).strip()


def _should_include_reference(top_entry: Mapping[str, Any]) -> bool:
    pdf_path = str(top_entry.get("pdf_path") or "").strip()
    full_screening = top_entry.get("full_screening") or {}
    included = full_screening.get("included")
    return bool(pdf_path) and included is True


def _format_ieee_reference(paper: Dict[str, Any], paper_id: str) -> str:
    authors = _format_authors_ieee(paper.get("authors"))
    title = _clean_title(paper.get("title") or paper_id)
    publisher = _clean_text(paper.get("publisher"))
    year = _extract_year(paper.get("published"))
    doi = _clean_text(paper.get("doi"))
    link = _clean_text(paper.get("link"))

    parts: List[str] = []
    if authors:
        parts.append(f"{authors},")
    parts.append(f"\"{title},\"")
    if publisher:
        parts.append(publisher + ",")
    if year:
        parts.append(year + ".")
    elif parts:
        parts[-1] = parts[-1].rstrip(",") + "."

    if doi:
        parts.append(f"doi: {doi}.")
    elif link:
        parts.append(link)

    return " ".join(part for part in parts if part).strip()


def _format_authors_ieee(authors: Any) -> str:
    if isinstance(authors, str):
        author_items = [authors] if authors.strip() else []
    elif isinstance(authors, Iterable):
        author_items = [str(author).strip() for author in authors if str(author).strip()]
    else:
        author_items = []

    formatted = [_format_single_author_ieee(author) for author in author_items if author]
    if not formatted:
        return ""
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return ", ".join(formatted[:-1]) + f", and {formatted[-1]}"


def _format_single_author_ieee(name: str) -> str:
    tokens = [token for token in str(name).replace(",", " ").split() if token]
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]

    family_name = tokens[-1]
    initials = []
    for token in tokens[:-1]:
        cleaned = token.strip(".-")
        if cleaned:
            initials.append(cleaned[0].upper() + ".")
    return " ".join(initials + [family_name])


def _extract_year(published: Any) -> str:
    text = _clean_text(published)
    if len(text) >= 4 and text[:4].isdigit():
        return text[:4]
    return ""


def _clean_title(value: Any) -> str:
    text = _clean_text(value)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    return " ".join(text.replace("\n", " ").replace("\r", " ").split())


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def test_references_cli() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raw_path = input(
        f"JSON path (press Enter to use default: {DEFAULT_JSON_PATH.as_posix()}): "
    ).strip()
    json_path = Path(raw_path) if raw_path else DEFAULT_JSON_PATH
    data = load_results_json(json_path)
    print(build_ieee_references_text(data))


def testCLI() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    _, references, theme_drafts = load_default_results_inputs()
    print(rewrite_results_findings(theme_drafts=theme_drafts, references=references))


if __name__ == "__main__":
    testCLI()
