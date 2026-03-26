from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


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

    for index, paper_id in enumerate(top_paper_ids.keys(), start=1):
        top_entry = top_paper_ids.get(paper_id) or {}
        paper = papers_by_id.get(paper_id) or {}
        merged = {**paper, **top_entry}
        references.append(f"[{index}] {_format_ieee_reference(merged, paper_id)}")

    return references


def build_ieee_references_text(data: Dict[str, Any]) -> str:
    return "\n".join(build_ieee_references_from_top_papers(data))


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


def testCLI() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raw_path = input(
        f"JSON path (press Enter to use default: {DEFAULT_JSON_PATH.as_posix()}): "
    ).strip()
    json_path = Path(raw_path) if raw_path else DEFAULT_JSON_PATH
    data = load_results_json(json_path)
    print(build_ieee_references_text(data))


if __name__ == "__main__":
    testCLI()
