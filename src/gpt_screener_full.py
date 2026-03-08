import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

try:
    from src.gpt_client import call_gpt_pdf
except ImportError:
    from gpt_client import call_gpt_pdf

DEFAULT_PDF_PATH = Path(
    "data/3_top_papers/pdf_papers_manual_53/Video-RAG - Visually Aligned Retrieval-Augmented Long Video Comprehension.pdf"
)

DEFAULT_CATEGORIES = [
    "Multimodal Representation Learning",
    "Video-Text Retrieval and Alignment",
    "Dataset Creation and Benchmarking",
    "Semantic Query Understanding and Mapping",
    "Efficiency and Scalability in Video Processing",
    "Domain-Specific Applications and Adaptations",
]

PROMPT_PATH = Path("prompts/screen_full.txt")
MAX_PDF_BYTES = 50 * 1024 * 1024


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


def load_default_question() -> str:
    fallback = (
        "How can AI systems efficiently index, retrieve, and semantically understand long-form video content at scale?\n"
        "What are the current techniques to retrieve a specific clip from a long video?"
    )

    path = Path("src/gpt_research_q.py")
    if not path.exists():
        return fallback

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "DEFAULT_QUESTION":
                        value = ast.literal_eval(node.value)
                        if isinstance(value, (tuple, list)):
                            return "\n".join(str(v) for v in value).strip()
                        if isinstance(value, str):
                            return value.strip()
    except Exception:
        pass

    return fallback


def build_prompt(question: str, categories: List[str]) -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_PATH}")
    template = PROMPT_PATH.read_text(encoding="utf-8")
    categories_block = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(categories))
    return (
        template.replace("{RESEARCH_QUESTION}", question)
        .replace("{CATEGORIES}", categories_block)
        .strip()
    )


def call_gpt_pdf_from_path(prompt: str, pdf_path: Path) -> str:
    pdf_bytes = pdf_path.read_bytes()
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise RuntimeError("PDF exceeds 50 MB limit for input_file.")
    return call_gpt_pdf(prompt, pdf_bytes, pdf_path.name)


def parse_tagged_output(text: str, categories: List[str]) -> Dict[str, object]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Empty output.")

    match = re.match(r"^DECISION:\s*(YES|NO)\s*$", lines[0], re.IGNORECASE)
    if not match:
        raise ValueError("Missing DECISION line.")

    decision = match.group(1).upper()
    if decision == "NO":
        return {"included": False}

    category_map: Dict[str, str] = {}
    i = 1
    while i < len(lines):
        line = lines[i].strip()
        if line.upper().startswith("CATEGORY:"):
            category_name = line.split(":", 1)[1].strip()
            if not category_name:
                raise ValueError("Empty CATEGORY name.")

            i += 1
            if i >= len(lines):
                raise ValueError(f"Missing SUMMARY for category '{category_name}'.")

            summary_line = lines[i].strip()
            if not summary_line.upper().startswith("SUMMARY:"):
                raise ValueError(f"Missing SUMMARY for category '{category_name}'.")

            summary = summary_line.split(":", 1)[1].strip()
            i += 1

            continuation: List[str] = []
            while i < len(lines) and not lines[i].strip().upper().startswith("CATEGORY:"):
                continuation.append(lines[i].strip())
                i += 1

            if continuation:
                summary = (summary + "\n" + "\n".join(continuation)).strip()

            category_map[category_name] = summary or "Not mentioned."
            continue

        i += 1

    if not category_map:
        raise ValueError("No categories parsed for YES decision.")

    for category in categories:
        if category not in category_map:
            category_map[category] = "Not mentioned."

    ordered_categories = {category: category_map[category] for category in categories}
    return {"included": True, "categories": ordered_categories}


def testCLU() -> None:
    pdf_input = input("PDF path (blank for default): ").strip()
    pdf_path = Path(pdf_input) if pdf_input else DEFAULT_PDF_PATH
    if not pdf_path.is_absolute():
        pdf_path = Path.cwd() / pdf_path

    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        return

    question = read_multiline_input(
        "Paste research question (end with empty line):\n(Press enter to select default)"
    )
    if not question:
        question = load_default_question()

    categories_text = read_multiline_input(
        "Paste categories, one per line (end with empty line):\n(Press enter to use default list)"
    )
    categories = (
        [line.strip() for line in categories_text.splitlines() if line.strip()]
        if categories_text
        else DEFAULT_CATEGORIES
    )
    if not categories:
        categories = DEFAULT_CATEGORIES

    prompt = build_prompt(question, categories)

    try:
        raw_output = call_gpt_pdf_from_path(prompt, pdf_path)
        parsed = parse_tagged_output(raw_output, categories)
        parsed["paper_file"] = str(pdf_path)
    except Exception as exc:
        print(f"PARSE_OR_CALL_ERROR: {exc}", file=sys.stderr)
        print("RAW_OUTPUT_START", file=sys.stderr)
        try:
            print(raw_output, file=sys.stderr)
        except Exception:
            print("<no raw output available>", file=sys.stderr)
        print("RAW_OUTPUT_END", file=sys.stderr)
        return

    print(json.dumps(parsed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    testCLU()
