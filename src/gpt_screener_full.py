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
try:
    from src.utils import read_multiline_input
except Exception:
    from utils import read_multiline_input

DEFAULT_PDF_PATH = Path(
    "data/3_top_papers/pdf_papers_manual_53/Video-RAG - Visually Aligned Retrieval-Augmented Long Video Comprehension.pdf"
)

DEFAULT_CATEGORIES: Dict[str, str] = {
    "Video Representation and Embedding Techniques": (
        "Approaches for creating robust video-text embeddings, including multimodal "
        "representation learning, semantic-preserving metric learning, and fine-grained "
        "feature alignment."
    ),
    "Dataset Construction and Utilization": (
        "Development and use of large-scale, diverse datasets for video-text retrieval, "
        "including methods for dataset creation, annotation, and leveraging weak "
        "supervision."
    ),
    "Multimodal Fusion and Alignment": (
        "Techniques for integrating and aligning information across modalities such as "
        "video, audio, text, and metadata to enhance retrieval and understanding."
    ),
    "Efficiency and Scalability in Video Retrieval": (
        "Methods to improve computational efficiency and scalability, including "
        "keyframe selection, modality-specific routing, and lightweight model "
        "architectures."
    ),
    "Fine-Grained and Contextual Understanding": (
        "Strategies for capturing fine-grained temporal, spatial, and semantic details "
        "in video content, including action localization, event detection, and "
        "query-to-event decomposition."
    ),
    "Domain-Specific Applications": (
        "Tailored approaches for specific domains such as contextual advertising, "
        "lecture video indexing, security, and human-centric scenes, addressing unique "
        "challenges and leveraging domain-specific data."
    ),
}

PROMPT_PATH = Path("prompts/screen_full.txt")
MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_OUTPUT_TOKENS = 6000


def load_default_question() -> str:
    fallback = (
        "How can AI systems efficiently retrieve and semantically understand relevant segments from long-form video content?\n"
        "How can long-form videos be segmented or represented to enable efficient querying?\n"
        "How can natural language queries be used to retrieve relevant segments from long videos?\n"
        "How can multimodal AI models maintain contextual understanding across long video sequences?\n"
        "What methods enable scalable querying and analysis of long-form video datasets?"
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


def build_prompt(question: str, categories: Dict[str, str]) -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_PATH}")
    template = PROMPT_PATH.read_text(encoding="utf-8")
    category_names = list(categories.keys())
    categories_block = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(category_names))
    category_explain_block = "\n".join(
        f"- {name}: {desc}" if desc else f"- {name}" for name, desc in categories.items()
    )
    return (
        template.replace("{CATEGORIES AND EXPLAINATION}", category_explain_block)
        .replace("{RESEARCH_QUESTION}", question)
        .replace("{CATEGORIES}", categories_block)
        .strip()
    )


def call_gpt_pdf_from_path(prompt: str, pdf_path: Path) -> str:
    pdf_bytes = pdf_path.read_bytes()
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise RuntimeError("PDF exceeds 50 MB limit for input_file.")
    return call_gpt_pdf(
        prompt,
        pdf_bytes,
        pdf_path.name,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )


def _clean_paragraph(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r'^\s*"+\s*', "", cleaned)
    cleaned = re.sub(r'\s*"+\s*$', "", cleaned)
    return cleaned.strip()


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

    category_map: Dict[str, Dict[str, object]] = {}
    i = 1
    while i < len(lines):
        line = lines[i].strip()
        if line.upper().startswith("CATEGORY:"):
            category_name = line.split(":", 1)[1].strip()
            if not category_name:
                raise ValueError("Empty CATEGORY name.")

            i += 1
            if i >= len(lines):
                raise ValueError(f"Missing Paragraph for category '{category_name}'.")

            paragraph_line = lines[i].strip()
            if not paragraph_line.upper().startswith("PARAGRAPH:"):
                raise ValueError(f"Missing Paragraph for category '{category_name}'.")

            inline = paragraph_line.split(":", 1)[1].strip()
            i += 1

            if inline:
                paragraph = _clean_paragraph(inline)
            else:
                parts: List[str] = []
                while i < len(lines):
                    nxt = lines[i].strip()
                    if (
                        nxt.upper().startswith("QUOTES:")
                        or nxt.upper().startswith("ANSWER:")
                        or nxt.upper().startswith("CATEGORY:")
                    ):
                        break
                    if nxt:
                        parts.append(_clean_paragraph(nxt))
                    i += 1
                paragraph = " ".join(parts).strip()

            if not paragraph:
                paragraph = "Not mentioned."

            if i >= len(lines):
                raise ValueError(f"Missing Quotes for category '{category_name}'.")

            quotes: List[str] = []
            quotes_line = lines[i].strip()
            if not quotes_line.upper().startswith("QUOTES:"):
                raise ValueError(f"Missing Quotes for category '{category_name}'.")

            inline_q = quotes_line.split(":", 1)[1].strip()
            i += 1

            if inline_q and inline_q.lower().startswith("none"):
                quotes = []
            else:
                if inline_q:
                    quotes.append(_clean_paragraph(inline_q))
                while i < len(lines):
                    nxt = lines[i].strip()
                    if nxt.upper().startswith("ANSWER:") or nxt.upper().startswith("CATEGORY:"):
                        break
                    if nxt.startswith("-"):
                        quotes.append(_clean_paragraph(nxt[1:].strip()))
                    i += 1

            category_map[category_name] = {
                "paragraph": paragraph,
                "quotes": quotes,
            }
            continue

        i += 1

    if not category_map:
        raise ValueError("No categories parsed for YES decision.")

    for category in categories:
        if category not in category_map:
            category_map[category] = {"paragraph": "Not mentioned.", "quotes": []}

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
        "Paste categories JSON (end with empty line):\n"
        "(Press enter to use default JSON)"
    )
    if categories_text:
        try:
            loaded = json.loads(categories_text)
        except json.JSONDecodeError as exc:
            print(f"ERROR: Invalid JSON for categories: {exc}", file=sys.stderr)
            return
        if not isinstance(loaded, dict):
            print("ERROR: Categories JSON must be an object of name -> explanation.", file=sys.stderr)
            return
        categories: Dict[str, str] = {
            str(k).strip(): str(v).strip() for k, v in loaded.items() if str(k).strip()
        }
    else:
        categories = DEFAULT_CATEGORIES.copy()

    if not categories:
        print("ERROR: Categories are empty.", file=sys.stderr)
        return

    prompt = build_prompt(question, categories)

    try:
        raw_output = call_gpt_pdf_from_path(prompt, pdf_path)
        category_names = list(categories.keys())
        parsed = parse_tagged_output(raw_output, category_names)
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
