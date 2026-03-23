import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

from atlas.read_paper.prompts import SCREEN_FULL_PROMPT
from atlas.utils.gpt_client import call_gpt_pdf
from atlas.utils.utils import read_multiline_input

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

    path = Path(__file__).resolve().parents[1] / "inital_fetch" / "gpt_research_q.py"
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


def build_prompt(
    question: str,
    categories: Dict[str, str],
    prompt_text: str = SCREEN_FULL_PROMPT,
) -> str:
    template = (prompt_text or "").strip() or SCREEN_FULL_PROMPT
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
    # Remove markdown code blocks if present
    text = re.sub(r"```[a-zA-Z]*\n", "", text)
    text = text.replace("```", "")
    
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Empty output.")

    # 1. Find Decision with improved regex
    decision = None
    decision_idx = -1
    for idx, line in enumerate(lines):
        # Match "DECISION: YES", "**DECISION:** NO", "Selected: YES", etc.
        match = re.search(r"(?:DECISION|SELECTED|INCLUDED?)\s*[:\-]?\s*(YES|NO|TRUE|FALSE)\b", line, re.IGNORECASE)
        if match:
            val = match.group(1).upper()
            decision = "YES" if val in ["YES", "TRUE"] else "NO"
            decision_idx = idx
            break
            
    if decision is None:
        # Fallback: check if the first line IS just "YES" or "NO"
        first_line = re.sub(r"^[#*\s-]+", "", lines[0]).strip().upper()
        if first_line in ["YES", "NO"]:
            decision = first_line
            decision_idx = 0

    if decision is None:
        raise ValueError("Missing DECISION line. Please ensure output starts with 'DECISION: YES' or 'DECISION: NO'.")

    if decision == "NO":
        return {"included": False}

    # 2. Extract Blocks
    category_map: Dict[str, Dict[str, object]] = {}
    i = decision_idx + 1
    current_category = None
    
    while i < len(lines):
        line = lines[i].strip()
        clean_line = re.sub(r"^[#*\s-]+", "", line).strip()
        
        # Check for Category tag (robustly)
        cat_match = re.search(r"^(?:CATEGORY|THEME|SECTION)\s*[:\-]?\s*(.*)$", clean_line, re.IGNORECASE)
        
        found_cat_name = None
        if cat_match:
            found_cat_name = cat_match.group(1).strip()
        else:
            # Fallback: Does the line start with one of our category names?
            for target_cat in categories:
                # Use escaped name for regex, allowing for leading numbers
                pattern = r"^(?:\d+[.)]\s*)?" + re.escape(target_cat) + r"\b"
                if re.search(pattern, clean_line, re.IGNORECASE):
                    found_cat_name = target_cat
                    break
        
        if found_cat_name:
            # Normalize the found name
            found_cat_name = re.sub(r"[*]+$", "", found_cat_name).strip()
            # If it's a number like "1. Name", strip number
            found_cat_name = re.sub(r"^\d+[.)]\s+", "", found_cat_name).strip()
            
            # Map to the closest canonical category name
            closest = None
            for c in categories:
                if c.lower() in found_cat_name.lower() or found_cat_name.lower() in c.lower():
                    closest = c
                    break
            
            current_category = closest or found_cat_name
            if current_category not in category_map:
                category_map[current_category] = {"paragraph": "Not mentioned.", "quotes": []}
            i += 1
            continue

        if current_category:
            # Parse Paragraph/Quotes for the current category
            if clean_line.upper().startswith("PARAGRAPH:"):
                para_part = clean_line.split(":", 1)[1].strip()
                if not para_part:
                    # Multi-line paragraph
                    parts = []
                    i += 1
                    while i < len(lines):
                        nxt = lines[i].strip()
                        nxt_clean = re.sub(r"^[#*\s-]+", "", nxt).strip()
                        if any(nxt_clean.upper().startswith(tag) for tag in ["QUOTES:", "CATEGORY:", "DECISION:"]):
                            break
                        parts.append(nxt)
                        i += 1
                    para_part = " ".join(parts).strip()
                    i -= 1 # adjust back
                category_map[current_category]["paragraph"] = para_part or "Not mentioned."
            
            elif clean_line.upper().startswith("QUOTES:"):
                quotes_part = clean_line.split(":", 1)[1].strip()
                quotes = []
                if quotes_part and not quotes_part.lower().startswith("none"):
                    quotes.append(quotes_part.strip('"'))
                
                i += 1
                while i < len(lines):
                    nxt = lines[i].strip()
                    nxt_clean = re.sub(r"^[#*\s-]+", "", nxt).strip()
                    if any(nxt_clean.upper().startswith(tag) for tag in ["CATEGORY:", "DECISION:"]):
                        break
                    if nxt.startswith("-") or nxt.startswith("*"):
                        q_text = re.sub(r"^[#*\s-]+", "", nxt).strip().strip('"')
                        if q_text: quotes.append(q_text)
                    i += 1
                category_map[current_category]["quotes"] = quotes
                i -= 1 # adjust back
        
        i += 1

    if not category_map:
        raise ValueError("No valid categories could be parsed from the response. Please ensure the model uses the specified tags.")

    # 3. Final Regularization
    final_category_map = {}
    for target_cat in categories:
        # Check for matching in parsed keys
        src = category_map.get(target_cat)
        if not src:
            for k, v in category_map.items():
                if k.lower() in target_cat.lower() or target_cat.lower() in k.lower():
                    src = v
                    break
        
        if src:
            final_category_map[target_cat] = src
        else:
            final_category_map[target_cat] = {"paragraph": "Not mentioned.", "quotes": []}

    return {"included": True, "categories": final_category_map}


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
