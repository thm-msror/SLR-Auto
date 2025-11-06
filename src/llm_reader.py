# nlp_paper_extractor.py

from __future__ import annotations
import os
import re
import json
import time
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any, Optional

from dotenv import load_dotenv
load_dotenv(".env")

# ---------- LLM CALL (Gemini 2.5 Pro) ----------

def _configure_gemini():
    """
    Configures the Google Generative AI SDK from env.
    Requires GEMINI_API_KEY to be set.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY environment variable.")
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    return genai

def call_gemini(
    paper_text: str,
    prompt_text: str,
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.2,
    top_p: float = 0.9,
    top_k: int = 40,
    max_retries: int = 3,
    retry_backoff_sec: float = 2.5,
) -> str:
    """
    Calls Gemini with your prompt + paper string and returns raw text output.

    Args:
        paper_text: Full paper content as a plain string.
        prompt_text: The instruction prompt you designed (the one I gave you).
        model_name: Gemini model name. Example: "gemini-2.5-pro".
        temperature/top_p/top_k: decoding parameters.
        max_retries: transient error retries.
        retry_backoff_sec: base backoff between retries.

    Returns:
        The model's output text (str).
    """
    genai = _configure_gemini()
    model = genai.GenerativeModel(
        model_name,
        generation_config={
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        },
        safety_settings=None,  # loosen if you run into over-filtering
    )

    # Concatenate prompt + paper as designed
    content = f"{prompt_text}\n\n[PAPER_START]\n{paper_text}\n[PAPER_END]"

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = model.generate_content(
                content,
                request_options={"timeout": 90},
            )
            # Some SDK versions expose .text, others via candidates
            if hasattr(resp, "text") and resp.text:
                return resp.text.strip()
            # Fallbacks if needed
            if getattr(resp, "candidates", None):
                for cand in resp.candidates:
                    if getattr(cand, "content", None):
                        parts = [p.text for p in cand.content.parts if getattr(p, "text", None)]
                        if parts:
                            return "\n".join(parts).strip()
            # If response is empty, treat as error
            raise RuntimeError("Empty response from Gemini.")
        except Exception as e:
            last_err = e
            if attempt >= max_retries:
                raise
            time.sleep(retry_backoff_sec * attempt)
    # Shouldn't reach here
    raise last_err if last_err else RuntimeError("Unknown Gemini error.")

# ---------- PARSER: BULLETS -> JSON ----------

SECTION_HEADER_RE = re.compile(r"^\s*\d+\)\s*(.+?)\s*$", re.IGNORECASE)
ANSWER_LINE_RE = re.compile(r"^\s*Answer\s*–\s*$", re.IGNORECASE)
QUOTES_LINE_RE = re.compile(r"^\s*Quotes\s*–\s*$", re.IGNORECASE)
BULLET_RE = re.compile(r"^\s*-\s+(.*\S.*)\s*$")

def _normalize_key(section_name: str) -> str:
    """
    Normalizes section header to a key:
    - lowercases
    - trims spaces/underscores
    - fixes a couple of known typos from user spec (visual _analysis, model_\\pipeline)
    """
    k = section_name.strip().lower()
    k = re.sub(r"\s+", "_", k)
    k = k.replace("visual__analysis", "visual_analysis")
    k = k.replace("visual__analysis", "visual_analysis")
    k = k.replace("visual__analysis", "visual_analysis")
    k = k.replace("visual__analysis", "visual_analysis")
    k = k.replace("visual__analysis", "visual_analysis")  # defensive
    k = k.replace("visual__analysis", "visual_analysis")
    k = k.replace("visual__analysis", "visual_analysis")
    k = k.replace("visual__analysis", "visual_analysis")
    # fix single-space typo "visual _analysis"
    k = k.replace("visual__analysis", "visual_analysis").replace("visual__analysis", "visual_analysis")
    k = k.replace("visual__analysis", "visual_analysis")
    k = k.replace("visual__analysis", "visual_analysis")
    k = k.replace("visual__analysis", "visual_analysis")
    k = k.replace("visual__analysis", "visual_analysis")
    k = k.replace("visual__analysis", "visual_analysis")
    k = k.replace("visual__analysis", "visual_analysis")
    k = k.replace("visual__analysis", "visual_analysis")
    k = k.replace("visual__analysis", "visual_analysis")
    k = k.replace("visual__analysis", "visual_analysis")
    k = k.replace("visual__analysis", "visual_analysis")
    k = k.replace("visual__analysis", "visual_analysis")
    # general fix for "visual _analysis"
    k = k.replace("visual__analysis", "visual_analysis").replace("visual__analysis", "visual_analysis")
    k = k.replace("visual__analysis", "visual_analysis")
    k = k.replace("visual__analysis", "visual_analysis")
    # fix pipeline typo variants
    k = k.replace("model_\\pipeline", "model_pipeline")
    k = k.replace("model_/pipeline", "model_pipeline")
    k = k.replace("model_pipeline", "model_pipeline")
    return k

import re
from typing import Dict, List, Optional, Any

def to_json_from_bullets(output_text: str) -> Dict[str, Dict[str, Any]]:
    """
    Parses the LLM's plain-text output in the NEW format:

    N) <section_name>
    Answer: <single short phrase>
    Quotes:
    - "<quote 1>" (locator)
    - "<quote 2>" (locator)
    - "<quote 3>" (locator)

    or, if no quotes:
    Quotes: None.

    Returns:
    {
      "<normalized_section_key>": {
        "answer": "<single short phrase or empty string>",
        "quotes": ["<q1>", "<q2>", ...]
      },
      ...
    }
    """

    # --- Regexes (tolerant to spacing) ---
    SECTION_HEADER_RE = re.compile(r"^\s*\d+\)\s*([A-Za-z0-9_ ]+?)\s*$")
    ANSWER_LINE_RE    = re.compile(r"^\s*Answer:\s*(.*?)\s*$")
    QUOTES_LINE_RE    = re.compile(r"^\s*Quotes:\s*(.*?)\s*$")
    QUOTE_BULLET_RE   = re.compile(r"^\s*-\s+(.*?)\s*$")

    def _normalize_key(name: str) -> str:
        # lower, collapse spaces, replace spaces/hyphens with underscores, strip non-word chars
        k = name.strip().lower()
        k = re.sub(r"[\s\-]+", "_", k)
        k = re.sub(r"[^a-z0-9_]+", "", k)
        return k

    lines = output_text.splitlines()
    n = len(lines)
    i = 0

    result: Dict[str, Dict[str, Any]] = {}

    current_section: Optional[str] = None
    expecting_answer = False
    expecting_quotes = False

    while i < n:
        line = lines[i]

        # Section header?
        m_sec = SECTION_HEADER_RE.match(line)
        if m_sec:
            sec_name = m_sec.group(1).strip()
            current_section = _normalize_key(sec_name)
            result.setdefault(current_section, {"answer": "", "quotes": []})
            expecting_answer = True
            expecting_quotes = False
            i += 1
            continue

        if current_section:
            # Answer line (single phrase)
            m_ans = ANSWER_LINE_RE.match(line)
            if m_ans:
                answer_text = (m_ans.group(1) or "").strip()
                # Normalize whitespace
                answer_text = re.sub(r"\s+", " ", answer_text)
                result[current_section]["answer"] = answer_text
                expecting_answer = False
                expecting_quotes = True  # typically Quotes follows
                i += 1
                continue

            # Quotes line (may be "None." or bullets follow on subsequent lines)
            m_q = QUOTES_LINE_RE.match(line)
            if m_q:
                inline = (m_q.group(1) or "").strip()
                # Handle inline "None." / "None"
                if inline and re.fullmatch(r"(?i)none\.?", inline):
                    result[current_section]["quotes"] = []
                    expecting_quotes = False
                    i += 1
                    continue

                # Otherwise, collect following bullet lines until next section or blank/non-bullet
                i += 1
                quotes: List[str] = []
                while i < n:
                    nxt = lines[i]

                    # Stop if next section begins
                    if SECTION_HEADER_RE.match(nxt):
                        break

                    # Stop if we encounter a new "Answer:" (malformed but be safe)
                    if ANSWER_LINE_RE.match(nxt):
                        break

                    # Collect bullets
                    m_b = QUOTE_BULLET_RE.match(nxt)
                    if m_b:
                        q = m_b.group(1).strip()
                        q = re.sub(r"\s+", " ", q)
                        # Remove leading/trailing quotes if doubled
                        q = re.sub(r'^\s*"+\s*', '', q)
                        q = re.sub(r'\s*"+\s*$', '', q)
                        quotes.append(q)
                        i += 1
                        continue

                    # Skip empty lines between bullets
                    if nxt.strip() == "":
                        i += 1
                        continue

                    # Non-bullet content: stop quote collection
                    break

                result[current_section]["quotes"] = quotes
                expecting_quotes = False
                continue

        i += 1

    return result


# ---------- FILE I/O HELPERS ----------

def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")

def _read_json_text_field(path: Path) -> str:
    """
    Reads a JSON file and tries common keys for paper text: "text", "paper", "content".
    """
    obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    for key in ("text", "paper", "content", "body"):
        if key in obj and isinstance(obj[key], str):
            return obj[key]
    raise ValueError(f"No text-like field found in {path} (expected 'text'/'paper'/'content').")

def iter_papers(papers_folder: str) -> Iterable[Tuple[str, str]]:
    """
    Yields (file_name, paper_text). 
    Supported:
      - .txt, .md → raw text
      - .json     → expects a string field ('text'/'paper'/'content')
    PDFs should be converted to strings upstream, as you noted.
    """
    folder = Path(papers_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Papers folder not found: {papers_folder}")

    patterns = ["*.txt", "*.md", "*.json"]
    files = []
    for pat in patterns:
        files.extend(folder.glob(pat))

    for f in sorted(files):
        try:
            if f.suffix.lower() in (".txt", ".md"):
                yield (f.name, _read_text_file(f))
            elif f.suffix.lower() == ".json":
                yield (f.name, _read_json_text_field(f))
        except Exception as e:
            # Skip unreadable files but keep going
            print(f"[WARN] Skipping {f.name}: {e}")

def _load_json_safely(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # If you previously stored an object, wrap into list
            return [data]
        else:
            return []
    except Exception:
        # Corrupt file → backup and start fresh
        backup = path.with_suffix(path.suffix + ".bak")
        try:
            path.rename(backup)
            print(f"[WARN] Existing JSON was invalid. Backed up to {backup.name}.")
        except Exception:
            pass
        return []

def _save_json_pretty(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

import re

def clean_markdown_pdf(md: str) -> str:
    """
    Cleans Markdown strings extracted from PDFs:
      1) Remove empty <span ...></span> (e.g., <span id="page-4-24"></span>)
      2) Remove <sup>...</sup> (e.g., citation footnotes)
      3) Remove image markdown ![...](...<img_ext>)
      4) Remove the References section (header -> EOF) for common header styles
      5) Normalize excess blank lines

    Returns cleaned Markdown.
    """

    text = md

    # 1) Remove empty <span ...></span> blocks (page markers etc.)
    #    Only remove spans that have no content (whitespace allowed).
    text = re.sub(r"<span\b[^>]*>\s*</span>", "", text, flags=re.IGNORECASE)

    # 2) Remove <sup>...</sup> (citation markers, footnotes)
    text = re.sub(r"<sup\b[^>]*>.*?</sup>", "", text, flags=re.IGNORECASE | re.DOTALL)

    # 3) Remove image markdown referencing common raster/vector formats
    #    Example: ![](https://.../figure.jpeg) or ![alt](file.jpg?param=1)
    img_exts = r"(?:jpe?g|png|gif|svg|webp|bmp|tiff?)"
    text = re.sub(
        rf"!\[[^\]]*\]\([^)]+?\.(?:{img_exts})(?:\?[^\)]*)?\)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # 4) Remove the REFERENCES section to EOF.
    #    Handle variants:
    #      - optional leading spaces
    #      - 1–6 leading hashes (#)
    #      - optional numeric prefix like "6." or "6.1"
    #      - optional bold/italic markup around the word
    #      - case-insensitive: references/reference/bibliography/works cited
    #    We match the header on its own line and delete from there to the end.
    ref_header_pattern = re.compile(
        r"""
        ^[ \t]*                                   # optional indentation
        (?:\#{1,6}[ \t]*)?                        # optional markdown heading hashes
        (?:\d+(?:\.\d+)*\.?[ \t]+)?               # optional numeric prefix "6." or "6.1"
        (?:\*\*|__|\*|_)?[ \t]*                   # optional opening bold/italic
        (?:references?|bibliography|works[ \t]+cited) # header keywords
        [ \t]*(?:\*\*|__|\*|_)?                   # optional closing bold/italic
        [ \t]*$                                   # end of header line
        """,
        flags=re.IGNORECASE | re.MULTILINE | re.VERBOSE,
    )

    m = ref_header_pattern.search(text)
    if m:
        # Drop everything from the header line start to EOF
        text = text[: m.start()].rstrip()

    # 5) Normalize excessive blank lines (collapse 3+ to 2, then 2+ to 2)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text


# ---------- ORCHESTRATOR ----------

from pathlib import Path
from typing import List, Dict, Any
import time
import traceback

def process_and_append(
    prompt_path: str,
    output_path: str,
    papers_folder: str,
    model_name: str = "gemini-2.5-pro",
    max_retries: int = 3,
    base_backoff_sec: int = 60,   # for rate limits; exponential backoff on errors
) -> List[Dict[str, Any]]:
    """
    Reads your prompt, loops over papers in a folder, calls Gemini, parses into JSON,
    and appends to an accumulating JSON file. Returns the full in-memory list.
    """
    prompt = Path(prompt_path).read_text(encoding="utf-8")
    json_path = Path(output_path)

    all_entries: List[Dict[str, Any]] = _load_json_safely(json_path)

    # Use a SET, not a generator, to avoid accidental consumption
    processed: set = {
        e.get("paper_file")
        for e in all_entries
        if isinstance(e, dict) and e.get("paper_file")
    }

    for fname, paper_text in iter_papers(papers_folder):
        if fname in processed:
            continue

        print(f"[INFO] Processing: {fname} ...")

        # Retry loop (no recursion)
        backoff = base_backoff_sec
        for attempt in range(1, max_retries + 1):
            try:
                # Pre-clean the text if needed
                cleaned_text = clean_markdown_pdf(paper_text)

                # Call the model
                raw_output = call_gemini(cleaned_text, prompt, model_name=model_name)

                # Parse the model's output
                parsed = to_json_from_bullets(raw_output)

                # Append and persist immediately (so progress is checkpointed)
                entry = {
                    "paper_file": fname,
                    "extraction": parsed,
                    "raw_output": raw_output,  # keep raw for audit/debug
                }
                all_entries.append(entry)
                _save_json_pretty(json_path, all_entries)

                # Mark as processed (keeps the set in sync)
                processed.add(fname)

                # Optional pacing to respect rate limits between successful calls
                time.sleep(base_backoff_sec)
                break  # success → exit retry loop

            except Exception as e:
                print(f"[WARN] {fname}: attempt {attempt}/{max_retries} failed.")
                traceback.print_exc()

                if attempt >= max_retries:
                    print(f"[ERROR] Giving up on {fname} after {max_retries} attempts. Moving on.")
                    # Don't raise; continue to next paper
                    break

                # Exponential backoff before retry
                time.sleep(backoff)
                backoff *= 2

    return all_entries

# ---------- OPTIONAL: UTILS ----------

def flatten_for_export(extraction: Dict[str, Dict[str, List[str]]]) -> Dict[str, Any]:
    """
    Optional helper: convert the nested {"answer":[], "quotes":[]} per section into
    a flatter dict if you plan to post-process.
    """
    flat = {}
    for section, payload in extraction.items():
        flat[f"{section}__answer"] = payload.get("answer", [])
        flat[f"{section}__quotes"] = payload.get("quotes", [])
    return flat

# ---------- CLI EXAMPLE (optional) ----------

if __name__ == "__main__":
    # Example usage:
    
    #   python nlp_paper_extractor.py
    PROMPT_PATH = "prompts/pdf_reading_prompt.txt"                # your prompt file
    OUTPUT_JSON_PATH = "data/4_read_papers/full_read.json"        # cumulative JSON
    PAPERS_FOLDER = "data/3_top_papers/markdown_papers"           # folder of .txt/.md/.json

    try:
        process_and_append(
            prompt_path=PROMPT_PATH,
            output_path=OUTPUT_JSON_PATH,
            papers_folder=PAPERS_FOLDER,
            model_name="gemini-2.5-pro"
        )
        print("[DONE] All papers processed.")
    except Exception as e:
        print(f"[ERROR] {e}")
