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
    Included: <boolean or yes/no>

    or, if no quotes:
    Quotes: None.

    Returns:
    {
      "<normalized_section_key>": {
        "answer": "<single short phrase or empty string>",
        "quotes": ["<q1>", "<q2>", ...],
        "included": bool
      },
      ...
    }
    """

    SECTION_HEADER_RE = re.compile(r"^\s*\d+\)\s*([A-Za-z0-9_ ]+?)\s*$")
    ANSWER_LINE_RE    = re.compile(r"^\s*Answer:\s*(.*?)\s*$")
    QUOTES_LINE_RE    = re.compile(r"^\s*Quotes:\s*(.*?)\s*$")
    QUOTE_BULLET_RE   = re.compile(r"^\s*-\s+(.*?)\s*$")
    INCLUDED_LINE_RE  = re.compile(r"^\s*Included:\s*(.*?)\s*$", re.IGNORECASE)

    def _normalize_key(name: str) -> str:
        k = name.strip().lower()
        k = re.sub(r"[\s\-]+", "_", k)
        k = re.sub(r"[^a-z0-9_]+", "", k)
        return k

    def _parse_bool(text: str) -> bool:
        t = text.strip().lower()
        return t in {"true", "yes", "1", "y", "included"}

    lines = output_text.splitlines()
    n = len(lines)
    i = 0

    result: Dict[str, Dict[str, Any]] = {}
    current_section: Optional[str] = None

    while i < n:
        line = lines[i]

        # Section header?
        m_sec = SECTION_HEADER_RE.match(line)
        if m_sec:
            sec_name = m_sec.group(1).strip()
            current_section = _normalize_key(sec_name)
            result.setdefault(current_section, {"answer": "", "quotes": [], "included": None})
            i += 1
            continue

        if current_section:
            # Answer
            m_ans = ANSWER_LINE_RE.match(line)
            if m_ans:
                answer_text = (m_ans.group(1) or "").strip()
                answer_text = re.sub(r"\s+", " ", answer_text)
                result[current_section]["answer"] = answer_text
                i += 1
                continue

            # Quotes
            m_q = QUOTES_LINE_RE.match(line)
            if m_q:
                inline = (m_q.group(1) or "").strip()
                if inline and re.fullmatch(r"(?i)none\.?", inline):
                    result[current_section]["quotes"] = []
                    i += 1
                    continue

                quotes: List[str] = []
                i += 1
                while i < n:
                    nxt = lines[i]
                    if SECTION_HEADER_RE.match(nxt) or ANSWER_LINE_RE.match(nxt) or INCLUDED_LINE_RE.match(nxt):
                        break
                    m_b = QUOTE_BULLET_RE.match(nxt)
                    if m_b:
                        q = m_b.group(1).strip()
                        q = re.sub(r"\s+", " ", q)
                        q = re.sub(r'^\s*"+\s*', '', q)
                        q = re.sub(r'\s*"+\s*$', '', q)
                        quotes.append(q)
                        i += 1
                        continue
                    if nxt.strip() == "":
                        i += 1
                        continue
                    break
                result[current_section]["quotes"] = quotes
                continue

            # Included
            m_inc = INCLUDED_LINE_RE.match(line)
            if m_inc:
                val = m_inc.group(1).strip()
                result[current_section]["included"] = _parse_bool(val)
                i += 1
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

def clean_markdown_pdf(md: str) -> str:
    """
    Cleans Markdown strings extracted from PDFs:
      - Strip empty spans, <sup>...</sup>, ALL HTML tags
      - Remove images
      - Remove links (inline, reference, autolinks, bare URLs)
      - Remove empty anchor links like [](#page-15-4)
      - Remove bare anchor parens like (#page-15-4)
      - Drop References/Bibliography/Works Cited section → EOF
      - Remove LaTeX wrappers and math blocks
      - Normalize whitespace/separators
    """
    text = md

    # 1) Empty <span ...></span>
    text = re.sub(r"<span\b[^>]*>\s*</span>", "", text, flags=re.IGNORECASE)

    # 2) <sup>...</sup>
    # text = re.sub(r"<sup\b[^>]*>.*?</sup>", "", text, flags=re.IGNORECASE | re.DOTALL)

    # 3) ALL HTML tags (preserve inner)
    text = re.sub(r"</?[^>\s]+(?:\s[^>]*?)?>", "", text, flags=re.IGNORECASE)

    # 4) Images ![...](...ext)
    img_exts = r"(?:jpe?g|png|gif|svg|webp|bmp|tiff?)"
    text = re.sub(
        rf"!\[[^\]]*\]\([^)]+?\.(?:{img_exts})(?:\?[^\)]*)?\)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # 5) Links
    # 5a) Inline: [text](url) -> keep 'text'
    text = re.sub(r"\[([^\]]+)\]\((?!\))[^)]+\)", r"\1", text)
    # 5b) Reference-style: [text][id] -> 'text'
    text = re.sub(r"\[([^\]]+)\]\[[^\]]+\]", r"\1", text)
    # 5c) Link defs: [id]: url  (remove whole line)
    # text = re.sub(r"^[ \t]*\[[^\]]+\]:\s+\S+\s*$", "", text, flags=re.MULTILINE)
    # 5d) Autolinks <http(s)://...> -> remove
    # text = re.sub(r"<https?://[^>]+>", "", text, flags=re.IGNORECASE)
    # 5e) Bare URLs -> remove
    # text = re.sub(r"https?://\S+", "", text, flags=re.IGNORECASE)
    # 5f) EMPTY inline anchor links: [](#page-15-4) / [](#anything)
    text = re.sub(r"\[\s*\]\(\s*#[^)]+\)", "", text)  # remove entirely
    # 5g) Bare anchor parens that sometimes appear: (#page-15-4)
    text = re.sub(r"\(\s*#[^)]+\)", "", text)

    # 6) Remove References/Bibliography/Works Cited → EOF
    ref_header_pattern = re.compile(
        r"""
        ^[ \t]*                                   # indentation
        (?:\#{1,6}[ \t]*)?                        # optional hashes
        (?:\d+(?:\.\d+)*\.?[ \t]+)?               # optional numeric prefix
        (?:\*\*|__|\*|_)?[ \t]*                   # optional bold/italic open
        (?:references?|bibliography|works[ \t]+cited) # keywords
        [ \t]*(?:\*\*|__|\*|_)?                   # optional bold/italic close
        [ \t]*$                                   # end of line
        """,
        flags=re.IGNORECASE | re.MULTILINE | re.VERBOSE,
    )
    m = ref_header_pattern.search(text)
    if m:
        text = text[: m.start()].rstrip()

    # 7) LaTeX math wrappers (keep inner content)
    text = re.sub(r"\\(mathcal|mathbf|mathrm|mathbb)\s*\{([^{}]*)\}", r"\2", text)

    # Strip inline/display math blocks entirely
    text = re.sub(r"\$\$(.*?)\$\$", "", text, flags=re.DOTALL)  # $$...$$
    text = re.sub(r"\$(?!\s)(.+?)(?<!\s)\$", "", text, flags=re.DOTALL)  # $...$
    text = re.sub(r"\\\((.*?)\\\)", "", text, flags=re.DOTALL)  # \( ... \)
    text = re.sub(r"\\\[(.*?)\\\]", "", text, flags=re.DOTALL)  # \[ ... \]

    # 8) Normalize whitespace & leftover separators
    # collapse runs created by removing anchors like [](#page-15-4)–[](#page-16-1)
    text = re.sub(r"[ \t]*[–—-][ \t]*([–—-][ \t]*)+", " – ", text)  # multi dashes → single en-dash
    text = re.sub(r"[ \t]+", " ", text)                              # collapse spaces
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)                     # tidy line breaks
    text = re.sub(r"\n{3,}", "\n\n", text).strip()                   # max 2 blank lines

    return text


# ---------- ORCHESTRATOR ----------

from pathlib import Path
from typing import List, Dict, Any
from collections import deque
import time, math, traceback
import re

# --- Heuristic token estimator (chars ≈ 4 per token) ---
def _estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)

# --- Rolling window trackers for RPM and TPM ---
class _RateWindow:
    """
    Tracks timestamps of requests (for RPM) and (timestamp, tokens) for TPM.
    Keeps only entries in the last 60 seconds.
    """
    def __init__(self):
        self.req_times = deque()          # for RPM
        self.token_times = deque()        # (t, tokens) for TPM
        self.window_s = 60

    def _prune(self, now: float):
        # drop entries older than window
        while self.req_times and now - self.req_times[0] >= self.window_s:
            self.req_times.popleft()
        while self.token_times and now - self.token_times[0][0] >= self.window_s:
            self.token_times.popleft()

    def add_request(self, now: float):
        self.req_times.append(now)

    def add_tokens(self, now: float, tokens: int):
        self.token_times.append((now, tokens))

    def rpm_used(self, now: float) -> int:
        self._prune(now)
        return len(self.req_times)

    def tpm_used(self, now: float) -> int:
        self._prune(now)
        return sum(t for _, t in self.token_times)

    def next_allowed_delay(self, now: float, rpm_limit: int, tpm_limit: int,
                           tokens_next_call: int) -> float:
        """
        Returns the seconds to wait so the *next* call will not exceed RPM or TPM.
        If 0, you can call immediately.
        """
        self._prune(now)

        # --- RPM guard: at most rpm_limit calls per 60s ---
        delay_rpm = 0.0
        if rpm_limit > 0 and len(self.req_times) >= rpm_limit:
            # when will the oldest call expire out of the 60s window?
            oldest = self.req_times[0]
            delay_rpm = max(0.0, (oldest + self.window_s) - now)

        # --- TPM guard: sum(tokens in last 60s) + next_call_tokens <= tpm_limit ---
        delay_tpm = 0.0
        if tpm_limit > 0:
            used = self.tpm_used(now)
            overflow = used + tokens_next_call - tpm_limit
            if overflow > 0:
                # need to wait until enough tokens expire from the window
                running = used
                for (tstamp, tok) in list(self.token_times):
                    # simulate expiring entries until we drop below threshold
                    if running + tokens_next_call <= tpm_limit:
                        break
                    running -= tok
                    # when this entry expires:
                    candidate_delay = max(0.0, (tstamp + self.window_s) - now)
                    delay_tpm = max(delay_tpm, candidate_delay)

        return max(delay_rpm, delay_tpm)


def read_paper_mds(
    prompt_path: str,
    output_path: str,
    papers_folder: str,
    model_name: str = "gemini-2.5-pro",
    max_retries: int = 3,
    base_backoff_sec: int = 60,   # exponential backoff for errors only
    # --- New pacing knobs ---
    RPM_LIMIT: int = 2,           # free Gemini 2.5 Pro default (requests/min)
    TPM_LIMIT: int = 125_000,     # free Gemini 2.5 Pro default (tokens/min)
    OUTPUT_TOKENS: int = 2_000,   # how many tokens you request in the response
    SAFETY_MARGIN: float = 0.90,  # keep headroom for tokenizer variance
) -> List[Dict[str, Any]]:
    """
    Reads your prompt, loops over papers in a folder, calls Gemini, parses into JSON,
    and appends to an accumulating JSON file. Returns the full in-memory list.

    NEW: Enforces RPM + TPM pacing with a rolling 60s window. It waits the minimum
    amount of time so the next call won't breach limits.
    """
    prompt = Path(prompt_path).read_text(encoding="utf-8")
    json_path = Path(output_path)

    all_entries: List[Dict[str, Any]] = _load_json_safely(json_path)

    processed: set = {
        e.get("paper_file")
        for e in all_entries
        if isinstance(e, dict) and e.get("paper_file")
    }

    # --- Pre-compute prompt tokens once (cleaned paper will be added per-doc) ---
    prompt_tokens = _estimate_tokens(prompt)

    # --- Rolling window trackers ---
    window = _RateWindow()

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

                # --- Compute tokens for this call (input + requested output) ---
                # Leave a safety margin so we don't flirt with the hard context limit.
                input_tokens = int(SAFETY_MARGIN * (prompt_tokens + _estimate_tokens(cleaned_text)))
                total_tokens_this_call = input_tokens + OUTPUT_TOKENS

                # --- Wait just enough to satisfy RPM + TPM ---
                now = time.time()
                wait_s = window.next_allowed_delay(
                    now,
                    rpm_limit=RPM_LIMIT,
                    tpm_limit=TPM_LIMIT,
                    tokens_next_call=total_tokens_this_call
                )
                if wait_s > 0:
                    time.sleep(wait_s)

                # --- Make the call ---
                raw_output = call_gemini(
                    cleaned_text,
                    prompt,
                    model_name=model_name
                    # max_output_tokens=OUTPUT_TOKENS  # pass through if your wrapper supports it
                )

                # Record success into the rolling windows
                call_time = time.time()
                window.add_request(call_time)
                window.add_tokens(call_time, total_tokens_this_call)

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

                # Optional: tiny breath between successful calls so logs are readable
                time.sleep(0.25)
                break  # success → exit retry loop

            except Exception as e:
                print(f"[WARN] {fname}: attempt {attempt}/{max_retries} failed.")
                traceback.print_exc()

                if attempt >= max_retries:
                    print(f"[ERROR] Giving up on {fname} after {max_retries} attempts. Moving on.")
                    break

                # Exponential backoff before retry (error handling only)
                time.sleep(backoff)
                backoff *= 2

    return all_entries

# ---------- OPTIONAL: UTILS ----------
from pathlib import Path
import json
import csv
from typing import List, Dict, Any

def _escape_md_cell(s: str) -> str:
    """Escape Markdown table pipes and normalize newlines."""
    if s is None:
        return ""
    s = str(s).replace("\n", "<br>")
    s = s.replace("|", r"\|")
    return s

def _flatten_answer_row(entry: Dict[str, Any], keys: List[str]) -> Dict[str, str]:
    """
    Turn one entry into a flat row mapping:
      'paper_file' -> str
      each key in keys -> extraction[key]['answer'] or ''
    """
    row = {"paper_file": entry.get("paper_file", "")}
    extraction = entry.get("extraction", {}) or {}
    for k in keys:
        ans = ""
        if isinstance(extraction, dict) and k in extraction and isinstance(extraction[k], dict):
            ans = extraction[k].get("answer", "")
        # collapse linebreaks for CSV friendliness
        if isinstance(ans, str):
            ans = ans.replace("\r", " ").replace("\n", " ").strip()
        row[k] = ans
    return row

def table_summary(
    papers: list,
    csv_output: str,
    md_output: str,
    column_order: List[str] = None,
) -> None:
    """
    Reads the JSON produced by your pipeline (list of entries with 'paper_file' and 'extraction'),
    writes a CSV and a Markdown table with 'answer' fields only.
    """

    # union of all extraction keys across entries
    all_keys = []
    seen = set()
    for e in papers:
        extraction = e.get("extraction", {}) or {}
        if isinstance(extraction, dict):
            for k in extraction.keys():
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)

    # user-specified order or default alphabetical for stability
    if column_order:
        # keep only known keys, preserve given order, then append any extras
        col_set = set(all_keys)
        ordered = [k for k in column_order if k in col_set]
        extras = [k for k in sorted(all_keys) if k not in set(ordered)]
        keys_final = ordered + extras
    else:
        keys_final = sorted(all_keys)

    # final headers
    headers = ["paper_file"] + keys_final

    # ---- CSV ----
    csv_path = Path(csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for e in papers:
            writer.writerow(_flatten_answer_row(e, keys_final))

    # ---- Markdown ----
    md_path = Path(md_output)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    # Build markdown table
    lines = []
    # header
    lines.append("| " + " | ".join(_escape_md_cell(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")

    for e in papers:
        # for markdown we keep basic line breaks as <br>
        extraction = e.get("extraction", {}) or {}
        row_vals = [e.get("paper_file", "")]
        for k in keys_final:
            ans = ""
            if isinstance(extraction, dict) and k in extraction and isinstance(extraction[k], dict):
                ans = extraction[k].get("answer", "")
            row_vals.append(ans if isinstance(ans, str) else "")
        lines.append("| " + " | ".join(_escape_md_cell(v) for v in row_vals) + " |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] CSV written to: {csv_path}")
    print(f"[OK] Markdown written to: {md_path}")
    print(f"[OK] Columns: {', '.join(headers)}")