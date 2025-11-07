from pathlib import Path
from typing import Dict, Any, List
from collections import deque
import math, time, json, re, traceback, os, random

# -------------------------
# Token estimation (heuristic)
# -------------------------
def _estimate_tokens(s: str) -> int:
    return math.ceil(len(s) / 4)  # ~4 chars per token

# -------------------------
# Simple 60s rolling windows for RPM/TPM
# -------------------------
class _RateWindow:
    def __init__(self):
        self.req_times = deque()
        self.token_times = deque()
        self.window = 60.0

    def _prune(self, now: float):
        while self.req_times and now - self.req_times[0] >= self.window:
            self.req_times.popleft()
        while self.token_times and now - self.token_times[0][0] >= self.window:
            self.token_times.popleft()

    def record(self, now: float, tokens: int):
        self.req_times.append(now)
        self.token_times.append((now, tokens))

    def next_delay(self, now: float, rpm_limit: int, tpm_limit: int, next_tokens: int) -> float:
        self._prune(now)
        delay_rpm = 0.0
        # RPM guard
        if rpm_limit and len(self.req_times) >= rpm_limit:
            delay_rpm = (self.req_times[0] + self.window) - now
            if delay_rpm < 0: delay_rpm = 0.0
        # TPM guard
        delay_tpm = 0.0
        if tpm_limit:
            used = sum(t for _, t in self.token_times)
            overflow = used + next_tokens - tpm_limit
            if overflow > 0:
                # wait until enough tokens expire
                running = used
                for (ts, tok) in list(self.token_times):
                    if running + next_tokens <= tpm_limit:
                        break
                    running -= tok
                    cand = (ts + self.window) - now
                    if cand > delay_tpm:
                        delay_tpm = max(0.0, cand)
        return max(delay_rpm, delay_tpm)

# -------------------------
# Prompts (you can keep yours)
# -------------------------
CHUNK_PROMPT = """You are an expert literature reviewer.
Given a set of paper notes for one gap, extract concise bullet points:

Output strictly as JSON with keys:
- "themes": [short bullets]
- "similarities": [short bullets]
- "differences": [short bullets]
- "notable_outliers": [each as "paper: short reason"]
- "quick_groups": [{"group_label": "...", "papers": ["paper1", "paper2", ...]}]

Be terse; no prose paragraph. Only JSON.
"""

FINAL_PROMPT = """You are an expert literature reviewer.
You will receive multiple JSON fragments, each produced from a subset of papers about the SAME gap.
1) Merge all fragments.
2) Produce a final JSON with:
{
  "gap": "<gap_name>",
  "literature_paragraph": "<one tight paragraph synthesizing the literature for this gap>",
  "themes": [...],
  "similarities": [...],
  "differences": [...],
  "notable_outliers": [...],
  "groups": [{"group_label": "...", "papers": ["..."]}]
}
- The paragraph should reference papers by short filename when useful (no citations).
- Keep it compact and information-dense.
- Only return JSON.
"""

# -------------------------
# Gemini config + call
# -------------------------
from dotenv import load_dotenv
load_dotenv(".env")

def _configure_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY environment variable.")
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    return genai

def _is_quota_error(exc: Exception) -> bool:
    """Detect rate/quota errors across SDK shapes."""
    msg = str(exc).lower()
    if any(k in msg for k in [
        "429", "rate limit", "quota", "resourceexhausted",
        "too many requests", "exceeded", "tokens per minute"
    ]):
        return True
    # Some SDKs attach HTTP status
    try:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 429:
            return True
    except Exception:
        pass
    return False

def call_gemini(
    paper_text: str,
    prompt_text: str,
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.2,
    top_p: float = 0.9,
    top_k: int = 40,
    max_retries: int = 6,
    retry_backoff_sec: float = 2.5,
    backoff_cap_sec: float = 180.0,   # cap a single sleep
) -> str:
    """
    Quota-safe Gemini call: exponential backoff + jitter on quota/rate errors.
    """
    genai = _configure_gemini()
    model = genai.GenerativeModel(
        model_name,
        generation_config={
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        },
        safety_settings=None,
    )

    content = f"{prompt_text}\n\n[PAPER_START]\n{paper_text}\n[PAPER_END]"

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = model.generate_content(
                content,
                request_options={"timeout": 90},
            )
            if hasattr(resp, "text") and resp.text:
                return resp.text.strip()
            if getattr(resp, "candidates", None):
                for cand in resp.candidates:
                    if getattr(cand, "content", None):
                        parts = [p.text for p in cand.content.parts if getattr(p, "text", None)]
                        if parts:
                            return "\n".join(parts).strip()
            raise RuntimeError("Empty response from Gemini.")
        except Exception as e:
            last_err = e
            # quota/rate → dynamic backoff with jitter; other errors → fast retry then raise
            if _is_quota_error(e):
                base = min(backoff_cap_sec, retry_backoff_sec * (2 ** (attempt - 1)))
                jitter = random.uniform(0.25 * base, 0.75 * base)
                sleep_s = min(backoff_cap_sec, base + jitter)
                print(f"[QUOTA] Attempt {attempt}/{max_retries} — sleeping {sleep_s:.1f}s … ({e})")
                time.sleep(sleep_s)
                continue
            else:
                print(f"[WARN] Non-quota error, attempt {attempt}/{max_retries}: {e}")
                if attempt >= max_retries:
                    raise
                time.sleep(min(backoff_cap_sec, retry_backoff_sec * attempt))
    # Shouldn't reach here
    raise last_err if last_err else RuntimeError("Unknown Gemini error.")

# -------------------------
# Helpers: autosave & safe JSON extraction
# -------------------------
def _extract_json_best_effort(s: str, fallback: dict) -> dict:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return fallback

def _save_checkpoint(json_path: Path, md_path: Path,
                     results: Dict[str, Any], md_lines: List[str],
                     tag: str = ""):
    tmp_json = json_path.with_suffix(".checkpoint.json")
    tmp_md   = md_path.with_suffix(".checkpoint.md")
    try:
        tmp_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        if tag:
            print(f"[SAVE] Checkpoint ({tag}) → {tmp_json} / {tmp_md}")
        else:
            print(f"[SAVE] Checkpoint → {tmp_json} / {tmp_md}")
    except Exception as e:
        print(f"[WARN] Failed to save checkpoint: {e}")

# -------------------------
# Core runner (no chunking, dynamic quota handling + autosave)
# -------------------------
def generate_gap_reviews(
    gap_answers: Dict[str, str],
    model_name: str = "gemini-2.5-pro",
    output_json: str = "data/markdown_papers/gap_reviews.json",
    output_md: str = "data/markdown_papers/gap_reviews.md",
    # pacing / limits (typical free/dev defaults for 2.5 Pro)
    RPM_LIMIT: int = 2,
    TPM_LIMIT: int = 125_000,
    MAX_INPUT_TOKENS: int = 1_048_576,   # not used directly here but kept for clarity
    OUTPUT_TOKENS_FINAL: int = 1800,
) -> Dict[str, Any]:

    results: Dict[str, Any] = {}
    window = _RateWindow()

    json_path = Path(output_json)
    md_path   = Path(output_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    md_lines = ["# Literature Review by Gap\n"]

    # adaptive cooldown factor: grows after quota hits, shrinks after successes
    cooldown_factor = 1.0
    COOLDOWN_MAX = 8.0
    COOLDOWN_MIN = 1.0

    for gap, text in gap_answers.items():
        print(f"[INFO] Gap: {gap} — single-call synthesis …")

        merge_payload = json.dumps({"gap": gap, "notes": text}, ensure_ascii=False)

        # pacing calc (input + prompt + output tokens)
        in_tokens_final = _estimate_tokens(merge_payload) + _estimate_tokens(FINAL_PROMPT)
        total_tokens_final = in_tokens_final + OUTPUT_TOKENS_FINAL

        # Respect RPM/TPM with adaptive cooldown
        now = time.time()
        wait_s = window.next_delay(now, RPM_LIMIT, TPM_LIMIT, total_tokens_final)
        wait_s *= cooldown_factor
        if wait_s > 0:
            time.sleep(wait_s)

        # quota-aware call with its own internal backoff
        try:
            final_json_str = call_gemini(
                merge_payload,
                FINAL_PROMPT,
                model_name=model_name,
            )
            window.record(time.time(), total_tokens_final)

            # success: gently relax cooldown
            cooldown_factor = max(COOLDOWN_MIN, cooldown_factor / 1.25)

        except Exception as e:
            # If this is a quota error that bubbled up (exhausted retries), increase cooldown and checkpoint
            if _is_quota_error(e):
                cooldown_factor = min(COOLDOWN_MAX, cooldown_factor * 1.75)
                print(f"[QUOTA] Escalated cooldown → x{cooldown_factor:.2f}. Saving checkpoint and continuing …")
                _save_checkpoint(json_path, md_path, results, md_lines, tag=f"quota-{gap}")
                # Put a longer cool-down before moving on to the next gap
                sleep_more = min(300, 30 * cooldown_factor + random.uniform(5, 20))
                time.sleep(sleep_more)
                # Store a placeholder so downstream knows this gap failed now
                results[gap] = {
                    "gap": gap,
                    "literature_paragraph": f"(Temporarily skipped due to quota limits: {e})",
                    "themes": [],
                    "similarities": [],
                    "differences": [],
                    "notable_outliers": [],
                    "groups": []
                }
                # write placeholder into MD too
                md_lines.append(f"## {gap}\n")
                md_lines.append(results[gap]["literature_paragraph"])
                md_lines.append("")
                continue
            else:
                print(f"[ERROR] Gap '{gap}' failed with non-quota error. Saving checkpoint and continuing …")
                traceback.print_exc()
                _save_checkpoint(json_path, md_path, results, md_lines, tag=f"error-{gap}")
                # Store placeholder
                results[gap] = {
                    "gap": gap,
                    "literature_paragraph": f"(Failed: {e})",
                    "themes": [],
                    "similarities": [],
                    "differences": [],
                    "notable_outliers": [],
                    "groups": []
                }
                md_lines.append(f"## {gap}\n")
                md_lines.append(results[gap]["literature_paragraph"])
                md_lines.append("")
                continue

        # Parse final JSON robustly
        final_obj = _extract_json_best_effort(
            final_json_str,
            fallback={
                "gap": gap,
                "literature_paragraph": final_json_str.strip(),
                "themes": [],
                "similarities": [],
                "differences": [],
                "notable_outliers": [],
                "groups": []
            }
        )

        # Store
        results[gap] = final_obj

        # Append to Markdown report
        md_lines.append(f"## {gap}\n")
        para = (final_obj.get("literature_paragraph") or "").strip()
        md_lines.append(para if para else "_No paragraph returned._")
        md_lines.append("")

        def _md_list(title, items):
            if not items: return
            md_lines.append(f"**{title}**")
            for it in items:
                if isinstance(it, dict):
                    if "group_label" in it and "papers" in it:
                        md_lines.append(f"- {it['group_label']}: {', '.join(it.get('papers', []))}")
                    else:
                        md_lines.append(f"- {json.dumps(it, ensure_ascii=False)}")
                else:
                    md_lines.append(f"- {it}")
            md_lines.append("")

        _md_list("Themes", final_obj.get("themes", []))
        _md_list("Similarities", final_obj.get("similarities", []))
        _md_list("Differences", final_obj.get("differences", []))
        _md_list("Notable outliers", final_obj.get("notable_outliers", []))
        _md_list("Groups", final_obj.get("groups", []))

        # Autosave after each successful gap (so you can resume any time)
        try:
            json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
            print(f"[OK] Gap '{gap}' synthesized and saved.")
        except Exception as e:
            print(f"[WARN] Could not write main outputs after '{gap}': {e}")
            _save_checkpoint(json_path, md_path, results, md_lines, tag=f"write-{gap}")

    # Final save (redundant but explicit)
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[OK] Saved JSON → {json_path}")
    print(f"[OK] Saved Markdown → {md_path}")

    return results

def gap_df(read_papers, gaps):
    gap_answers = {}
    for gap in gaps:
        gap_answers[f"{gap}"] = ""
        for paper in read_papers:
            gap_answers[f"{gap}"] += f'In {paper.get("paper_file")}, {paper.get("extraction").get(gap).get("answer")} \n\n'
    return gap_answers
    


