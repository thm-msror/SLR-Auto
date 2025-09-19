import json
import sys
from pathlib import Path
import time
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from src.utils import save_checkpoint

load_dotenv(".env")
client = OpenAI(
    base_url="https://api.fanar.qa/v1",
    api_key=os.getenv("FANAR_API_KEY"),
)


def screen_papers(papers, prompt_path: str, track = None, batch_size=50):
    """
    Runs LLM-assisted screening on abstracts using FANAR API.

    Args:
        input_json_path (str): Path to input JSON with abstracts.
        prompt_path (str): Path to a text file containing the base prompt.

    Returns: the same list of papers but with a new llm_screening notes   
    """
    # Handle track as a directory
    track_dir = None
    if track:
        track_dir = "track" if track is True else str(track)
        os.makedirs(track_dir, exist_ok=True)

    # Load screening prompt
    with open(prompt_path, "r", encoding="utf-8") as f:
        base_prompt = f.read().strip()

    results = []

    for p, paper in enumerate(papers):
        print(f"> Screening paper {p+1}:", paper.get("title"))
        # Build final prompt
        full_prompt = f"{base_prompt}\n\nPaper:\n{paper}"

        try:
            response = client.chat.completions.create(
                model="Fanar", 
                messages=[
                    {"role": "system", "content": "You are an assistant that outputs structured JSON only."},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.0, # More coherent
            )

            llm_output = response.choices[0].message.content.strip()

            try:
                parsed_output = json.loads(llm_output)
            except json.JSONDecodeError: # if the JSON can't be parsed
                fix_prompt = f"""
                The following text was supposed to be valid JSON but contains unnecessary text or errors.
                Clean it so it becomes valid JSON only. Do not add explanations, only output JSON.

                Raw text:
                {llm_output}
                """

                fix_response = client.chat.completions.create(
                    model="Fanar",
                    messages=[
                        {"role": "system", "content": "You are a JSON fixer. You ONLY output valid JSON."},
                        {"role": "user", "content": fix_prompt},
                    ],
                    temperature=0.0,
                )

                fixed_output = fix_response.choices[0].message.content.strip()

                try:
                    parsed_output = json.loads(fixed_output)
                except json.JSONDecodeError:
                    try:
                        parsed_output = json.loads(fixed_output.replace("\n","").replace("\\", ""))
                    except json.JSONDecodeError:
                        parsed_output = {"error": "Failed to clean JSON", "raw_output": llm_output}

            results.append({
                "paper": paper,
                "llm_screening": parsed_output,
            })

        except Exception as e:
            results.append({
                "paper": paper,
                "error": str(e),
            })
        
        if p%batch_size == batch_size-1:
            if track_dir: backup_path = save_checkpoint(results, track_dir, f".llm_backup_{p}_")
            print(f"{p+1} of {len(papers)} done screening. Checkpoint saved in {backup_path}")

    return results


# def sanitize_parsed_output(parsed_output):
#     """
#     Normalize the parsed LLM output into a stable screening dict:
#     - unwrap nested 'llm_screening' layers
#     - remove accidental 'paper' key
#     - ensure keys exist with sensible defaults
#     NOTE: This does NOT discard raw_output if present in parsed_output.
#     """
#     # Unwrap nested llm_screening entries (sometimes LLM wraps)
#     while isinstance(parsed_output, dict) and "llm_screening" in parsed_output:
#         parsed_output = parsed_output["llm_screening"]

#     # Preserve raw_output if present
#     raw_output = None
#     if isinstance(parsed_output, dict) and "raw_output" in parsed_output:
#         raw_output = parsed_output.get("raw_output")
#         # do not pop it — we will put it back into final dict

#     # Remove accidental 'paper' key if present
#     if isinstance(parsed_output, dict) and "paper" in parsed_output:
#         parsed_output = {k: v for k, v in parsed_output.items() if k != "paper"}

#     # Stable default screening structure
#     default = {
#         "review_paper": False,
#         "inclusion_criteria": {},
#         "exclusion_criteria": {},
#         "included": False,
#         "relevance": 0,
#         "reason_of_relevance": "",
#         "key_technologies": [],
#         "modalities": [],
#         "task_type": None,
#         "datasets": [],
#         "application": [],
#         "limitations": [],
#         "notes": ""
#     }

#     # If parsed_output not a dict, return defaults (but allow caller to attach raw_output)
#     if not isinstance(parsed_output, dict):
#         merged = default.copy()
#         if raw_output:
#             merged["raw_output"] = raw_output
#         return merged

#     # Merge parsed keys over defaults (keep only expected keys + inclusion/exclusion criteria)
#     merged = default.copy()
#     for k, v in parsed_output.items():
#         if k in merged:
#             merged[k] = v
#         elif k in ("inclusion_criteria", "exclusion_criteria") and isinstance(v, dict):
#             merged[k] = v
#         elif k == "raw_output":
#             # will attach below
#             raw_output = v
#         else:
#             # ignore unexpected keys (or add small notes)
#             pass

#     # Ensure numeric relevance is int and within 0-10
#     try:
#         merged["relevance"] = int(merged.get("relevance", 0) or 0)
#         if merged["relevance"] < 0:
#             merged["relevance"] = 0
#         if merged["relevance"] > 10:
#             merged["relevance"] = 10
#     except Exception:
#         merged["relevance"] = 0

#     # Ensure lists are lists
#     for list_key in ("key_technologies", "modalities", "datasets", "application", "limitations"):
#         if merged.get(list_key) is None:
#             merged[list_key] = []
#         elif not isinstance(merged[list_key], list):
#             merged[list_key] = [merged[list_key]]

#     # Attach raw_output if available (very helpful for debugging)
#     if raw_output:
#         merged["raw_output"] = raw_output

#     return merged
# def screen_papers_batch(papers_list, prompt_path):
#     """
#     Batch-friendly LLM screening with JSON fixer and consistent output format.
#     Returns a list of screening dicts (one per input paper). Does NOT return
#     wrappers containing the 'paper' key. If parsing fails, `raw_output` and
#     `fixed_output` (when available) are preserved inside the screening dict.
#     """
#     results = []

#     # Load prompt
#     with open(prompt_path, "r", encoding="utf-8") as f:
#         base_prompt = f.read().strip()

#     for paper in papers_list:
#         print("> Screening:", paper.get("title"))
#         paper_text = json.dumps(paper, ensure_ascii=False)
#         full_prompt = f"{base_prompt}\n\nPaper:\n{paper_text}"

#         try:
#             response = client.chat.completions.create(
#                 model="Fanar",
#                 messages=[
#                     {"role": "system", "content": "You are an assistant that outputs structured JSON only."},
#                     {"role": "user", "content": full_prompt},
#                 ],
#                 temperature=0.0,
#             )
#             llm_output = response.choices[0].message.content.strip()

#             parsed_output = None
#             fixed_output = None

#             # Try direct JSON parse first
#             try:
#                 parsed_output = json.loads(llm_output)
#             except json.JSONDecodeError:
#                 # Fix invalid JSON with LLM (your existing fix prompt)
#                 fix_prompt = f"""
#                 The following text was supposed to be valid JSON but contains unnecessary text or errors.
#                 Clean it so it becomes valid JSON only. Do not add explanations, only output JSON.

#                 Raw text:
#                 {llm_output}
#                 """
#                 fix_response = client.chat.completions.create(
#                     model="Fanar",
#                     messages=[
#                         {"role": "system", "content": "You are a JSON fixer. You ONLY output valid JSON."},
#                         {"role": "user", "content": fix_prompt},
#                     ],
#                     temperature=0.0,
#                 )
#                 fixed_output = fix_response.choices[0].message.content.strip()
#                 # Try parsing fixed output
#                 try:
#                     parsed_output = json.loads(fixed_output)
#                 except json.JSONDecodeError:
#                     # Heuristic fallback: extract first {...} block from the fixed_output or the original
#                     m = re.search(r"\{.*\}", fixed_output or "", flags=re.S)
#                     if not m:
#                         m = re.search(r"\{.*\}", llm_output, flags=re.S)
#                     if m:
#                         candidate = m.group(0)
#                         try:
#                             parsed_output = json.loads(candidate)
#                         except json.JSONDecodeError:
#                             parsed_output = {"error": "Failed to clean JSON", "raw_output": llm_output}
#                     else:
#                         parsed_output = {"error": "Failed to clean JSON", "raw_output": llm_output}

#             # If parsed_output exists, attach raw/fixed strings so sanitiser can keep them
#             if isinstance(parsed_output, dict):
#                 # if the fixer produced something but parsing still failed earlier, store fixed_output
#                 if fixed_output and "raw_output" not in parsed_output:
#                     parsed_output["fixed_output"] = fixed_output
#                 # also keep original raw LLM output for debugging if parsing was messy
#                 if "raw_output" not in parsed_output:
#                     parsed_output["raw_output"] = llm_output
#             else:
#                 # completely non-dict parsed_output -> preserve raw
#                 parsed_output = {"error": "Non-dict parsed output", "raw_output": llm_output}
#                 if fixed_output:
#                     parsed_output["fixed_output"] = fixed_output

#             # Sanitize/normalize the parsed output and preserve raw text
#             screening_dict = sanitize_parsed_output(parsed_output)

#             # Ensure raw_output and fixed_output are preserved in the sanitized dict (if present)
#             if isinstance(parsed_output, dict) and parsed_output.get("raw_output"):
#                 screening_dict["raw_output"] = parsed_output["raw_output"]
#             if isinstance(parsed_output, dict) and parsed_output.get("fixed_output"):
#                 screening_dict["fixed_output"] = parsed_output["fixed_output"]

#             results.append(screening_dict)

#         except Exception as e:
#             # If LLM call fails entirely, append a stable error screening dict but with the error string
#             error_dict = {
#                 "review_paper": False,
#                 "inclusion_criteria": {},
#                 "exclusion_criteria": {},
#                 "included": False,
#                 "relevance": 0,
#                 "reason_of_relevance": f"LLM call failed: {e}",
#                 "key_technologies": [],
#                 "modalities": [],
#                 "task_type": None,
#                 "datasets": [],
#                 "application": [],
#                 "limitations": [],
#                 "notes": ""
#             }
#             error_dict["raw_output"] = str(e)
#             results.append(error_dict)

#     return results
