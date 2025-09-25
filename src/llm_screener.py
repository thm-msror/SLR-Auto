#src.llm_screener.py
import json
import sys
from pathlib import Path
import time
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from src.utils import save_json, save_checkpoint, strip_json_comments

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
        all_papers_dir = save_json(papers, track_dir, ".all_papers_to_screen_") 

    # Load screening prompt
    with open(prompt_path, "r", encoding="utf-8") as f:
        base_prompt = f.read().strip()

    results = []

    for p, paper in enumerate(papers):
        print(f"> Screening paper {p+1}:", paper.get("title"))
        # Build final prompt
        full_prompt = f'''{base_prompt} Paper: {paper.get("title")}: {paper.get("abstract")}'''

        try:
            response = client.chat.completions.create(
                model="Fanar", 
                messages=[
                    {"role": "system", "content": "You are an assistant that outputs structured JSON only."},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.0, # More coherent
            )

            llm_output = strip_json_comments(response.choices[0].message.content.strip())

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

                fixed_output = strip_json_comments(fix_response.choices[0].message.content.strip())

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
                "relevance_score": relevance_scorer(parsed_output),
            })

        except Exception as e:
            results.append({
                "paper": paper,
                "error": str(e),
            })
        
        if track_dir and p%batch_size == 0:
            papers_left = save_checkpoint(papers[p:], track_dir, ".all_papers_to_screen_remaining")
            backup_path = save_checkpoint(results, track_dir, f".llm_backup_{p}")
            print(f"{p+1} of {len(papers)} done screening. Checkpoint saved in {backup_path}")

    return results

def relevance_scorer(paper_data):
    """
    Adds a 'relevance' score to the given JSON-like dictionary based on the
    sum of values in the 'inclusion_exclusion_criteria' section.
    """
    criteria = paper_data.get("inclusion_exclusion_criteria", {})
    relevance_score = sum(criteria.values())

    return relevance_score