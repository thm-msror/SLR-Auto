import json
import sys
from pathlib import Path

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env")
client = OpenAI(
    base_url="https://api.fanar.qa/v1",
    api_key=os.getenv("FANAR_API_KEY"),
)


def screen_papers(input_json_path: str, prompt_path: str):
    """
    Runs LLM-assisted screening on abstracts using FANAR API.

    Args:
        input_json_path (str): Path to input JSON with abstracts.
        prompt_path (str): Path to a text file containing the base prompt.

    Returns: the same list of papers but with a new llm_screening notes
        {
        "paper": {same as the input},
        "llm_screening": {
            "relevance": 10,
            "reason_of_relevance": "1–2 sentence summary of why this is or isn’t relevant",
            "key_technologies": [list of methods, models, or approaches mentioned],
            "modalities": [video, audio, text, multimodal],
            "task_type": "video QA / clip retrieval / event localization / other",
            "datasets": [if any],
            "application": [list of applications with results],
            "limitations": [list of limitations if any],
            "notes":"the highlight of this paper in 1-2 sentences"
            }
        }

        
    """
    # Load abstracts
    with open(input_json_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    # Load screening prompt
    with open(prompt_path, "r", encoding="utf-8") as f:
        base_prompt = f.read().strip()

    results = []

    for paper in papers:
        print("> Screening:", paper.get("title"))
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

    return results



