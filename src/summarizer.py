from openai import OpenAI
import os, json

client = OpenAI(api_key=os.getenv("FANAR_API_KEY"), base_url="https://api.fanar.qa/v1")

def summarize_screened(json_path, prompt_path="data/summarization_prompt.txt"):
    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    # Filter for highly relevant
    relevant = [p for p in papers if p.get("relevance") == 10]

    # Load summarization prompt
    with open(prompt_path, "r", encoding="utf-8") as f:
        summarization_prompt = f.read()

    # Prepare text for LLM
    content = summarization_prompt + "\n\n"
    for p in relevant[:50]:  # don’t overload context
        content += f"- {p['title']} ({p.get('published','N/A')}) | Methods: {p.get('methods','?')} | Dataset: {p.get('dataset','?')}\n"

    # Call LLM
    response = client.chat.completions.create(
        model="Fanar",
        messages=[{"role": "user", "content": content}]
    )

    return response.choices[0].message.content.strip()
