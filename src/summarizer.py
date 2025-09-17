from openai import OpenAI
import os, json

client = OpenAI(api_key=os.getenv("FANAR_API_KEY"), base_url="https://api.fanar.qa/v1")

def summarize_screened(json_path, prompt_path="data/summarization_prompt.txt"):
    import json

    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    # Filter for relevant papers (relevance 10, 9, 8) # Using only relevance 10 papers fix the LLM context size better
    relevant = [p for p in papers if p.get("llm_screening", {}).get("relevance") in [10, 9, 8]]

    if not relevant:
        return "No relevant papers (relevance 8, 9, or 10) found to summarize."

    # Load summarization prompt
    with open(prompt_path, "r", encoding="utf-8") as f:
        summarization_prompt = f.read()

    # Prepare text for LLM
    content = summarization_prompt + "\n\n"
    for p in relevant[:50]:  # limit context
        title = p.get("paper", {}).get("title", "N/A")
        published = p.get("paper", {}).get("published", "N/A")
        methods = ", ".join(p.get("llm_screening", {}).get("key_technologies", [])) or "?"
        datasets = ", ".join(p.get("llm_screening", {}).get("datasets", [])) or "?"
        task_type = p.get("llm_screening", {}).get("task_type", "N/A")
        content += f"- **Title:** {title}\n  - Published: {published}\n  - Methods: {methods}\n  - Datasets: {datasets}\n  - Task: {task_type}\n\n"

    # Call LLM
    response = client.chat.completions.create(
        model="Fanar",
        messages=[{"role": "user", "content": content}],
        temperature=0,   # deterministic
        top_p=1
    )

    return response.choices[0].message.content.strip()