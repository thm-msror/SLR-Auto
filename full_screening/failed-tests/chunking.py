import os
import re
from dotenv import load_dotenv
from openai import OpenAI
import textwrap

def clean_paper_md(md):

    # Remove References 
    md = re.sub(r'(?i)^#\s*References\b.*?(?=^#|\Z)', '', md, flags=re.DOTALL | re.MULTILINE).strip()

    # Remove Markdown links like [something](#page-4-0)
    md = re.sub(r'\[.*?\]\(.*?page.*?\)', '', md).strip()
    
    # Remove HTML tags like <span> or </span>
    md = re.sub(r'<.*?>', '', md).strip()

    # Remove heading tag # 
    md = md.replace("#", "").strip()

    return md

# =============== CONFIG ===============
load_dotenv(".env")

client = OpenAI(
    base_url="https://api.fanar.qa/v1",
    api_key=os.getenv("FANAR_API_KEY"),
)

PAPER_TITLE = "Audio Does Matter Importance-Aware Multi-Granularity Fusion for Video Moment Retrieval"

PROMPT_PATH = r"full_screening\failed-tests\full_prompt.txt"
PAPER_PATH = fr"data\markdown_papers\{PAPER_TITLE}.md"
OUTPUT_PATH = fr"full_screening\full_reads\{PAPER_TITLE}.txt"

# =============== CHUNKING ===============
def chunk_markdown(md_text, max_words=1500):
    """Split markdown into roughly token-safe chunks (by word count)."""
    paragraphs = md_text.split("\n\n")
    chunks, current_chunk, current_len = [], [], 0

    for para in paragraphs:
        words = len(para.split())
        if current_len + words > max_words:
            chunks.append("\n\n".join(current_chunk))
            current_chunk, current_len = [], 0
        current_chunk.append(para)
        current_len += words

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


# =============== FANAR CALL ===============
def call_fanar(prompt_text, chunk, client):
    """Send a single chunk through FANAR API using a strict extractive prompt."""
    full_prompt = f"""
{prompt_text}

Text:
{chunk}

Output only as bullet points directly based on the text.
Do NOT infer or add new information.
If the text has no extractable facts, return "No relevant content found."
"""
    try:
        response = client.chat.completions.create(
            model="Fanar",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Error processing chunk: {e}")
        return ""


# =============== POSTPROCESSING ===============
def enforce_bullet_format(text):
    """Ensure every line starts with a bullet."""
    lines = text.splitlines()
    bullets = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not line.startswith("-") and not line.startswith("*"):
            line = f"- {line}"
        bullets.append(line)
    return "\n".join(bullets)


# =============== MAIN PIPELINE ===============
def process_paper(prompt_path, paper_path, client, output_path):
    # Load the extraction prompt
    with open(prompt_path, "r", encoding="utf-8") as f:
        base_prompt = f.read().strip()

    # Load the Markdown paper
    with open(paper_path, "r", encoding="utf-8") as f:
        paper_md = f.read()

    paper_md = clean_paper_md(paper_md)

    # Split into manageable chunks
    chunks = chunk_markdown(paper_md)
    print(f"✅ Split paper into {len(chunks)} chunks")

    bullet_outputs = []

    # Process each chunk
    for i, chunk in enumerate(chunks):
        print(f"🔹 Processing chunk {i+1}/{len(chunks)}...")
        result = call_fanar(base_prompt, chunk, client)
        print(result)
        bullet_outputs.append(enforce_bullet_format(result))

    # Merge all results
    merged_output = "\n".join(bullet_outputs)

    # Optional: second-pass summary
    print("🔁 Generating final consolidated bullet summary...")
    final_summary = call_fanar(
        "Combine the following extracted bullet points into a concise, unified list without adding new information:",
        merged_output,
        client,
    )
    final_summary = enforce_bullet_format(final_summary)

    # Save output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_summary)

    print(f"✅ Saved final bullet summary to: {output_path}")
    return final_summary


# =============== RUN ===============
if __name__ == "__main__":
    process_paper(PROMPT_PATH, PAPER_PATH, client, OUTPUT_PATH)
