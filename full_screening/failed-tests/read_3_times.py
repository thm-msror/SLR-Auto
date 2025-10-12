import os
import re
from openai import OpenAI
from dotenv import load_dotenv


# Load API key from .env
load_dotenv(".env")

client = OpenAI(
    base_url="https://api.fanar.qa/v1",
    api_key=os.getenv("FANAR_API_KEY"),
)

model_name = "Fanar"

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


def paper_reader(paper_md: str, prompts: object) -> str:
    """
    Extracts metadata, summaries, relevance from a paper by reading it three times

    Returns: 
    - a string with bullet points
    """
    outputs = []

    for _ , prompt_path in prompts.items():

        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read()

        messages = [
            {
                "role": "system",
                "content": '''
                    From now on, you will play the role of “Expert SLR Screener Assistant – Fanar Precision Model.”
                    You are a ChatGPT-based AI model specialized in systematic literature review data extraction.
                    You are able to analyze long academic papers (converted from PDF to text) and extract data only from explicitly mentioned information — never infer, never assume, never hallucinate.
                    To achieve this, you will read through the paper content and locate textual evidence before producing outputs.
                    If a human expert in SLR data extraction has a knowledge level of 10, you have level 300.
                    Be careful: If you hallucinate or output in a wrong format, it may break the entire data pipeline.
                    So give your best — your accuracy ensures the pipeline works and prevents project failure.
                '''
                
            },
            {
                "role": "user",
                "content": f"{prompt}\n\n---\nPaper:\n{paper_md}\n---"
            }
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_tokens=1500,
        )

        outputs.append(response.choices[0].message.content.strip())

    final_output = "\n".join(outputs)
    return final_output


def batch_paper_reader(input_folder: str, output_folder: str, prompts: object):
    """
    Reads all Markdown files from input_folder, processes them with paper_reader(),
    and writes the results to output_folder as .txt files.

    Parameters:
        input_folder (str): Path to the folder containing markdown files.
        output_folder (str): Path to the folder where text outputs will be saved.
        prompts (object): The prompts object to be passed into paper_reader().

    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through each file in the input directory
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".md"):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + ".txt"
            output_path = os.path.join(output_folder, output_filename)
            
            # Read the Markdown file
            with open(input_path, "r", encoding="utf-8") as f:
                paper_md = f.read()

            # Apply the paper_reader function
            try:
                result = paper_reader(clean_paper_md(paper_md), prompts)
            except Exception as e:
                print(f"ERROR processing {filename}")
                continue

            # Save the output
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)

            print(f"Read {output_filename}")

    print("✅ All files processed successfully.")

prompts = {
    "metadata": r"full_screening\prompts\metadata.txt",
    "summaries": r"full_screening\prompts\summarizer.txt",
    "relevance": r"data\screening_prompt.txt",
}
batch_paper_reader(r"data\markdown_papers", r"full_screening\full_reads", prompts)