import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

try:
    from config import GAPS
except Exception:
    GAPS = [
        "video_segmentation",
        "frame_sampling_method",
        "input_video_length",
        "spatiotemporal_analysis",
        "visual_analysis",
        "speech_audio_analysis",
        "sound_audio_analysis",
        "qa_interaction",
        "retrieval_level",
        "inferring_method",
        "video_representation",
        "model_pipeline",
        "dataset",
        "comparisons",
        "hyperparameters",
        "environment",
        "repository",
        "authors",
    ]


PDF_PATH = Path(
    "data/3_top_papers/pdf_papers/Video-RAG Visually-aligned Retrieval-Augmented Long Video Comprehension.pdf"
)
RESPONSES_API_VERSION = os.getenv("GPT_RESPONSES_VERSION", "2025-03-01-preview")


def test_gpt4o_pdf_gap_connection() -> None:
    load_dotenv(".env")

    endpoint = os.getenv("GPT_ENDPOINT")
    deployment = os.getenv("GPT_DEPLOYMENT")
    subscription_key = os.getenv("GPT_KEY")

    if not all([endpoint, deployment, subscription_key]):
        raise RuntimeError(
            "Missing one or more required env vars: GPT_ENDPOINT, GPT_DEPLOYMENT, GPT_KEY"
        )
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF file not found: {PDF_PATH}")

    client = AzureOpenAI(
        api_version=RESPONSES_API_VERSION,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

    gaps_list = "\n".join(f"- {gap}" for gap in GAPS)
    prompt_text = f"""
You are reading an academic paper from the attached PDF file.
Determine whether the paper talks about each GAP below.

GAPs:
{gaps_list}

Instructions:
1) For each GAP, output exactly one line in this format:
   <gap_name>: YES or NO | <very short evidence phrase>
2) Use only evidence from the attached PDF.
3) If not mentioned clearly, mark NO.
4) After all GAPs, add one final line:
   OVERALL_GAPS_COVERED: YES or NO
5) Output plain text only. No headings, no explanation, no markdown, no code fences.
"""

    with PDF_PATH.open("rb") as file_obj:
        uploaded_file = client.files.create(file=file_obj, purpose="assistants")

    response = client.responses.create(
        model=deployment,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_text},
                    {"type": "input_file", "file_id": uploaded_file.id},
                ],
            }
        ],
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=1500,
    )

    print(f"PDF: {PDF_PATH}")
    print(f"Responses API version: {RESPONSES_API_VERSION}")
    print(f"Uploaded file id: {uploaded_file.id}")
    print("\nModel output:\n")
    print((response.output_text or "").strip())

    try:
        client.files.delete(uploaded_file.id)
        print("\nUploaded file deleted from Azure Files API.")
    except Exception as exc:
        print(f"\nWarning: could not delete uploaded file {uploaded_file.id}: {exc}")


if __name__ == "__main__":
    test_gpt4o_pdf_gap_connection()
