import io
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv(".env")

DEFAULT_GPT_ENDPOINT = "https://60099-m1xc2jq0-australiaeast.openai.azure.com/"
DEFAULT_GPT_DEPLOYMENT = "gpt-4o-kairos"
DEFAULT_GPT_VERSION = "2024-12-01-preview"
DEFAULT_GPT_RESPONSES_VERSION = "2025-03-01-preview"

_client: Optional[AzureOpenAI] = None
_responses_client: Optional[AzureOpenAI] = None


def get_gpt_deployment(model_name: Optional[str] = None) -> str:
    deployment = model_name or os.getenv("GPT_DEPLOYMENT", DEFAULT_GPT_DEPLOYMENT)
    if not deployment:
        raise RuntimeError("Missing GPT_DEPLOYMENT environment variable.")
    return deployment


def get_gpt_client() -> AzureOpenAI:
    global _client
    if _client is None:
        endpoint = os.getenv("GPT_ENDPOINT", DEFAULT_GPT_ENDPOINT).rstrip("/")
        api_key = os.getenv("GPT_KEY")
        api_version = os.getenv("GPT_VERSION", DEFAULT_GPT_VERSION)

        if not api_key:
            raise RuntimeError(
                "Missing GPT_KEY environment variable. Set GPT_KEY in .env."
            )

        _client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    return _client


def get_gpt_responses_client() -> AzureOpenAI:
    global _responses_client
    if _responses_client is None:
        endpoint = os.getenv("GPT_ENDPOINT", DEFAULT_GPT_ENDPOINT).rstrip("/")
        api_key = os.getenv("GPT_KEY")
        api_version = os.getenv("GPT_RESPONSES_VERSION", DEFAULT_GPT_RESPONSES_VERSION)

        if not api_key:
            raise RuntimeError(
                "Missing GPT_KEY environment variable. Set GPT_KEY in .env."
            )

        _responses_client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    return _responses_client


def call_gpt_chat(
    messages: List[Dict[str, str]],
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    top_p: Optional[float] = None,
) -> str:
    client = get_gpt_client()
    deployment = get_gpt_deployment(model_name)

    payload: Dict[str, Any] = {
        "model": deployment,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        payload["top_p"] = top_p

    response = client.chat.completions.create(**payload)
    content = response.choices[0].message.content if response.choices else None
    if not content:
        raise RuntimeError("Empty response from GPT.")
    return content.strip()


def _extract_response_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text:
        text = response.output_text.strip()
        if text:
            return text

    try:
        data = response.model_dump()
    except Exception:
        data = response

    texts: List[str] = []
    if isinstance(data, dict):
        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"} and content.get("text"):
                    texts.append(content["text"])

    combined = "\n".join(texts).strip()
    if not combined:
        raise RuntimeError("Empty response from GPT.")
    return combined


def call_gpt_pdf(
    prompt: str,
    pdf_bytes: bytes,
    filename: str,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_output_tokens: int = 800,
) -> str:
    client = get_gpt_responses_client()
    deployment = get_gpt_deployment(model_name)
    uploaded_file = None

    try:
        file_obj = io.BytesIO(pdf_bytes)
        file_obj.name = filename
        uploaded_file = client.files.create(file=file_obj, purpose="assistants")

        response = client.responses.create(
            model=deployment,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_file", "file_id": uploaded_file.id},
                    ],
                }
            ],
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        return _extract_response_text(response)
    finally:
        if uploaded_file is not None:
            try:
                client.files.delete(uploaded_file.id)
            except Exception:
                pass
