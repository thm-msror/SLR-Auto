import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv(".env")

DEFAULT_GPT_ENDPOINT = "https://60099-m1xc2jq0-australiaeast.openai.azure.com/"
DEFAULT_GPT_DEPLOYMENT = "gpt-4o-kairos"
DEFAULT_GPT_VERSION = "2024-12-01-preview"

_client: Optional[AzureOpenAI] = None


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
