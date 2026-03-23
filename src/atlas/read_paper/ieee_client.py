import os
from typing import Dict, List

from dotenv import load_dotenv

from atlas.inital_fetch.fetch_ieee import fetch_papers as _fetch_ieee_papers

load_dotenv(".env")


def get_ieee_api_key() -> str | None:
    return os.getenv("IEEE_API")


def fetch_ieee_papers(
    queries: List[str],
    max_results: int = 50,
    start_index: int = 1,
    delay: float = 1.0,
    track=False,
) -> List[Dict]:
    api_key = get_ieee_api_key()
    if not api_key:
        print("IEEE_API not configured. Skipping IEEE search.")
        return []

    return _fetch_ieee_papers(
        queries,
        api_key=api_key,
        max_results=max_results,
        start_index=start_index,
        delay=delay,
        track=track,
    )
