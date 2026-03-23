from pathlib import Path


def _read_prompt(filename: str) -> str:
    return (Path(__file__).resolve().parent / filename).read_text(encoding="utf-8").strip()


MAKE_CATEGORIES_PROMPT = _read_prompt("make_categories.txt")
SCREEN_FULL_PROMPT = _read_prompt("screen_full.txt")
