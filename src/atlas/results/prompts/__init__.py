from pathlib import Path


def _read_prompt(filename: str) -> str:
    return (Path(__file__).resolve().parent / filename).read_text(encoding="utf-8").strip()


SYNTHESIZE_CATEGORY_PROMPT = _read_prompt("synthesize_category.txt")
