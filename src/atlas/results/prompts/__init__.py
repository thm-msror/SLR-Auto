from pathlib import Path


def _read_prompt(filename: str) -> str:
    return (Path(__file__).resolve().parent / filename).read_text(encoding="utf-8").strip()


ATLAST_METHODOLOGY_CONTEXT = _read_prompt("atlast_methodology_context.txt")
SYNTHESIZE_CATEGORY_PROMPT = _read_prompt("synthesize_category.txt")
