from pathlib import Path


def _read_prompt(filename: str) -> str:
    return (Path(__file__).resolve().parent / filename).read_text(encoding="utf-8").strip()


RQ_CRITERIA_PROMPT = _read_prompt("rq_criteria.txt")
SCREEN_INITIAL_PROMPT = _read_prompt("screen_initial.txt")
