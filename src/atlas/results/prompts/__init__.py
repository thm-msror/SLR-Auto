from pathlib import Path


def _read_prompt(filename: str) -> str:
    return (Path(__file__).resolve().parent / filename).read_text(encoding="utf-8").strip()


ATLAST_METHODOLOGY_CONTEXT = _read_prompt("atlast_methodology_context.txt")
ABSTRACT_PROMPT = _read_prompt("abstract.txt")
DISCUSSION_CONCLUSION_PROMPT = _read_prompt("discussion_conclusion.txt")
INTRODUCTION_PROMPT = _read_prompt("introduction.txt")
METHODOLOGY_PROMPT = _read_prompt("methodology.txt")
RESULTS_FINDINGS_PROMPT = _read_prompt("results_findings.txt")
SYNTHESIZE_CATEGORY_PROMPT = _read_prompt("synthesize_category.txt")
