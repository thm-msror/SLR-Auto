from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from atlas.results.gpt_abstract import generate_title_abstract_keywords
from atlas.results.gpt_discussion_conclusion import generate_discussion_conclusion
from atlas.results.gpt_introduction import build_introduction_from_questions
from atlas.results.gpt_methodology import build_methodology_section
from atlas.results.gpt_results import build_ieee_references_text, rewrite_results_findings
from atlas.results.prisma import build_prisma_svg


def generate_full_draft(run: Dict[str, Any], save_path: str | Path) -> Dict[str, Any]:
    inputs = run.setdefault("inputs", {})
    syntheses = run.setdefault("syntheses", {})

    research_questions = (inputs.get("research_questions") or "").strip()
    boolean_query_used = (inputs.get("boolean_query_used") or "").strip()
    criteria_used = inputs.get("criteria_used") or []
    queries = inputs.get("queries") or []
    prisma = run.get("prisma") or {}
    theme_drafts = run.get("categories") or {}

    if not research_questions:
        raise ValueError("Missing research questions for full draft generation.")
    if not boolean_query_used:
        raise ValueError("Missing confirmed Boolean query for full draft generation.")
    if not criteria_used:
        raise ValueError("Missing confirmed screening criteria for full draft generation.")
    if not theme_drafts:
        raise ValueError("Missing confirmed themes for full draft generation.")

    references = build_ieee_references_text(run)
    if not references.strip():
        raise ValueError("References could not be generated from top papers.")

    introduction = build_introduction_from_questions(research_questions)
    methodology = build_methodology_section(
        research_questions=research_questions,
        boolean_query_used=boolean_query_used,
        criteria_used=criteria_used,
        prisma=prisma,
        queries=queries,
    )
    results = rewrite_results_findings(
        theme_drafts=theme_drafts,
        references=references,
    )
    discussion_conclusion = generate_discussion_conclusion(
        introduction=introduction,
        results=results,
    )

    title_abstract = generate_title_abstract_keywords(
        introduction=introduction,
        methodology=methodology,
        results=results,
        discussion=discussion_conclusion["discussion"],
        conclusion=discussion_conclusion["conclusion"],
    )

    report_path, prisma_svg_path = _resolve_output_paths(save_path)
    prisma_svg = build_prisma_svg(prisma)
    prisma_svg_path.parent.mkdir(parents=True, exist_ok=True)
    prisma_svg_path.write_text(prisma_svg, encoding="utf-8")

    methodology_with_prisma = _inject_prisma_before_last_paragraph(methodology, prisma_svg)
    draft_report = _build_draft_report_markdown(
        title=title_abstract["title"],
        abstract=title_abstract["abstract"],
        keywords=title_abstract["keywords"],
        introduction=introduction,
        methodology=methodology_with_prisma,
        results=results,
        discussion=discussion_conclusion["discussion"],
        conclusion=discussion_conclusion["conclusion"],
        references=references,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(draft_report, encoding="utf-8")

    syntheses["title"] = title_abstract["title"]
    syntheses["abstract"] = title_abstract["abstract"]
    syntheses["keywords"] = list(title_abstract["keywords"])
    syntheses["introduction"] = introduction
    syntheses["methodology"] = methodology
    syntheses["references"] = references
    syntheses["results"] = results
    syntheses["discussion"] = discussion_conclusion["discussion"]
    syntheses["conclusion"] = discussion_conclusion["conclusion"]
    syntheses["draft_report"] = draft_report
    syntheses["draft_report_path"] = str(report_path)
    syntheses["prisma_svg"] = prisma_svg
    syntheses["prisma_svg_path"] = str(prisma_svg_path)

    return {
        "title": title_abstract["title"],
        "abstract": title_abstract["abstract"],
        "keywords": list(title_abstract["keywords"]),
        "introduction": introduction,
        "methodology": methodology,
        "references": references,
        "results": results,
        "discussion": discussion_conclusion["discussion"],
        "conclusion": discussion_conclusion["conclusion"],
        "prisma_svg": prisma_svg,
        "draft_report": draft_report,
        "draft_report_path": str(report_path),
        "prisma_svg_path": str(prisma_svg_path),
    }


def _resolve_output_paths(save_path: str | Path) -> tuple[Path, Path]:
    path = Path(save_path)
    if path.exists() and path.is_dir():
        report_path = path / "draft_report.md"
    elif path.suffix.lower() == ".md":
        report_path = path
    else:
        report_path = path / "draft_report.md"

    prisma_svg_path = report_path.with_name(f"{report_path.stem}_prisma.svg")
    return report_path, prisma_svg_path


def _inject_prisma_before_last_paragraph(methodology: str, prisma_svg: str) -> str:
    paragraphs = [part.strip() for part in methodology.split("\n\n") if part.strip()]
    svg_block = "\n".join(
        [
            '<div class="prisma-flow">',
            prisma_svg,
            "</div>",
        ]
    )

    if not paragraphs:
        return svg_block
    if len(paragraphs) == 1:
        return f"{svg_block}\n\n{paragraphs[0]}"

    head = "\n\n".join(paragraphs[:-1]).strip()
    tail = paragraphs[-1]
    return f"{head}\n\n{svg_block}\n\n{tail}"


def _build_draft_report_markdown(
    title: str,
    abstract: str,
    keywords: List[str],
    introduction: str,
    methodology: str,
    results: str,
    discussion: str,
    conclusion: str,
    references: str,
) -> str:
    keyword_line = "; ".join(keyword.strip() for keyword in keywords if keyword.strip())
    sections = [
        f"# {title.strip()}",
        "## Abstract\n\n" + abstract.strip(),
        f"**Keywords:** {keyword_line}" if keyword_line else "",
        "## Introduction\n\n" + introduction.strip(),
        "## Methodology\n\n" + methodology.strip(),
        "## Results and Findings\n\n" + results.strip(),
        "## Discussion\n\n" + discussion.strip(),
        "## Conclusion\n\n" + conclusion.strip(),
        "## References\n\n" + references.strip(),
    ]
    return "\n\n".join(section for section in sections if section.strip()).strip()
