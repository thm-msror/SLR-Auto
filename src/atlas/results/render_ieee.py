from __future__ import annotations

import re
from html import escape
from pathlib import Path
from typing import Iterable, Sequence


def render_ieee_html_document(
    *,
    title: str,
    abstract: str,
    keywords: Sequence[str],
    introduction: str,
    methodology: str,
    results: str,
    discussion: str,
    conclusion: str,
    references: str,
    prisma_svg: str = "",
) -> str:
    keyword_line = ", ".join(keyword.strip() for keyword in keywords if keyword.strip())
    methodology_html = _render_methodology_html(methodology, prisma_svg)
    references_html = _render_reference_lines_html(references)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title.strip() or "IEEE Draft")}</title>
  <style>
    :root {{
      --paper-width: 8.27in;
      --paper-min-height: 11.69in;
      --page-padding-x: 0.72in;
      --page-padding-top: 0.7in;
      --page-padding-bottom: 0.78in;
      --column-gap: 0.28in;
      --border: #d4d4d4;
      --shadow: 0 14px 36px rgba(0, 0, 0, 0.12);
      --toolbar-bg: #1f2937;
      --toolbar-text: #f9fafb;
      --accent: #0f172a;
    }}

    * {{
      box-sizing: border-box;
    }}

    html, body {{
      margin: 0;
      padding: 0;
      background: #ece8e1;
      color: #000;
      font-family: "Times New Roman", Times, serif;
    }}

    body {{
      padding: 24px;
    }}

    .ieee-toolbar {{
      width: min(var(--paper-width), 100%);
      margin: 0 auto 14px;
      display: flex;
      justify-content: flex-end;
      gap: 10px;
    }}

    .ieee-toolbar button {{
      border: 0;
      background: var(--toolbar-bg);
      color: var(--toolbar-text);
      padding: 10px 14px;
      font: 600 13px/1.1 Arial, sans-serif;
      cursor: pointer;
      border-radius: 999px;
    }}

    .ieee-paper {{
      width: min(var(--paper-width), 100%);
      min-height: var(--paper-min-height);
      margin: 0 auto;
      background: #fff;
      box-shadow: var(--shadow);
      border: 1px solid var(--border);
      padding:
        var(--page-padding-top)
        var(--page-padding-x)
        var(--page-padding-bottom);
    }}

    .paper-kicker {{
      margin: 0 0 14px;
      text-align: center;
      font: 600 10px/1.2 Arial, sans-serif;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #475569;
    }}

    .paper-title {{
      margin: 0 auto 12px;
      max-width: 6.8in;
      text-align: center;
      font-size: 20pt;
      line-height: 1.18;
      font-weight: 700;
    }}

    .paper-author {{
      margin: 0 0 20px;
      text-align: center;
      font-size: 10pt;
      font-weight: 700;
    }}

    .abstract-title {{
      margin: 0 0 4px;
      font-size: 9pt;
      font-weight: 700;
      font-style: italic;
    }}

    .abstract-text,
    .keywords {{
      margin: 0 0 10px;
      font-size: 9pt;
      line-height: 1.35;
      text-align: justify;
    }}

    .keywords strong {{
      font-style: italic;
    }}

    .paper-columns {{
      column-count: 2;
      column-gap: var(--column-gap);
      column-fill: balance;
      font-size: 10pt;
      line-height: 1.3;
      text-align: justify;
    }}

    .paper-section {{
      break-inside: avoid;
      margin: 0 0 10px;
    }}

    .paper-section h2 {{
      margin: 0 0 6px;
      text-align: center;
      font-size: 10pt;
      font-weight: 700;
      letter-spacing: 0.02em;
    }}

    .paper-section p,
    .paper-section .ieee-reference {{
      margin: 0 0 0.7em;
      orphans: 3;
      widows: 3;
    }}

    .prisma-figure {{
      margin: 0.9em 0 1em;
      break-inside: avoid;
      text-align: center;
    }}

    .prisma-figure svg {{
      width: 100%;
      height: auto;
    }}

    .prisma-figure figcaption {{
      margin-top: 0.45em;
      font-size: 8.5pt;
      line-height: 1.25;
    }}

    @page {{
      size: A4;
      margin: 0.55in 0.6in;
    }}

    @media print {{
      body {{
        padding: 0;
        background: #fff;
      }}

      .ieee-toolbar {{
        display: none;
      }}

      .ieee-paper {{
        width: auto;
        min-height: 0;
        border: 0;
        box-shadow: none;
        margin: 0;
        padding: 0;
      }}
    }}

    @media (max-width: 900px) {{
      body {{
        padding: 12px;
      }}

      .ieee-paper {{
        padding: 18px 16px 22px;
      }}

      .paper-title {{
        font-size: 16pt;
      }}

      .paper-columns {{
        column-count: 1;
      }}
    }}
  </style>
</head>
<body>
  <div class="ieee-toolbar">
    <button type="button" onclick="window.print()">Print / Save as PDF</button>
  </div>
  <article class="ieee-paper">
    <p class="paper-kicker">ATLAS Generated Draft</p>
    <h1 class="paper-title">{escape(title.strip())}</h1>
    <p class="paper-author">ATLAS Generated Draft</p>
    <section class="paper-abstract">
      <p class="abstract-title">Abstract</p>
      <p class="abstract-text">{_escape_inline_text(abstract)}</p>
      <p class="keywords"><strong>Index Terms</strong>{_format_keywords_html(keyword_line)}</p>
    </section>
    <div class="paper-columns">
      {_render_section_html("I. INTRODUCTION", introduction)}
      <section class="paper-section">
        <h2>II. METHODOLOGY</h2>
        {methodology_html}
      </section>
      {_render_section_html("III. RESULTS AND FINDINGS", results)}
      {_render_section_html("IV. DISCUSSION", discussion)}
      {_render_section_html("V. CONCLUSION", conclusion)}
      <section class="paper-section">
        <h2>REFERENCES</h2>
        {references_html}
      </section>
    </div>
  </article>
</body>
</html>
"""


def render_ieee_tex_document(
    *,
    title: str,
    abstract: str,
    keywords: Sequence[str],
    introduction: str,
    methodology: str,
    results: str,
    discussion: str,
    conclusion: str,
    references: str,
) -> str:
    keyword_line = ", ".join(keyword.strip() for keyword in keywords if keyword.strip())
    methodology_tex = _render_methodology_tex(methodology)
    bibliography = _render_thebibliography_tex(references)

    keyword_block = ""
    if keyword_line:
        keyword_block = (
            "\\begin{IEEEkeywords}\n"
            f"{_escape_latex_text(keyword_line)}\n"
            "\\end{IEEEkeywords}\n\n"
        )

    return (
        "\\documentclass[conference]{IEEEtran}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage{graphicx}\n\n"
        f"\\title{{{_escape_latex_text(title)}}}\n"
        "\\author{\\IEEEauthorblockN{ATLAS Generated Draft}}\n\n"
        "\\begin{document}\n"
        "\\maketitle\n\n"
        "\\begin{abstract}\n"
        f"{_escape_latex_text(abstract)}\n"
        "\\end{abstract}\n\n"
        f"{keyword_block}"
        "\\section{Introduction}\n"
        f"{_render_paragraphs_tex(introduction)}\n\n"
        "\\section{Methodology}\n"
        f"{methodology_tex}\n\n"
        "\\section{Results and Findings}\n"
        f"{_render_paragraphs_tex(results)}\n\n"
        "\\section{Discussion}\n"
        f"{_render_paragraphs_tex(discussion)}\n\n"
        "\\section{Conclusion}\n"
        f"{_render_paragraphs_tex(conclusion)}\n\n"
        "\\begin{thebibliography}{99}\n"
        f"{bibliography}\n"
        "\\end{thebibliography}\n\n"
        "\\end{document}\n"
    )


def ieee_output_paths(report_path: str | Path) -> tuple[Path, Path]:
    path = Path(report_path)
    stem = path.stem or "SLR_draft"
    return path.with_name(f"{stem}_ieee.html"), path.with_name(f"{stem}_ieee.tex")


def _render_section_html(title: str, body: str) -> str:
    return (
        '<section class="paper-section">\n'
        f"  <h2>{escape(title)}</h2>\n"
        f"{_render_paragraphs_html(body)}\n"
        "</section>"
    )


def _render_methodology_html(methodology: str, prisma_svg: str) -> str:
    paragraphs = _split_paragraphs(methodology)
    if not prisma_svg.strip():
        return _render_paragraphs_html(methodology)
    if len(paragraphs) <= 1:
        return f"{_render_prisma_figure_html(prisma_svg)}\n{_render_paragraphs_html(methodology)}"

    head = _render_paragraphs_html_from_parts(paragraphs[:-1])
    tail = _render_paragraphs_html_from_parts([paragraphs[-1]])
    return f"{head}\n{_render_prisma_figure_html(prisma_svg)}\n{tail}"


def _render_prisma_figure_html(prisma_svg: str) -> str:
    return (
        '<figure class="prisma-figure">\n'
        f"{prisma_svg.strip()}\n"
        "<figcaption>Fig. 1. PRISMA 2020 flow diagram of the study selection process.</figcaption>\n"
        "</figure>"
    )


def _render_reference_lines_html(references: str) -> str:
    lines = _split_reference_lines(references)
    if not lines:
        return ""
    return "\n".join(f'<p class="ieee-reference">{_escape_inline_text(line)}</p>' for line in lines)


def _render_methodology_tex(methodology: str) -> str:
    paragraphs = _split_paragraphs(methodology)
    figure_placeholder = (
        "\\begin{figure}[!t]\n"
        "\\centering\n"
        "\\fbox{\\parbox{0.95\\columnwidth}{\\centering "
        "PRISMA figure omitted in this export.\\\\"
        "Convert prisma\\_2020.svg to PDF or PNG and insert it here if compiling.}}\n"
        "\\caption{PRISMA 2020 flow diagram of the study selection process.}\n"
        "\\label{fig:prisma}\n"
        "\\end{figure}"
    )

    if not paragraphs:
        return figure_placeholder
    if len(paragraphs) == 1:
        return f"{figure_placeholder}\n\n{_render_paragraphs_tex_from_parts(paragraphs)}"

    head = _render_paragraphs_tex_from_parts(paragraphs[:-1])
    tail = _render_paragraphs_tex_from_parts([paragraphs[-1]])
    return f"{head}\n\n{figure_placeholder}\n\n{tail}"


def _render_thebibliography_tex(references: str) -> str:
    lines = _split_reference_lines(references)
    if not lines:
        return "\\bibitem{ref0} References unavailable."

    entries = []
    for idx, line in enumerate(lines, start=1):
        cleaned = re.sub(r"^\[\d+\]\s*", "", line).strip()
        entries.append(f"\\bibitem{{ref{idx}}}\n{_escape_latex_text(cleaned)}")
    return "\n\n".join(entries)


def _render_paragraphs_html(body: str) -> str:
    return _render_paragraphs_html_from_parts(_split_paragraphs(body))


def _render_paragraphs_html_from_parts(paragraphs: Iterable[str]) -> str:
    items = [f"<p>{_escape_inline_text(paragraph)}</p>" for paragraph in paragraphs if paragraph.strip()]
    return "\n".join(items)


def _render_paragraphs_tex(body: str) -> str:
    return _render_paragraphs_tex_from_parts(_split_paragraphs(body))


def _render_paragraphs_tex_from_parts(paragraphs: Iterable[str]) -> str:
    return "\n\n".join(_escape_latex_text(paragraph) for paragraph in paragraphs if paragraph.strip())


def _split_paragraphs(body: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", body.strip()) if part.strip()]


def _split_reference_lines(references: str) -> list[str]:
    return [line.strip() for line in references.splitlines() if line.strip()]


def _format_keywords_html(keyword_line: str) -> str:
    if not keyword_line:
        return ""
    return f": {_escape_inline_text(keyword_line)}"


def _escape_inline_text(text: str) -> str:
    return escape(" ".join(text.strip().split()))


def _escape_latex_text(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = text.strip()
    for original, replacement in replacements.items():
        escaped = escaped.replace(original, replacement)
    return escaped
