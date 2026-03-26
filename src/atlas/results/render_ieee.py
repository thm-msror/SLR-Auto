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
    paper_blocks = _render_paper_blocks_html(
        title=title,
        abstract=abstract,
        keyword_line=keyword_line,
        introduction=introduction,
        methodology=methodology,
        results=results,
        discussion=discussion,
        conclusion=conclusion,
        references=references,
        prisma_svg=prisma_svg,
    )

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
    }}

    * {{
      box-sizing: border-box;
    }}

    html, body {{
      margin: 0;
      padding: 0;
      background: #F0F2F6;
      color: #000;
      font-family: "Times New Roman", Times, serif;
    }}

    body {{
      padding: 24px;
    }}

    .paper-source {{
      display: none;
    }}

    .ieee-pages {{
      display: grid;
      gap: 22px;
      justify-items: center;
    }}

    .ieee-page {{
      width: min(var(--paper-width), 100%);
      min-height: var(--paper-min-height);
      background: #fff;
      box-shadow: var(--shadow);
      border: 1px solid var(--border);
      padding:
        var(--page-padding-top)
        var(--page-padding-x)
        var(--page-padding-bottom);
    }}

    .ieee-page-flow {{
      height: calc(var(--paper-min-height) - var(--page-padding-top) - var(--page-padding-bottom) - 2px);
      font-size: 10pt;
      line-height: 1.3;
      text-align: justify;
    }}

    .paper-block {{
      break-inside: avoid-column;
    }}

    .paper-frontmatter,
    .paper-abstract,
    .paper-heading {{
      display: block;
    }}

    .paper-frontmatter {{
      margin: 0 0 18px;
      text-align: center;
    }}

    .paper-kicker {{
      margin: 0 0 14px;
      font: 600 10px/1.2 Arial, sans-serif;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #475569;
    }}

    .paper-title {{
      margin: 32px auto 12px;
      max-width: 6.8in;
      font-size: 20pt;
      line-height: 1.18;
      font-weight: 700;
    }}

    .paper-author {{
      margin: 0;
      font-size: 10pt;
      font-weight: 700;
    }}

    .paper-abstract {{
      margin: 0 0 14px;
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

    .paper-heading {{
      margin: 16px 0 6px;
      text-align: left;
      font-size: 9pt;
      font-weight: 700;
      letter-spacing: 0.02em;
    }}

    .paper-paragraph,
    .paper-reference {{
      margin: 0 0 0.7em;
      orphans: 3;
      widows: 3;
    }}

    .prisma-figure {{
      margin: 0.9em 0 1em;
      break-inside: avoid-column;
      text-align: center;
    }}

    .prisma-figure svg {{
      width: min(100%, 0.95in);
      max-width: 100%;
      height: auto;
      display: block;
      margin: 0 auto;
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

      .ieee-pages {{
        gap: 0;
      }}

      .ieee-page {{
        width: auto;
        border: 0;
        box-shadow: none;
        margin: 0 auto;
        break-after: page;
      }}

      .ieee-page:last-child {{
        break-after: auto;
      }}
    }}

    @media (max-width: 900px) {{
      body {{
        padding: 12px;
      }}

      .ieee-page {{
        padding: 18px 16px 22px;
      }}

      .paper-title {{
        font-size: 16pt;
      }}

      .ieee-page-flow {{
        height: auto;
      }}
    }}
  </style>
</head>
<body>
  <div id="paper-source" class="paper-source">
    {paper_blocks}
  </div>
  <div id="paper-pages" class="ieee-pages"></div>
  <script>
    (() => {{
      const source = document.getElementById("paper-source");
      const host = document.getElementById("paper-pages");
      if (!source || !host) {{
        return;
      }}

      const blocks = Array.from(source.children);
      if (!blocks.length) {{
        return;
      }}

      function createPage() {{
        const page = document.createElement("article");
        page.className = "ieee-page";
        const flow = document.createElement("div");
        flow.className = "ieee-page-flow";
        page.appendChild(flow);
        host.appendChild(page);
        return flow;
      }}

      function paginate() {{
        host.innerHTML = "";
        let flow = createPage();

        for (const original of blocks) {{
          const block = original.cloneNode(true);
          flow.appendChild(block);
          if (flow.scrollHeight > flow.clientHeight + 1) {{
            flow.removeChild(block);
            flow = createPage();
            flow.appendChild(block);
            if (flow.scrollHeight > flow.clientHeight + 1) {{
              flow.style.height = "auto";
              flow.parentElement.style.minHeight = "auto";
            }}
          }}
        }}
      }}

      paginate();
      let resizeTimer = null;
      window.addEventListener("resize", () => {{
        window.clearTimeout(resizeTimer);
        resizeTimer = window.setTimeout(paginate, 120);
      }});
    }})();
  </script>
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


def _render_paper_blocks_html(
    *,
    title: str,
    abstract: str,
    keyword_line: str,
    introduction: str,
    methodology: str,
    results: str,
    discussion: str,
    conclusion: str,
    references: str,
    prisma_svg: str,
) -> str:
    blocks = [
        _render_frontmatter_block_html(title),
        _render_abstract_block_html(abstract, keyword_line),
        _render_section_blocks_html("I. INTRODUCTION", introduction),
        _render_methodology_blocks_html(methodology, prisma_svg),
        _render_section_blocks_html("III. RESULTS AND FINDINGS", results),
        _render_section_blocks_html("IV. DISCUSSION", discussion),
        _render_section_blocks_html("V. CONCLUSION", conclusion),
        _render_reference_blocks_html(references),
    ]
    return "\n".join(block for block in blocks if block.strip())


def _render_frontmatter_block_html(title: str) -> str:
    return (
        '<header class="paper-block paper-frontmatter">\n'
        f'  <h1 class="paper-title">{escape(title.strip())}</h1>\n'
        '  <p class="paper-author">ATLAS Generated Draft</p>\n'
        "</header>"
    )


def _render_abstract_block_html(abstract: str, keyword_line: str) -> str:
    return (
        '<section class="paper-block paper-abstract">\n'
        '  <p class="abstract-title">Abstract</p>\n'
        f'  <p class="abstract-text">{_escape_inline_text(abstract)}</p>\n'
        f'  <p class="keywords"><strong>Index Terms</strong>{_format_keywords_html(keyword_line)}</p>\n'
        "</section>"
    )


def _render_section_blocks_html(title: str, body: str) -> str:
    blocks = [f'<h2 class="paper-block paper-heading">{escape(title)}</h2>']
    blocks.extend(
        f'<p class="paper-block paper-paragraph">{_escape_inline_text(paragraph)}</p>'
        for paragraph in _split_paragraphs(body)
    )
    return "\n".join(blocks)


def _render_methodology_blocks_html(methodology: str, prisma_svg: str) -> str:
    paragraphs = _split_paragraphs(methodology)
    blocks = ['<h2 class="paper-block paper-heading">II. METHODOLOGY</h2>']
    if not paragraphs:
        if prisma_svg.strip():
            blocks.append(_render_prisma_figure_block_html(prisma_svg))
        return "\n".join(blocks)

    if prisma_svg.strip() and len(paragraphs) > 1:
        body_parts = paragraphs[:-1]
        tail = paragraphs[-1]
    else:
        body_parts = paragraphs
        tail = ""

    blocks.extend(
        f'<p class="paper-block paper-paragraph">{_escape_inline_text(paragraph)}</p>'
        for paragraph in body_parts
    )

    if prisma_svg.strip():
        blocks.append(_render_prisma_figure_block_html(prisma_svg))

    if tail:
        blocks.append(f'<p class="paper-block paper-paragraph">{_escape_inline_text(tail)}</p>')

    return "\n".join(blocks)


def _render_prisma_figure_block_html(prisma_svg: str) -> str:
    resized_svg = _make_svg_resizable(prisma_svg)
    return (
        '<figure class="paper-block prisma-figure">\n'
        f"{resized_svg}\n"
        "<figcaption>Fig. 1. PRISMA 2020 flow diagram of the study selection process.</figcaption>\n"
        "</figure>"
    )


def _render_reference_blocks_html(references: str) -> str:
    lines = _split_reference_lines(references)
    blocks = ['<h2 class="paper-block paper-heading">REFERENCES</h2>']
    blocks.extend(
        f'<p class="paper-block paper-reference">{_escape_inline_text(line)}</p>'
        for line in lines
    )
    return "\n".join(blocks)


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


def _make_svg_resizable(svg_text: str) -> str:
    svg = svg_text.strip()
    if not svg.startswith("<svg"):
        return svg

    width_match = re.search(r'\bwidth="([\d.]+)"', svg)
    height_match = re.search(r'\bheight="([\d.]+)"', svg)
    has_viewbox = re.search(r"\bviewBox=", svg, flags=re.IGNORECASE)

    if width_match and height_match and not has_viewbox:
        viewbox = f' viewBox="0 0 {width_match.group(1)} {height_match.group(1)}"'
        svg = re.sub(r"<svg\b", f"<svg{viewbox}", svg, count=1)

    svg = re.sub(r'\swidth="[^"]*"', "", svg, count=1)
    svg = re.sub(r'\sheight="[^"]*"', "", svg, count=1)

    if re.search(r'\bstyle="[^"]*"', svg):
        svg = re.sub(
            r'style="([^"]*)"',
            lambda match: f'style="{match.group(1).rstrip(";")};width:100%;height:auto;display:block;"',
            svg,
            count=1,
        )
    else:
        svg = re.sub(r"<svg\b", '<svg style="width:100%;height:auto;display:block;"', svg, count=1)

    return svg
