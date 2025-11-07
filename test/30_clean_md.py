import re 
from pathlib import Path
import csv
import math

def clean_markdown_pdf(md: str) -> str:
    """
    Cleans Markdown strings extracted from PDFs:
      - Strip empty spans, <sup>...</sup>, ALL HTML tags
      - Remove images
      - Remove links (inline, reference, autolinks, bare URLs)
      - Remove empty anchor links like [](#page-15-4)
      - Remove bare anchor parens like (#page-15-4)
      - Drop References/Bibliography/Works Cited section → EOF
      - Remove LaTeX wrappers and math blocks
      - Normalize whitespace/separators
    """
    text = md

    # 1) Empty <span ...></span>
    text = re.sub(r"<span\b[^>]*>\s*</span>", "", text, flags=re.IGNORECASE)

    # 2) <sup>...</sup>
    # text = re.sub(r"<sup\b[^>]*>.*?</sup>", "", text, flags=re.IGNORECASE | re.DOTALL)

    # 3) ALL HTML tags (preserve inner)
    text = re.sub(r"</?[^>\s]+(?:\s[^>]*?)?>", "", text, flags=re.IGNORECASE)

    # 4) Images ![...](...ext)
    img_exts = r"(?:jpe?g|png|gif|svg|webp|bmp|tiff?)"
    text = re.sub(
        rf"!\[[^\]]*\]\([^)]+?\.(?:{img_exts})(?:\?[^\)]*)?\)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # 5) Links
    # 5a) Inline: [text](url) -> keep 'text'
    text = re.sub(r"\[([^\]]+)\]\((?!\))[^)]+\)", r"\1", text)
    # 5b) Reference-style: [text][id] -> 'text'
    text = re.sub(r"\[([^\]]+)\]\[[^\]]+\]", r"\1", text)
    # 5c) Link defs: [id]: url  (remove whole line)
    # text = re.sub(r"^[ \t]*\[[^\]]+\]:\s+\S+\s*$", "", text, flags=re.MULTILINE)
    # 5d) Autolinks <http(s)://...> -> remove
    # text = re.sub(r"<https?://[^>]+>", "", text, flags=re.IGNORECASE)
    # 5e) Bare URLs -> remove
    # text = re.sub(r"https?://\S+", "", text, flags=re.IGNORECASE)
    # 5f) EMPTY inline anchor links: [](#page-15-4) / [](#anything)
    text = re.sub(r"\[\s*\]\(\s*#[^)]+\)", "", text)  # remove entirely
    # 5g) Bare anchor parens that sometimes appear: (#page-15-4)
    text = re.sub(r"\(\s*#[^)]+\)", "", text)

    # 6) Remove References/Bibliography/Works Cited → EOF
    ref_header_pattern = re.compile(
        r"""
        ^[ \t]*                                   # indentation
        (?:\#{1,6}[ \t]*)?                        # optional hashes
        (?:\d+(?:\.\d+)*\.?[ \t]+)?               # optional numeric prefix
        (?:\*\*|__|\*|_)?[ \t]*                   # optional bold/italic open
        (?:references?|bibliography|works[ \t]+cited) # keywords
        [ \t]*(?:\*\*|__|\*|_)?                   # optional bold/italic close
        [ \t]*$                                   # end of line
        """,
        flags=re.IGNORECASE | re.MULTILINE | re.VERBOSE,
    )
    m = ref_header_pattern.search(text)
    if m:
        text = text[: m.start()].rstrip()

    # 7) LaTeX math wrappers (keep inner content)
    text = re.sub(r"\\(mathcal|mathbf|mathrm|mathbb)\s*\{([^{}]*)\}", r"\2", text)

    # Strip inline/display math blocks entirely
    text = re.sub(r"\$\$(.*?)\$\$", "", text, flags=re.DOTALL)  # $$...$$
    text = re.sub(r"\$(?!\s)(.+?)(?<!\s)\$", "", text, flags=re.DOTALL)  # $...$
    text = re.sub(r"\\\((.*?)\\\)", "", text, flags=re.DOTALL)  # \( ... \)
    text = re.sub(r"\\\[(.*?)\\\]", "", text, flags=re.DOTALL)  # \[ ... \]

    # 8) Normalize whitespace & leftover separators
    # collapse runs created by removing anchors like [](#page-15-4)–[](#page-16-1)
    text = re.sub(r"[ \t]*[–—-][ \t]*([–—-][ \t]*)+", " – ", text)  # multi dashes → single en-dash
    text = re.sub(r"[ \t]+", " ", text)                              # collapse spaces
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)                     # tidy line breaks
    text = re.sub(r"\n{3,}", "\n\n", text).strip()                   # max 2 blank lines

    return text



# ---- Token estimators ----
def estimate_tokens(text: str) -> dict:
    """
    Returns multiple estimates, since exact tokenizer may differ by provider/model.
    - char_based: ~1 token ≈ 4 chars (common rough rule)
    - word_based: ~1 token ≈ 0.75 words (English-ish heuristic)
    If you later add a true tokenizer, plug it in and add a 'model_tokens' field.
    """
    chars = len(text)
    words = len(re.findall(r"\S+", text))

    char_based = math.ceil(chars / 4) if chars else 0
    word_based = math.ceil(words / 0.75) if words else 0

    return {
        "char_based": char_based,
        "word_based": word_based,
    }

def main():
    in_dir = Path(r"data\3_top_papers\markdown_papers")
    out_dir = in_dir / "cleaned"
    out_dir.mkdir(parents=True, exist_ok=True)

    md_exts = {".md", ".markdown", ".mdown", ".mkd", ".mkdn", ".txt"}  # include .txt if you keep markdown there
    files = sorted(p for p in in_dir.glob("**/*") if p.is_file() and p.suffix.lower() in md_exts and out_dir not in p.parents)

    report_rows = []
    for p in files:
        try:
            raw = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"[SKIP] Could not read {p}: {e}")
            continue

        orig_chars = len(raw)
        orig_tokens = estimate_tokens(raw)

        cleaned = clean_markdown_pdf(raw)
        cleaned_chars = len(cleaned)
        cleaned_tokens = estimate_tokens(cleaned)

        # mirror the relative structure under 'cleaned/'
        rel = p.relative_to(in_dir)
        cleaned_path = out_dir / rel
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_path.write_text(cleaned, encoding="utf-8")

        report_rows.append({
            "file": str(rel),
            "orig_chars": orig_chars,
            "cleaned_chars": cleaned_chars,
            "delta_chars": cleaned_chars - orig_chars,
            "shrink_ratio": round((cleaned_chars / orig_chars), 4) if orig_chars else None,
            "est_tokens_orig_char4": orig_tokens["char_based"],
            "est_tokens_orig_word075": orig_tokens["word_based"],
            "est_tokens_clean_char4": cleaned_tokens["char_based"],
            "est_tokens_clean_word075": cleaned_tokens["word_based"],
        })

        # find the largest file size (add before loop)
        max_orig_chars = max((len(p.read_text(encoding="utf-8", errors="replace")) for p in files), default=0)
        min_orig_chars = min((len(p.read_text(encoding="utf-8", errors="replace")) for p in files), default=0)

        # inside your for-loop, after you compute orig_chars and cleaned_chars:
        reduction_pct = 100 * (1 - cleaned_chars / orig_chars) if orig_chars else 0
        star = "⭐" if orig_chars == max_orig_chars or orig_chars == min_orig_chars  else " "

        print(
            f"[OK] {str(rel)[:30]:<30} "
            f"| chars: {orig_chars:>8,} → {cleaned_chars:>8,} "
            f"| Δ {reduction_pct:>6.2f}% "
            f"(~tokens: {orig_tokens['char_based']:>8,} → {cleaned_tokens['char_based']:>8,}) {star} "
        )

    # Write CSV report
    csv_path = in_dir / "clean_report.csv"
    fieldnames = [
        "file",
        "orig_chars",
        "cleaned_chars",
        "delta_chars",
        "shrink_ratio",
        "est_tokens_orig_char4",
        "est_tokens_orig_word075",
        "est_tokens_clean_char4",
        "est_tokens_clean_word075",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    # Quick summary
    total_files = len(report_rows)
    total_orig = sum(r["orig_chars"] for r in report_rows)
    total_clean = sum(r["cleaned_chars"] for r in report_rows)
    print("\n=== Summary ===")
    print(f"Files processed: {total_files}")
    print(f"Total chars: {total_orig} -> {total_clean} "
          f"(Δ {total_clean - total_orig}, {round((total_clean/total_orig)*100, 2) if total_orig else 0}% of original)")
    print(f"CSV report: {csv_path}")
    print(f"Cleaned files saved under: {out_dir}")

if __name__ == "__main__":
    main()

'''
[OK] A Simple Transformer-Based Mod | chars: 15,729 → 11,762 | Δ 25.22% (~tokens: 3,933 → 2,941)
[OK] Audio Does Matter Importance-A | chars: 87,246 → 43,481 | Δ 50.16% (~tokens: 21,812 → 10,871)
[OK] Bridging High-Quality Audio an | chars: 35,552 → 26,176 | Δ 26.37% (~tokens: 8,888 → 6,544)
[OK] Bridging_the_Semantic_Gap_A_De | chars: 27,621 → 23,339 | Δ 15.50% (~tokens: 6,906 → 5,835)
[OK] CLaMR Contextualized Late-Inte | chars: 80,304 → 43,783 | Δ 45.48% (~tokens: 20,076 → 10,946)
[OK] Condensed Movies Story Based R | chars: 71,101 → 38,360 | Δ 46.05% (~tokens: 17,776 → 9,590)
[OK] Conditional Cross Correlation  | chars: 46,825 → 34,927 | Δ 25.41% (~tokens: 11,707 → 8,732)
[OK] ContextIQ A Multimodal Expert- | chars: 82,519 → 38,881 | Δ 52.88% (~tokens: 20,630 → 9,721)
[OK] ECLIPSE Efficient Long-range V | chars: 69,988 → 44,858 | Δ 35.91% (~tokens: 17,497 → 11,215)
[OK] End-to-End Audio Visual Scene- | chars: 39,335 → 30,784 | Δ 21.74% (~tokens: 9,834 → 7,696)
[OK] Florence - A New Foundation Mo | chars: 83,137 → 55,956 | Δ 32.69% (~tokens: 20,785 → 13,989)
[OK] HumanOmni - A Large Vision-Spe | chars: 64,180 → 45,189 | Δ 29.59% (~tokens: 16,045 → 11,298)
[OK] Improving semantic video retri | chars: 135,170 → 80,642 | Δ 40.34% (~tokens: 33,793 → 20,161)
[OK] Known-Item Video Search via Qu | chars: 23,584 → 18,009 | Δ 23.64% (~tokens: 5,896 → 4,503)
[OK] Local-Global Video-Text Intera | chars: 52,292 → 28,125 | Δ 46.22% (~tokens: 13,073 → 7,032)
[OK] Masking Modalities for Cross-m | chars: 50,877 → 40,428 | Δ 20.54% (~tokens: 12,720 → 10,107)
[OK] MDMMT Multidomain Multimodal T | chars: 94,357 → 45,917 | Δ 51.34% (~tokens: 23,590 → 11,480)
[OK] MMMORRF Multimodal Multilingua | chars: 37,416 → 22,858 | Δ 38.91% (~tokens: 9,354 → 5,715)
[OK] Multi-granularity Corresponden | chars: 88,069 → 38,944 | Δ 55.78% (~tokens: 22,018 → 9,736)
[OK] PreMind Multi-Agent Video Unde | chars: 88,787 → 37,636 | Δ 57.61% (~tokens: 22,197 → 9,409)
[OK] Question-Aware_Global-Local_Vi | chars: 67,615 → 52,284 | Δ 22.67% (~tokens: 16,904 → 13,071)
[OK] Semantic Mapping in Video Retr | chars: 5,855 → 5,132 | Δ 12.35% (~tokens: 1,464 → 1,283) ⭐
[OK] Semantic Multimedia Retrieval  | chars: 20,892 → 19,006 | Δ 9.03% (~tokens: 5,223 → 4,752)
[OK] Smart Routing for Multimodal V | chars: 39,889 → 35,539 | Δ 10.91% (~tokens: 9,973 → 8,885)
[OK] Spoken Moments - Learning Joi  | chars: 153,041 → 49,369 | Δ 67.74% (~tokens: 38,261 → 12,343)
[OK] TACo Token-aware Cascade Contr | chars: 77,079 → 45,268 | Δ 41.27% (~tokens: 19,270 → 11,317)
[OK] TLDW Summarizing Instructional | chars: 69,938 → 54,627 | Δ 21.89% (~tokens: 17,485 → 13,657)
[OK] Towards Fast Adaptation of Pre | chars: 64,551 → 33,117 | Δ 48.70% (~tokens: 16,138 → 8,280)
[OK] Towards Holistic Language-vide | chars: 61,046 → 47,778 | Δ 21.73% (~tokens: 15,262 → 11,945)
[OK] Traffic Video Event Retrieval  | chars: 35,762 → 28,480 | Δ 20.36% (~tokens: 8,941 → 7,120)
[OK] Type-to-Track - Retrieve Any   | chars: 116,768 → 50,625 | Δ 56.64% (~tokens: 29,192 → 12,657)
[OK] Understanding Co-speech Gestur | chars: 78,359 → 43,856 | Δ 44.03% (~tokens: 19,590 → 10,964)
[OK] Unified Static and Dynamic Net | chars: 155,198 → 102,882 | Δ 33.71% (~tokens: 38,800 → 25,721) ⭐
[OK] Unified Video-Language Pre-tra | chars: 81,746 → 36,920 | Δ 54.84% (~tokens: 20,437 → 9,230)
[OK] Use What You Have - Video Retr | chars: 72,395 → 38,393 | Δ 46.97% (~tokens: 18,099 → 9,599)
[OK] VALOR Vision-Audio-Language Om | chars: 119,973 → 74,620 | Δ 37.80% (~tokens: 29,994 → 18,655)
[OK] VAST A Vision-Audio-Subtitle-T | chars: 106,104 → 42,160 | Δ 60.27% (~tokens: 26,526 → 10,540)
[OK] Video-RAG - Visually Aligned R | chars: 67,670 → 29,550 | Δ 56.33% (~tokens: 16,918 → 7,388)
[OK] VideoStory Embeddings Recogniz | chars: 91,988 → 74,966 | Δ 18.50% (~tokens: 22,997 → 18,742)
[OK] Zero-Shot Event Detection by M | chars: 54,702 → 42,150 | Δ 22.95% (~tokens: 13,676 → 10,538)
'''