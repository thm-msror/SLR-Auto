# test/21_test_pdf_to_md.py
"""
PDF to Markdown Converter
-------------------------
This script demonstrates how to take a PDF file, extract text using PyMuPDF (fitz),
and optionally convert that text into Markdown using Pandoc (via pypandoc).

Why PyMuPDF?
- Pandoc cannot directly read PDFs (PDF is only an output format for Pandoc).
- PyMuPDF (`fitz`) allows direct text extraction page by page.

Workflow:
1. Load the PDF with fitz.
2. Extract text content from each page.
3. Try to clean/convert the text to Markdown using Pandoc.
4. Save the result into a `.md` file.
"""

from pathlib import Path
import fitz  # PyMuPDF for extracting text from PDFs
import pypandoc  # Optional text → Markdown polishing

# Input PDF and output Markdown paths
PDF_PATH = Path("data/40_papers_pdfs/Semantic Multimedia Retrieval using Lexical Query Expansion and Model-Based Reranking.pdf")
OUT_PATH = Path("test/semantic_multimedia.md")

def pdf_to_text(pdf_path: Path) -> str:
    """
    Extracts raw text from a PDF using PyMuPDF (fitz).
    
    Args:
        pdf_path (Path): Path to the input PDF file.

    Returns:
        str: Concatenated text extracted from all pages.
    """
    doc = fitz.open(pdf_path)  # Open PDF
    text = ""
    for page in doc:
        # Extract plain text from each page
        text += page.get_text("text") + "\n"
    return text

def text_to_markdown(text: str, out_path: Path):
    """
    Converts extracted text to Markdown using Pandoc.
    If Pandoc fails, falls back to saving raw text.

    Args:
        text (str): The plain text extracted from PDF.
        out_path (Path): Path to save the output `.md` file.
    """
    try:
        # Pandoc attempt: convert plain text → Markdown
        md = pypandoc.convert_text(text, "md", format="plain")
    except Exception as e:
        print(f"[WARN] Pandoc failed, saving raw text instead: {e}")
        md = text

    # Save result to file
    out_path.write_text(md, encoding="utf-8")
    print(f"[OK] Saved markdown to {out_path}")

if __name__ == "__main__":
    # Ensure PDF exists before running
    if not PDF_PATH.exists():
        print(f"[ERROR] File not found: {PDF_PATH}")
    else:
        print(f"[INFO] Converting {PDF_PATH.name} → Markdown")
        raw_text = pdf_to_text(PDF_PATH)  # Step 1: Extract text
        text_to_markdown(raw_text, OUT_PATH)  # Step 2: Convert + save
