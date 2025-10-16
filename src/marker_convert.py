import os
import time
from typing import Dict, Optional

# Set environment variable before any TensorFlow imports
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Pure conversion helpers; no filesystem writes happen here.

_GLOBAL_MODELS = None
_GLOBAL_CONVERTER = None

def load_marker_models():
    """Load and cache Marker models.

    Returns:
        Tuple of (artifact models dict, PdfConverter instance).
    """
    global _GLOBAL_MODELS, _GLOBAL_CONVERTER
    if _GLOBAL_MODELS is None or _GLOBAL_CONVERTER is None:
        print(" Loading Marker models... (this may take a few minutes)")
        # Lazy import to avoid hard dependency at module import time
        from marker.models import create_model_dict
        from marker.converters.pdf import PdfConverter
        _GLOBAL_MODELS = create_model_dict()
        _GLOBAL_CONVERTER = PdfConverter(artifact_dict=_GLOBAL_MODELS)
        print(" Models loaded successfully!")
    return _GLOBAL_MODELS, _GLOBAL_CONVERTER

def convert_pdf_to_markdown(pdf_path: str, converter: Optional[object] = None) -> str:
    """Convert a single PDF to Markdown text.

    This function performs no filesystem writes; it only returns the converted
    Markdown text so that the caller (e.g., `main.py`) can decide where and how
    to save it.

    Args:
        pdf_path: Path to the input PDF file.
        converter: Optional preloaded PdfConverter. If not provided, a cached
            global instance is used.

    Returns:
        Markdown text extracted from the PDF.
    """
    if converter is None:
        _, converter = load_marker_models()
    # Lazy import for output helper
    from marker.output import text_from_rendered
    rendered = converter(pdf_path)
    text, _, _ = text_from_rendered(rendered)
    return text

def convert_pdfs_in_directory(input_dir: str, converter: Optional[object] = None) -> Dict[str, str]:
    """Convert all PDFs in a directory.

    No writes happen here. The returned mapping allows the caller to save the
    results in the desired location and format.

    Args:
        input_dir: Directory containing PDF files.
        converter: Optional preloaded PdfConverter.

    Returns:
        Dict mapping original PDF file names to their Markdown text.
    """
    if converter is None:
        _, converter = load_marker_models()
    outputs: Dict[str, str] = {}
    t1 = time.time()
    for file in os.listdir(input_dir):
        if not file.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(input_dir, file)
        try:
            t0 = time.time()
            md_text = convert_pdf_to_markdown(pdf_path, converter=converter)
            outputs[file] = md_text
            print(f" Converted: {file} ({time.time() - t0:.2f} sec)")
        except Exception as e:
            print(f" Error converting {file}: {e}")
    print(f" Done! Converted {len(outputs)} PDFs in {time.time() - t1:.2f} seconds.")
    return outputs

def run_marker_batch(input_dir: str, output_dir: str) -> int:
    """Convert PDFs in input_dir and write Markdown files to output_dir.

    Takes paths as parameters so they live in config/main. Returns number of files written.
    """
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    _, converter = load_marker_models()
    t1 = time.time()
    written = 0
    for file in os.listdir(input_dir):
        if not file.endswith(".pdf"):
            continue
        pdf_path = os.path.join(input_dir, file)
        out_path = os.path.join(output_dir, file.replace(".pdf", ".md"))
        if os.path.exists(out_path):
            print(f"⏭ Skipping {file}, already converted.")
            continue
        print(f" Converting: {file}")
        try:
            t0 = time.time()
            md_text = convert_pdf_to_markdown(pdf_path, converter=converter)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(md_text)
            written += 1
            print(f" Saved: {out_path} ({time.time() - t0:.2f} sec)")
        except Exception as e:
            print(f" Error converting {file}: {e}")
    print(f" Done! Converted all new PDFs in {time.time() - t1:.2f} seconds.")
    return written
