import os
import time
from pdfminer.high_level import extract_text

input_dir = "data/40_papers_PDFs"
output_dir = "test/PDFMiner/40_papers_filtered_txt"
os.makedirs(output_dir, exist_ok=True)

t1 = time.time()

import re

import re

def smart_filter_lines(text, short_threshold=30, max_consecutive_empty=2, max_prev_short=3):
    """
    Filters lines based on:
    1. Avoid too many consecutive empty lines.
    2. Avoid multiple consecutive short lines (< short_threshold).
    """
    cleaned_lines = []
    empty_count = 0
    prev_short_lines = []

    for line in text.splitlines():
        stripped = line.strip()

        # Handle empty lines
        if not stripped:
            empty_count += 1
            if empty_count <= max_consecutive_empty:
                cleaned_lines.append(line)
            continue
        elif len(stripped) < 5:
            empty_count += 1
            continue
        else:
            empty_count = 0

        # Check if line is short
        is_short = len(stripped) < short_threshold

        # Check letters presence
        if not re.search(r'[A-Za-z]', stripped):
            continue  # skip lines with no letters

        # Skip if too many previous lines were short
        if is_short and len([s for s in prev_short_lines[-max_prev_short:] if s]) >= max_prev_short:
            prev_short_lines.append(is_short)
            continue

        # Add the line
        cleaned_lines.append(line)
        prev_short_lines.append(is_short)

    return "\n".join(cleaned_lines)

for file in os.listdir(input_dir):
    t0 = time.time()
    if not file.endswith(".pdf"):
        continue

    pdf_path = os.path.join(input_dir, file)
    out_path = os.path.join(output_dir, file.replace(".pdf", ".txt"))

    print(f"📄 Converting: {file}")
    try:
        # Extract text from the PDF
        text = smart_filter_lines(extract_text(pdf_path))

        # Save to file
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"✅ Saved: {out_path}")

    except Exception as e:
        print(f"❌ Error converting {file}: {e}")

    print(f"Took {time.time() - t0:2f} seconds to convert {file}")

print(f"🎉 All PDFs processed! Took {time.time() - t1:2f} seconds")

