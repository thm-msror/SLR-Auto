import os
import time
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# import torch
# print(torch.cuda.is_available())

'''
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. 
This behaviour is the source of the following dependency conflicts.
marker-pdf 1.10.1 requires torch<3.0.0,>=2.7.0, but you have torch 2.5.1+cu121 which is incompatible.
surya-ocr 0.17.0 requires torch<3.0.0,>=2.7.0, but you have torch 2.5.1+cu121 which is incompatible.
Successfully installed sympy-1.13.1 torch-2.5.1+cu121 torchaudio-2.5.1+cu121 torchvision-0.20.1+cu121
'''

# For testing a single PDF file
input_pdf = "data/40_papers_PDFs/Improving semantic video retrieval models by training with a relevance-aware online mining strategy.pdf"
output_dir = "test/marker40_papers_markdown"
os.makedirs(output_dir, exist_ok=True)

print("Loading Marker models (this may take a few minutes)...")
models = create_model_dict()
converter = PdfConverter(artifact_dict=models)
print("Models loaded successfully.")

t1 = time.time()

try:
    t0 = time.time()
    file_name = os.path.basename(input_pdf)
    out_path = os.path.join(output_dir, file_name.replace(".pdf", ".md"))

    print(f"Converting single file: {file_name}")
    rendered = converter(input_pdf)
    text, _, _ = text_from_rendered(rendered)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved: {out_path}")

    print(f"Took {time.time() - t0 :2f} seconds to convert {file_name}")

except Exception as e:
    print(f"Error converting {input_pdf}: {e}")

print(f"Conversion finished! Total time: {time.time() - t1 :2f} seconds")

# --- 💬 Original directory-based loop (commented for later use) ---
"""
input_dir = "data/40_papers_PDFs"
output_dir = "test/marker40_papers_markdown"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    t0 = time.time()
    if not file.endswith(".pdf"):
        continue

    pdf_path = os.path.join(input_dir, file)
    out_path = os.path.join(output_dir, file.replace(".pdf", ".md"))

    print(f"Converting: {file}")
    try:
        rendered = converter(pdf_path)
        text, _, _ = text_from_rendered(rendered)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved: {out_path}")
    except Exception as e:
        print(f"Error converting {file}: {e}")

    print(f"Took {time.time() - t0 :2f} second to convert {file}")

print(f"All PDFs processed!. Took {time.time() - t1 :2f} second")
"""
