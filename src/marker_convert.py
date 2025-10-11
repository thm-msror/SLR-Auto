import os
import time
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# === Paths ===
input_dir = "data/40_papers_pdfs"
output_dir = "data/markdown_papers"
os.makedirs(output_dir, exist_ok=True)

# === Load Models ===
print("🔄 Loading Marker models... (this may take a few minutes)")
models = create_model_dict()
converter = PdfConverter(artifact_dict=models)
print("✅ Models loaded successfully!")

t1 = time.time()

# === Convert all PDFs ===
for file in os.listdir(input_dir):
    if not file.endswith(".pdf"):
        continue
    pdf_path = os.path.join(input_dir, file)
    out_path = os.path.join(output_dir, file.replace(".pdf", ".md"))

    print(f"📄 Converting: {file}")
    try:
        t0 = time.time()
        rendered = converter(pdf_path)
        text, _, _ = text_from_rendered(rendered)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"✅ Saved: {out_path} ({time.time() - t0:.2f} sec)")

    except Exception as e:
        print(f"❌ Error converting {file}: {e}")

print(f"🎉 Done! Converted all PDFs in {time.time() - t1:.2f} seconds.")
