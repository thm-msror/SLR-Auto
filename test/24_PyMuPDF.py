import os
import time
import fitz

input_dir = "data/40_papers_PDFs"
output_dir = "test/PyMuPDF/40_papers_txt"
os.makedirs(output_dir, exist_ok=True)

t1 = time.time()

for file in os.listdir(input_dir):
    t0 = time.time()
    if not file.endswith(".pdf"):
        continue

    pdf_path = os.path.join(input_dir, file)
    out_path = os.path.join(output_dir, file.replace(".pdf", ".txt"))

    print(f"📄 Converting: {file}")
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✅ Saved: {out_path}")

    except Exception as e:
        print(f"❌ Error converting {file}: {e}")

    print(f"Took {time.time() - t0 :2f} second to convert {file}")

print(f"🎉 All PDFs processed!. Took {time.time() - t1 :2f} second")
