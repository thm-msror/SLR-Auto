import os
import time
import pymupdf4llm
import pathlib

input_dir = "data/40_papers_PDFs"
output_dir = "test/PyMuPDF/40_papers_MD"
os.makedirs(output_dir, exist_ok=True)

t1 = time.time()

for file in os.listdir(input_dir):
    t0 = time.time()
    if not file.endswith(".pdf"):
        continue

    pdf_path = os.path.join(input_dir, file)
    out_path = os.path.join(output_dir, file.replace(".pdf", ".md"))

    print(f"Converting: {file}")
    try:
        text =  pymupdf4llm.to_markdown(pdf_path)


        pathlib.Path(out_path).write_bytes(text.encode())

    except Exception as e:
        print(f"Error converting {file}: {e}")

    print(f"Took {time.time() - t0 :2f} second to convert {file}")

print(f"All PDFs processed!. Took {time.time() - t0 :2f} second")
