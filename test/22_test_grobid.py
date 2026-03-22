"""
==============================================================
GROBID PDF → TEI XML + Markdown Converter
==============================================================

This script automatically sends a PDF to a running GROBID server,
receives its structured TEI XML output, and converts that XML into
a readable Markdown (.md) file.

--------------------------------------------------------------
HOW GROBID WORKS
--------------------------------------------------------------
GROBID (GeneRation Of BIbliographic Data) is a machine learning
tool that extracts, parses, and structures scholarly documents.
It runs as a REST API server — you send a PDF to an endpoint
(`/api/processFulltextDocument`), and it returns the full
structured content in XML (TEI format).

From there, this script:
  1. Sends a PDF to the local GROBID server.
  2. Saves the XML response.
  3. Extracts title, abstract, and text paragraphs.
  4. Generates a Markdown version of the paper.

--------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------
1. Install Docker Desktop on your system.
2. Run the GROBID server using this command:

    docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0

3. Install Python dependencies (preferably in a virtual environment):

    python -m venv venv
    venv\Scripts\activate          (Windows)
    pip install requests beautifulsoup4 lxml

--------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------
1. Place your target PDF inside the `40_papers_pdfs/` folder.
2. Open a terminal in this script's folder.
3. Activate your virtual environment.
4. Run:

    python test_grobid.py

Output:
    - `results/grobid_results/grobid_output.xml`
    - `results/grobid_results/grobid_output.md`

--------------------------------------------------------------
HOW TO ADD TO A REPO
--------------------------------------------------------------
Include the following in your GitHub repo:
     40_papers_pdfs/           → folder with test PDFs
     results/grobid_results/   → auto-created output folder
     test_grobid.py            → this script
     requirements.txt          → add dependencies

Example `requirements.txt`:
    requests
    beautifulsoup4
    lxml

--------------------------------------------------------------
RUNNING OUTSIDE A VENV
--------------------------------------------------------------
You can run this script system-wide (not inside venv) if:
  - Python 3.9+ is installed
  - You install dependencies globally:

        pip install requests beautifulsoup4 lxml

However, using a `venv` is recommended to avoid version conflicts.
==============================================================
"""

import os
import requests
from bs4 import BeautifulSoup

def convert_with_grobid(pdf_path, output_dir="results/grobid_results"):
    """
    Send a PDF to the local GROBID server and save both XML and Markdown outputs.

    Args:
        pdf_path (str): Path to the input PDF.
        output_dir (str): Directory to save the XML and Markdown results.

    Returns:
        None
    """
    # Ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    # GROBID server API endpoint
    url = "http://localhost:8070/api/processFulltextDocument"

    # Send PDF to the GROBID server
    with open(pdf_path, "rb") as f:
        files = {"input": f}
        try:
            response = requests.post(url, files=files)
        except requests.exceptions.RequestException as e:
            print(f" GROBID request failed: {e}")
            return

    # Check for valid response
    if response.status_code != 200 or not response.text.strip():
        print(f" GROBID failed or returned empty response. Status: {response.status_code}")
        return

    # Save the XML output
    xml_path = os.path.join(output_dir, "grobid_output.xml")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    print(f" GROBID XML saved: {xml_path}")

    # Convert XML → Markdown safely
    try:
        # Use lxml parser if available, fallback to built-in XML parser
        try:
            soup = BeautifulSoup(response.text, "lxml-xml")
        except Exception:
            soup = BeautifulSoup(response.text, "xml")
    except Exception as e:
        print(f" BeautifulSoup parser error: {e}")
        return

    # Extract sections from TEI XML
    title = soup.find("title").get_text() if soup.find("title") else "Untitled"
    abstract = soup.find("abstract").get_text(separator="\n") if soup.find("abstract") else ""
    paragraphs = [p.get_text(separator="\n") for p in soup.find_all("p")]

    # Compose Markdown content
    md_content = f"# {title}\n\n"
    if abstract:
        md_content += f"## Abstract\n{abstract}\n\n"
    if paragraphs:
        md_content += "## Main Text\n" + "\n\n".join(paragraphs)

    # Save Markdown output
    md_path = os.path.join(output_dir, "grobid_output.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f" Markdown saved: {md_path}")

if __name__ == "__main__":
    pdf_file = "data/40_papers_pdfs/HumanOmni - A Large Vision-Speech Language Model for Human-Centric Video Understanding.pdf"
    convert_with_grobid(pdf_file)
