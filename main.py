from src.process_arxiv import run as process_arxiv
from src.process_crossref import process_crossref

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


if __name__ == "__main__":
    # Run arXiv pipeline
    # process_arxiv()

    # Run Crossref pipeline
    process_crossref()