import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fetch_ieee import IEEEFetcher
from dotenv import load_dotenv

load_dotenv(".env")

def main():
    API_KEY = os.getenv("IEEE_API_KEY")

    fetcher = IEEEFetcher(API_KEY)
    try:
        results = fetcher.fetch("graph neural networks", max_records=3)
        print(json.dumps(results, indent=2))
    except Exception as e:
        print("❌ Error:", e)


if __name__ == "__main__":
    main()
