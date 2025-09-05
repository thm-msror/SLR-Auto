import json
import sys
import os

# Ensure src is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from fetch_ieee import IEEEFetcher

def main():
    API_KEY = "bnd6jb3whqgqbcw3tkph34pm"

    fetcher = IEEEFetcher(API_KEY)
    try:
        results = fetcher.fetch("graph neural networks", max_records=3)
        print(json.dumps(results, indent=2))
    except Exception as e:
        print("❌ Error:", e)


if __name__ == "__main__":
    main()
