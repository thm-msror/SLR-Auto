import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.config as config

def main():
    print("Search queries:", config.QUERIES)
    print("Max results:", config.MAX_RESULTS)

if __name__ == "__main__":
    main()
