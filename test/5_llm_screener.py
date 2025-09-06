import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm_screener import screen_papers

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python src/llm_screener.py <input.json> <prompt.txt> <output.json>")
        sys.exit(1)

    input_json, prompt_file, output_json = sys.argv[1], sys.argv[2], sys.argv[3]
    screen_papers(input_json, prompt_file, output_json)