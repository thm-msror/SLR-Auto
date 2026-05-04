# scripts/generate_queries_for_config.py
import sys
from pathlib import Path

# Ensure root and 'src' are in path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))

import config
from atlas.inital_fetch.gpt_research_q import build_boolean_query_from_questions, boolean_to_queries

def main():
    print("[*] Generating queries from Research Questions in config.py...")
    rq = config.RESEARCH_QUESTIONS.strip()
    
    # 1. Build Boolean Query
    print("[*] Calling GPT to build Boolean query...")
    boolean_query = build_boolean_query_from_questions(rq)
    print(f"\n[BOOLEAN QUERY]\n{boolean_query}\n")
    
    # 2. Build Expanded Queries
    print("[*] Calling GPT to expand into query list...")
    queries = boolean_to_queries(boolean_query, max_queries=config.MAX_QUERIES)
    print(f"\n[EXPANDED QUERIES] Generated {len(queries)} queries.")
    
    # 3. Output as Python list for config.py
    print("\n--- COPY AND PASTE INTO config.py ---")
    print(f"BOOLEAN_QUERY = {repr(boolean_query)}")
    print("\nQUERIES = [")
    for q in queries:
        print(f"    {repr(q)},")
    print("]")
    print("--------------------------------------")

if __name__ == "__main__":
    main()
