# src/retry_connection_errors.py
import json
from pathlib import Path
from src.fetch_utils import load_json, save_json
from src.screen_crossref import screen_sequential


def retry_connection_errors(
    crossref_file: Path,
    in_process_file: Path,
    fixed_file: Path,
    output_file: Path,
    prompt_path: Path,
    batch_size: int = 5
):
    """
    Automatically retry LLM screening for papers with llm_screening.raw_output == 'Connection error.'.

    Steps:
      1. Extract failed entries from crossref_file → in_process_file.
      2. Re-run screening on failed entries → fixed_file.
      3. Merge fixed results back into full dataset → output_file.
    """

    # --- Step 1: Load original screened data ---
    data = load_json(crossref_file) or []
    print(f"📄 Total papers in {crossref_file}: {len(data)}")

    # --- Step 2: Extract entries with connection error ---
    connection_errors = [
        e["paper"] for e in data
        if e.get("llm_screening", {}).get("raw_output") == "Connection error."
    ]
    print(f"⚠️ Papers with connection error: {len(connection_errors)}")

    if not connection_errors:
        print("✅ No connection errors found, nothing to retry.")
        return

    # Save subset
    save_json(connection_errors, in_process_file.parent, in_process_file.name)
    print(f"💾 Saved {len(connection_errors)} entries to {in_process_file}")

    # --- Step 3: Re-screen subset using screen_sequential ---
    print("\n⚡ Re-screening connection errors...")
    screen_sequential(
        input_json_path=in_process_file,
        output_json_path=fixed_file,
        prompt_path=prompt_path,
        batch_size=batch_size
    )

    # Load re-screened results
    fixed_data = load_json(fixed_file) or []
    print(f"🔄 Loaded {len(fixed_data)} re-screened entries from {fixed_file}")

    # --- Step 4: Merge results back ---
    fixed_lookup = {
        d["paper"].get("doi", d["paper"].get("title")): d
        for d in fixed_data
    }
    merged, replaced_count = [], 0

    for entry in data:
        key = entry["paper"].get("doi", entry["paper"].get("title"))
        if key in fixed_lookup:
            merged.append(fixed_lookup[key])
            replaced_count += 1
        else:
            merged.append(entry)

    print(f"✅ Replaced {replaced_count} entries out of {len(data)}")

    # --- Step 5: Save merged dataset ---
    save_json(merged, output_file.parent, output_file.name)
    print(f"💾 Saved merged dataset to {output_file}")