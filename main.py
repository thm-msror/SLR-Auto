# main.py
import time
from pathlib import Path
from src.utils import load_json, save_json
from src.fetch_arxiv import fetch_papers as fetch_arvix
from src.fetch_crossref import fetch_papers as fetch_crossref
from src.enrich_openalex import enrich
from src.llm_screener_bullets import screen_papers
from src.utils import append_to_json
import config as config

SCREENED_ARTICLES_DIR = Path("data/screened_articles")
CHECKPOINT_DIR = SCREENED_ARTICLES_DIR / "checkpoints"
SCREENED_ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

ALL_JSON_FILE = SCREENED_ARTICLES_DIR / "all_screened_papers.json"
ALL_BULLETS_FILE = SCREENED_ARTICLES_DIR / "all_screened_bullets.txt"

# ---------------- Helper ----------------
def get_latest_checkpoint(track_dir):
    checkpoints = sorted(Path(track_dir).glob("papers_remaining_*.json"))
    return checkpoints[-1] if checkpoints else None

# ---------------- Main ----------------
if __name__ == "__main__":

    start_time = time.time()

    # ---------------- FETCHING ARTICLES ----------------
    if config.all_fetched_path:
        all_fetched_papers = load_json(config.all_fetched_path)
    else:
        # Fetch arXiv
        raw_arvix_fetch = fetch_arvix(
            config.QUERIES,
            max_results=config.MAX_QUERIES,
            track=f"{config.FETCHED_PAPERS_FOLDER}/raw_fetch/checkpoints",
        )
        # Fetch Crossref
        raw_crossref_fetch = fetch_crossref(
            config.QUERIES,
            max_results=config.MAX_QUERIES,
            track=f"{config.FETCHED_PAPERS_FOLDER}/raw_fetch/checkpoints",
        )

        # Combine and enrich
        combined_fetch = raw_crossref_fetch + raw_arvix_fetch
        all_fetched_papers = enrich(combined_fetch,
                                    track=f"{config.FETCHED_PAPERS_FOLDER}/enrich/checkpoints")
        save_json(all_fetched_papers, folder=config.FETCHED_PAPERS_FOLDER,
                  filename=f"fetched_{len(all_fetched_papers)}_")

    # ---------------- LLM SCREENING ----------------
    if config.all_screened_path:
        all_screened_papers = load_json(config.all_screened_path)
    else:
        # Resume from last checkpoint if available
        latest_checkpoint = get_latest_checkpoint(CHECKPOINT_DIR)
        if latest_checkpoint:
            print(f"🔄 Resuming from checkpoint: {latest_checkpoint}")
            remaining_papers = load_json(latest_checkpoint)
        else:
            remaining_papers = all_fetched_papers

        # Screen papers
        all_screened_papers = screen_papers(
            remaining_papers,
            batch_size=100,
            model="Fanar",
            prompt_txt_path=config.LLM_SCREENING_PROMPT_TXT,
        )

    # ---------------- SAVE BULLETS ----------------
    # Append raw bullets in readable format for reference
    with open(ALL_BULLETS_FILE, "a", encoding="utf-8") as f:
        for entry in all_screened_papers:
            paper = entry.get("paper", {})
            llm_screening = entry.get("llm_screening", {})
            bullet_text = llm_screening.get("llm_screening_raw") or ""
            if not bullet_text:
                # Fallback: reconstruct from parsed JSON
                bullet_lines = []
                for key in [
                    "notes", "reason_of_relevance", "key_technologies",
                    "datasets", "application", "limitations", "decision", "top_evidence"
                ]:
                    val = llm_screening.get(key)
                    if isinstance(val, list):
                        val_text = "; ".join(map(str, val))
                    else:
                        val_text = str(val or "")
                    if val_text:
                        bullet_lines.append(f"- {key.replace('_', ' ').title()}: {val_text}")
                bullet_text = "\n".join(bullet_lines)

            f.write(f"Title: {paper.get('title', 'N/A')}\n")
            f.write(bullet_text.strip() + "\n\n")

    # ---------------- SAVE JSON ----------------
    append_to_json(all_screened_papers, ALL_JSON_FILE)
    print(f"\n💾 Final JSON saved: {ALL_JSON_FILE}")
    print(f"💾 Bullets TXT saved: {ALL_BULLETS_FILE}")
    print(f"\n🏁 Total pipeline time: {time.time() - start_time:.2f} sec")

    # ---------------- LLM SUMMARIZATION ----------------
    # t0 = time.time()
    # print("\n📊 LLM Summary of Relevant Papers (from bullet points):\n")
    # summary = summarize_screened_from_bullets()
    # save_md(summary, folder=config.SUMMARY_FOLDER, filename=f"llm_summary_{config.MAX_QUERIES}_")
    # print(f"⏱️ Summarization took {time.time() - t0:.2f} sec")

    # ---------------- PAPER TABLE ----------------
    # t0 = time.time()
    # print("\n📄 Paper Table of Top Relevant Papers:\n")
    # top_papers = get_relevant(all_screened_papers, 100)
    # table_md = paper_table(top_papers)
    # save_md(table_md, folder=config.SUMMARY_FOLDER, filename=f"paper_summary_{config.MAX_QUERIES}")
    # print(f"⏱️ Paper table generation took {time.time() - t0:.2f} sec")

    # print(f"\n🏁 Total pipeline time: {time.time() - start_time:.2f} sec")
