# main.py
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Ensure 'src' is in path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import config
from atlas.utils.app_helpers import (
    new_run, save_run, run_initial_screening, 
    select_top_ids, run_full_screening, run_category_synthesis,
    update_counts, update_prisma, sync_prisma_from_top_papers
)
from atlas.inital_fetch.gpt_research_q import build_boolean_query_from_questions, boolean_to_queries
from atlas.inital_screen.gpt_criteria import build_criteria_from_question, criteria_to_list
from atlas.read_paper.gpt_categories import build_taxonomy_categories
from atlas.read_paper.pdf_downloader import download_pdfs
from atlas.utils.streamlit_pipeline import fetch_and_enrich
from atlas.utils.utils import safe_filename, deduplicate_papers_by_title_authors
from atlas.results.generate_full_draft import generate_full_draft

def get_latest_run_dir():
    runs_dir = Path(config.RUNS_DIR)
    if not runs_dir.exists():
        return None
    dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not dirs:
        return None
    return max(dirs, key=lambda x: x.name)

def main():
    # 1. Initialize/Resume Run
    latest_dir = get_latest_run_dir()
    resume = False
    
    if latest_dir:
        print(f"[*] Found existing run: {latest_dir.name}")
        ans = input(">>> Do you want to resume this run? (Y/n): ").strip().lower()
        if ans != 'n':
            resume = True
            
    if resume:
        run_dir = latest_dir
        run_id = run_dir.name.replace("run_", "")
        run_path = run_dir / "log.json"
        with open(run_path, "r", encoding="utf-8") as f:
            run = json.load(f)
        print(f"[*] Resuming Auto-SLR Run: {run_id} at stage: {run.get('stage')}")
    else:
        run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        run_dir = Path(config.RUNS_DIR) / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_path = run_dir / "log.json"
        print(f"[*] Starting NEW Auto-SLR Run: {run_id}")
        run = new_run()
        run["stage"] = "started"
    
    # 2. Setup Inputs
    if not resume or run["stage"] == "started":
        print("[*] Setting up research questions and criteria...")
        rq = config.RESEARCH_QUESTIONS.strip()
        run["inputs"]["research_questions"] = rq
        
        # Generate/Confirm Queries (Use saved ones if available)
        if hasattr(config, "BOOLEAN_QUERY") and config.BOOLEAN_QUERY:
            suggested_boolean = config.BOOLEAN_QUERY
            print("[+] Using saved Boolean query from config.")
        else:
            suggested_boolean = build_boolean_query_from_questions(rq)
            
        run["inputs"]["boolean_query_used"] = suggested_boolean
        
        if hasattr(config, "QUERIES") and config.QUERIES:
            queries = config.QUERIES
            print(f"[+] Using {len(queries)} saved queries from config.")
        else:
            queries = boolean_to_queries(suggested_boolean, max_queries=config.MAX_QUERIES)
            
        run["inputs"]["queries"] = queries
        
        # Generate/Confirm Criteria
        criteria = config.CRITERIA if config.CRITERIA else criteria_to_list(build_criteria_from_question(rq))
        run["inputs"]["criteria_used"] = criteria
        
        save_run(run, run_path)
    else:
        rq = run["inputs"].get("research_questions")
        queries = run["inputs"].get("queries")
        criteria = run["inputs"].get("criteria_used")

    # 3. Fetching and Enriching
    if run["stage"] in ("started", "setup_complete"):
        print("[*] Fetching papers from IEEE, Crossref, and Semantic Scholar...")
        app_limits = {
            "max_queries": config.MAX_QUERIES,
            "per_query_results": config.PER_QUERY_RESULTS,
            "top_n": config.TOP_N
        }
        
        enriched_papers, fetch_logs = fetch_and_enrich(
            queries, run, app_limits, raw_boolean=run["inputs"].get("boolean_query_used", "")
        )
        
        # Map to run structure
        from atlas.utils.app_helpers import paper_id_from
        papers_by_id = {}
        for paper in enriched_papers:
            pid = paper_id_from(paper)
            paper["paper_id"] = pid
            if pid not in papers_by_id:
                papers_by_id[pid] = paper
                
        run["papers_by_id"] = papers_by_id
        run["stage"] = "fetch_complete"
        save_run(run, run_path)
        print(f"[+] Fetched {len(papers_by_id)} unique papers.")
    else:
        papers_by_id = run.get("papers_by_id", {})
        print(f"[OK] Skipping fetch. {len(papers_by_id)} papers already loaded.")

    # 4. Initial Screening
    if run["stage"] == "fetch_complete":
        print("[*] Running LLM screening (Abstract/Title)...")
        run_initial_screening(run, run_path, criteria)
        run["stage"] = "screening_complete"
        save_run(run, run_path)
        print("[+] Screening complete.")
    else:
        print("[OK] Skipping screening. Already complete.")

    # 5. Top Paper Selection and Download
    if run["stage"] == "screening_complete":
        print("[*] Selecting top papers for full-text retrieval...")
        top_ids = select_top_ids(papers_by_id, max_n=config.TOP_N, min_score=3)
        run["top_paper_ids"] = {pid: {"title": papers_by_id[pid].get("title") or pid} for pid in top_ids}
        update_counts(run, top_selected=len(top_ids))
        run["stage"] = "top_selected"
        save_run(run, run_path)
    else:
        top_ids = list(run.get("top_paper_ids", {}).keys())
        print(f"[OK] {len(top_ids)} top papers already selected.")

    pdf_dir = run_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    if run["stage"] == "top_selected":
        download_list = [{"paper": papers_by_id[pid]} for pid in top_ids]
        print(f"[*] Attempting to download {len(download_list)} PDFs...")
        download_pdfs(download_list, pdf_dir)
        run["stage"] = "download_attempted"
        save_run(run, run_path)

    # 6. Check for missing PDFs and PAUSE
    print("\n" + "="*50)
    print("PDF DOWNLOAD STATUS CHECK")
    print("="*50)
    
    missing_papers = []
    for pid in top_ids:
        title = papers_by_id[pid].get("title") or pid
        pdf_path = pdf_dir / f"{safe_filename(title)}.pdf"
        if pdf_path.exists():
            run["top_paper_ids"][pid]["pdf_path"] = str(pdf_path)
        else:
            missing_papers.append((pid, title))
            
    if missing_papers:
        print(f"\n[!] WARNING: {len(missing_papers)} PDFs could not be downloaded automatically:")
        for i, (pid, title) in enumerate(missing_papers, 1):
            doi = papers_by_id[pid].get("doi") or "No DOI"
            print(f"  {i}. {title} (DOI: {doi})")
        
        print(f"\n[ACTION REQUIRED]")
        print(f"1. Please try to find these PDFs manually.")
        print(f"2. Place them in the following folder:")
        print(f"   {pdf_dir.absolute()}")
        print(f"3. Important: The filename must match the pattern: 'Safe_Title.pdf'")
        print(f"   Pattern for first missing: '{safe_filename(missing_papers[0][1])}.pdf'")
        print("\nOnce you have added the PDFs you could find, press ENTER to resume.")
        input(">>> Press ENTER to resume...")
        
        # Re-check after resume
        still_missing = 0
        for pid, title in missing_papers:
            pdf_path = pdf_dir / f"{safe_filename(title)}.pdf"
            if pdf_path.exists():
                run["top_paper_ids"][pid]["pdf_path"] = str(pdf_path)
            else:
                still_missing += 1
        print(f"[+] Resuming. Total PDFs available: {len(top_ids) - still_missing} / {len(top_ids)}")
    else:
        print("[+] All PDFs downloaded successfully!")

    run["stage"] = "proxy_download_done"
    save_run(run, run_path)

    # 7. Taxonomy and Full Reading
    print("[*] Using fixed extraction categories for systematic review...")
    from config import FIXED_CATEGORIES
    categories = FIXED_CATEGORIES
    run["categories"] = categories
    save_run(run, run_path)
    
    print("[*] Reading full papers (PDF analysis)...")
    run_full_screening(run, run_path)
    print("[+] Full reading complete.")

    # 8. Synthesis and Final Draft
    print("[*] Synthesizing results across categories...")
    run_category_synthesis(run, run_path)
    
    print("[*] Generating final draft report...")
    draft_path = run_dir / "SLR_draft.md"
    generate_full_draft(run, draft_path)
    
    print("[*] Exporting results to Excel...")
    try:
        from atlas.results.generate_session_report import export_run_to_excel
        excel_path = run_dir / "SLR_Report.xlsx"
        export_run_to_excel(run, excel_path)
        print(f"[+] Excel report saved to: {excel_path.absolute()}")
    except Exception as e:
        print(f"[WARN] Could not generate Excel report: {e}")
    
    print("\n" + "="*50)
    print("SLR WORKFLOW COMPLETE")
    print("="*50)
    print(f"Final Run ID: {run_id}")
    print(f"Report saved to: {draft_path.absolute()}")
    print(f"Log saved to: {run_path.absolute()}")
    print("="*50)

if __name__ == "__main__":
    main()
