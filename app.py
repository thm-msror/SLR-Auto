import argparse
import json
from datetime import datetime

from src.utils import (
    deduplicate_papers_by_title_authors,
    iso_now,
    print_section,
    read_prefixed_lines,
    safe_filename,
)
from src.gpt_research_q import build_boolean_query_from_questions, boolean_to_queries
from src.gpt_criteria import build_criteria_from_question, criteria_to_list
from src.fetch_arxiv import fetch_papers as fetch_arxiv
from src.fetch_crossref import fetch_papers as fetch_crossref
from src.enrich_openalex import enrich as enrich_openalex
from src.gpt_categories import build_taxonomy_categories
from src.pdf_downloader import download_pdfs
from src.app_helpers import (
    RUNS_DIR,
    RUN_FILE,
    ensure_run_shape,
    new_run,
    paper_id_from,
    resolve_run_path,
    run_full_screening,
    run_initial_screening,
    run_step,
    save_run,
    select_top_ids,
    summarize_run,
    update_counts,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="SLR Auto interactive runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Start a new run")
    run_p.add_argument("--run-id", default=None, help="Optional run id")

    resume_p = sub.add_parser("resume", help="Resume an existing run")
    resume_p.add_argument("run_id_or_path")

    args = parser.parse_args()

    if args.cmd == "run":
        run_id = args.run_id or datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        run_dir = RUNS_DIR / run_id
        run_path = run_dir / RUN_FILE
        run = new_run()
        ensure_run_shape(run)
        run["stage"] = "init"
        save_run(run, run_path)
        print(f"SLR Auto — New Run ({run_id})")
    else:
        run_path = resolve_run_path(args.run_id_or_path)
        run_dir = run_path.parent
        run = json.loads(run_path.read_text(encoding="utf-8"))
        ensure_run_shape(run)
        print(f"SLR Auto — Resume ({run_dir.name})")
        summarize_run(run)

    steps = run.setdefault("steps", {})

    # ---------------- STEP 1: Research Questions ----------------
    if not (run.get("inputs") or {}).get("research_questions"):
        run["stage"] = "research_questions"
        print_section("STEP 1/9: RESEARCH QUESTIONS")
        lines = read_prefixed_lines(
            "Enter one or more lines. Blank line to finish.",
            "RQ> ",
        )
        while not lines:
            lines = read_prefixed_lines("Please enter at least one line.", "RQ> ")
        run["inputs"]["research_questions"] = "\n".join(lines).strip()
        steps["research_questions"] = {
            "done": True,
            "updated_keys": ["inputs.research_questions"],
            "elapsed_sec": 0,
            "ts": iso_now(),
        }
        save_run(run, run_path)

    # ---------------- STEP 2: Boolean Query ----------------
    if not (run.get("inputs") or {}).get("boolean_query_used"):
        run["stage"] = "boolean_query"
        print_section("STEP 2/9: BOOLEAN QUERY")
        suggested, _ = run_step(
            run,
            steps,
            "generate_boolean_query",
            build_boolean_query_from_questions,
            updated_keys=["inputs.boolean_query_suggested", "stats.timings_sec.generate_boolean_query"],
            questions_text=run["inputs"]["research_questions"],
        )
        run["inputs"]["boolean_query_suggested"] = suggested
        print("Suggested:")
        print(f"  {suggested}")
        edited = input("Press Enter to accept or type an edit:\nQUERY> ").strip()
        used = edited if edited else suggested
        run["inputs"]["boolean_query_used"] = used
        steps["generate_boolean_query"]["updated_keys"].append("inputs.boolean_query_used")
        save_run(run, run_path)

    if not (run.get("inputs") or {}).get("queries"):
        print("\nExpanding to queries...")
        used = run["inputs"]["boolean_query_used"]
        while True:
            queries = boolean_to_queries(used, max_queries=200)
            if len(queries) <= 50:
                break
            print(f"Found {len(queries)} queries (cap is 50).")
            new_q = input("Enter a shorter boolean query (blank to accept first 50): ").strip()
            if not new_q:
                queries = queries[:50]
                break
            used = new_q
            run["inputs"]["boolean_query_used"] = used

        run["inputs"]["queries"] = queries
        update_counts(run, queries=len(queries))
        steps["boolean_to_queries"] = {
            "done": True,
            "updated_keys": ["inputs.queries", "stats.counts.queries"],
            "elapsed_sec": 0,
            "ts": iso_now(),
        }
        save_run(run, run_path)

    # ---------------- STEP 3: Criteria ----------------
    if not (run.get("inputs") or {}).get("criteria_used"):
        run["stage"] = "criteria"
        print_section("STEP 3/9: INCLUSION / EXCLUSION CRITERIA")
        raw, _ = run_step(
            run,
            steps,
            "build_criteria",
            build_criteria_from_question,
            updated_keys=["inputs.criteria_suggested", "stats.timings_sec.build_criteria"],
            question_text=run["inputs"]["research_questions"],
        )
        suggested = criteria_to_list(raw)
        run["inputs"]["criteria_suggested"] = suggested
        print("Suggested criteria (one per line):")
        for line in suggested:
            print(f"  {line}")

        custom_lines = read_prefixed_lines(
            "Press Enter to accept, or paste replacements (blank line to finish):",
            "CRIT> ",
        )
        if custom_lines:
            used = criteria_to_list("\n".join(custom_lines))
        else:
            used = suggested
        run["inputs"]["criteria_used"] = used
        steps["build_criteria"]["updated_keys"].append("inputs.criteria_used")
        save_run(run, run_path)

    # ---------------- STEP 4: Fetch + Enrich ----------------
    if not (run.get("papers_by_id") or {}):
        run["stage"] = "fetch_enrich"
        print_section("STEP 4/9: FETCH + ENRICH")
        queries = run["inputs"]["queries"]

        arxiv, _ = run_step(
            run,
            steps,
            "fetch_arxiv",
            fetch_arxiv,
            ["stats.timings_sec.fetch_arxiv"],
            queries,
            max_results=100,
        )
        crossref, _ = run_step(
            run,
            steps,
            "fetch_crossref",
            fetch_crossref,
            ["stats.timings_sec.fetch_crossref"],
            queries,
            max_results=100,
        )
        update_counts(
            run,
            fetched_arxiv=len(arxiv),
            fetched_crossref=len(crossref),
        )

        combined = arxiv + crossref
        deduped = deduplicate_papers_by_title_authors(combined, paper_type="fetched")
        update_counts(run, deduped_total=len(deduped))

        enriched, _ = run_step(
            run,
            steps,
            "enrich_openalex",
            enrich_openalex,
            ["stats.timings_sec.enrich_openalex"],
            deduped,
        )
        update_counts(run, enriched_total=len(enriched))

        papers_by_id = {}
        for paper in enriched:
            pid = paper_id_from(paper)
            paper["paper_id"] = pid
            if pid not in papers_by_id:
                papers_by_id[pid] = paper

        run["papers_by_id"] = papers_by_id
        save_run(run, run_path)

    # ---------------- STEP 5: Initial Screening ----------------
    if run.get("papers_by_id") and any(
        not p.get("screening") for p in run["papers_by_id"].values()
    ):
        run["stage"] = "initial_screening"
        print_section("STEP 5/9: INITIAL SCREENING")
        criteria = run["inputs"]["criteria_used"]
        run_step(
            run,
            steps,
            "initial_screening",
            lambda: run_initial_screening(run, run_path, criteria),
            updated_keys=["papers_by_id.*.screening", "stats.timings_sec.initial_screening"],
        )
        save_run(run, run_path)

    # ---------------- STEP 6: Top Selection ----------------
    if not run.get("top_paper_ids"):
        run["stage"] = "top_selection"
        print_section("STEP 6/9: TOP SELECTION")
        selected_ids = select_top_ids(run["papers_by_id"], max_n=50)
        top_map = {}
        for pid in selected_ids:
            paper = run["papers_by_id"].get(pid, {})
            top_map[pid] = {"title": paper.get("title") or pid}
        run["top_paper_ids"] = top_map
        update_counts(run, top_selected=len(top_map))
        save_run(run, run_path)

    # ---------------- STEP 7: Download PDFs ----------------
    if run.get("top_paper_ids"):
        run["stage"] = "download_pdfs"
        print_section("STEP 7/9: DOWNLOAD PDFS")
        pdf_dir = run_dir / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        download_list = []
        for pid in run["top_paper_ids"].keys():
            paper = run["papers_by_id"].get(pid)
            if paper:
                download_list.append({"paper": paper})

        if download_list:
            run_step(
                run,
                steps,
                "download_pdfs",
                download_pdfs,
                ["top_paper_ids.*.pdf_path", "stats.timings_sec.download_pdfs"],
                download_list,
                pdf_dir,
            )

        downloaded = 0
        for pid, entry in run["top_paper_ids"].items():
            paper = run["papers_by_id"].get(pid, {})
            title = paper.get("title") or pid
            pdf_path = pdf_dir / f"{safe_filename(title)}.pdf"
            if pdf_path.exists():
                entry["pdf_path"] = str(pdf_path)
                downloaded += 1

        update_counts(run, pdf_downloaded=downloaded)
        save_run(run, run_path)

    # ---------------- STEP 8: Categories ----------------
    if not run.get("categories"):
        run["stage"] = "build_categories"
        print_section("STEP 8/9: CATEGORIES")
        abstracts = []
        for pid in run.get("top_paper_ids", {}).keys():
            paper = run["papers_by_id"].get(pid, {})
            abstract = paper.get("abstract") or ""
            if abstract.strip():
                abstracts.append(abstract.strip())

        if abstracts:
            categories, _ = run_step(
                run,
                steps,
                "build_categories",
                build_taxonomy_categories,
                updated_keys=["categories", "stats.timings_sec.build_categories"],
                research_question=run["inputs"]["research_questions"],
                abstracts=abstracts,
            )
            run["categories"] = categories
            save_run(run, run_path)
        else:
            print(" No abstracts available for categories.")

    # ---------------- STEP 9: Full Screening ----------------
    if run.get("categories") and run.get("top_paper_ids"):
        run["stage"] = "full_screening"
        print_section("STEP 9/9: FULL SCREENING")
        run_step(
            run,
            steps,
            "full_screening",
            lambda: run_full_screening(run, run_path),
            updated_keys=["top_paper_ids.*.full_screening", "stats.timings_sec.full_screening"],
        )
        save_run(run, run_path)

    run["stage"] = "done"
    save_run(run, run_path)
    print("Done.")


if __name__ == "__main__":
    main()
