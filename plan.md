**App.py Plan (Original Flow + Option A JSON Structure)**

**Summary**
Implement `app.py` as the interactive orchestrator. Use `_RUNS/<run_id/_run.json` as the single source of truth, with all papers stored in `papers_by_id`, and top selection tracked by `top_paper_ids`. PDFs are saved in the same run folder.

- `python app.py run` starts a new interactive run.
- `python app.py resume <run_id_or_path>` resumes.

**CLI Print Flow (Option A: Guided)**

```text
SLR Auto — New Run
==================================================
STEP 1/6: RESEARCH QUESTIONS
==================================================
Enter one or more lines. Blank line to finish.
RQ> How do we retrieve long-form video segments?
RQ> How do multimodal embeddings affect search?
RQ>
Saved 2 lines.

==================================================
STEP 2/6: BOOLEAN QUERY
==================================================
Suggested:
  "long video" AND (retrieval OR search) AND (multimodal OR audio)
Press Enter to accept or type an edit:
QUERY> "long video" AND (retrieval OR search) AND multimodal

Expanding to queries...
Found 12 queries. Proceed? [Enter=yes]
> 

==================================================
STEP 3/6: INCLUSION / EXCLUSION CRITERIA
==================================================
Suggested criteria (one per line):
  INCLUDE: Does the study focus on AI systems for long-form video?
  EXCLUDE: Is it only short-form video?
Press Enter to accept, or paste replacements (blank line to finish):
CRIT> INCLUDE: Does the study focus on AI systems for long-form video?
CRIT> EXCLUDE: Is it only short-form video?
CRIT>
Saved 2 criteria.
```

**Src Functions Used (Inputs / Outputs)**

- `src/gpt_research_q.build_boolean_query_from_questions(questions_text: str) -> str` returns a boolean query string.
- `src/gpt_research_q.boolean_to_queries(boolean_query: str, max_queries: int = 50) -> List[str]` returns expanded query list.
- `src/gpt_criteria.build_criteria_from_question(question_text: str) -> str` returns raw criteria text.
- `src/gpt_criteria.criteria_to_list(raw: str) -> List[str]` returns normalized criteria list.
- `src/fetch_arxiv.fetch_papers(queries: list[str], max_results=100, start=0, per_query=50, delay=3, track=False) -> list[dict]` returns arXiv papers list.
- `src/fetch_crossref.fetch_papers(queries: list[str], max_results=100, per_page=20, delay=3, track=False) -> list[dict]` returns Crossref papers list.
- `src/enrich_openalex.enrich(papers: list[dict], track=None) -> list[dict]` returns enriched papers list.
- `src/utils.deduplicate_papers_by_title_authors(papers: list[dict], paper_type="fetched") -> list[dict]` returns deduped list.
- `src/gpt_screener_initial.screen_paper(paper: dict, criteria: list[str], prompt_txt_path="prompts/screen_initial.txt") -> dict` returns screening result with `relevance_score`, `answers`, `raw`.
- `src/gpt_categories.build_taxonomy_categories(research_question: str, abstracts: list[str], ...) -> dict[str, str]` returns category map.
- `src/pdf_downloader.download_pdfs(papers: list, output_dir: str|Path) -> None` downloads PDFs in place (expects each item to have `paper` key; may need an adapter).
- `src/gpt_screener_full.build_prompt(question: str, categories: dict[str, str]) -> str` returns a prompt.
- `src/gpt_screener_full.call_gpt_pdf_from_path(prompt: str, pdf_path: Path) -> str` returns raw LLM output.
- `src/gpt_screener_full.parse_tagged_output(text: str, categories: list[str]) -> dict` returns full-screening JSON.

**Option 1: Minimal Wrapper (value + step log, explicit updated keys)**

```python
# app.py (helper)
import time
from datetime import datetime

def iso_now():
    return datetime.now().isoformat(timespec="seconds")

def run_step(run, step, name, func, updated_keys, *args, **kwargs):
    t0 = time.time()
    result = func(*args, **kwargs)
    elapsed = round(time.time() - t0, 2)

    run["updated_at"] = iso_now()
    run.setdefault("stats", {}).setdefault("timings_sec", {})[name] = elapsed

    step[name] = {
        "done": True,
        "updated_keys": updated_keys,
        "elapsed_sec": elapsed,
        "ts": iso_now(),
    }
    return result, step[name]
```

Usage (example):

```python
if not run["inputs"].get("queries"):
    print("")
    print_section("Expanding to queries...")
    run["inputs"]["queries"], step["queries"] = run_step(
        run,
        step,
        "boolean_to_queries",
        boolean_to_queries,
        updated_keys=["inputs.queries", "stats.timings_sec.boolean_to_queries"],
        boolean_query=run["inputs"]["boolean_query_used"],
        max_queries=50,
    )
    save_run(run, step)
```


**Run JSON Structure (core)**

```json
{
  "created_at": "2026-03-09T12:34:56",
  "updated_at": "2026-03-09T12:40:12",
  "stage": "initial_screening",

  "inputs": {
    "research_questions": "Line 1...\nLine 2...\nLine 3...",
    "boolean_query_suggested": "\"long video\" AND (retrieval OR search) AND (multimodal OR audio)",
    "boolean_query_used": "\"long video\" AND (retrieval OR search) AND (multimodal OR audio)",
    "queries": [
      "\"long video\" retrieval multimodal",
      "\"long video\" search audio"
    ],
    "criteria_suggested": [
      "INCLUDE: Does the study focus on AI systems for long-form video?",
      "EXCLUDE: Is it only short-form video?"
    ],
    "criteria_used": [
      "INCLUDE: Does the study focus on AI systems for long-form video?",
      "EXCLUDE: Is it only short-form video?"
    ]
  },

  "stats": {
    "timings_sec": {
      "generate_boolean_query": 6.2,
      "fetch_arxiv": 120.4,
      "fetch_crossref": 95.7,
      "enrich_openalex": 210.1,
      "initial_screening": 980.5,
      "top_selection": 0.4,
      "download_pdfs": 140.2,
      "build_categories": 18.7,
      "full_screening": 1880.6
    },
    "counts": {
      "queries": 12,
      "fetched_arxiv": 850,
      "fetched_crossref": 630,
      "deduped_total": 1260,
      "enriched_total": 1260,
      "screened_total": 1260,
      "top_selected": 50,
      "pdf_downloaded": 47,
      "full_screened": 47,
      "errors": 3
    }
  },
  

  "papers_by_id": {
    "10.1234/abc": {
      "paper_id": "10.1234/abc",
      "title": "Paper title",
      "authors": ["A. Author", "B. Author"],
      "abstract": "Abstract...",
      "published": "2024-05-01",
      "publisher": "Publisher",
      "doi": "10.1234/abc",
      "link": "https://doi.org/10.1234/abc",
      "from_query": "\"long video\" retrieval multimodal",

      "enrich": {
        "citations": 12,
        "openalex_id": "https://openalex.org/W123",
        "crossref_lookup": true
      },

      "screening": {
        "relevance_score": 6,
        "answers": [
          { "criterion": "INCLUDE: ...", "answer": "YES" },
          { "criterion": "EXCLUDE: ...", "answer": "NO" }
        ],
        "raw": "C1: YES\nC2: NO\n..."
    }
  },

"categories": {
    "Category 1": "Explanation...",
    "Category 2": "Explanation..."
    },

  "top_paper_ids": {
    "10.1234/abc":
        {   "title": "Paper title",
            "pdf_path": "_RUNS/2026-03-09T12-34-56/Paper title.pdf",

            "full_screening": {
                "included": true,
                "categories": {
                "Category 1": {
                    "paragraph": "Summary paragraph...",
                    "quotes": ["Quote 1", "Quote 2"]
                }
                }
            }
  
        }
  }

  "errors": [
    {
      "stage": "download_pdfs",
      "paper_id": "10.1234/abc",
      "error": "PDF not found"
    }
  ]
}
```

**Chronological Process**

1. **Startup / Resume**

   - Ask if user wants to resume from an existing `run.json`.
   - If yes: load it, show summary (research question, stage, counts), continue from next stage.
   - If no: create `_RUNS/<timestamp>/run.json` with `stage="init"`.
2. **Research Question**

   - Prompt multiline research question(s).
   - Save to `inputs.research_questions`.
3. **Boolean Query**

   - Generate boolean query with `src/gpt_research_q.build_boolean_query_from_questions`.
   - Show it, allow edit. If edited, use the new one.
   - Expand with `boolean_to_queries`.
   - If >50 queries and user presses Enter, reprompt until ≤50.
   - Save suggested and used boolean queries plus query list.
4. **Fetch + Enrich (background)**

   - In background: fetch arXiv + Crossref, dedupe, enrich via OpenAlex.
   - Save results into `papers_by_id` as they arrive (all updates in place).
5. **Inclusion / Exclusion Criteria**

   - Generate criteria via `src/gpt_criteria.build_criteria_from_question`.
   - Show line-based criteria and allow edits (empty input keeps).
   - Save suggested and used criteria lists.
6. **Initial Screening (parallel GPT streaming)**

   - For each paper in `papers_by_id`, run GPT initial screening in parallel (workers via `GPT_MAX_WORKERS` or CPU default).
   - Print `title — score` when each paper completes.
   - Store results in place at `papers_by_id[id].screening`.
7. **Top Selection (≤50)**

   - Bucket by `relevance_score` descending.
   - Add full buckets while total ≤50.
   - If top bucket alone >50, cap to first 50.
   - Save IDs to `top_paper_ids`.
8. **Download PDFs**

   - Download PDFs only for `top_paper_ids`.
   - Save `pdf_path` in each top paper entry.
9. **Categories**

   - Use abstracts from top papers to build taxonomy via `gpt_categories`.
   - Save categories map on top papers.
10. **Full PDF Screening**

- For each top paper with PDF, call `gpt_screener_full`.
- Store output in place at `papers_by_id[id].full_screening`.

11. **Finalize**

- Set `stage="done"` and finalize `stats`.

**Test Plan**

1. New run end-to-end with ≤50 queries.
2. > 50 queries triggers reprompt until ≤50.
   >
3. Resume mid-screening continues correctly.
4. Top bucket >50 caps to 50.
5. PDFs only for top papers, and `pdf_path` saved.
6. Full screening writes results into the same objects.

**Assumptions**

1. `paper_id = DOI if available; else hash(title+authors)`.
2. Only `papers_by_id` is authoritative; top set is represented by `top_paper_ids`.
3. PDFs and `run.json` live in the same run folder.
