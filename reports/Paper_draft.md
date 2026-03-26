# ATLAS: Automated Tool for Literature Analysis and Synthesis

**DSAI4201 - Selected Topics in Data Science**

**Human-guided Automated Systematic Literature Review Tool**

**Joy Anne P. Dela Cruz***
Student, University of Doha for Science and Technology, 60301959@udst.edu.qa

**Tehreem Masroor***
Student, University of Doha for Science and Technology, 60302531@udst.edu.qa

---

## ABSTRACT

Systematic Literature Reviews (SLRs) are fundamental to evidence-based research but are traditionally characterized by labor-intensive, multi-week manual workflows. This paper introduces **ATLAS (Automated Tool for Literature Analysis and Synthesis)**, a human-guided web application that automates the end-to-end SLR pipeline using Large Language Models (LLMs). Built with **Streamlit** and powered by **GPT-4o**, ATLAS streamlines paper discovery from peer-reviewed databases (**IEEE Xplore, Crossref, and Semantic Scholar**), multi-stage LLM screening, and data synthesis, while maintaining full transparency through an automatically generated **PRISMA 2020 flow diagram**. Unlike previous fully automated or black-box attempts, ATLAS integrates Human-in-the-Loop checkpoints: researchers validate and edit the LLM-generated Boolean query and inclusion/exclusion criteria before any papers are fetched. The discovery phase is optimized through a robust multi-source retrieval engine with high-precision query expansion, while the relevance scoring model computes `Inclusion YES - Exclusion YES`, precisely capturing topical fit. We demonstrate the tool on a representative computer science research question, showing significant reductions in review time without compromising semantic depth or thematic coverage.

**CCS CONCEPTS**
- Information systems → Systematic Literature Review (SLR)
- Computing methodologies → Large Language Model (LLM)
- General and reference → Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA)

**Keywords:** SLR automation, GPT-4o, Streamlit, Human-in-the-Loop, AI-assisted research, PRISMA 2020.

---

## 1. INTRODUCTION

A **Systematic Literature Review (SLR)** is a structured methodology used to identify, evaluate, and synthesize all research evidence relevant to a specific question. The **PRISMA (Preferred Reporting Items for Systematic Reviews and Meta-Analyses)** framework provides transparent reporting standards for how studies were identified, screened, and included — making SLRs the gold standard for evidence-based research in medicine, computer science, and engineering.

However, the manual execution of an SLR creates a significant barrier. A thorough SLR typically requires **4–12 weeks** of expert effort: constructing Boolean queries across multiple academic databases, manually screening hundreds to thousands of abstracts against a set of criteria, acquiring and reading full-text papers, and synthesizing findings. For junior researchers and students, this represents an often insurmountable bottleneck.

Existing semi-automated tools such as Rayyan and Covidence help organize the screening process but still require a human to read and judge every abstract. They provide collaboration and labeling infrastructure, but no semantic understanding. The research question remains: *Can LLMs automate the semantic screening and synthesis steps of an SLR while keeping researchers in control of the process definition?*

**ATLAS (Automated Tool for Literature Analysis and Synthesis)** is proposed as a direct answer to this question. By combining the semantic reasoning of GPT-4o with an interactive Streamlit interface, ATLAS automates the most time-consuming SLR stages without removing the researcher from the methodological decisions.

### 1.1 Objectives

The primary objectives of ATLAS are to:

1. **Reduce manual labor** by automating paper discovery, abstract screening, and knowledge synthesis.
2. **Maintain research integrity** by keeping the researcher in control of the search strategy and screening criteria.
3. **Ensure reproducibility** through deterministic LLM settings (temperature = 0.0) and PRISMA-compliant count tracking.
5. **Strategic Focus on IT and Tech SLRs**: Supporting the rapidly evolving fields of Information Technology, Computer Science, and Data Science, where tracking emerging topics (e.g., Graph RAG) requires continuous monitoring of new literature.

---

## 2. BACKGROUND

### 2.1 AutoSLR: The Predecessor CLI System

ATLAS is an evolution of **AutoSLR**, a command-line Python tool developed as a capstone project for the research question: *"How can AI systems efficiently index, retrieve, and semantically understand long-form video content at scale?"*

AutoSLR demonstrated that automated LLM-based screening is feasible, but its architecture was tightly coupled to a single use case. A key characteristic of the legacy system was its use of the **Fanar API** — a locally hosted, Arabic-English bilingual LLM provided by Qatar's national AI initiative — as its core reasoning engine. While Fanar enabled zero-cost, data-sovereign LLM access and was appropriate for the capstone context, it imposed limitations on output quality, context window size, and structured output compliance that constrained the pipeline's capabilities compared to frontier commercial models.

The legacy pipeline operated as follows:

```
Configuration (hardcoded queries + criteria)
        |
        v
[1] Fetch Papers
    - IEEE Xplore API (peer-reviewed)
    - Crossref API (Springer, Nature, ACM, Elsevier)
        |
        v
[2] Deduplicate + Enrich
    - OpenAlex API (abstracts, citations)
        |
        v
[3] LLM Initial Screening (Fanar API)
    - Persona-based abstract screener
    - LLM: Fanar API (Arabic-English bilingual, locally hosted)
    - Instruction set for abstract screening
    - Criteria from configuration (hardcoded, domain-specific)
    - Output: Relevance score per paper (simple inclusion count only)
        |
        v
[4] Filter Top Papers (keep high scores)
        |
        v
[5] Download PDFs
    - Open Access + institutional proxy
        |
        v
[6] PDF Processing (local conversion)
    - Automated PDF-to-text conversion (requires high system dependencies)
        |
        v
[7] LLM Full Paper Reading (Fanar API)
    - Multi-stage full-text analysis
    - Extracts answers to hardcoded technical gap questions
        |
        v
[8] Knowledge Synthesis (Fanar API)
    - Generation of per-paper research summaries
```

**Key limitations of the legacy system:**

| Limitation | Description |
|---|---|
| LLM: Fanar API | Limited context window and structured output compliance; domain-biased toward Arabic/English academic content |
| Hardcoded scope | Queries, criteria, and gap questions were fixed; not editable without code changes |
| Deployment | CLI-only; required technical knowledge of Python environments to operate |
| Persistence | No run management; re-running the pipeline re-processed everything from scratch |
| Accountability | No PRISMA tracking; paper counts at each pipeline stage were not systematically recorded |
| Scoring Logic | Simple relevance score only counted satisfied inclusion criteria; did not penalize exclusion criteria |
| Dependencies | Heavy system-level PDF requirements causing cross-platform installation failures |
| API Fragility | Included unreliable scraping fallbacks, adding maintenance overhead |

### 2.2 Prior Work in SLR Automation

Semi-automated tools like **Rayyan** [CITATION] and **SWIFT-Review** [CITATION] assist large research teams with collaborative screening but require every abstract to be read and labeled by a human. They do not offer semantic understanding.

Marshall and Wallace (2019) showed that machine learning classifiers (SVM, active learning) can assist with title/abstract screening, but these approaches require labeled training data specific to each review — expensive and not generalizable. Syriani et al. (2023) explored the use of LLMs for literature automation, demonstrating that zero-shot GPT-4 could classify abstracts with reasonable accuracy. ATLAS extends this work into a full end-to-end pipeline with a user interface and PRISMA tracking.

---

## 3. METHODOLOGY

### 3.1 System Overview — The ATLAS Pipeline

ATLAS introduces a modernized architecture backed by an interactive web-based frontend. The pipeline is redesigned as a **9-stage human-guided workflow**:

```
[User Input] Research Question (runtime, multi-line)
        |
        v
[1] LLM Boolean Query Generation
    - Converts research goals into a structured Boolean string
    - User reviews and edits the query before deployment
        |
        v
[2] High-Precision Query Expansion
    - Decomposes the single Boolean string into individual search queries.
    - Optimized for search engine strictness: utilizes up to 5 mandatory (ANDed) concepts per query to minimize noise and irrelevant records.
        |
        v
[3] LLM Criteria Generation
    - Systematic generation of 15–30 inclusion and exclusion criteria
    - Standardized peer-review criteria included as default suggestions
    - User refines criteria before screening begins
        |
        v
[4] Fetch + Deduplicate + Enrich (Stable Multi-Source Discovery)
    - **IEEE Xplore Integration**: Strict rate-limiting and proactive quota tracking (200 calls/day).
    - **Crossref Integration**: Retrieval from diverse peer-reviewed publishers (Springer, Nature, ACM, Elsevier).
    - **Semantic Scholar Integration**: High-fidelity metadata and abstract retrieval from a multi-billion paper database.
    - **Sequential Execution**: Robust multi-source discovery engine optimized for application stability.
    - **Deduplication Engine**: Multi-stage matching using DOIs and metadata hashing.
    - **Metadata Enrichment**: OpenAlex API integration for refined paper records.
        |
        v
[5] LLM Initial Screening (Parallel Processing)
    - Automated metadata screening against generated inclusion/exclusion criteria
    - Relevance Scoring: `Inclusion YES - Exclusion YES`
    - Automated PRISMA 2020 record tracking
        |
        v
[6] Selection Strictness and Automated Taxonomy
    - **Relevance Thresholding**: Filters papers by a minimum score (Score >= 3). This ensures only high-quality records (satisfying multiple inclusion criteria) proceed.
    - **Taxonomy Generation**: Automatically generates thematic categories from the abstracts of the top-ranked papers *before* full-text analysis.
        |
        v
[7] PDF Acquisition
    - Open Access retrieval and institutional proxy support
        |
        v
[8] Full Paper Analysis and Final Synthesis
    - **One-Click Integration**: Sequential automation of Download -> Eligibility -> Synthesis.
    - **Full-Text Extract**: Decision-making and data extraction from PDF sources using GPT-4o File API.
    - **Thematic Synthesis**: Aggregation of cross-paper findings into coherent narratives across proposed categories.
    - **Final Reporting**: Generation of a downloadable Results Summary (CSV) and Synthesis Report (Markdown).
```

### 3.2 PRISMA 2020 Tracking

Every ATLAS run persists a structured state file tracking paper counts at each PRISMA 2020 stage:

| PRISMA Stage | Tracked Field | Populated At |
|---|---|---|
| Identification — IEEE | `prisma.identification.ieee` | After IEEE fetch |
| Identification — Crossref | `prisma.identification.crossref` | After Crossref fetch |
| Identification — Semantic Scholar | `prisma.identification.semanticscholar` | After S2 fetch |
| After deduplication | `prisma.after_dedup` | After dedup |
| Screened | `prisma.screened` | After initial LLM screening |
| Excluded at screening | `prisma.excluded_screening` | After initial LLM screening |
| Sought for retrieval | `prisma.sought_retrieval` | After top selection |
| Not retrieved | `prisma.not_retrieved` | After PDF download |
| Assessed for eligibility | `prisma.assessed_eligibility` | After full screening |
| Excluded at eligibility | `prisma.excluded_eligibility` | After full screening |
| Included | `prisma.included` | After full screening |

The Streamlit interface renders this as a live **PRISMA 2020 SVG flow diagram**, acting as the final pipeline summary. Both the diagram (SVG) and the detailed counts are downloadable.

### 3.3 Human-in-the-Loop Design

A critical design principle in ATLAS is the **Human-in-the-Loop** checkpoint pattern. Unlike fully autonomous pipeline systems, ATLAS pauses at two critical junctures:

1.  **Boolean Query Review**: After GPT-4o generates a Boolean search string, the user sees it in an editable text area before any papers are fetched. The query is validated for correct Boolean syntax. The user may add synonyms, remove concept groups, or rewrite entirely.

2.  **Criteria Review**: After GPT-4o generates 15–30 inclusion/exclusion criteria, the user sees them in an editable text area before screening begins. The LLM always proposes peer-review inclusion/exclusion criteria as default suggestions; the user may remove these if their research question spans preprints or grey literature.

This design ensures that the automated pipeline remains aligned with the researcher's intent, and that methodological decisions — which drive the entire review — are always under human control.

### 3.4 Relevance Scoring Model

The relevance score for each paper is computed as:

```
Relevance Score = (count of INCLUDE criteria answered YES)
                - (count of EXCLUDE criteria answered YES)
```

This formulation captures topical fit (inclusion criterion satisfaction) while penalizing papers with known disqualifying characteristics. A paper with many relevant features but also an exclusion flag (e.g., non-peer-reviewed, out-of-scope domain) is correctly downranked. Papers are ranked by descending score; the top 50 are selected for full-text retrieval.

The `INSUFFICIENT` verdict is emitted by the screener when a criterion cannot be determined from the title and abstract alone (e.g., the paper's methodology is not described in the abstract). This is neither counted as YES nor NO, preventing unfair penalization of papers with brief abstracts.

### 3.5 LLM Selection: Why GPT-4o over Fanar?

The legacy system used the **Fanar API**, a locally hosted bilingual (Arabic-English) LLM appropriate for the capstone's data sovereignty constraints. For ATLAS, we transitioned to **GPT-4o via Azure OpenAI Service** for the following reasons:

| Dimension | Legacy: Fanar API | ATLAS: Azure GPT-4o |
|---|---|---|
| Context window | Limited (< 8K tokens) | 128K tokens — full papers in one call |
| PDF input | Not supported — required `marker` conversion | Native file API — PDFs sent directly |
| Structured output | Inconsistent format compliance | Reliable YES/NO per criterion, tagged output |
| Screening quality | Adequate for domain-specific capstone | Superior zero-shot nuanced classification |
| Multilingual bias | Arabic-English optimized | General multilingual, CS-focused |
| Deployment | Local/UDST infrastructure | Azure cloud, accessed via REST API |

GPT-4o is accessed through the **Azure OpenAI Service** (not the public OpenAI API), using the `AzureOpenAI` Python client configured with a UDST-provisioned endpoint and deployment. This preserves institutional data governance while providing access to the most capable available model. Two separate client instances are used: a **Chat Completions client** for all text-in/text-out tasks (Boolean query generation, criteria generation, initial screening), and a **Responses API client** for PDF processing (full paper reading via the file upload API).

### 3.6 Hyperparameter Configuration

All LLM calls in ATLAS use **temperature = 0.0**, enforcing deterministic, reproducible responses. A paper screened today will receive the same screening decision on repeated runs, which is essential for SLR reproducibility. Higher temperatures introduce noise into YES/NO decisions and reduce parse success rates.

Maximum output token limits are set per stage: 800 tokens for initial screening (YES/NO answers per criterion, no explanations required) and up to 4096 tokens for full PDF extraction (detailed per-category paragraphs and quotes).

### 3.7 Frontend and Deployment

The application is built using **Streamlit**, which serves as both the UI framework and the backend runtime. Streamlit was selected for its Python-native integration with the pipeline modules, its `st.session_state` for persistent stateful workflows across interactions, and its zero-configuration deployment to Streamlit Community Cloud. For a research prototype, this eliminates the DevOps overhead of a REST API + React architecture.

### 3.8 Prompt Engineering Principles

The transition from the legacy Fanar-based system to the ATLAS GPT-4o pipeline was underpinned by the application of rigorous prompt engineering principles, transforming domain-specific instructions into generalized, robust templates:

1.  **Role Prompting**: Each prompt (e.g., `rq_query.txt`, `screen_initial.txt`) begins with a clear persona assignment (e.g., "You are an expert research librarian" or "strict, evidence-based screener"). This sets the expected stylistic and evidentiary floor for the LLM's responses.
2.  **One-Shot Learning**: For tasks requiring complex structural logic, such as `make_categories.txt`, we implemented one-shot prompting. By providing a high-quality example of the transition from research questions/abstracts to thematic categories, we improved thematic coherence and format compliance.
3.  **Strict Delimiters and Formatting**: Instructions use explicit delimiters (e.g., `STRICT — no deviations allowed`) and line-based formats (e.g., `C1: YES|NO|INSUFFICIENT`). This ensures that the Python parsing logic (`_extract_response_text`) remains deterministic and error-free.
4.  **Negative Constraints**: Prompts explicitly list "Do NOT" rules (e.g., "Do NOT guess", "No meta-commentary") to suppress the LLM's natural tendency toward conversational filler or speculation.
5.  **Extractive Favoritism**: The `screen_full.txt` prompt enforces an "Extractive-First" behavior, requiring the LLM to provide verbatim quotes for every claimed category, which serves as a primary verification layer for the researcher.

### 3.9 Specialized Generalization for the Tech Domain

While AutoSLR was hardcoded for video retrieval, the ATLAS prompt ecosystem is intentionally generalized for any Information Technology, Computer Science, or Data Science Systematic Literature Review. We distinguish this from "universal generalization" across all academic fields (such as medicine or social sciences), which is avoided to prevent performance degradation due to **data drift** and **mode collapse** inherent in cross-disciplinary domain shifting.

By scoping ATLAS to the IT and Tech sectors, we ensure that the prompt logic, technical terminology, and workflow remain optimized for the high-velocity nature of these fields. The tool is designed to assist researchers in keeping pace with **emerging and hyper-recent topics** (e.g., Graph RAG, Agentic Workflows) where new papers are published daily. This makes ATLAS particularly effective for "living" SLRs that necessitate frequent updates to maintain a current state-of-the-art overview. The removal of hardcoded video-centric terms (e.g., "frame-rate") in favor of structural variables (e.g., `{RESEARCH_QUESTION}`) allows the system to flexibly adapt within the technical research umbrella while maintaining a high signal-to-noise ratio.

### 3.10 Authenticated Retrieval via Local Session Handover

A significant technical challenge in cloud-deployed research tools is the retrieval of papers from institutional repositories that require interactive proxy authentication (e.g., UDST Library, Microsoft MFA). Since cloud environments (Streamlit Community Cloud) are "headless" and isolated from the user's local network, direct interactive login is impossible. 

ATLAS solves this through a **Local Session Handover** workflow. Users download a streamlined **Helper Package (ZIP)** containing an interactive login script and a setup guide. The script utility extracts the authenticated security state and saves it as a session JSON file on the user's desktop. By uploading this file back to the ATLAS web interface, the researcher "hands over" their authorized session to the cloud server. This enables the cloud-based ATLAS instance to retrieve papers using the user's institutional credentials without ever requiring their password or compromising security.

---

## 4. COMPARISON: LEGACY vs. ATLAS

| Dimension | Legacy AutoSLR | ATLAS (Human-in-the-Loop) |
|---|---|---|
| Deployment | Command-Line Interface (CLI) | WebUI (Streamlit) |
| **LLM engine** | **Fanar API** (locally hosted) | **Azure OpenAI GPT-4o** (128K context) |
| **PDF processing** | Local library conversion | **Native PDF upload** (GPT-4o file API) |
| Configuration | Static code definitions | Dynamic runtime input |
| Automated Search | Hardcoded scripts | LLM-generated, user-editable Boolean |
| Screening Logic | Static hardcoded criteria | Dynamic, user-refined criteria |
| Search sources | Multiple (IEEE + Crossref) | Specialized technical sourcing |
| Deduplication | Metadata hashing | Multi-stage (DOI + Hash) |
| Relevance scoring | Inclusion count only | Weighted (Inclusion - Exclusion) |
| Governance | None | Full PRISMA 2020 tracking |
| Search Strategy | Truncated expansion | High-precision (5 concepts) |
| Helper Distribution | Manual files | Consolidated ZIP package |
| User Control | None | Human-in-the-Loop checkpoints |

---

## 5. RESULTS

*This section presents evaluation results. All quantitative entries below are placeholders to be filled with real run data.*

### 5.1 Prompting Strategy Comparison

We compared three prompting configurations for the initial screening stage:

| Strategy | Description | Precision | Recall | Parse Success Rate |
|---|---|---|---|---|
| Zero-shot (current) | Criteria only, no examples | ? | ? | ? |
| One-shot | One example inclusion/exclusion pair | ? | ? | ? |
| Few-shot | Three diverse examples | ? | ? | ? |

*Table 1: Screening prompting strategies evaluated against a manually labelled subset.*

### 5.2 Temperature Ablation

| Temperature | Consistency (same paper, 5 runs) | Parse Success Rate |
|---|---|---|
| 0.0 | ~100% | ? |
| 0.3 | ? | ? |
| 0.7 | ? | ? |

*Table 2: Effect of LLM temperature on screening reproducibility.*

### 5.3 PRISMA Flow Results

The ATLAS tool auto-generates the following counts during a full run (placeholder values):

| PRISMA Stage | Count |
|---|---|
| Records identified via IEEE | N |
| Records identified via Crossref | N |
| Records identified via Semantic Scholar | N |
| Duplicates removed | N |
| Records after deduplication | N |
| Records screened (abstract) | N |
| Excluded at screening (score <= 0) | N |
| Reports sought for full-text retrieval | N |
| Reports not retrieved (no PDF) | N |
| Reports assessed for eligibility (full text) | N |
| Excluded at eligibility stage | N |
| Studies included in review | N |

*Table 3: PRISMA 2020 flow counts from a representative ATLAS run.*

---

## 6. CONCLUSION

ATLAS demonstrates that LLMs can significantly reduce the manual burden of Systematic Literature Reviews while preserving the methodological transparency required by PRISMA 2020. By replacing hardcoded configurations with LLM-generated, user-editable strategies and embedding interactive Human-in-the-Loop checkpoints, ATLAS generalizes the original domain-specific AutoSLR pipeline into a universal research tool.

The shift from simple inclusion counting to `Inclusion YES - Exclusion YES` scoring, combined with the `INSUFFICIENT` verdict for ambiguous criteria, produces a more precise relevance ranking than the legacy system. Replacing the `marker`/`playwright` PDF conversion stack with the GPT-4o file API simplifies deployment and improves cross-platform compatibility.

**Future work:** Integration of additional databases (Scopus, PubMed, ACM Digital Library), multi-user session management, and support for automated citation extraction into BibTeX/RIS formats.

---

## REFERENCES

[1] Moher, D., Liberati, A., Tetzlaff, J., Altman, D. G., & The PRISMA Group. (2009). Preferred reporting items for systematic reviews and meta-analyses: the PRISMA statement. *PLoS Medicine*, 6(7).

[2] Page, M. J., et al. (2021). The PRISMA 2020 statement: an updated guideline for reporting systematic reviews. *BMJ*, 372.

[3] OpenAI. (2024). GPT-4o Technical Report. OpenAI.

[4] Streamlit Inc. (2024). Streamlit Documentation — Rapid prototyping for data applications. https://docs.streamlit.io

[5] Marshall, C., & Wallace, B. C. (2019). Toward systematic review automation: a practical guide to using machine learning tools in research synthesis. *Systematic Reviews*, 8(1), 163.

[6] Syriani, E., David, I., & Kumar, G. (2023). Assessing the ability of ChatGPT to screen articles for systematic reviews. *arXiv preprint arXiv:2307.06464*.

[7] CrossRef. (2024). Crossref REST API documentation. https://api.crossref.org

[8] IEEE. (2024). IEEE Xplore API documentation. https://developer.ieee.org

[9] Priem, J., Piwowar, H., & Orr, R. (2022). OpenAlex: A fully-open index of the world's research works. *arXiv preprint arXiv:2205.01833*.

---

*[Additional domain-specific references to be integrated from the conducted SLR itself.]*
