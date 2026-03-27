# ATLAS: Automated Tool for Literature Analysis and Synthesis
## Slide-by-Slide Presentation Outline (15 Minutes)

### 1. Title Slide
- **Title:** ATLAS: Human-Guided Automated Systematic Literature Review
- **Subtitle:** DSAI4201 - Selected Topics in Data Science
- **Presented by:** Joy Anne P. Dela Cruz & Tehreem Masroor
- **Date:** March 2026

---

### 2. The Problem: The "SLR Drudge Work"
- **Context:** Systematic Literature Reviews (SLRs) are the gold standard for research.
- **Pain Points:** 
  - Manual, tedious, and labor-intensive (4–12 weeks).
  - High volume of papers to screen (often 1000+ initially).
  - Prone to human error and fatigue.
- **Why it matters:** In fast-evolving fields like AI/IT, manual reviews are outdated by the time they are published.

---

### 3. Our Solution: ATLAS
- **Overview:** An automated tool that mimics a human researcher’s workflow.
- **Key Philosophy:** Human-in-the-Loop.
  - The AI suggests, the human approves/edits.
  - Transparent, repeatable, and scalable.
- **Tech Stack:** GPT-4o, Streamlit, Scholarly APIs (IEEE, Semantic Scholar, Crossref).

---

### 4. System Architecture: The Pipeline
- **Visual:** [Insert Mermaid Flowchart from Technical Guide]
- **Five Major Steps:**
  1. **Query Generation:** Research Question -> Boolean Query.
  2. **Fetch:** Programmatic retrieval from 3+ databases.
  3. **Screen:** LLM-based abstract grading (Inclusion/Exclusion).
  4. **Read:** PDF downloading and taxonomy (theme) extraction.
  5. **Synthesize:** Evidence-based draft report generation.

---

### 5. Detailed AI Approach
- **LLM Selection:** Why GPT-4o? (Context window, PDF handling, speed).
- **Prompting Strategies:** 
  - Zero-shot/One-shot for extraction.
  - Chain-of-Thought for complex screening.
- **Stability:** Deterministic settings (Temperature = 0) to prevent hallucinations.

---

### 6. Demo / Visual Showcase: PRISMA Diagram
- **Highlight:** Automatic PRISMA 2020 Flow Diagram.
- **Benefit:** Instantly shows the paper selection journey (Identification -> Screening -> Eligibility -> Included).
- **Trust:** Maintains rigorous count tracking at every stage.

---

### 7. Results: Efficiency Gained
- **Manual vs. ATLAS:**
  - Manual Screening: Hours/Days per batch.
  - ATLAS Screening: Minutes per batch.
- **Output:**
  - Full SLR Markdown Draft.
  - Formatted Excel Report (Audit Trail).
  - SVG PRISMA Diagram.

---

### 8. Handling "Human-in-the-Loop"
- **Interaction points:**
  - Editing Boolean queries before searching.
  - Modifying screening criteria before filtering.
  - Approving research themes before final extraction.
- **Outcome:** High research integrity with low manual effort.

---

### 9. Challenges & Limitations
- **PDF Access:** Relying on proxy helpers for paywalled content.
- **LLM Quotas:** Managing API token limits for large-scale reviews.
- **Hallucinations:** Mitigated by low temperature and human oversight.

---

### 10. Conclusion & Future Work
- **Summary:** ATLAS transforms SLRs from a burden into a fast, manageable tool.
- **Next Steps:**
  - Adding more database adapters (Scopus, PubMed).
  - Multi-agent collaboration for cross-review verification.
  - Advanced sentiment and trend analysis for synthesis.

---

### 11. Q&A Session
- "Thank you for your attention. We are happy to answer any questions about our AI methodology or system design."
