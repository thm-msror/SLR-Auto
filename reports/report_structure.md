# ATLAS: Automated Tool for Literature Analysis and Synthesis

**DSAI4201 - Selected Topics in Data Science**  
**Human-guided Automated Systematic Literature Review Tool**

**Authors:**  
Joy Anne P. Dela Cruz (60301959@udst.edu.qa)  
Tehreem Masroor (60302531@udst.edu.qa)  
*University of Doha for Science and Technology*

---

## 1. Introduction & Motivation
Systematic Literature Reviews (SLRs) are the "gold standard" for evidence-based research, requiring a rigorous, repeatable process to identify and synthesize all relevant studies on a specific topic. However, the traditional SLR process is notoriously manual, tedious, and labor-intensive. A thorough review typically spans **4 to 12 weeks**, involving multiple stages: defining research questions, identifying papers, screening titles/abstracts, and finally synthesizing data to identify research gaps.

**ATLAS (Automated Tool for Literature Analysis and Synthesis)** addresses these challenges by automating the repetitive components of the SLR pipeline while maintaining research integrity through a **human-in-the-loop** design. By leveraging Large Language Models (LLMs) and scholarly APIs, ATLAS reduces weeks of manual work into a streamlined, automated process.

## 2. Problem Definition
The primary problem in current SLR workflows is the **scalability-thoroughness tradeoff**. Researchers often struggle to keep up with the rapidly growing volume of scientific literature, leading to biased or incomplete reviews. Existing tools like Rayyan or Covidence provide management platforms but still require the researcher to manually read and screen every abstract.

**Objectives of ATLAS:**
1. **Automate Discovery:** Programmatically fetch papers from high-quality databases (IEEE, Crossref, Semantic Scholar).
2. **Intelligent Screening:** Use LLMs to score papers against specific inclusion/exclusion criteria.
3. **Synthesis & Gap Analysis:** Automatically extract themes and generate a draft report and PRISMA diagram.
4. **Maintain Control:** Ensure the researcher reviews and approves every major decision (queries, criteria, themes).

## 3. AI Approach & Methodology
ATLAS employs a multi-agentic pipeline powered by **GPT-4o**, chosen for its superior context window, speed, and native PDF processing capabilities.

### 3.1 The Pipeline Steps
1.  **Boolean Query Generation:** The system translates the user's Research Question into complex Boolean search strings.
2.  **Multi-Source Fetching:** Data is retrieved via APIs, with metadata enrichment from **OpenAlex** to fill gaps in abstracts or author details.
3.  **Heuristic Screening:** The LLM evaluates abstracts against user-defined criteria, assigning a **Relevancy Score (RS)**.
4.  **Taxonomy Discovery:** The system analyzes the top-ranked abstracts to suggest research themes.
5.  **Full-Text Synthesis:** The LLM reads the full PDF content of the most relevant papers, extracting evidence-based insights for each theme.

### 3.2 Prompting Strategies
To ensure stability and accuracy, ATLAS uses specialized prompting techniques:
-   **Zero-Shot / One-Shot:** Used for structured data extraction.
-   **Chain-of-Thought:** Employed during full-screening to ensure the model justifies its decisions with supporting quotes.
-   **Deterministic Settings:** The model temperature is set to `0` for screening and extraction to eliminate hallucinations and ensure reproducibility.

## 4. System Implementation
ATLAS is built using **Streamlit** for a responsive, interactive frontend. The backend is a modular Python architecture:
-   **Fetching Layer:** Individual modules for IEEE Xplore, Semantic Scholar, and Crossref.
-   **Screening Layer:** Logic to handle bulk processing of papers against LLM-driven criteria.
-   **Reporting Layer:** Automated generation of a **PRISMA 2020 diagram (SVG)**, an Excel report, and a Markdown SLR draft.

## 5. Results & Discussion
The implementation successfully demonstrates a full end-to-end SLR. 
-   **Efficiency:** Preliminary runs show that screening 100 papers takes minutes compared to hours of manual effort.
-   **Accuracy:** The use of **RS (Relevancy Score)** provides a transparent ranking system.
-   **Transparency:** The generated PRISMA diagram provides the quantitative evidence required for publishing systematic reviews.

## 6. Conclusion
ATLAS bridges the gap between manual rigor and AI efficiency. By automating the "drudge work" of SLR—fetching, filtering, and initial drafting—it allows researchers to focus on high-level analysis and critical thinking. Future work includes integrating more databases (e.g., Scopus, PubMed) and improving the proxy-based PDF downloader for broader access.

---

## References
1. Kitchenham, B., & Charters, S. (2007). *Guidelines for performing Systematic Literature Reviews in Software Engineering*.
2. Page, M. J., et al. (2021). *The PRISMA 2020 statement*. BMJ.
3. Syriani, E., et al. (2024). *Screening articles for systematic reviews with ChatGPT*. Journal of Computer Languages.
