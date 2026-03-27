# ATLAS Frontend: Features & Implementation

This document explains the user interface of ATLAS, built using **Streamlit**. It details the interactive features, state management, and custom components used to provide a seamless Systematic Literature Review (SLR) experience.

## 1. Interface Overview
The ATLAS interface is designed as a **progressive pipeline**. Each section expands as the previous one is completed, guiding the user through the complex SLR journey.

### Key Sections:
1.  **Research Question:** The starting point.
2.  **Initial Search:** Handles boolean query generation and multi-source API fetching.
3.  **Initial Screening:** Displays results in a table and allows LLM-based filtering.
4.  **Paper Reading:** Manages PDF access (via Proxy) and theme identification.
5.  **Review Synthesis:** Generates the final project draft and PRISMA diagram.

---

## 2. Interactive Features

### 2.1 The "Human-in-the-Loop" Design
Unlike "black-box" AI tools, ATLAS prioritizes user control.
-   **Editable Suggestions:** Every LLM-generated output (Boolean queries, screening criteria, and research themes) is presented in a `st.text_area`.
-   **Confirmation Buttons:** Users must explicitly confirm these items before the pipeline proceeds. This ensures the researcher remains the "pilot" of the review.

### 2.2 Relevancy Score (RS) Column
The screening table features a **Relevancy Score (RS)** column.
-   This score is dynamically calculated by the LLM based on user-approved inclusion/exclusion criteria.
-   It uses a "point-based" heuristic (e.g., +1 for meeting inclusion criteria, -1 for exclusion violations).
-   This provides a fast, visual way for researchers to identify the most promising papers in a large batch.

### 2.3 PRISMA 2020 Visualization
ATLAS uses a custom SVG component to render the **PRISMA flow diagram**.
-   **Integration:** Uses `streamlit.components.v1.html` to inject SVG code directly into the UI.
-   **Dynamic Updates:** The diagram updates as papers are identified, screened, and finally "Included."
-   **Downloadable:** Users can download the diagram as a high-quality SVG file for their research papers.

### 2.4 Resume Session (Log Persistence)
-   Users can upload a `log.json` from a previous session using `st.file_uploader`.
-   The app parses this log and restores the exact state of the pipeline, allowing researchers to pick up where they left off without re-running expensive LLM calls.

---

## 3. Technical Implementation

### 3.1 Session State Management
ATLAS heavily utilizes `st.session_state` to track the "stage" of the SLR.
-   **Flags:** `started`, `queries_confirmed`, `criteria_confirmed`, etc.
-   **Persistence:** The `run` dictionary (stored in session state) is synchronized with a local `log.json` file on every major interaction.

### 3.2 UI Modalities (Normal vs. Fast)
The app supports a `--mode` flag (handled via `argparse`):
-   **Normal:** Full-scale SLR (up to 100 papers per source).
-   **Fast:** Quick demonstration mode (limited to 5 papers/queries), perfect for presenting or testing.

### 3.3 Access Helper (Proxy Integration)
Since many PDF databases are paywalled, the UI provides a "Download Access Helper" button.
-   This generates a zip file containing a script that the user runs in their browser (via a university proxy or manual login).
-   The user then uploads a `session.json` file to ATLAS, enabling the tool to "impersonate" the researcher’s session and fetch full-text PDFs.

---

## 4. Design Aesthetics
ATLAS follows modern web aesthetics:
-   **Vibrant UI:** Custom CSS for the PRISMA diagram.
-   **Micro-animations:** Loading spinners and status columns.
-   **Layout:** Wide-mode layout for data-intensive tables.
-   **Typography:** Clear, hierarchical headers and helpful captions.
