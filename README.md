# 📊 SLR Automation Pipeline

### **1. Define Research Questions & Keywords**

`query → API → JSON → LLM tagging`

* **Input:**  SLR question(s) → e.g., *“What are the applications of Graph Neural Networks in healthcare?”*
* **Output:** A list of **queries/keywords** →

  ```json
  ["graph neural networks", "healthcare", "medicine", "GNN applications"]
  ```

---

### **2. Collect Papers (via APIs)**

* **Input:** Queries from step 1.
* **Process:** Call APIs like Semantic Scholar, CrossRef, PubMed, or arXiv.
* **Output:** Raw metadata JSON for each paper, e.g.:

  ```json
  {
    "title": "Graph Neural Networks in Healthcare",
    "authors": ["Smith, J.", "Lee, K."],
    "abstract": "This paper explores applications of GNNs in drug discovery...",
    "doi": "10.1000/xyz123",
    "year": 2023,
    "url": "https://doi.org/10.1000/xyz123",
    "citations": 54
  }
  ```

---

### **3. Store & Clean Metadata**

* **Input:** Raw API JSON.
* **Process:**

  * Remove duplicates (based on DOI/title).
  * Normalise fields (consistent keys, clean strings).
* **Output:** A structured JSON database, e.g.:

  | title                               | year | doi            | abstract                          | citations |
  | ----------------------------------- | ---- | -------------- | --------------------------------- | --------- |
  | Graph Neural Networks in Healthcare | 2023 | 10.1000/xyz123 | This paper explores applications… | 54        |

---

### **4. Apply Inclusion/Exclusion Criteria**

* **Input:** Clean dataset of papers.
* **Process:** Filter by:

  * Publication year range
  * Language (English only, etc.)
  * Domain relevance (can use keyword matching or LLM check)
* **Output:** Smaller dataset of **candidate papers** (e.g., 500 → 120).

---

### **5. LLM-Assisted Screening**

* **Input:** Candidate paper abstracts.
* **Process:** Send each abstract into an LLM with a prompt like:

  > “Given this abstract, decide if it addresses *applications of GNNs in healthcare*. Respond with YES or NO.”
* **Output:** Labels for each paper, e.g.:

  ```json
  {
    "doi": "10.1000/xyz123",
    "include": "YES",
    "reason": "Paper explicitly mentions healthcare applications of GNNs"
  }
  ```

---

### **6. Information Extraction**

* **Input:** Included abstracts.
* **Process:** Use LLM to extract structured attributes:

  * **Problem domain**
  * **Methodology**
  * **Datasets used**
  * **Findings & limitations**
* **Output:** JSON records per paper, e.g.:

  ```json
  {
    "doi": "10.1000/xyz123",
    "methodology": "Graph Neural Network",
    "application": "Drug discovery",
    "dataset": "DrugBank",
    "findings": "Improved prediction accuracy by 15%",
    "limitations": "Limited generalisability to other diseases"
  }
  ```

---

### **7. Aggregation & Analysis**

* **Input:** Extracted structured dataset.
* **Process:**

  * Group by themes (e.g., “Drug discovery”, “Medical imaging”, “EHR analysis”).
  * Count frequencies (e.g., 40% use GNNs for drug discovery).
  * Identify trends (years, citations).
* **Output:** A **summary dataset** for analysis, e.g.:

  | Application     | Papers | Most Common Dataset | Avg Citations |
  | --------------- | ------ | ------------------- | ------------- |
  | Drug discovery  | 25     | DrugBank            | 60            |
  | Medical imaging | 18     | ImageNet, MRI scans | 45            |
  | EHR analysis    | 12     | MIMIC-III           | 30            |

---

### **8. Final Synthesis**

* **Input:** The aggregated results + extracted details.
* **Process:** Write structured summaries:

  * What’s been done?
  * Where are the gaps?
  * How has the field evolved?
* **Output:**

  * A **narrative SLR report**
  * Visuals (bar charts, timelines, word clouds)
  * Exported dataset (CSV/JSON) for reproducibility.

---

⚡ In short:

* **Raw input:** Research question → queries → abstracts
* **Pipeline:** API → clean → filter → LLM screening → LLM extraction → aggregation
* **Final output:** A structured evidence base + synthesis you can use in your SLR.
