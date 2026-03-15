# ATLAS Execution Instructions

This guide details how to set up and run the ATLAS Automated SLR tool in various environments.

## 1. Local Environment Setup

ATLAS requires Python 3.9+ and several system-level dependencies.

### Step A: Create a Virtual Environment
It is highly recommended to use a virtual environment to avoid dependency conflicts.
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step B: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m playwright install chromium
```

---

## 2. Launching the Application

### Option 1: Streamlit Web UI (Recommended)
This is the interactive version with a dashboard and PRISMA diagram.
```bash
streamlit run streamlit.py
```

### Option 2: CLI Mode (For large batches)
If you prefer a terminal-based workflow:
```bash
python app.py run
```

---

## 3. Environment Variables (Secrets)
Create a `.env` file in the root directory with the following keys:
```text
IEEE_API=your_ieee_key
GPT_ENDPOINT=your_azure_endpoint
GPT_DEPLOYMENT=your_azure_deployment_name
GPT_KEY=your_azure_api_key
GPT_VERSION=2024-05-01-preview
```

---

## 4. Running on Streamlit Cloud
1. Connect your GitHub repository to Streamlit Community Cloud.
2. Select `streamlit.py` as the Main File Path.
3. Paste the contents of your `.env` file into the "Secrets" dashboard in TOML format:
   ```toml
   IEEE_API = "..."
   GPT_ENDPOINT = "..."
   # and so on...
   ```
