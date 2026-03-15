# ATLAS Deployment and Setup Guide

This guide details how to set up ATLAS for local development and how to deploy it to the web using Streamlit Cloud.

## 1. Local Setup

Follow these steps to run ATLAS on your own machine.

### Prerequisites
- Python 3.9 or higher
- An IEEE Xplore API Key
- An Azure OpenAI Service endpoint and key

### Step 1: Clone and Create Virtual Environment
```powershell
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt

# Install Playwright browsers (needed for institutional proxy downloads)
playwright install chromium
```

### Step 3: Configure Environment Variables
Create a file named `.env` in the root directory (copy from `.env.example` if available) and add your keys:

```ini
IEEE_API = "your_ieee_key_here"

# Azure OpenAI Configuration
GPT_ENDPOINT = "https://your-endpoint.openai.azure.com/"
GPT_DEPLOYMENT = "your-model-deployment-name"
GPT_KEY = "your-api-key"
GPT_VERSION = "2024-12-01-preview"
```

### Step 4: Run the Application
You can run ATLAS in two modes:

**Web UI (Recommended):**
```powershell
streamlit run streamlit.py
```

**Interactive CLI:**
```powershell
python app.py run
```

---

## 2. Deploying to Streamlit Cloud

Streamlit Cloud is the easiest way to share ATLAS with your instructor.

### Step 1: Push to GitHub
Ensure your code is in a public or private GitHub repository. **Do NOT push your `.env` file to GitHub.** Add `.env` to your `.gitignore`.

### Step 2: Connect to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Connect your GitHub account and select your repository.
3. Set the main file path to `streamlit.py`.

### Step 3: Deployment Files (Automatic)
ATLAS includes two files crucial for Streamlit Cloud:
- `requirements.txt`: Standard Python dependencies.
- `packages.txt`: System-level dependencies for the Chromium browser.

### Step 4: Configure Secrets (Crucial)
Since you won't have a `.env` file on the cloud, you must add your keys to the Streamlit "Secrets" dashboard:
1. In the Streamlit Cloud app settings, go to **Secrets**.
2. Paste the contents of your `.env` file into the text area in the following format:

```toml
IEEE_API = "..."
GPT_ENDPOINT = "..."
GPT_DEPLOYMENT = "..."
GPT_KEY = "..."
GPT_VERSION = "..."
```

Streamlit will automatically pipe these into `os.getenv`, and the `python-dotenv` library in ATLAS will handle them just like the local `.env` file.

---

## 3. Important: Playwright on Streamlit Cloud

ATLAS uses Playwright for high-quality PDF downloads. When deployed to the cloud:
- **No Interactive Login**: You cannot perform the "Manual Proxy Login" step on the cloud because it needs a visible browser window.
- **Best Practice**: If you need to access institutional papers (IEEE, etc.) via proxy on the cloud, you should first run the app **locally**, log in to the proxy, and then upload your `data/.udst_playwright_session.json` file to your GitHub repository (or use Streamlit Secrets to manage the cookie data).
- **Auto-Install**: The application is configured to automatically run `playwright install` on the first run in the cloud environment.

---

## 4. Troubleshooting

- **Authentication Errors**: Ensure your `GPT_KEY` and `IEEE_API` are correctly pasted without extra spaces.
- **Missing Resource**: If you get a "Resource not found" error for GPT, verify your `GPT_DEPLOYMENT` matches the name in your Azure OpenAI Studio.
- **PDF Download Timeout**: If institutional papers fail to download, ensure your machine has access to the university proxy if required, or run the app on a machine with direct library access.
