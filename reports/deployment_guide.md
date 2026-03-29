# Deploying ATLAS to Streamlit Cloud

This guide explains how to deploy the ATLAS tool to **Streamlit Cloud**, focusing on how to securely manage your API keys (OpenAI, IEEE, etc.) using **Streamlit Secrets**.

## 1. Prerequisites
-   A GitHub repository containing the ATLAS code.
-   A [Streamlit Cloud](https://share.streamlit.io/) account connected to your GitHub.
-   OpenAI API Key (Required).
-   IEEE Xplore API Key (Optional).

---

## 2. Preparing your Repository
Before deploying, ensure your repo has the following files:
1.  **`streamlit.py`**: The main app file.
2.  **`pyproject.toml`** (or `requirements.txt`): To install dependencies automatically.
3.  **`.streamlit/config.toml`**: (Optional) For theme settings.

**Important:** Do **NOT** upload your `.env` file to GitHub. Add it to your `.gitignore`.

---

## 3. Deployment Steps

1.  **Login to Streamlit Cloud:** Go to [share.streamlit.io](https://share.streamlit.io/).
2.  **Create a New App:**
    -   Click the **"New app"** button.
    -   Select your repository (`joey-en/SLR-Auto`).
    -   Select the branch (usually `main`).
    -   Set the main file path to `streamlit.py`.
3.  **Advanced Settings (Secrets):**
    -   Before clicking "Deploy," click on **"Advanced settings..."**.
    -   In the **Secrets** section, paste the contents of your `.env` file in the following format:
    ```toml
    OPENAI_API_KEY = "your-api-key-here"
    IEEE_API_KEY = "your-ieee-api-key-here"
    # Add any other environment variables from your .env here
    ```
4.  **Deploy:** Click the **"Deploy!"** button. Streamlit will now build the environment and launch your app.

---

## 4. How Secrets Work in Streamlit
In a local environment, ATLAS uses `python-dotenv` to load secrets from a `.env` file. However, Streamlit Cloud provides a built-in secrets management system that is more secure.

Your code automatically handles this via the `st.secrets` object. In `src/atlas/utils/app_helpers.py` (or similar), ATLAS is designed to check for both:
1.  **Local:** `os.getenv("OPENAI_API_KEY")` (loads from `.env`).
2.  **Cloud:** `st.secrets["OPENAI_API_KEY"]` (loads from the Cloud settings).

### Updating Secrets Later
If you need to change your API keys after deployment:
1.  Go to your app's dashboard on Streamlit Cloud.
2.  Click the three dots **(...)** next to your app.
3.  Select **Settings** -> **Secrets**.
4.  Edit the TOML file and click **Save**.

---

## 5. Troubleshooting Deployment
-   **Dependencies Error:** Ensure all packages used (like `openai`, `playwright`, `pandas`) are listed in your `pyproject.toml` or `requirements.txt`.
-   **Playwright Issues:** ATLAS automatically runs `playwright install chromium` on the first run. If this fails on Streamlit Cloud, you may need to add `playwright` to your dependencies and add a custom `packages.txt` with `libgbm-dev`, `libnss3`, etc., though Streamlit Cloud usually handles common browser dependencies.
-   **Logs:** If the app crashes, check the **"Manage app"** console at the bottom-right corner of the screen for error logs.
