# Technical Brief: Proxy Authentication in Cloud Environments

## 1. The Core Problem: "Headless" Deployment
When ATLAS is deployed to **Streamlit Cloud**, **Render**, or **Vercel**, it runs on a remote Linux server that has no display (it is "headless").

- **GUI Limitation**: The line `browser.launch(headless=False)` in our code works on a laptop because it can open a window. On a cloud server, this command will crash because there is no desktop environment to draw a window.
- **Human-in-the-Middle**: University proxies (UDST/Microsoft) often require Multi-Factor Authentication (MFA). Since the server can't show the user the login screen, the server gets stuck at the "Login Required" barrier.
- **Client vs. Server Separation**: If the user clicks a link and logs in on their own browser, those login cookies stay on **their** computer. The ATLAS server in the cloud doesn't have access to the user's local browser cookies unless they are explicitly uploaded.

## 2. Platform Comparison (Render vs. Vercel vs. Streamlit)
Switching platforms does **not** solve this specific issue, as all serverless/cloud providers operate in headless mode:

| Platform | Popup Support? | Persistence? | Result |
|---|---|---|---|
| **Streamlit Cloud** | No | Ephemeral (Temp) | Headless only; needs manual session injection. |
| **Render** | No | Persistent Disk | Same problem; no GUI for human interaction. |
| **Vercel** | No | Read-only | Worse environment for long-running Playwright tasks. |

## 3. Workaround Options for the Teammate

### Option A: The "Local Helper" Script (Recommended Solution)
This is the most secure and reliable way to bridge the gap between a user's laptop and the cloud server.

**The Improved 5-Step Workflow:**
1. **Reach Step 7**: The user proceeds through the ATLAS pipeline until they reach **"Step 7: PDF Download & Proxy Check"**.
2. **Download Helper**: The user clicks the **"📥 Download Login Helper Script"** button directly in the web app.
3. **Run Locally**: The user runs the downloaded script on their laptop (`python get_session.py`).
4. **Hand Over**: Once they log in on their laptop, they upload the resulting `udst_session.json` to the web app's upload box.
5. **Unlock Pipeline**: The "Start Full Pipeline" button unlocks, and ATLAS on the cloud can now fetch all PDFs.

**Pros**: Completely secure (no passwords shared); integrated directly into the UI; easy for teammates.
**Cons**: Session JSONs expire (usually after 24-48 hours), requiring a quick re-run of the helper script if the review takes multiple days.

### Option B: The "Proxy Bridge" (Advanced)
A specialized tool like **Browserless.io** allows you to connect the Streamlit app to a "Browser-as-a-Service."
- **Pros**: It can handle some sessions longer.
- **Cons**: Expensive and still requires a complex setup for MFA.

### Option C: The "Stage-Based SLR"
Split the responsibility between Cloud and Local.
- **Cloud (Stages 1-5)**: Use the web app for Research Questions, Query Tuning, and Abstract Screening (None of this needs a proxy).
- **Local (Stages 6-10)**: Once the user has their "Top 50 Papers" list, they download the `run.json` and finish the PDF Download and Full-Paper reading on their local PC.

## 4. Why the "User Link" Idea is Hard
You mentioned: *"Can we open the window on the user's end like a link?"*
- **The Catch**: While the user can log into the library website via a link, there is no "automatic" way for their browser to send those private security cookies back to our Streamlit server for security reasons (Same-Origin Policy).
- **The "Real" Way**: The user would need to use a browser extension (like "EditThisCookie") to export their cookies to JSON and then upload that file to ATLAS. This is effectively the same as **Option A**.

## Summary for Development
"We cannot open a login popup on the cloud. Our pipeline depends on PDFs, and PDFs depend on being 'The User.' To fix this, we must build a way for the user to 'Hand Over' their login cookies to the server via an upload button."
