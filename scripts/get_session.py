"""
ATLAS PROXY SESSION HELPER
===========================

Use this script to log in to your institutional library (UDST/IEEE) 
on your local computer so that the ATLAS Web App can use your session.

REQUIREMENTS:
1. Python installed.
2. Run: pip install playwright
3. Run: playwright install chromium

HOW TO USE:
1. Run this script: python get_session.py
2. A browser will open. Log in to your library account.
3. Once you see the IEEE home page, return to this terminal and press ENTER.
4. A file 'udst_session.json' will appear in this same folder.
5. Upload that file to the ATLAS Web App.
"""
import json
import os
import sys
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Error: Playwright not installed.")
    print("Please run: pip install playwright && playwright install chromium")
    sys.exit(1)

def main():
    print("\n=== ATLAS Proxy Session Helper ===")
    print("This script will help you log in to the UDST Library proxy and save your session.")
    
    # Save in the same folder as the script for reliability
    output_path = Path(__file__).parent / "udst_session.json"
    
    proxy_url = "https://ieeexplore-ieee-org.udstlibrary.idm.oclc.org/Xplore/home.jsp"
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        
        print(f"\n1. Opening browser to: {proxy_url}")
        page.goto(proxy_url)
        
        print("\n2. ACTION REQUIRED:")
        print("   - Log in to UDST/Microsoft in the browser window.")
        print("   - Wait until you see the IEEE Xplore home page.")
        
        print("\n3. Once you are logged in, come back here and press ENTER...")
        input("   [Press ENTER to continue]")
        
        # Save storage state
        state = context.storage_state()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            
        print(f"\nSUCCESS! Session saved to: {output_path}")
        print("You can now upload this file to the ATLAS web app.")
        
        browser.close()

if __name__ == "__main__":
    main()
