from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import re
import json
import random
from bs4 import BeautifulSoup

def get_acm_abstract(doi, delay=6):
    """
    Fetch the abstract from an ACM Digital Library article.

    Parameters:
        doi (str): The ACM DOI.
        delay (int): Seconds to wait for the page to load before scraping.

    Returns:
        str: The abstract text, or None if not found.
    """
    # Set up Chrome options (optional: run headless)
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Uncomment to run without opening a browser

    # Initialize WebDriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    try:
        url = f"https://dl.acm.org/doi/{doi}"
        driver.get(url)
        time.sleep(random.uniform(delay-2, delay+2))  # Wait for page content to load

        # Find the abstract section
        abstract_element = driver.find_element(By.ID, "abstract")
        abstract_text = abstract_element.text.strip()
        return abstract_text

    except Exception as e:
        print("Error:", e)
        return None

    finally:
        driver.quit()

# Example usage
# url = "10.1145/3731715.3733460"
# abstract = get_acm_abstract(url, delay=5)
# print("Abstract:", abstract)

def get_ieee_abstract(url, delay=6):
    """
    Fetch the abstract from an IEEE Xplore document using page metadata.

    Parameters:
        url (str): The IEEE Xplore document URL.
        delay (int): Seconds to wait for the page to load before scraping.

    Returns:
        str: The abstract text, or None if not found.
    """
    # Set up Chrome options
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Optional: run headless

    # Initialize WebDriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    try:
        driver.get(url)
        time.sleep(random.uniform(delay-2, delay+2))  # wait for page to load

        # Option 1: try meta tag first
        meta_abstract = driver.find_element(By.CSS_SELECTOR, 'meta[property="twitter:description"]')
        if meta_abstract:
            return meta_abstract.get_attribute('content').strip()

        # Option 2: fallback to parsing xplGlobal.document.metadata
        page_source = driver.page_source
        match = re.search(r'xplGlobal\.document\.metadata\s*=\s*(\{.*?\});', page_source, re.DOTALL)
        if match:
            metadata_json = match.group(1)
            metadata = json.loads(metadata_json)
            return metadata.get('abstract', '').strip()

        return None

    except Exception as e:
        print("Error:", e)
        return None

    finally:
        driver.quit()

# Example usage
# url = "https://ieeexplore.ieee.org/document/11086420"
# abstract = get_ieee_abstract(url, delay=5)
# print("Abstract:", abstract)

# DOESNT WORKKK
def get_elsevier_abstract(url, delay=6):
    """
    Fetch the abstract from an Elsevier (ScienceDirect) article using Chromium.

    Works for both 'abssecXXXX' and 'd1eXXXX' structures.

    Parameters:
        url (str): Elsevier article URL
        delay (int): Seconds to wait for page to fully load

    Returns:
        str: The abstract text, or None if not found
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")  # run Chromium headless
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    try:
        driver.get(url)
        time.sleep(random.uniform(delay-2, delay+2))

        soup = BeautifulSoup(driver.page_source, "html.parser")

        # 1️⃣ Try the common abssec pattern
        abstract_div = soup.find("div", id=lambda x: x and x.startswith("abssec"))
        if abstract_div:
            return abstract_div.get_text(separator=" ", strip=True)

        # 2️⃣ Fallback: find <h2>Abstract</h2> and get the next <div>
        h2_tag = soup.find("h2", string=lambda text: text and "Abstract" in text)
        if h2_tag:
            next_div = h2_tag.find_next("div")
            if next_div:
                return next_div.get_text(separator=" ", strip=True)

        return None

    except Exception as e:
        print("Error:", e)
        return None

    finally:
        driver.quit()


# Example usage:
url = "https://www.sciencedirect.com/science/article/pii/S2405452624000114"  # replace with your Elsevier link
abstract = get_elsevier_abstract(url, delay=5)
print("\nAbstract:\n", abstract)

