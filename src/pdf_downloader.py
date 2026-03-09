import re
import json
import requests
import time
from pathlib import Path
from bs4 import BeautifulSoup
from src.utils import safe_filename

try:
    from playwright.sync_api import sync_playwright
except Exception:
    sync_playwright = None

# ---------------- CONFIGURATION ----------------
SESSION_STATE_PATH = "data/.udst_playwright_session.json"
UDST_PROXY_PREFIX = "https://ieeexplore-ieee-org.udstlibrary.idm.oclc.org"
ACM_PROXY_PREFIX = "https://dl-acm-org.udstlibrary.idm.oclc.org"
SD_PROXY_PREFIX = "https://www-sciencedirect-com.udstlibrary.idm.oclc.org"
SPRINGER_PROXY_PREFIX = "https://link-springer-com.udstlibrary.idm.oclc.org"
UNPAYWALL_EMAIL = "60302531@udst.edu.qa" 

# ---------------- CORE UTILS ----------------

def clean_title(title):
    return " ".join(title.replace('\n', ' ').split())

def is_valid_pdf(path: Path):
    if not path.exists() or path.stat().st_size < 5000: return False
    try:
        with open(path, "rb") as f: return f.read(4) == b"%PDF"
    except: return False

def download_file(url, out_path, source_name, session=None):
    if not url: return False
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/pdf,application/octet-stream",
        "Referer": f"{UDST_PROXY_PREFIX}/Xplore/home.jsp"
    }
    try:
        fetcher = session if session else requests
        r = fetcher.get(url, stream=True, timeout=30, headers=headers, allow_redirects=True)
        
        if "login" in r.url.lower() and "getpdf" not in r.url.lower():
            return False

        content_type = r.headers.get('Content-Type', '').lower()
        if r.status_code == 200 and ('pdf' in content_type or 'octet-stream' in content_type):
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(8192): f.write(chunk)
            if is_valid_pdf(out_path): 
                print(f"   ✅ Success: {source_name}")
                return True
    except: pass
    if out_path.exists(): out_path.unlink()
    return False

# ---------------- SESSION MANAGEMENT (RESTORED) ----------------

def load_udst_session():
    session = requests.Session()
    if Path(SESSION_STATE_PATH).exists():
        with open(SESSION_STATE_PATH, "r", encoding="utf-8") as f:
            storage = json.load(f)
            for c in storage.get("cookies", []):
                session.cookies.set(c["name"], c["value"], domain=c["domain"])
    return session

def ensure_udst_session():
    session_file = Path(SESSION_STATE_PATH)
    if session_file.exists():
        print('[INFO] Deep-testing UDST proxy access...')
        test_session = load_udst_session()
        try:
            test_url = f'{UDST_PROXY_PREFIX}/stampPDF/getPDF.jsp?arnumber=1'
            r = test_session.get(test_url, timeout=10, allow_redirects=True)
            if 'login' not in r.url.lower() and 'idp' not in r.url.lower():
                print('[OK] Session valid. Proceeding...')
                return
        except: pass
        session_file.unlink()

    if sync_playwright is None:
        print('[WARN] Playwright not installed; skipping UDST proxy login.')
        print('       Install with: pip install playwright && playwright install')
        return

    print('\n[LOGIN] Manual login required to UDST Library.')
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(f'{UDST_PROXY_PREFIX}/Xplore/home.jsp')
        print('\n--- ACTION ---\n1. Log in to UDST/Microsoft.\n2. Wait for IEEE search bar.\n-> Press ENTER here after login...')
        input()
        context.storage_state(path=SESSION_STATE_PATH)
        browser.close()

# ---------------- EXTRACTION HELPERS (RESTORED) ----------------

def extract_ieee_arnumber(paper):
    arnum = paper.get("ieee_arnumber") or paper.get("arnumber")
    if arnum: return str(arnum).strip()
    doi = str(paper.get("doi", "")).strip()
    if "10.1109" in doi:
        match = re.search(r'\.(\d+)$', doi)
        if match: return match.group(1)
    return None

def get_arxiv_link(doi):
    if not doi: return None
    if 'arxiv' in doi.lower():
        match = re.search(r'(\d{4}\.\d{4,5})', doi)
        if match: return f"https://arxiv.org/pdf/{match.group(1)}.pdf"
    return None

# ---------------- ENFORCED TITLE SEARCHES ----------------

def search_sciencedirect_by_title(title, session):
    """ENFORCED: Searches ScienceDirect via proxy using Title."""
    print(f"      🧪 ScienceDirect Enforced Title Search...")
    try:
        search_url = f"{SD_PROXY_PREFIX}/search?qs={requests.utils.quote(title)}"
        r = session.get(search_url, timeout=20)
        soup = BeautifulSoup(r.text, 'html.parser')
        result = soup.find('a', href=re.compile(r'/science/article/pii/'))
        if result:
            pii = result['href'].split('/')[-1].split('?')[0]
            return f"{SD_PROXY_PREFIX}/science/article/pii/{pii}/pdfft?is_pdf=true"
    except: pass
    return None

def google_search_fallback(title):
    """ENFORCED: General Internet Search specifically for ResearchGate."""
    print(f"      🌐 ResearchGate/Google Enforced Title Hunt...")
    if sync_playwright is None:
        print('      [WARN] Skipping Google fallback (playwright not installed).')
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            search_query = f'"{title}" site:researchgate.net OR filetype:pdf'
            page.goto(f"https://www.google.com/search?q={requests.utils.quote(search_query)}")
            links = page.locator('a').all()
            for link in links:
                href = link.get_attribute('href')
                if href and ('.pdf' in href.lower() or 'researchgate.net' in href.lower()):
                    browser.close(); return href
            browser.close()
    except: pass
    return None

# ---------------- OTHER METHODS (RETAINED) ----------------

def hunt_cvf(title):
    print(f"      🖼️  Checking CVF Open Access...")
    clean = re.sub(r'[^\w\s]', '', title).split()
    search_query = "+".join(clean[:8]) 
    try:
        r = requests.get(f"https://openaccess.thecvf.com/search?q={search_query}", timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        for a in soup.find_all('a', href=True):
            if '.pdf' in a['href'] and 'content' in a['href']:
                return "https://openaccess.thecvf.com/" + a['href'].lstrip('/')
    except: pass
    return None

def fetch_from_scihub(doi, pdf_path):
    if not doi or len(doi) < 5: return False
    print(f"      🏴‍☠️  Trying Sci-Hub...")
    mirrors = ["https://sci-hub.se", "https://sci-hub.st", "https://sci-hub.ru"]
    for base_url in mirrors:
        try:
            target_url = f"{base_url}/{doi}"
            response = requests.get(target_url, timeout=15)
            soup = BeautifulSoup(response.text, 'html.parser')
            pdf_element = soup.find('button', onclick=re.compile(r"location.href='(.*?)'")) or \
                          soup.find('embed', type='application/pdf') or \
                          soup.find('iframe', id='pdf')
            if pdf_element:
                if pdf_element.name == 'button':
                    pdf_link = re.search(r"location.href='(.*?)'", pdf_element['onclick']).group(1)
                else:
                    pdf_link = pdf_element.get('src') or pdf_element.get('href')
                if pdf_link:
                    if pdf_link.startswith('//'): pdf_link = "https:" + pdf_link
                    elif not pdf_link.startswith('http'): pdf_link = base_url + pdf_link
                    if download_file(pdf_link, pdf_path, f"Sci-Hub ({base_url.split('.')[-1]})"): return True
        except: continue
    return False

def search_openalex(title):
    print(f"      🔍 Searching OpenAlex...")
    try:
        url = f"https://api.openalex.org/works?filter=title.search:{title}&mailto={UNPAYWALL_EMAIL}"
        r = requests.get(url, timeout=10)
        data = r.json()
        if data.get('results'):
            result = data['results'][0]
            return result.get('best_oa_location', {}).get('pdf_url'), result.get('doi', '').replace('https://doi.org/', '')
    except: pass
    return None, None

def search_semantic_scholar(title):
    print(f"      🎓 Searching Semantic Scholar...")
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={title}&fields=openAccessPdf,title&limit=1"
        r = requests.get(url, timeout=10)
        data = r.json()
        if data.get('data'):
            pdf_info = data['data'][0].get('openAccessPdf')
            if pdf_info: return pdf_info.get('url')
    except: pass
    return None

# ---------------- MAIN PIPELINE ----------------

def download_pdfs(papers, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_udst_session()
    session = load_udst_session()

    for item in papers:
        paper = item["paper"]
        title = clean_title(paper.get("title", "unknown"))
        pdf_path = output_dir / f"{safe_filename(title)}.pdf"
        doi = str(paper.get("doi", "")).lower().strip().replace("https://doi.org/", "")
        pub_field = str(paper.get("publisher", "")).lower()

        if is_valid_pdf(pdf_path):
            print(f"⏭️ Skip: {title[:50]}..."); continue

        print(f"\n🚀 Processing: {title[:60]}...")

        # 1. ArXiv & CVF Direct (Fastest)
        arxiv_url = get_arxiv_link(doi) or get_arxiv_link(paper.get("link", ""))
        if arxiv_url and download_file(arxiv_url, pdf_path, "ArXiv Direct"): continue
        
        cvf_url = hunt_cvf(title)
        if cvf_url and download_file(cvf_url, pdf_path, "CVF"): continue

        # 2. Publisher Proxies
        if "10.1109" in doi or "ieee" in pub_field:
            arnum = extract_ieee_arnumber(paper)
            if arnum and download_file(f"{UDST_PROXY_PREFIX}/stampPDF/getPDF.jsp?arnumber={arnum}", pdf_path, "IEEE Proxy", session): continue

        if "10.1145" in doi or "acm" in pub_field:
            if download_file(f"{ACM_PROXY_PREFIX}/doi/pdf/{doi}", pdf_path, "ACM Proxy", session): continue

        if "10.1016" in doi or "elsevier" in pub_field or "sciencedirect" in pub_field:
            sd_url = search_sciencedirect_by_title(title, session)
            if sd_url and download_file(sd_url, pdf_path, "ScienceDirect", session): continue

        if "10.1007" in doi or "springer" in pub_field:
            springer_url = f"{SPRINGER_PROXY_PREFIX}/content/pdf/{doi}.pdf"
            if download_file(springer_url, pdf_path, "Springer Proxy", session): continue

        # 3. Enforced ResearchGate/Google Hunt
        google_url = google_search_fallback(title)
        if google_url and download_file(google_url, pdf_path, "ResearchGate/Google"): continue

        # 4. Sci-Hub
        if doi and fetch_from_scihub(doi, pdf_path): continue

        # 5. OA Fallbacks
        oa_pdf, oa_doi = search_openalex(title)
        if oa_pdf and download_file(oa_pdf, pdf_path, "OpenAlex"): continue
        if oa_doi and not doi: doi = oa_doi 

        if doi:
            try:
                up_r = requests.get(f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}", timeout=5)
                up_url = up_r.json().get("best_oa_location", {}).get("url_for_pdf")
                if up_url and download_file(up_url, pdf_path, "Unpaywall"): continue
            except: pass

        ss_url = search_semantic_scholar(title)
        if ss_url and download_file(ss_url, pdf_path, "Semantic Scholar"): continue

        print(f"❌ Failed all methods: {title[:40]}")
