import requests

# Test DOI
doi = "10.1109/iccsp60870.2024.10543566"
url = f"https://api.unpaywall.org/v2/{doi}?email=tehreem.masroor@gmail.com"

resp = requests.get(url)
if resp.status_code == 200:
    data = resp.json()
    
    print("Title:", data.get("title"))
    print("Is Open Access?", data.get("is_oa"))
    
    if data.get("best_oa_location"):
        oa = data["best_oa_location"]
        print("OA Type:", oa.get("host_type"))
        print("OA URL:", oa.get("url_for_pdf") or oa.get("url"))
    else:
        print("No Open Access location found for this DOI")
else:
    print("Failed to query Unpaywall:", resp.status_code)
