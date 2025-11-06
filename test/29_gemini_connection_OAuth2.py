from vertexai import init
from vertexai.generative_models import GenerativeModel
# pip install google-cloud-aiplatform
# Replace with your project and region that supports Gemini (e.g., us-central1, europe-west1)
PROJECT_ID = "prj-udst-prod-oussama-1"
LOCATION = "europe-west4"

init(project=PROJECT_ID, location=LOCATION)

model = GenerativeModel("gemini-2.5-pro")
resp = model.generate_content("Hello from Vertex AI OAuth route.")
print(resp.text)

# ValueError: Unsupported region for Vertex AI, select from frozenset({'me-west1', 'australia-southeast1', 'us-central1', 'us-west4', 'europe-southwest1', 'asia-southeast2', 'europe-west6', 'us-south1', 'asia-east2', 'europe-west2', 'europe-north1', 'asia-south2', 'northamerica-northeast2', 'asia-east1', 'australia-southeast2', 'europe-west8', 'asia-south1', 'europe-west12', 'asia-southeast1', 'europe-west4', 'asia-northeast1', 'europe-west9', 'me-central2', 'southamerica-west1', 'us-west1', 'global', 'europe-west3', 'northamerica-northeast1', 'africa-south1', 'me-central1', 'europe-central2', 'us-west3', 'southamerica-east1', 'europe-west1', 'us-east4', 'us-east7', 'asia-northeast2', 'asia-northeast3', 'us-east1', 'us-east5', 'us-west2'})   
