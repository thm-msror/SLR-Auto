import requests
import json


class IEEEFetcher:
    BASE_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch(self, query: str, max_records: int = 5, sort_field: str = "publication_year") -> dict:
        """
        Fetch articles from IEEE Xplore API.
        """
        url = (
            f"{self.BASE_URL}"
            f"?apikey={self.api_key}"
            f"&querytext={query}"
            f"&max_records={max_records}"
            f"&sort_order=desc"
            f"&sort_field={sort_field}"
        )

        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError(
                f"Error {response.status_code}: {response.text}"
            )
