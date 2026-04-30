"""HTTP utilities for AI_devs tasks."""

import csv
import io
import time
import requests
from typing import Any

from .config import get_api_key, HUB_VERIFY_URL, HUB_DATA_URL

MAX_RETRIES = 3
RETRY_BACKOFF = 3  # seconds, doubled each retry


def _retry_on_429(make_request, retries=MAX_RETRIES, backoff=RETRY_BACKOFF):
    """Retry a request function on 429 Too Many Requests."""
    for attempt in range(retries + 1):
        response = make_request()
        if response.status_code != 429:
            return response
        wait = backoff * (2 ** attempt)
        print(f"Rate limited (429). Retrying in {wait}s... (attempt {attempt + 1}/{retries})")
        time.sleep(wait)
    return response  # return last response even if still 429


def post_request(url: str, data: dict, raise_on_error: bool = True, **kwargs) -> dict:
    """Send a POST request and return the JSON response.
    
    Args:
        url: The endpoint URL.
        data: JSON-serializable payload.
        raise_on_error: If True (default), raise on HTTP errors.
            If False, return the error body as a dict instead.
    """
    response = _retry_on_429(lambda: requests.post(url, json=data, **kwargs))
    if not response.ok:
        try:
            error_body = response.json()
        except Exception:
            error_body = {"error": response.text}
        print(f"HTTP {response.status_code} from {url}: {error_body}")
        if raise_on_error:
            response.raise_for_status()
        return {"http_status": response.status_code, **error_body}
    return response.json()


def get_request(url: str, **kwargs) -> requests.Response:
    """Send a GET request and return the response."""
    response = _retry_on_429(lambda: requests.get(url, **kwargs))
    response.raise_for_status()
    return response


def send_report(task: str, answer: Any, verify_url: str = HUB_VERIFY_URL) -> dict:
    """Submit a task answer to the AI_devs hub.
    
    Args:
        task: Task name (e.g., 'people')
        answer: The answer to submit (can be any JSON-serializable type)
        verify_url: The verification endpoint URL
        
    Returns:
        The hub's response as a dict
    """
    api_key = get_api_key()
    payload = {
        "apikey": api_key,
        "task": task,
        "answer": answer,
    }
    print(f"Sending report for task '{task}' to {verify_url}...")
    result = post_request(verify_url, payload)
    print(f"Hub response: {result}")
    return result


def get_hub_data(filename: str) -> requests.Response:
    """Download a data file from the AI_devs hub.
    
    Args:
        filename: The filename to download (e.g., 'people.csv')
        
    Returns:
        The raw response object
    """
    api_key = get_api_key()
    url = f"{HUB_DATA_URL}/{api_key}/{filename}"
    print(f"Downloading {filename} from hub...")
    return get_request(url)


def parse_csv(text: str) -> list[dict]:
    """Parse CSV text into a list of dicts."""
    reader = csv.DictReader(io.StringIO(text))
    return [row for row in reader]
