"""HTTP utilities for AI_devs tasks."""

import requests
from typing import Any

from .config import get_api_key, HUB_VERIFY_URL, HUB_DATA_URL


def post_request(url: str, data: dict, **kwargs) -> dict:
    """Send a POST request and return the JSON response."""
    response = requests.post(url, json=data, **kwargs)
    response.raise_for_status()
    return response.json()


def get_request(url: str, **kwargs) -> requests.Response:
    """Send a GET request and return the response."""
    response = requests.get(url, **kwargs)
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
