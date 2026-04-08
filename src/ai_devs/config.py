"""Configuration and environment variable management."""

import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))


def get_api_key() -> str:
    """Get the AI_DEVS API key from environment."""
    key = os.getenv("AI_DEVS_API_KEY")
    if not key:
        raise ValueError("AI_DEVS_API_KEY not set in .env")
    return key


def get_open_router_api_key() -> str:
    """Get the OpenRouter API key from environment."""
    key = os.getenv("OPEN_ROUTER_API_KEY")
    if not key:
        raise ValueError("OPEN_ROUTER_API_KEY not set in .env")
    return key


def get_open_router_base_url() -> str:
    """Get the OpenRouter API base URL."""
    return "https://openrouter.ai/api/v1"


# Hub URLs
HUB_BASE_URL = "https://hub.ag3nts.org"
HUB_VERIFY_URL = f"{HUB_BASE_URL}/verify"
HUB_DATA_URL = f"{HUB_BASE_URL}/data"
