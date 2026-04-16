"""Configuration and environment variable management."""

import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))


def _require_env(name: str) -> str:
    """Get a required environment variable or raise with a clear message."""
    val = os.getenv(name)
    if not val:
        raise ValueError(f"{name} not set in .env")
    return val


def get_api_key() -> str:
    """Get the AI_DEVS API key from environment."""
    return _require_env("AI_DEVS_API_KEY")


def get_open_router_api_key() -> str:
    """Get the OpenRouter API key from environment."""
    return _require_env("OPEN_ROUTER_API_KEY")


def get_open_router_base_url() -> str:
    """Get the OpenRouter API base URL."""
    return _require_env("OPEN_ROUTER_BASE_URL")


def get_ai_gateway_api_key() -> str:
    """Get the AI Gateway API key from environment."""
    return _require_env("AI_GATEWAY_KEY")


def get_ai_gateway_base_url() -> str:
    """Get the AI Gateway API base URL."""
    return _require_env("AI_GATEWAY_BASE_URL")


# Hub URLs
HUB_BASE_URL = _require_env("HUB_BASE_URL")
HUB_VERIFY_URL = f"{HUB_BASE_URL}/verify"
HUB_DATA_URL = f"{HUB_BASE_URL}/data"
HUB_API_URL = f"{HUB_BASE_URL}/api"


def get_openai_api_key() -> str:
    """Get the OpenAI API key from environment."""
    return _require_env("OPENAI_API_KEY")


def get_openai_base_url() -> str:
    """Get the OpenAI API base URL from environment."""
    return _require_env("OPENAI_BASE_URL")
