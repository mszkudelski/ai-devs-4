"""Pre-built reusable Tool instances for common agent operations."""

import base64

import requests

from .agent import Tool
from .openai_service import LLMService


def _read_doc(url: str) -> str:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        return f"ERROR fetching {url}: {e}"


READ_DOC_TOOL = Tool(
    name="read_doc",
    description=(
        "Fetch a text document (Markdown, plain text, etc.) by its full URL. "
        "Returns the raw text content."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Full URL of the document"},
        },
        "required": ["url"],
    },
    callback=_read_doc,
)


def make_analyze_image_tool(service: LLMService) -> Tool:
    """Create an analyze_image Tool backed by the given vision LLMService."""

    def _analyze_image(url: str) -> str:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            return f"ERROR fetching image {url}: {e}"

        content_type = resp.headers.get("Content-Type", "image/png").split(";")[0].strip()
        b64 = base64.b64encode(resp.content).decode("utf-8")
        image_data_url = f"data:{content_type};base64,{b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract ALL text, numbers, tables, codes, and any other data "
                            "visible in this image. Return a complete structured description "
                            "— nothing should be omitted."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ]
        return service.chat(messages, temperature=0.0)

    return Tool(
        name="analyze_image",
        description=(
            "Download an image file and extract ALL information from it using a vision model. "
            "Use for .png, .jpg, .gif, and other image files. "
            "Returns a structured text description of everything visible in the image."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL of the image"},
            },
            "required": ["url"],
        },
        callback=_analyze_image,
    )
