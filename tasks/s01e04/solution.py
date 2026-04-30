"""S01E04 — Sendit task.

Downloads SPK documentation (including images via vision model), then uses a
Function Calling agent to parse docs, fill a transport declaration, and submit
it to the Hub. If the Hub rejects the declaration, the agent reads the error
hint and self-corrects within the same loop.

Usage:
    python -m tasks.s01e04.solution
    python -m tasks.s01e04.solution --max-iterations 10
"""

import argparse
import base64
import sys
import os
from typing import Optional

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ai_devs import LLMService, send_report
from src.ai_devs.config import HUB_BASE_URL, get_ai_gateway_api_key, get_ai_gateway_base_url
from src.ai_devs.agent import Tool, run_agent


DOC_BASE_URL = f"{HUB_BASE_URL}/dane/doc"
DOC_INDEX_URL = f"{DOC_BASE_URL}/index.md"

_vision_service: Optional[LLMService] = None


def _get_vision_service() -> LLMService:
    global _vision_service
    if _vision_service is None:
        _vision_service = LLMService(provider="gateway", model="gemini-2.5-flash")
    return _vision_service


# ── Tool callbacks ───────────────────────────────────────────────────

def _read_doc(url: str) -> str:
    """Fetch a text document by URL."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        return f"ERROR fetching {url}: {e}"


def _analyze_image(url: str) -> str:
    """Download an image and extract all its content using a vision model."""
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
                        "This image is part of SPK (System Przesyłek Konduktorskich) documentation. "
                        "Extract ALL text, numbers, tables, codes, routes, tariffs, and any other "
                        "information visible in the image. Return a complete structured description "
                        "of every piece of data — nothing should be omitted."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url},
                },
            ],
        }
    ]
    return _get_vision_service().chat(messages, temperature=0.0)


def _submit_declaration(declaration: str) -> str:
    """Submit the completed SPK declaration to the Hub and return the response."""
    result = send_report("sendit", {"declaration": declaration})
    return str(result)


# ── Tool definitions ─────────────────────────────────────────────────

TOOLS = [
    Tool(
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
    ),
    Tool(
        name="analyze_image",
        description=(
            "Download an image file and extract ALL information from it using a vision model. "
            "Use for .png, .jpg, .gif, and other image files found in the documentation. "
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
    ),
    Tool(
        name="submit_declaration",
        description=(
            "Submit the completed SPK transport declaration to the Hub for verification. "
            "Returns the hub response. If rejected, the response contains an error hint "
            "describing what needs to be fixed — read it carefully, correct the declaration, "
            "and call this tool again."
        ),
        parameters={
            "type": "object",
            "properties": {
                "declaration": {
                    "type": "string",
                    "description": "Full declaration text, formatted exactly as the template",
                },
            },
            "required": ["declaration"],
        },
        callback=_submit_declaration,
    ),
]


# ── System prompt ────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""\
You are an expert at filling SPK (System Przesyłek Konduktorskich) transport declarations.

Your task is to fill out and submit a transport declaration using the official SPK documentation.

Steps:
1. Read the documentation index: {DOC_INDEX_URL}
2. Follow ALL links in the docs — read EVERY referenced file (use read_doc for text/markdown, \
analyze_image for any image files such as .png .jpg .gif)
3. Find the declaration template — note its exact format, all field names, separators, and order
4. Find the route code for Gdańsk → Żarnowiec (check railway connection tables or route lists)
5. Determine the correct package category whose fee is 0 PP (system-funded) — check the tariff tables
6. Fill the declaration EXACTLY per the template — preserve all separators, field order, and formatting
7. Submit with submit_declaration
8. If the Hub rejects the declaration, read the error message carefully — it tells you what to fix. \
Correct only what the error specifies and resubmit. Do NOT retry with the same content.

Declaration data to fill in:
- Sender ID (identyfikator nadawcy): 450202122
- Origin (punkt nadawczy): Gdańsk
- Destination (punkt docelowy): Żarnowiec
- Weight (waga): 2800 kg
- Budget (budżet): 0 PP — select the category whose fee is zero (system-funded)
- Contents (zawartość): kasety z paliwem do reaktora
- Special notes (uwagi specjalne): NONE — leave this field empty or omit it

Critical rules:
- Always analyze image files — they may contain tariff tables or route maps with essential data
- The declaration format must match the template character-for-character (Hub validates formatting)
- Do NOT invent route codes or fees — derive them strictly from the documentation
"""


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="S01E04 Sendit task")
    parser.add_argument("--max-iterations", type=int, default=25)
    args = parser.parse_args()

    result = run_agent(
        system_prompt=SYSTEM_PROMPT,
        user_message=f"Fill and submit the SPK declaration. Start by reading: {DOC_INDEX_URL}",
        tools=TOOLS,
        model="gpt-4.1-mini",
        max_iterations=args.max_iterations,
        max_tokens=4096,
        api_key=get_ai_gateway_api_key(),
        base_url=get_ai_gateway_base_url(),
    )
    print(f"\n{'='*60}")
    print(f"Agent finished:\n{result}")


if __name__ == "__main__":
    main()
