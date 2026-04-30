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
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ai_devs import LLMService, Tool, run_agent, READ_DOC_TOOL, make_analyze_image_tool
from src.ai_devs.api import post_request
from src.ai_devs.config import HUB_BASE_URL, HUB_VERIFY_URL, get_api_key, get_ai_gateway_api_key, get_ai_gateway_base_url


DOC_BASE_URL = f"{HUB_BASE_URL}/dane/doc"
DOC_INDEX_URL = f"{DOC_BASE_URL}/index.md"


# ── Tool callbacks ───────────────────────────────────────────────────

def _submit_declaration(declaration: str) -> str:
    payload = {
        "apikey": get_api_key(),
        "task": "sendit",
        "answer": {"declaration": declaration},
    }
    result = post_request(HUB_VERIFY_URL, payload, raise_on_error=False)
    return str(result)


SUBMIT_DECLARATION_TOOL = Tool(
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
)


# ── System prompt ────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""\
You are an SPK (System Przesyłek Konduktorskich) transport declaration expert. \
Your goal is to submit a valid, accepted declaration to the Hub.

The official SPK documentation is available starting at {DOC_INDEX_URL}. \
It spans multiple files and may include images — treat all of it as your source of truth. \
Use read_doc for text files and analyze_image for any image files.

Constraints:
- Read the complete documentation before attempting to submit — do not submit until you have read every file referenced in the docs, including images
- All field values must come from the documentation — never invent data
- The declaration must match the template format exactly — Hub validates both values and formatting
- Budget must result in 0 PP (choose the system-funded category)
- Special notes field must be empty
- If the Hub rejects your submission, go back to the documentation to find the correct value — do not guess

Shipment data:
- Sender ID: 450202122
- Route: Gdańsk → Żarnowiec
- Weight: 2800 kg
- Contents: kasety z paliwem do reaktora

If the Hub rejects your submission, use the error message to identify and fix the specific \
problem, then resubmit.
"""


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="S01E04 Sendit task")
    parser.add_argument("--max-iterations", type=int, default=25)
    args = parser.parse_args()

    vision_service = LLMService(provider="gateway", model="gemini-2.5-flash")

    result = run_agent(
        system_prompt=SYSTEM_PROMPT,
        user_message=f"Fill and submit the SPK declaration. Start by reading: {DOC_INDEX_URL}",
        tools=[READ_DOC_TOOL, make_analyze_image_tool(vision_service), SUBMIT_DECLARATION_TOOL],
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
