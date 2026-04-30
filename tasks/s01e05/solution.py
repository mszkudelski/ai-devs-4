"""s01e05 — Railway task.

Activates route X-01 via the self-documenting railway API.
Demonstrates: 503 retry with exponential backoff, rate limit header monitoring,
API self-discovery via help action.

Usage:
    python -m tasks.s01e05.solution
"""

import json
import re
import sys
import os
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ai_devs.config import get_api_key, HUB_VERIFY_URL

MAX_RETRIES = 12
BASE_BACKOFF = 2  # seconds, doubles each 503 retry, capped at 60s


# ── HTTP layer ───────────────────────────────────────────────────────

def railway_post(action_payload: dict, api_key: str, max_retries: int = MAX_RETRIES) -> tuple[dict, dict]:
    """POST to the railway endpoint. Returns (response_body, response_headers).

    Retries automatically on 503 with exponential backoff.
    Respects Retry-After header if present.
    Raises on non-503 HTTP errors or if retries are exhausted.
    """
    payload = {"apikey": api_key, "task": "railway", "answer": action_payload}
    backoff = BASE_BACKOFF

    for attempt in range(max_retries + 1):
        resp = requests.post(HUB_VERIFY_URL, json=payload, timeout=30)

        if resp.status_code in (429, 503):
            try:
                body = resp.json()
            except Exception:
                body = {}
            retry_after = (
                body.get("retry_after")
                or resp.headers.get("Retry-After")
                or min(backoff * (2 ** attempt), 60)
            )
            print(f"[{resp.status_code}] attempt {attempt + 1}/{max_retries}, waiting {retry_after}s...")
            time.sleep(int(retry_after))
            continue

        headers = dict(resp.headers)
        if not resp.ok:
            try:
                body = resp.json()
            except Exception:
                body = {"error": resp.text}
            print(f"[HTTP {resp.status_code}] {body}")
            resp.raise_for_status()

        return resp.json(), headers

    raise RuntimeError(f"Exhausted {max_retries} retries on 429/503")


# ── Rate-limit monitoring ────────────────────────────────────────────

def _log_rate_limit_headers(headers: dict) -> None:
    relevant = {k: v for k, v in headers.items() if "rate" in k.lower() or "limit" in k.lower() or "retry" in k.lower()}
    if relevant:
        print(f"[rate-limit] {relevant}")


def _wait_if_needed(headers: dict) -> None:
    remaining = headers.get("X-RateLimit-Remaining") or headers.get("RateLimit-Remaining")
    if remaining is not None and int(remaining) <= 1:
        wait_raw = headers.get("Retry-After") or headers.get("X-RateLimit-Reset") or headers.get("RateLimit-Reset")
        wait = int(wait_raw) if wait_raw else 10
        print(f"[rate-limit] approaching limit (remaining={remaining}), waiting {wait}s...")
        time.sleep(wait)


# ── Flag extraction ──────────────────────────────────────────────────

def _extract_flag(data: dict) -> str | None:
    match = re.search(r'\{FLG:[^}]+\}', json.dumps(data))
    return match.group(0) if match else None


# ── Doc parsing ──────────────────────────────────────────────────────

def _parse_docs(help_response: dict) -> list[dict]:
    """Extract the ordered action sequence from the help response.

    Returns a list of action specs: [{"name": str, "params": dict}, ...]
    """
    # Try structured JSON parse first; the API nests actions under a "help" key
    if isinstance(help_response, dict):
        candidates = [help_response, help_response.get("help", {})]
        for node in candidates:
            if not isinstance(node, dict):
                continue
            for key in ("actions", "steps", "sequence", "endpoints"):
                if key in node and isinstance(node[key], list):
                    specs = [_normalise_action_spec(item) for item in node[key]]
                    return [s for s in specs if s["name"] != "help"]

        content = json.dumps(help_response)
    else:
        content = str(help_response)

    # LLM fallback: parse unstructured prose
    return _parse_docs_with_llm(content)


def _normalise_action_spec(item: dict | str) -> dict:
    if isinstance(item, str):
        return {"name": item, "params": {}}
    return {
        "name": item.get("name") or item.get("action") or item.get("id") or str(item),
        "params": item.get("params") or item.get("parameters") or item.get("body") or {},
        "required_params": item.get("required_params") or item.get("requires") or item.get("required") or [],
    }


def _parse_docs_with_llm(docs_text: str) -> list[dict]:
    from pydantic import BaseModel
    from src.ai_devs import LLMService

    class ActionSpec(BaseModel):
        name: str
        required_params: list[str]

    class RailwayDocs(BaseModel):
        sequence: list[ActionSpec]

    service = LLMService(provider="gateway", model="gpt-4.1-mini")
    result = service.structured_output(
        messages=[
            {"role": "system", "content": "Extract the ordered sequence of API actions from the following railway API documentation."},
            {"role": "user", "content": docs_text},
        ],
        response_schema=RailwayDocs,
    )
    return [{"name": a.name, "params": {}, "required_params": a.required_params} for a in result.sequence]


# ── State management ─────────────────────────────────────────────────

def _build_action_body(action_spec: dict, state: dict) -> dict:
    body: dict = {"action": action_spec["name"]}
    for param in action_spec.get("required_params", []):
        if param in state:
            body[param] = state[param]
    # Also merge any explicitly declared params (non-dynamic defaults)
    for k, v in action_spec.get("params", {}).items():
        if k not in body:
            body[k] = v
    return body


def _update_state(state: dict, response_body: dict) -> dict:
    updated = {**state}
    if isinstance(response_body, dict):
        updated.update(response_body)
    return updated


# ── Main orchestration ───────────────────────────────────────────────

def activate_route(api_key: str) -> str:
    print("Step 1: fetching API documentation...")
    help_body, help_headers = railway_post({"action": "help"}, api_key)
    _log_rate_limit_headers(help_headers)
    print(f"Help response: {json.dumps(help_body, indent=2, ensure_ascii=False)}")

    flag = _extract_flag(help_body)
    if flag:
        return flag

    _wait_if_needed(help_headers)

    action_list = _parse_docs(help_body)
    print(f"\nParsed action sequence ({len(action_list)} steps): {[a['name'] for a in action_list]}\n")

    state: dict = {"route": "x-01", "value": "RTOPEN"}

    for i, action_spec in enumerate(action_list, start=1):
        body = _build_action_body(action_spec, state)
        print(f"Step {i + 1}: {body}")

        result, headers = railway_post(body, api_key)
        _log_rate_limit_headers(headers)
        print(f"  → {result}")

        flag = _extract_flag(result)
        if flag:
            return flag

        state = _update_state(state, result)
        _wait_if_needed(headers)

    raise RuntimeError("Completed all documented actions but no flag was found in any response")


def main():
    api_key = get_api_key()
    flag = activate_route(api_key)
    print(f"\n{'=' * 60}")
    print(f"FLAG: {flag}")


if __name__ == "__main__":
    main()
