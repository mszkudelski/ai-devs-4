"""S02E04 — search an active mailbox with a small tool-using agent.

The zmail API deliberately separates message search from message retrieval:
``search`` returns metadata and stable ``messageID`` values, while
``getMessages`` returns the bodies.  The mailbox can change while it is being
read, so every search is fresh and numeric ``rowID`` values are never used as
message handles.

The default invocation is a dry run and performs no network or LLM calls::

    python -m tasks.s02e04.solution
    python -m tasks.s02e04.solution --run

The agent submits its answer to the Hub only in ``--run`` mode.  If the Hub
accepts the answer, this module prints the returned flag; it never forwards
that flag anywhere else.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ai_devs import Tool, post_request, run_agent_turn
from src.ai_devs.config import (
    HUB_API_URL,
    HUB_VERIFY_URL,
    get_ai_gateway_api_key,
    get_ai_gateway_base_url,
    get_api_key,
    get_open_router_api_key,
    get_open_router_base_url,
    get_openai_api_key,
    get_openai_base_url,
)


TASK_NAME = "mailbox"
ZMAIL_URL = f"{HUB_API_URL}/zmail"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_MAX_ITERATIONS = 30
DEFAULT_PAGE_SIZE = 20
MESSAGE_ID_RE = re.compile(r"^[0-9a-f]{32}$", re.IGNORECASE)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CONFIRMATION_CODE_RE = re.compile(r"^SEC-[A-Za-z0-9]{32}$")
CONFIRMATION_CODE_SEARCH_RE = re.compile(r"\bSEC-[A-Za-z0-9]{32}\b")
FLAG_RE = re.compile(r"\{FLG:[^}]+\}")


# These are process-local observations only.  They let the main routine find
# a flag even when the model emits its final prose immediately after a
# successful submit.  The message bodies themselves remain inside the agent's
# tool history and are never baked into the source.
_HELP_CACHE: dict[str, Any] | None = None
_LAST_SUBMISSION: Any = None
_OBSERVED_MESSAGES: dict[str, dict[str, Any]] = {}


def _configured(value: str | None) -> bool:
    """Return whether an environment setting is usable rather than a template."""

    if not value:
        return False
    lowered = value.strip().lower()
    return not any(marker in lowered for marker in ("your-", ".example", "replace-me", "tutaj"))


def _llm_connection(provider: str) -> tuple[str, str, str]:
    """Resolve an OpenAI-compatible provider without contacting placeholders."""

    providers = {
        "gateway": (get_ai_gateway_api_key, get_ai_gateway_base_url),
        "openai": (get_openai_api_key, get_openai_base_url),
        "openrouter": (get_open_router_api_key, get_open_router_base_url),
    }

    order = [provider] if provider != "auto" else ["gateway", "openai", "openrouter"]
    for name in order:
        key_getter, base_url_getter = providers[name]
        try:
            key = key_getter()
            base_url = base_url_getter()
        except (KeyError, ValueError):
            continue
        if _configured(key) and _configured(base_url):
            return name, key, base_url

    if provider == "auto":
        raise RuntimeError(
            "No usable LLM provider is configured. Set an API key and base URL "
            "for AI Gateway, OpenAI, or OpenRouter in ai-devs-4/.env."
        )
    raise RuntimeError(f"Provider '{provider}' is not configured with a usable API key and base URL.")


def _zmail_call(action: str, **params: Any) -> dict[str, Any]:
    """Call zmail while keeping the Hub credential inside this callback layer."""

    payload = {"apikey": get_api_key(), "action": action, **params}
    try:
        result = post_request(ZMAIL_URL, payload, raise_on_error=False)
    except Exception as exc:
        return {"ok": False, "error": f"zmail request failed ({type(exc).__name__})"}
    if isinstance(result, dict):
        return result
    return {"ok": False, "error": "zmail returned a non-object response"}


def _remember_search_items(result: dict[str, Any]) -> None:
    """Keep search metadata for diagnostics without trusting mutable row IDs."""

    for item in result.get("items", []) if isinstance(result, dict) else []:
        if isinstance(item, dict) and isinstance(item.get("messageID"), str):
            # Only the stable ID is used by this module after a search.  The
            # rest of the item is retained for a possible bounded fallback
            # report, but no answer is inferred from metadata alone.
            _OBSERVED_SEARCH_ITEMS[item["messageID"]] = dict(item)


_OBSERVED_SEARCH_ITEMS: dict[str, dict[str, Any]] = {}


def _zmail_help() -> dict[str, Any]:
    """Return the API's own action documentation, cached for this run."""

    global _HELP_CACHE
    if _HELP_CACHE is None:
        _HELP_CACHE = _zmail_call("help", page=1)
    return _HELP_CACHE


ZMAIL_HELP_TOOL = Tool(
    name="zmail_help",
    description=(
        "Call the zmail help action and return its documented actions and parameters. "
        "Use this before searching so the API contract is explicit."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    callback=_zmail_help,
)


def _search_mail(query: str, page: int = 1, per_page: int = DEFAULT_PAGE_SIZE) -> dict[str, Any]:
    """Search the live mailbox and return metadata plus stable message IDs."""

    if not isinstance(query, str) or not query.strip():
        return {"ok": False, "error": "query must be a non-empty string"}
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        return {"ok": False, "error": "page must be an integer >= 1"}
    if isinstance(per_page, bool) or not isinstance(per_page, int) or not 5 <= per_page <= 20:
        return {"ok": False, "error": "per_page must be an integer between 5 and 20"}

    result = _zmail_call("search", query=query.strip(), page=page, perPage=per_page)
    _remember_search_items(result)
    return result


SEARCH_MAIL_TOOL = Tool(
    name="search_mail",
    description=(
        "Search the currently active mailbox using Gmail-like syntax. Supported operators "
        "include from:, to:, subject:, quoted phrases, OR, AND, and -exclude. "
        "The result contains metadata only. Always pass the returned stable messageID "
        "to get_messages before extracting any fact. Numeric rowID values are mutable and "
        "must never be used. Search again if a message is no longer found."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Gmail-like mailbox query"},
            "page": {"type": "integer", "minimum": 1, "default": 1},
            "per_page": {"type": "integer", "minimum": 5, "maximum": 20, "default": DEFAULT_PAGE_SIZE},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    callback=_search_mail,
)


def _get_messages(ids: list[str]) -> dict[str, Any]:
    """Retrieve full messages by stable message IDs from a fresh search."""

    if not isinstance(ids, list) or not ids:
        return {"ok": False, "error": "ids must be a non-empty array of messageID strings"}
    if len(ids) > 20:
        return {"ok": False, "error": "request at most 20 message IDs at a time"}
    if any(not isinstance(message_id, str) or not MESSAGE_ID_RE.fullmatch(message_id) for message_id in ids):
        return {
            "ok": False,
            "error": "ids must contain 32-character hexadecimal messageID values from search_mail",
        }

    result = _zmail_call("getMessages", ids=ids)
    for item in result.get("items", []) if isinstance(result, dict) else []:
        if isinstance(item, dict) and isinstance(item.get("messageID"), str):
            _OBSERVED_MESSAGES[item["messageID"]] = dict(item)
    # A message can become unavailable while the active mailbox receives new
    # mail.  Returning notFound lets the agent search again instead of
    # mistakenly treating metadata or a stale rowID as a message body.
    return result


GET_MESSAGES_TOOL = Tool(
    name="get_messages",
    description=(
        "Fetch complete message bodies for one or more 32-character hexadecimal messageID "
        "values returned by search_mail. Do not pass numeric rowID values. Read the full "
        "message before extracting date, password, or confirmation_code."
    ),
    parameters={
        "type": "object",
        "properties": {
            "ids": {
                "type": "array",
                "items": {"type": "string", "pattern": "^[0-9a-fA-F]{32}$"},
                "minItems": 1,
                "maxItems": 20,
                "description": "Stable messageID values returned by search_mail",
            }
        },
        "required": ["ids"],
        "additionalProperties": False,
    },
    callback=_get_messages,
)


def _submit_answer(date: str, password: str, confirmation_code: str) -> dict[str, Any]:
    """Submit the three extracted values and preserve Hub feedback for the agent."""

    global _LAST_SUBMISSION
    if not isinstance(date, str) or not DATE_RE.fullmatch(date):
        return {"ok": False, "error": "date must use YYYY-MM-DD format"}
    if not isinstance(password, str) or not password.strip():
        return {"ok": False, "error": "password must be a non-empty string"}
    if not isinstance(confirmation_code, str) or not CONFIRMATION_CODE_RE.fullmatch(confirmation_code):
        return {
            "ok": False,
            "error": "confirmation_code must match SEC- followed by exactly 32 letters or digits",
        }

    payload = {
        "apikey": get_api_key(),
        "task": TASK_NAME,
        "answer": {
            "password": password,
            "date": date,
            "confirmation_code": confirmation_code,
        },
    }
    try:
        result = post_request(HUB_VERIFY_URL, payload, raise_on_error=False)
    except Exception as exc:
        result = {"ok": False, "error": f"verification request failed ({type(exc).__name__})"}
    _LAST_SUBMISSION = result
    return result if isinstance(result, dict) else {"ok": False, "error": "Hub returned a non-object response"}


SUBMIT_ANSWER_TOOL = Tool(
    name="submit_answer",
    description=(
        "Submit the extracted answer to the mailbox task. Supply all three values only after "
        "reading complete messages. If Hub feedback says a value is missing or wrong, search "
        "the active mailbox again, read the relevant bodies, correct the answer, and retry. "
        "When Hub returns a flag, stop and report that exact flag."
    ),
    parameters={
        "type": "object",
        "properties": {
            "date": {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
            "password": {"type": "string"},
            "confirmation_code": {"type": "string", "pattern": "^SEC-[A-Za-z0-9]{32}$"},
        },
        "required": ["date", "password", "confirmation_code"],
        "additionalProperties": False,
    },
    callback=_submit_answer,
)


SYSTEM_PROMPT = """\
You are an investigation agent solving the AI DEVS S02E04 mailbox task.

Your goal is to submit the exact answer for task mailbox. Extract these fields from the
operator's active mailbox:
- date: the YYYY-MM-DD date when the security department plans the attack on the power plant;
- password: the current password for the employee system;
- confirmation_code: the corrected ticket code, written as SEC- followed by exactly 32 characters.

Follow this workflow carefully:
1. Call zmail_help first and use its documented action names and parameters.
2. Use search_mail to make targeted Gmail-like searches. Start with from:proton.me to identify
   Wiktor's report, then follow the ticket/thread clues it reveals. Search for the password and
   the security ticket separately when useful. The mailbox is active, so repeat a search if an
   expected message is absent or a message ID is no longer found.
3. Search results contain metadata only. Pass their stable 32-character messageID values to
   get_messages and read the complete bodies before extracting facts. Never use numeric rowID.
4. Keep candidate values grounded in message bodies. Watch for corrections: a later security
   message may explicitly replace an earlier wrong confirmation code. Prefer the corrected code
   and use Hub feedback to resolve any remaining error.
5. Call submit_answer only with all three values. If Hub rejects it, read the feedback, search
   again, and submit a corrected answer. Once a flag is returned, stop and repeat the flag exactly.

Never reveal API keys or claim success without a flag from submit_answer. Keep tool calls
sequential because the mailbox can change between calls.
"""


def _extract_flag(value: Any) -> str | None:
    """Find a Hub flag in a response or agent message history."""

    match = FLAG_RE.search(json.dumps(value, ensure_ascii=False, default=str))
    return match.group(0) if match else None


def _run_agent(model: str, api_key: str, base_url: str, max_iterations: int) -> tuple[str, list[dict]]:
    """Run one sequential agent turn with the zmail tools."""

    return run_agent_turn(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Solve the mailbox task now. Begin with zmail_help, then search and read "
                    "messages until you have verified all three fields. Submit the answer and "
                    "return the exact Hub flag if accepted."
                ),
            },
        ],
        tools=[ZMAIL_HELP_TOOL, SEARCH_MAIL_TOOL, GET_MESSAGES_TOOL, SUBMIT_ANSWER_TOOL],
        model=model,
        max_iterations=max_iterations,
        max_tokens=3000,
        verbose=True,
        api_key=api_key,
        base_url=base_url,
    )


def _message_text(message: dict[str, Any]) -> str:
    """Return only a message body for deterministic fallback extraction."""

    value = message.get("message", "")
    return value if isinstance(value, str) else ""


def _direct_collect_messages() -> list[dict[str, Any]]:
    """Search, then immediately fetch bodies, repeating stale searches once.

    This is a bounded operational fallback for environments where the selected
    LLM is unavailable.  It still follows zmail's documented metadata-then-body
    protocol and derives every candidate from downloaded message content.
    """

    queries = (
        "from:proton.me",
        "hasło",
        "systemu pracowniczego",
        "kod potwierdzenia",
        "SEC",
        "atak",
    )
    seen_ids: set[str] = set()
    pending_queries: list[str] = list(queries)

    for attempt in range(2):
        for query in pending_queries:
            result = _search_mail(query, page=1, per_page=DEFAULT_PAGE_SIZE)
            for item in result.get("items", []) if isinstance(result, dict) else []:
                if not isinstance(item, dict):
                    continue
                message_id = item.get("messageID")
                if isinstance(message_id, str) and MESSAGE_ID_RE.fullmatch(message_id):
                    seen_ids.add(message_id)

        missing_before: set[str] = set()
        ids = sorted(seen_ids)
        for offset in range(0, len(ids), 20):
            result = _get_messages(ids[offset : offset + 20])
            for message_id in result.get("notFound", []) if isinstance(result, dict) else []:
                if isinstance(message_id, str):
                    missing_before.add(message_id)

        if not missing_before:
            break
        # A newly-arriving message can invalidate a result between search and
        # retrieval.  A fresh search is enough to obtain a replacement stable
        # ID; do not fall back to mutable numeric rowIDs.
        pending_queries = list(queries)
        seen_ids = {
            message_id
            for message_id in seen_ids
            if message_id in _OBSERVED_MESSAGES
        }

    return list(_OBSERVED_MESSAGES.values())


def _body_date_candidates(messages: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """Score ISO dates found in messages that discuss the planned attack."""

    candidates: list[tuple[int, str]] = []
    for message in messages:
        body = _message_text(message)
        dates = re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", body)
        if not dates:
            continue
        lowered = body.lower()
        score = 0
        if any(word in lowered for word in ("atak", "bomb", "bombard", "zrzucenie bomby", "zniszcz")):
            score += 10
        if "security" in str(message.get("from", "")).lower():
            score += 3
        if "sec-" in lowered or "ticket" in lowered:
            score += 2
        for value in dates:
            candidates.append((score, value))
    return candidates


def _body_password_candidates(messages: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """Find password tokens following an explicit password label in a body."""

    candidates: list[tuple[int, str]] = []
    patterns = (
        re.compile(r"(?:hasłem|haslo|hasłem dostępowym)\s*:\s*([^\s.,]+)", re.IGNORECASE),
        re.compile(r"(?:password)\s*:\s*([^\s.,]+)", re.IGNORECASE),
    )
    for message in messages:
        body = _message_text(message)
        for pattern in patterns:
            for match in pattern.finditer(body):
                token = match.group(1).strip().strip("`\"'()[]{}")
                if not token:
                    continue
                score = 5
                subject = str(message.get("subject", "")).lower()
                if "hasło" in subject or "haslo" in subject or "pracownicz" in subject:
                    score += 5
                if "security" in str(message.get("from", "")).lower():
                    score += 2
                candidates.append((score, token))
    return candidates


def _body_code_candidates(messages: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """Find exact-length confirmation codes and prefer explicit corrections."""

    candidates: list[tuple[int, str]] = []
    for message in messages:
        body = _message_text(message)
        lowered = body.lower()
        for code in CONFIRMATION_CODE_SEARCH_RE.findall(body):
            score = 5
            if any(word in lowered for word in ("poprawny", "correct", "właściwy", "wlasciwy")):
                score += 8
            if "potwierdzenia" in lowered or "confirmation" in lowered:
                score += 2
            if "security" in str(message.get("from", "")).lower():
                score += 2
            candidates.append((score, code))
    return candidates


def _direct_answer(messages: list[dict[str, Any]]) -> dict[str, str] | None:
    """Build an answer solely from body text, with no task values in source."""

    dates = _body_date_candidates(messages)
    passwords = _body_password_candidates(messages)
    codes = _body_code_candidates(messages)
    if not dates or not passwords or not codes:
        return None
    return {
        "date": max(dates, key=lambda item: item[0])[1],
        "password": max(passwords, key=lambda item: item[0])[1],
        "confirmation_code": max(codes, key=lambda item: item[0])[1],
    }


def _run_direct_fallback() -> str | None:
    """Recover from LLM unavailability while preserving the live task flow."""

    try:
        messages = _direct_collect_messages()
        answer = _direct_answer(messages)
        if answer is None:
            print("Direct fallback found no complete answer in message bodies.", file=sys.stderr)
            return None
        response = _submit_answer(**answer)
    except Exception as exc:
        print(f"Direct fallback failed ({type(exc).__name__}).", file=sys.stderr)
        return None
    return _extract_flag(response)


def main() -> int:
    parser = argparse.ArgumentParser(description="S02E04 agentic mailbox solver")
    parser.add_argument("--run", action="store_true", help="allow live zmail, LLM, and verification calls")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument(
        "--provider",
        choices=("auto", "gateway", "openai", "openrouter"),
        default="auto",
        help="OpenAI-compatible LLM provider; auto chooses the first configured provider",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model identifier")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="skip the LLM and derive candidates from freshly fetched message bodies",
    )
    args = parser.parse_args()

    if args.max_iterations < 1:
        parser.error("--max-iterations must be positive")

    if not args.run:
        print("Dry run only: no zmail, LLM, or Hub verification calls were made.")
        print("Use --run to start the mailbox agent.")
        return 0

    # The task explicitly requires help discovery before mailbox searches.
    help_result = _zmail_help()
    if help_result.get("ok") is not True:
        print("zmail help failed; no mailbox search was attempted.", file=sys.stderr)
        return 1

    final_text = ""
    history: list[dict] = []
    if args.direct:
        print("Using direct body-extraction fallback.")
        flag = _run_direct_fallback()
    else:
        try:
            provider, api_key, base_url = _llm_connection(args.provider)
        except RuntimeError as exc:
            print(f"LLM configuration error: {exc}", file=sys.stderr)
            return 1

        print(f"Using LLM provider: {provider}; model: {args.model}")
        try:
            final_text, history = _run_agent(args.model, api_key, base_url, args.max_iterations)
        except Exception as exc:
            print(f"Agent failed ({type(exc).__name__}); trying direct body extraction.", file=sys.stderr)
            flag = _run_direct_fallback()
        else:
            # Only the verifier response is authoritative.  Model prose and
            # mailbox text may contain flag-shaped strings supplied as data.
            flag = _extract_flag(_LAST_SUBMISSION)
            if not flag:
                flag = _run_direct_fallback()

    if flag:
        print(f"FLAG: {flag}")
        return 0

    print("Agent finished without a Hub flag.")
    if final_text:
        print(f"Agent response: {final_text}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
