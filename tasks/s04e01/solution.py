"""S04E01 — apply the requested edits in the OKO editor.

The OKO web application is useful for read-only reconnaissance, but the task
API is the write boundary.  This solver keeps that boundary small and
deterministic: it calls ``help`` first, applies three idempotent updates, and
then calls ``done``.  The default invocation only prints the planned answer
objects and performs no network calls::

    python -m tasks.s04e01.solution

Use ``--run`` to perform the live Hub calls.  The response from ``done`` is
searched locally for the verifier flag and the flag is printed; it is never
sent to a second endpoint or submitted through the OKO UI.

The IDs below are the stable OKO records for this lesson.  They can be
overridden with command-line options (or the corresponding environment
variables) when a lesson account has a different seeded dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


# Allow ``python tasks/s04e01/solution.py`` as well as module execution from
# the repository root, while keeping imports lazy so dry-run mode never loads
# credentials or the HTTP client.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


TASK_NAME = "okoeditor"
DEFAULT_SKOLWIN_REPORT_ID = "380792b2c86d9c5be670b3bde48e187b"
DEFAULT_SKOLWIN_TASK_ID = DEFAULT_SKOLWIN_REPORT_ID
DEFAULT_KOMAROWO_REPORT_ID = "351c0d9c90d66b4c040fff1259dd191d"
FLAG_RE = re.compile(r"\{FLG:[^}]+\}")
DEFAULT_TIMEOUT = 120
MAX_ID_LENGTH = 128
MAX_TEXT_LENGTH = 10_000


class HubAPIError(RuntimeError):
    """Raised when the Hub rejects an action or returns malformed JSON."""

    def __init__(self, message: str, response: Any = None) -> None:
        super().__init__(message)
        self.response = response


def extract_flag(value: Any) -> str | None:
    """Return the first verifier flag in a Hub response, if one is present."""

    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(value)
    match = FLAG_RE.search(text)
    return match.group(0) if match else None


def _response_summary(value: Any, limit: int = 700) -> str:
    """Render bounded diagnostics without echoing request data or credentials."""

    if isinstance(value, Mapping):
        fields = {
            key: value[key]
            for key in ("http_status", "code", "message", "error")
            if key in value
        }
        text = json.dumps(fields or {"type": "unexpected_response"}, ensure_ascii=False)
    else:
        text = json.dumps({"type": type(value).__name__}, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "..."


def _response_is_error(value: Any) -> bool:
    """Recognise HTTP and task-level error responses from the shared helper."""

    if not isinstance(value, Mapping):
        return True
    status = value.get("http_status")
    if isinstance(status, int) and status >= 400:
        return True
    code = value.get("code")
    if isinstance(code, int) and code < 0:
        return True
    return bool(value.get("error"))


def _validated_id(value: str, field: str) -> str:
    """Validate an OKO record ID before placing it in an API payload."""

    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    result = value.strip()
    if not result or len(result) > MAX_ID_LENGTH:
        raise ValueError(f"{field} must be 1-{MAX_ID_LENGTH} characters")
    # IDs are supplied by the user or environment; keep them to URL/JSON-safe
    # identifier characters and reject control characters and shell syntax.
    if not re.fullmatch(r"[A-Za-z0-9_-]+", result):
        raise ValueError(f"{field} contains unsupported characters")
    return result


def _validated_text(value: str, field: str) -> str:
    """Validate editable Polish text without changing the intended content."""

    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    result = value.strip()
    if not result or len(result) > MAX_TEXT_LENGTH:
        raise ValueError(f"{field} must be 1-{MAX_TEXT_LENGTH} characters")
    if any(char in result for char in ("\x00", "\r")):
        raise ValueError(f"{field} contains unsupported control characters")
    return result


def build_help_answer() -> dict[str, str]:
    """Build the mandatory discovery action."""

    return {"action": "help"}


def build_skolwin_report_update(
    report_id: str = DEFAULT_SKOLWIN_REPORT_ID,
    *,
    title: str = "MOVE04 Trudne do klasyfikacji ruchy nieopodal miasta Skolwin",
    content: str = (
        "W okolicach miasta Skolwin wykryto ruch zwierząt. "
        "Analiza danych wskazuje na obecność dzikiej fauny, "
        "prawdopodobnie bobrów lub innych zwierząt wodnych "
        "poruszających się w pobliżu rzeki."
    ),
) -> dict[str, str]:
    """Build the update that reclassifies the Skolwin incident as animals."""

    return {
        "action": "update",
        "page": "incydenty",
        "id": _validated_id(report_id, "Skolwin report ID"),
        "title": _validated_text(title, "Skolwin report title"),
        "content": _validated_text(content, "Skolwin report content"),
    }


def build_skolwin_task_update(
    task_id: str = DEFAULT_SKOLWIN_TASK_ID,
    *,
    content: str = (
        "Zadanie zakończone. Zaobserwowano ruch zwierząt (bobry) "
        "w okolicach Skolwina. Reklasyfikacja incydentu z MOVE03 na MOVE04."
    ),
) -> dict[str, str]:
    """Build the update that completes the Skolwin task."""

    return {
        "action": "update",
        "page": "zadania",
        "id": _validated_id(task_id, "Skolwin task ID"),
        "content": _validated_text(content, "Skolwin task content"),
        "done": "YES",
    }


def build_komarowo_report_update(
    report_id: str = DEFAULT_KOMAROWO_REPORT_ID,
    *,
    title: str = "MOVE01 Wykrycie ruchu ludzi w okolicach miasta Komarowo",
    content: str = (
        "W okolicach niezamieszkałego miasta Komarowo wykryto ruch ludzi. "
        "Czujniki zarejestrowały obecność osób przemieszczających się "
        "w pobliżu opuszczonych budynków."
    ),
) -> dict[str, str]:
    """Build the update that records human movement around Komarowo."""

    return {
        "action": "update",
        "page": "incydenty",
        "id": _validated_id(report_id, "Komarowo report ID"),
        "title": _validated_text(title, "Komarowo report title"),
        "content": _validated_text(content, "Komarowo report content"),
    }


def build_done_answer() -> dict[str, str]:
    """Build the final verifier action."""

    return {"action": "done"}


def build_plan(
    *,
    skolwin_report_id: str = DEFAULT_SKOLWIN_REPORT_ID,
    skolwin_task_id: str = DEFAULT_SKOLWIN_TASK_ID,
    komarowo_report_id: str = DEFAULT_KOMAROWO_REPORT_ID,
) -> tuple[tuple[str, dict[str, str]], ...]:
    """Return the ordered, verifier-ready action plan."""

    return (
        ("help", build_help_answer()),
        (
            "Skolwin incident",
            build_skolwin_report_update(skolwin_report_id),
        ),
        (
            "Skolwin task",
            build_skolwin_task_update(skolwin_task_id),
        ),
        (
            "Komarowo incident",
            build_komarowo_report_update(komarowo_report_id),
        ),
        ("done", build_done_answer()),
    )


ActionCallable = Callable[[dict[str, str]], Any]


class HubClient:
    """Authenticated client for the S04E01 ``/verify`` action protocol.

    ``action_callable`` is injectable for local checks and keeps the payload
    construction and response validation testable without a live request.
    """

    def __init__(
        self,
        *,
        action_callable: ActionCallable | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._action_callable = action_callable
        self._timeout = timeout
        self._post_request: Callable[..., Any] | None = None
        self._verify_url: str | None = None
        self._api_key: str | None = None

    def _load_live_dependencies(self) -> None:
        if self._post_request is not None:
            return
        from src.ai_devs.api import post_request
        from src.ai_devs.config import HUB_VERIFY_URL, get_api_key

        self._post_request = post_request
        self._verify_url = HUB_VERIFY_URL
        self._api_key = get_api_key()

    def call(self, answer: Mapping[str, str]) -> dict[str, Any]:
        """Execute one action, raising a concise error for rejected calls."""

        if not isinstance(answer, Mapping) or not answer.get("action"):
            raise ValueError("answer must contain a non-empty action")
        request_answer = dict(answer)

        if self._action_callable is not None:
            raw_result = self._action_callable(request_answer)
        else:
            self._load_live_dependencies()
            assert self._post_request is not None
            assert self._verify_url is not None
            assert self._api_key is not None
            raw_result = self._post_request(
                self._verify_url,
                {
                    "apikey": self._api_key,
                    "task": TASK_NAME,
                    "answer": request_answer,
                },
                raise_on_error=False,
                timeout=self._timeout,
            )

        if not isinstance(raw_result, Mapping):
            raise HubAPIError(
                f"{request_answer['action']} returned a non-object response",
                raw_result,
            )
        result = dict(raw_result)
        if _response_is_error(result):
            raise HubAPIError(
                f"{request_answer['action']} rejected: {_response_summary(result)}",
                result,
            )
        return result


@dataclass(frozen=True)
class LiveRun:
    """Summary of one live run, including a locally captured flag if present."""

    completed_actions: tuple[str, ...]
    flag: str | None


def run_live(
    plan: tuple[tuple[str, dict[str, str]], ...],
    *,
    client: HubClient | None = None,
) -> LiveRun:
    """Run the ordered plan and capture a verifier flag from any response."""

    live_client = client or HubClient()
    completed: list[str] = []
    for index, (label, answer) in enumerate(plan, start=1):
        print(f"[{index}/{len(plan)}] {label} ({answer['action']})")
        response = live_client.call(answer)
        completed.append(label)
        flag = extract_flag(response)
        if flag:
            print(f"FLAG: {flag}")
            return LiveRun(tuple(completed), flag)
        print(f"  Hub response: {_response_summary(response)}")
    return LiveRun(tuple(completed), None)


def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _print_dry_run(plan: tuple[tuple[str, dict[str, str]], ...]) -> None:
    """Print a reviewable plan without importing credentials or calling Hub."""

    print("S04E01 OKOEDITOR dry run (no network calls)")
    for index, (label, answer) in enumerate(plan, start=1):
        print(f"[{index}/{len(plan)}] {label}")
        print(json.dumps(answer, ensure_ascii=False, sort_keys=True))
    print("Use --run to execute this plan against the live Hub.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="perform live Hub updates; without this flag only print the plan",
    )
    parser.add_argument(
        "--skolwin-report-id",
        default=_env_or_default("OKO_SKOLWIN_REPORT_ID", DEFAULT_SKOLWIN_REPORT_ID),
        help="OKO incident ID for the Skolwin report",
    )
    parser.add_argument(
        "--skolwin-task-id",
        default=_env_or_default("OKO_SKOLWIN_TASK_ID", DEFAULT_SKOLWIN_TASK_ID),
        help="OKO task ID for the Skolwin task",
    )
    parser.add_argument(
        "--komarowo-report-id",
        default=_env_or_default("OKO_KOMAROWO_REPORT_ID", DEFAULT_KOMAROWO_REPORT_ID),
        help="OKO incident ID for the Komarowo report",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"per-request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the offline plan printer or the explicit live action sequence."""

    args = _parser().parse_args(argv)
    try:
        plan = build_plan(
            skolwin_report_id=args.skolwin_report_id,
            skolwin_task_id=args.skolwin_task_id,
            komarowo_report_id=args.komarowo_report_id,
        )
        if not args.run:
            _print_dry_run(plan)
            return 0

        result = run_live(plan, client=HubClient(timeout=args.timeout))
        if result.flag is None:
            print("Live plan completed, but no {FLG:...} flag was returned.", file=sys.stderr)
            return 1
        return 0
    except (HubAPIError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
