"""S02E03 — condense the power-plant failure log.

The Hub serves a large, noisy ``failure.log``.  This solver extracts the
first terminal escalation for every subsystem, keeps the original event text
on one line, and submits the resulting digest.  The selection is derived from
the downloaded records; subsystem names and a flag are never embedded in the
solution.

The default invocation is a dry run so importing or reviewing this module
cannot contact the Hub::

    python -m tasks.s02e03.solution
    python -m tasks.s02e03.solution --run

When the Hub rejects a digest, ``--run`` tries the next deterministic
selection policy and prints only the Hub's feedback and status.  API keys are
kept inside the shared helpers and are never printed by this module.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ai_devs import get_api_key, get_hub_data, post_request
from src.ai_devs.config import HUB_VERIFY_URL


TASK_NAME = "failure"
LOG_FILENAME = "failure.log"
MAX_LOG_TOKENS = 1500

# Hub records currently use ``[YYYY-MM-DD HH:MM:SS] [ERRO] message``.  Keep
# seconds optional so the parser also works with the minute precision shown in
# the lesson examples and with future log exports.
_LINE_RE = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?)\]"
    r"\s+\[(?P<severity>[^]]+)\]\s+(?P<message>\S.*)$"
)
_IDENTIFIER_RE = re.compile(r"\b[A-Z][A-Z0-9]{2,}(?:_[A-Z0-9]+)?\b")
_TIMESTAMP_MINUTE_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}")

# ``ERRO`` is the spelling used by the Hub fixture.  Include common aliases
# for portability without changing the original severity in rendered output.
_SEVERITY_RANK = {
    "TRACE": 0,
    "DEBUG": 0,
    "INFO": 0,
    "NOTICE": 1,
    "WARN": 1,
    "WARNING": 1,
    "ERROR": 2,
    "ERRO": 2,
    "ERR": 2,
    "ALERT": 3,
    "CRIT": 3,
    "CRITICAL": 3,
    "FATAL": 3,
    "EMERG": 3,
    "EMERGENCY": 3,
}


@dataclass(frozen=True)
class LogEntry:
    """One structured log record, retaining its source ordering."""

    line_number: int
    timestamp: str
    severity: str
    message: str
    raw: str

    @property
    def minute(self) -> str:
        """Return the required ``YYYY-MM-DD HH:MM`` timestamp precision."""
        match = _TIMESTAMP_MINUTE_RE.match(self.timestamp)
        return match.group(0) if match else self.timestamp


def parse_log(text: str) -> list[LogEntry]:
    """Parse structured records while ignoring blank or malformed lines.

    A malformed line cannot safely satisfy the required timestamp/severity
    fields, so it is excluded from the digest rather than guessed at.
    """

    entries: list[LogEntry] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        match = _LINE_RE.match(line)
        if match is None:
            continue
        entries.append(
            LogEntry(
                line_number=line_number,
                timestamp=match.group("timestamp"),
                severity=match.group("severity").strip(),
                message=match.group("message").strip(),
                raw=line,
            )
        )
    return entries


def _message_identifiers(message: str) -> list[str]:
    """Return uppercase identifier tokens in their source order."""

    return _IDENTIFIER_RE.findall(message)


def derive_components(entries: Sequence[LogEntry], minimum_occurrences: int = 2) -> set[str]:
    """Derive recurring subsystem identifiers from the log itself.

    One-off uppercase markers (for example a configuration key embedded in an
    error message) are deliberately excluded.  This avoids hard-coding the
    current subsystem list and prevents those markers becoming false events.
    """

    counts = Counter(
        identifier
        for entry in entries
        for identifier in _message_identifiers(entry.message)
    )
    return {
        identifier
        for identifier, count in counts.items()
        if count >= minimum_occurrences
    }


def _entry_component(entry: LogEntry, components: set[str]) -> str | None:
    """Choose the subsystem token for an entry.

    The first recurring identifier is the primary component in the Hub log.
    If a message references multiple components, preferring the first one
    preserves the source's subject (e.g. a tank message mentioning ECCS8).
    """

    for identifier in _message_identifiers(entry.message):
        if identifier in components:
            return identifier
    return None


def _severity_rank(severity: str) -> int:
    return _SEVERITY_RANK.get(severity.strip().upper(), 0)


def _first_by_component(
    entries: Sequence[LogEntry],
    components: set[str],
    minimum_rank: int,
    *,
    maximum_rank_only: bool = True,
) -> list[LogEntry]:
    """Select one earliest escalation record for every derived component."""

    grouped: dict[str, list[LogEntry]] = defaultdict(list)
    for entry in entries:
        component = _entry_component(entry, components)
        if component is not None and _severity_rank(entry.severity) >= minimum_rank:
            grouped[component].append(entry)

    selected: list[LogEntry] = []
    for component, component_entries in grouped.items():
        if maximum_rank_only:
            highest = max(_severity_rank(entry.severity) for entry in component_entries)
            component_entries = [
                entry
                for entry in component_entries
                if _severity_rank(entry.severity) == highest
            ]
        selected.append(min(component_entries, key=lambda entry: entry.line_number))

    return sorted(selected, key=lambda entry: entry.line_number)


def _last_by_component(
    entries: Sequence[LogEntry],
    components: set[str],
    minimum_rank: int,
) -> list[LogEntry]:
    """Select the final highest-severity event for each component."""

    grouped: dict[str, list[LogEntry]] = defaultdict(list)
    for entry in entries:
        component = _entry_component(entry, components)
        if component is not None and _severity_rank(entry.severity) >= minimum_rank:
            grouped[component].append(entry)

    selected: list[LogEntry] = []
    for component, component_entries in grouped.items():
        highest = max(_severity_rank(entry.severity) for entry in component_entries)
        selected.append(
            max(
                (
                    entry
                    for entry in component_entries
                    if _severity_rank(entry.severity) == highest
                ),
                key=lambda entry: entry.line_number,
            )
        )
    return sorted(selected, key=lambda entry: entry.line_number)


def _unique_events(
    entries: Sequence[LogEntry],
    components: set[str],
    minimum_rank: int,
) -> list[LogEntry]:
    """Keep the first record for every distinct severe event template."""

    seen: set[tuple[str, str, str]] = set()
    selected: list[LogEntry] = []
    for entry in entries:
        component = _entry_component(entry, components)
        if component is None or _severity_rank(entry.severity) < minimum_rank:
            continue
        key = (component, entry.severity.strip().upper(), entry.message)
        if key in seen:
            continue
        seen.add(key)
        selected.append(entry)
    return selected


def _render_entry(entry: LogEntry) -> str:
    """Render one event with minute precision and its original message."""

    return f"[{entry.minute}] [{entry.severity}] {entry.message}"


def _token_count(text: str) -> tuple[int, str]:
    """Count tokens with the closest available Hub tokenizer.

    ``tiktoken`` is optional in this small repository.  The fallback is
    intentionally conservative and only matters for unusually large future
    digests; the normal seven-event digest is far below the limit.
    """

    try:
        import tiktoken  # type: ignore

        encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(text)), "o200k_base"
    except Exception:
        return max(1, (len(text) + 2) // 3), "conservative character estimate"


def _fit_token_budget(events: Iterable[LogEntry], limit: int = MAX_LOG_TOKENS) -> str:
    """Render events in source order, never exceeding the strict token limit."""

    rendered: list[str] = []
    for event in events:
        candidate = "\n".join([*rendered, _render_entry(event)])
        tokens, _ = _token_count(candidate)
        if tokens >= limit:
            break
        rendered.append(_render_entry(event))
    return "\n".join(rendered)


def build_digest(
    entries: Sequence[LogEntry],
    policy: str = "first-critical",
    limit: int = MAX_LOG_TOKENS,
) -> tuple[str, dict[str, Any]]:
    """Build a bounded digest using a named, deterministic policy.

    Policies are deliberately generic so feedback can select a new view of
    the same downloaded log without embedding the answer:

    ``first-critical``
        earliest event at the highest severity observed for each component;
    ``first-severe``
        earliest ERRO/CRIT event for each component;
    ``first-warning``
        earliest WARN or more severe event for each component;
    ``last-critical``
        latest event at each component's highest severity.
    ``unique-severe``
        one chronological example of every distinct ERRO/CRIT event template;
        this gives the Hub enough context without repeating hundreds of lines.
    """

    components = derive_components(entries)
    if policy == "first-critical":
        # Select the highest level observed for every component.  On the
        # current fixture each component eventually reaches CRIT; allowing a
        # lower terminal level keeps the parser useful for partial exports.
        selected = _first_by_component(entries, components, 1)
    elif policy == "first-severe":
        selected = _first_by_component(
            entries,
            components,
            2,
            maximum_rank_only=False,
        )
    elif policy == "first-warning":
        selected = _first_by_component(
            entries,
            components,
            1,
            maximum_rank_only=False,
        )
    elif policy == "last-critical":
        selected = _last_by_component(entries, components, 3)
    elif policy == "unique-severe":
        selected = _unique_events(entries, components, 2)
    else:
        raise ValueError(f"unknown digest policy: {policy}")

    digest = _fit_token_budget(selected, limit)
    tokens, tokenizer = _token_count(digest)
    metadata = {
        "policy": policy,
        "parsed_records": len(entries),
        "components": sorted(components),
        "selected_records": len(selected),
        "digest_lines": len(digest.splitlines()) if digest else 0,
        "tokens": tokens,
        "tokenizer": tokenizer,
    }
    return digest, metadata


def _feedback_text(response: Any) -> str:
    """Return bounded, non-secret feedback suitable for a progress message."""

    if isinstance(response, dict):
        for key in ("feedback", "message", "error", "hint", "detail"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:500]
    if isinstance(response, str):
        return response.strip()[:500]
    return json.dumps(response, ensure_ascii=False, sort_keys=True)[:500]


def _extract_flag(value: Any) -> str | None:
    """Extract a Hub flag for the caller without ever manufacturing one."""

    match = re.search(r"\{FLG:[^}]+\}", json.dumps(value, ensure_ascii=False))
    return match.group(0) if match else None


def _download_log() -> list[LogEntry]:
    response = get_hub_data(LOG_FILENAME)
    text = getattr(response, "text", response)
    if not isinstance(text, str):
        raise TypeError("Hub log response did not contain text")
    entries = parse_log(text)
    if not entries:
        raise ValueError("failure.log contained no structured records")
    return entries


def _submit(answer: dict[str, str]) -> dict[str, Any]:
    """Submit through the stable shared HTTP helper without raising on errors."""

    payload = {
        "apikey": get_api_key(),
        "task": TASK_NAME,
        "answer": answer,
    }
    result = post_request(HUB_VERIFY_URL, payload, raise_on_error=False)
    return result if isinstance(result, dict) else {"response": result}


def _next_policies(response: Any, current: str) -> list[str]:
    """Use Hub feedback to order deterministic fallback policies."""

    feedback = _feedback_text(response).lower()
    policies = [
        "first-critical",
        "unique-severe",
        "first-severe",
        "first-warning",
        "last-critical",
    ]
    if any(word in feedback for word in ("short", "more line", "more context", "more event")):
        preferred = "unique-severe"
    elif any(word in feedback for word in ("latest", "last", "final", "terminal")):
        preferred = "last-critical"
    elif "warning" in feedback or "warn" in feedback:
        preferred = "first-warning"
    elif any(word in feedback for word in ("error", "erro", "severe")):
        preferred = "first-severe"
    else:
        preferred = "first-critical"
    return [preferred, *[policy for policy in policies if policy != preferred and policy != current]]


def run(max_attempts: int = 4) -> str | None:
    """Download once and iterate bounded candidate digests until accepted."""

    entries = _download_log()
    policies = [
        "first-critical",
        "unique-severe",
        "first-severe",
        "first-warning",
        "last-critical",
    ]
    attempted: set[str] = set()
    response: Any = None

    for _ in range(max(1, max_attempts)):
        if not policies:
            break
        policy = policies.pop(0)
        if policy in attempted:
            continue
        attempted.add(policy)
        digest, metadata = build_digest(entries, policy=policy)
        if not digest:
            print(f"[{policy}] no candidate events found; continuing")
            continue
        print(
            f"[{policy}] parsed={metadata['parsed_records']} "
            f"components={len(metadata['components'])} "
            f"lines={metadata['digest_lines']} tokens<{MAX_LOG_TOKENS} "
            f"({metadata['tokens']}, {metadata['tokenizer']})"
        )
        response = _submit({"logs": digest})
        flag = _extract_flag(response)
        if flag:
            return flag
        feedback = _feedback_text(response)
        print(f"[{policy}] Hub feedback: {feedback}")
        policies = [
            fallback
            for fallback in _next_policies(response, policy)
            if fallback not in attempted
        ]

    if response is not None:
        print("Hub did not accept a digest within the attempt limit.")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="AI DEVS S02E03 failure-log digest")
    parser.add_argument("--run", action="store_true", help="allow live Hub download and verification")
    parser.add_argument("--max-attempts", type=int, default=4, help="maximum feedback iterations")
    args = parser.parse_args()

    if not args.run:
        print("Dry run only: no Hub request was made.")
        print("Use --run to download failure.log and submit a bounded digest.")
        return

    try:
        flag = run(max_attempts=args.max_attempts)
    except Exception as exc:
        print(f"S02E03 failed ({type(exc).__name__}); check Hub configuration and network.", file=sys.stderr)
        raise SystemExit(1)
    if flag:
        print(f"FLAG: {flag}")
        return
    raise SystemExit(1)


if __name__ == "__main__":
    main()
