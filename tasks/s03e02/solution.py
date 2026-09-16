"""S03E02 — repair and start the ECCS cooling firmware.

The challenge VM exposes a small command interpreter through the Hub rather
than a normal shell.  This solver keeps that protocol in one guarded tool,
reads the safe task files, repairs the deliberately broken ``settings.ini``
with ``editline``, starts the firmware with the password found in the VM, and
submits the returned ECCS confirmation.

The default invocation is a dry run and makes no network calls::

    python -m tasks.s03e02.solution
    python -m tasks.s03e02.solution --run

``--run`` is intentionally the only mode that can reboot the VM, edit its
configuration, execute the firmware, or send the final answer to ``/verify``.
The API key is kept inside the shared configuration and HTTP helpers.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ai_devs import post_request, send_report
from src.ai_devs.config import HUB_API_URL


TASK_NAME = "firmware"
SHELL_URL = f"{HUB_API_URL}/shell"
FIRMWARE_PATH = "/opt/firmware/cooler/cooler.bin"
SETTINGS_PATH = "/opt/firmware/cooler/settings.ini"
IGNORE_FILE_PATH = "/opt/firmware/cooler/.gitignore"
LOCK_PATH = "/opt/firmware/cooler/cooler-is-blocked.lock"

# These are the only paths needed by the task.  They deliberately exclude the
# ignored ``.env``, ``storage.cfg``, and ``logs/`` entries in the firmware
# directory.  The history and notes are ordinary operator files and are safe
# sources for the access password.
PASSWORD_PATHS = (
    "/home/operator/notes/pass.txt",
    "/tmp/aidevs4.txt",
    "/home/operator/.bash_history",
)

PROTECTED_ROOTS = ("/etc", "/root", "/proc")
IGNORED_VM_PATHS = (
    "/opt/firmware/cooler/.env",
    "/opt/firmware/cooler/storage.cfg",
    "/opt/firmware/cooler/logs",
)

FLAG_RE = re.compile(r"\{FLG:[^}]+\}")
CONFIRMATION_RE = re.compile(r"\bECCS-[0-9A-Fa-f]{40}\b", re.IGNORECASE)

# The VM's command set is intentionally small.  Deletion is limited to the
# firmware's own lock file, which the task requires removing after repairing
# the configuration.  The safety policy rejects shell syntax that could turn
# one request into an arbitrary command sequence.
COMMAND_RE = re.compile(r"^(?:help|ls|cat|cd|pwd|editline|reboot|date|uptime|history|whoami)(?:\s+.*)?$")
FIRMWARE_COMMAND_RE = re.compile(
    r"^/opt/firmware/cooler/cooler\.bin(?:\s+[-A-Za-z0-9_./=]+)*$"
)
LOCK_REMOVE_COMMAND_RE = re.compile(
    r"^rm /opt/firmware/cooler/cooler-is-blocked\.lock$"
)
SHELL_META_CHARS = frozenset(";|&<>$`\n\r")


@dataclass(frozen=True)
class ShellResult:
    """Normalised result from one custom VM command."""

    command: str
    response: Any
    data: str


class ShellPolicyError(ValueError):
    """Raised when a command would violate the challenge's VM rules."""


def _json_text(value: Any) -> str:
    """Convert a Hub value to bounded printable text without leaking secrets."""

    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _extract_data(value: Any) -> str:
    """Extract command output from the Hub response shape."""

    if isinstance(value, dict) and "data" in value:
        return _json_text(value["data"])
    return _json_text(value)


def _contains_forbidden_path(command: str) -> str | None:
    """Return a forbidden path marker found in a command, if any."""

    normalised = command.replace("\\", "/")
    for root in PROTECTED_ROOTS:
        if re.search(rf"(?<![A-Za-z0-9_.-]){re.escape(root)}(?:/|$)", normalised):
            return root
    for path in IGNORED_VM_PATHS:
        if path in normalised:
            return path
    # Any traversal through the ignored logs directory is forbidden, including
    # a trailing slash or a child path.
    if re.search(r"/opt/firmware/cooler/logs(?:/|$)", normalised):
        return "/opt/firmware/cooler/logs"
    return None


def validate_command(command: str) -> str:
    """Validate one VM command before it is sent to the external challenge."""

    if not isinstance(command, str) or not command.strip():
        raise ShellPolicyError("command must be a non-empty string")
    command = command.strip()
    if any(char in command for char in SHELL_META_CHARS):
        raise ShellPolicyError("shell chaining and redirection are disabled")
    forbidden = _contains_forbidden_path(command)
    if forbidden is not None:
        raise ShellPolicyError(f"challenge policy forbids access to {forbidden}")
    if command.startswith("/opt/firmware/cooler/cooler.bin"):
        if not FIRMWARE_COMMAND_RE.fullmatch(command):
            raise ShellPolicyError("firmware command contains unsupported arguments")
        return command
    if command.startswith("rm "):
        if not LOCK_REMOVE_COMMAND_RE.fullmatch(command):
            raise ShellPolicyError("deletion is limited to the firmware lock file")
        return command
    if not COMMAND_RE.fullmatch(command):
        raise ShellPolicyError("unsupported VM command")
    return command


def _shell_request(command: str) -> ShellResult:
    """Execute one validated custom VM command and preserve Hub diagnostics."""

    command = validate_command(command)
    payload = {"apikey": _api_key(), "cmd": command}
    response = post_request(SHELL_URL, payload, raise_on_error=False, timeout=120)
    return ShellResult(command=command, response=response, data=_extract_data(response))


def _api_key() -> str:
    """Resolve the Hub key lazily so importing the dry-run module is harmless."""

    from src.ai_devs.config import get_api_key

    return get_api_key()


def _response_is_error(response: Any) -> bool:
    """Recognise both HTTP errors and error-shaped successful JSON responses."""

    if not isinstance(response, dict):
        return True
    status = response.get("http_status")
    if isinstance(status, int) and status >= 400:
        return True
    code = response.get("code")
    if isinstance(code, int) and code < 0:
        return True
    return bool(response.get("error"))


def _extract_confirmation(value: Any) -> str | None:
    """Extract the exact 40-hex-digit ECCS confirmation from command output."""

    match = CONFIRMATION_RE.search(_json_text(value))
    return match.group(0) if match else None


def _extract_flag(value: Any) -> str | None:
    """Extract a returned Hub flag without manufacturing one."""

    match = FLAG_RE.search(_json_text(value))
    return match.group(0) if match else None


def _read(path: str) -> ShellResult:
    """Read one explicitly allowlisted, non-ignored VM file."""

    return _shell_request(f"cat {path}")


def _read_text(path: str) -> str:
    """Read a file and raise a bounded diagnostic on a Hub error."""

    result = _read(path)
    if _response_is_error(result.response):
        raise RuntimeError(f"VM could not read {path}: {_response_summary(result.response)}")
    return result.data


def _bounded(value: Any, limit: int = 900) -> str:
    """Keep VM error output useful without flooding the terminal."""

    text = _json_text(value).replace("\x00", "")
    return text if len(text) <= limit else text[:limit] + "..."


def _safe_output_hint(value: Any, secrets: Iterable[str] = ()) -> str:
    """Return bounded firmware diagnostics with credentials and URLs removed."""

    text = _bounded(value)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<redacted>")
    text = re.sub(r"https?://[^\s\"']+", "<url-redacted>", text, flags=re.IGNORECASE)
    text = re.sub(
        r"(?i)((?:api[_-]?key|password|passphrase|secret|token)\s*[:=]\s*)[^,}\s]+",
        r"\1<redacted>",
        text,
    )
    return _bounded(text)


def _response_summary(
    value: Any,
    *,
    include_data: bool = False,
    secrets: Iterable[str] = (),
) -> str:
    """Summarise an error, optionally exposing sanitized VM diagnostics."""

    if isinstance(value, dict):
        safe = {
            key: value[key]
            for key in ("http_status", "code", "message", "error")
            if key in value
        }
        if include_data and "data" in value:
            safe["data"] = _safe_output_hint(value["data"], secrets)
        return _bounded(safe or {"type": "unexpected_response"})
    return f"unexpected response type: {type(value).__name__}"


def _parse_ignored_patterns(text: str) -> tuple[str, ...]:
    """Parse a simple gitignore file and reject patterns we cannot enforce."""

    patterns: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            # The challenge fixture uses basename, glob, and directory rules.
            # Refuse a more expressive rule rather than accidentally touching
            # a path whose protection we cannot model precisely.
            if not re.fullmatch(r"/?[A-Za-z0-9._*?/-]+", line):
                raise ShellPolicyError(f"unsupported VM ignore pattern: {line!r}")
            patterns.append(line)
    return tuple(patterns)


def _ignored_by_pattern(path: str, pattern: str) -> bool:
    """Apply the subset of gitignore semantics used by the challenge VM."""

    root = PurePosixPath("/opt/firmware/cooler")
    candidate = PurePosixPath(path)
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError:
        return False

    rule = pattern.lstrip("/")
    directory_rule = rule.endswith("/")
    rule = rule.rstrip("/")
    if not rule:
        return False
    if directory_rule and (relative == rule or relative.startswith(f"{rule}/")):
        return True
    # A basename rule such as ``.env`` or ``*.log`` applies at any depth;
    # slash-containing rules match from the firmware directory root.
    if "/" in rule:
        return fnmatch.fnmatchcase(relative, rule)
    return fnmatch.fnmatchcase(PurePosixPath(relative).name, rule)


def _assert_not_ignored(path: str, patterns: Sequence[str]) -> None:
    """Fail closed before touching a path covered by the fetched ignore file."""

    for pattern in patterns:
        if _ignored_by_pattern(path, pattern):
            raise ShellPolicyError(f"challenge .gitignore forbids access to {path}")


def _password_candidates(texts: Iterable[tuple[str, str]]) -> list[str]:
    """Return plausible password tokens in source order, without hardcoding one."""

    primary: list[str] = []
    fallback: list[str] = []
    excluded = {
        "http",
        "https",
        "www",
        "youtube",
        "com",
        "watch",
        "opt",
        "firmware",
        "cooler",
        "bin",
        "admin",
    }
    for path, text in texts:
        # A pass.txt file is a stronger source than a history file.  Keep its
        # first non-empty line first, then use token extraction as a fallback.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if path.endswith("/pass.txt") and lines:
            candidate = lines[0]
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,63}", candidate):
                primary.append(candidate)
        for token in re.findall(r"\b[A-Za-z0-9][A-Za-z0-9_.-]{2,63}\b", text):
            lowered = token.casefold()
            if lowered not in excluded and token not in primary and token not in fallback:
                fallback.append(token)
    # The dedicated pass.txt source is authoritative.  Falling back to
    # arbitrary history/URL tokens would create repeated wrong-password runs
    # and can trigger the VM lockout, so callers must prefer this first group.
    return primary or fallback[:1]


def _settings_lines(text: str) -> list[str]:
    """Return settings lines while preserving line numbers for editline."""

    # ``splitlines`` drops the final newline, which is harmless because the VM
    # replacement command addresses existing lines one at a time.
    return text.splitlines()


def _line_in_section(lines: Sequence[str], section: str, key: str) -> int | None:
    """Find a one-based line number for ``key`` inside an INI section."""

    current: str | None = None
    for index, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1].strip()
            continue
        if current == section and re.match(rf"^#?\s*{re.escape(key)}\s*=", stripped):
            return index
    return None


def _line_value(lines: Sequence[str], line_number: int) -> str:
    """Return the unindented content of one one-based settings line."""

    return lines[line_number - 1].strip()


def _desired_setting_changes(text: str) -> list[tuple[int, str]]:
    """Identify the deliberately broken settings that must be repaired."""

    lines = _settings_lines(text)
    changes: list[tuple[int, str]] = []

    safety_line = next(
        (
            index
            for index, raw_line in enumerate(lines, start=1)
            if re.match(r"^\s*#?\s*SAFETY_CHECK\s*=", raw_line)
        ),
        None,
    )
    if safety_line is not None and _line_value(lines, safety_line) != "SAFETY_CHECK=pass":
        changes.append((safety_line, "SAFETY_CHECK=pass"))

    test_line = _line_in_section(lines, "test_mode", "enabled")
    if test_line is not None and _line_value(lines, test_line) != "enabled=false":
        changes.append((test_line, "enabled=false"))

    cooling_line = _line_in_section(lines, "cooling", "enabled")
    if cooling_line is not None and _line_value(lines, cooling_line) != "enabled=true":
        changes.append((cooling_line, "enabled=true"))

    return changes


def _edit_settings(changes: Sequence[tuple[int, str]]) -> list[ShellResult]:
    """Apply validated line replacements through the VM's custom edit command."""

    results: list[ShellResult] = []
    for line_number, content in changes:
        if not 1 <= line_number <= 200:
            raise ValueError(f"invalid settings line number: {line_number}")
        if any(char in content for char in SHELL_META_CHARS):
            raise ValueError("settings replacement contains shell syntax")
        result = _shell_request(f"editline {SETTINGS_PATH} {line_number} {content}")
        results.append(result)
        if _response_is_error(result.response):
            raise RuntimeError(
                f"VM rejected settings line {line_number}: {_response_summary(result.response)}"
            )
    return results


def _remove_block_lock(ignored_patterns: Sequence[str]) -> None:
    """Remove only the task's generated lock after settings are repaired."""

    _assert_not_ignored(LOCK_PATH, ignored_patterns)
    listing = _shell_request("ls /opt/firmware/cooler")
    if _response_is_error(listing.response):
        raise RuntimeError(f"VM could not inspect firmware directory: {_response_summary(listing.response)}")
    # ``ls`` may return either newline-delimited text or a JSON list; a
    # filename substring is sufficient because the name is exact and fixed.
    if PurePosixPath(LOCK_PATH).name not in listing.data:
        return
    result = _shell_request(f"rm {LOCK_PATH}")
    if _response_is_error(result.response):
        raise RuntimeError(f"VM could not remove firmware lock: {_response_summary(result.response)}")


def _print_dry_run() -> None:
    """Describe the live flow without resolving credentials or contacting Hub."""

    print("Dry run only: no Hub or VM calls were made.")
    print(f"Shell endpoint: {SHELL_URL}")
    print(f"Firmware: {FIRMWARE_PATH}")
    print(f"Settings: {SETTINGS_PATH}")
    print("Live flow: read help and safe files, repair settings, run firmware, verify confirmation.")


def solve_live() -> str:
    """Repair the VM and return the exact confirmation emitted by the firmware."""

    # A previous failed attempt leaves a lock file.  Reboot rebuilds the VM
    # state before the solver makes any edits and is explicitly supported by
    # the task.  It also makes rerunning a failed local attempt predictable.
    reboot = _shell_request("reboot")
    if _response_is_error(reboot.response):
        raise RuntimeError(f"VM reboot failed: {_response_summary(reboot.response)}")

    help_result = _shell_request("help")
    if _response_is_error(help_result.response):
        raise RuntimeError(f"VM help failed: {_response_summary(help_result.response)}")

    ignore_text = _read_text(IGNORE_FILE_PATH)
    ignored_patterns = _parse_ignored_patterns(ignore_text)
    print(f"VM ignore rules respected: {', '.join(ignored_patterns) or '(none)'}")

    _assert_not_ignored(SETTINGS_PATH, ignored_patterns)
    settings_text = _read_text(SETTINGS_PATH)
    changes = _desired_setting_changes(settings_text)
    if changes:
        print(f"Repairing {len(changes)} firmware setting line(s).")
        _edit_settings(changes)
    else:
        print("Firmware settings already match the safe target configuration.")
    _remove_block_lock(ignored_patterns)

    source_texts: list[tuple[str, str]] = []
    for path in PASSWORD_PATHS:
        try:
            source_texts.append((path, _read_text(path)))
        except RuntimeError:
            # The VM may omit one of the secondary copies.  Keep looking; the
            # final error below remains bounded and contains no credentials.
            continue
    passwords = _password_candidates(source_texts)
    if not passwords:
        raise RuntimeError("No usable firmware password was found in safe VM files.")

    _assert_not_ignored(FIRMWARE_PATH, ignored_patterns)
    last_run: ShellResult | None = None
    for password in passwords[:1]:
        # Passwords come from VM content, but constrain their shape before
        # placing one in the custom command string.
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,63}", password):
            continue
        run_result = _shell_request(f"{FIRMWARE_PATH} {password}")
        last_run = run_result
        # The shell endpoint has returned both a top-level response object and
        # a normalized ``data`` string in different challenge deployments.
        # Inspect both representations so a valid detached-mode confirmation
        # is not lost to a response-shape difference.
        confirmation = _extract_confirmation(run_result.response)
        if confirmation is None:
            confirmation = _extract_confirmation(run_result.data)
        if confirmation is not None:
            return confirmation
        if _response_is_error(run_result.response):
            continue

    if last_run is None:
        raise RuntimeError("No safe password candidate could be executed.")
    raise RuntimeError(
        "Firmware did not emit an ECCS confirmation. "
        f"Last VM response: {_response_summary(last_run.response, include_data=True, secrets=passwords)}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dry-run or authorized live firmware flow."""

    parser = argparse.ArgumentParser(description="S03E02 ECCS firmware solver")
    parser.add_argument(
        "--run",
        action="store_true",
        help="reboot and repair the challenge VM, then submit the confirmation",
    )
    args = parser.parse_args(argv)

    if not args.run:
        _print_dry_run()
        return 0

    try:
        confirmation = solve_live()
    except Exception as exc:
        # Keep diagnostics useful for a live VM failure while removing URLs,
        # credential-shaped values, and any implementation traceback details.
        detail = _safe_output_hint(str(exc))
        print(f"Solver failed ({type(exc).__name__}): {detail}", file=sys.stderr)
        return 1

    print(f"RETURNED FROM CODE: {confirmation}")
    try:
        response = send_report(TASK_NAME, {"confirmation": confirmation})
    except Exception as exc:
        print(f"Verification failed ({type(exc).__name__}).", file=sys.stderr)
        return 1

    flag = _extract_flag(response)
    if flag is None:
        print(f"Hub response: {_response_summary(response)}")
        print("Hub did not return a flag.", file=sys.stderr)
        return 1
    print(f"RETURNED FROM CODE: {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
