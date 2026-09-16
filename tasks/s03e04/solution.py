"""S03E04 — a narrow catalog-search callback for the negotiations agent.

The Hub's agent sends a natural-language item request as
``{"params": "..."}`` to a public callback.  This module serves one endpoint
which matches that request against the public S03E04 catalog and returns the
city names connected to the matching item as ``{"output": "..."}``.

The default invocation is a dry run.  Starting the callback server, creating
the verification payload, submitting it, and polling verification are separate
explicit commands::

    python -m tasks.s03e04.solution
    python -m tasks.s03e04.solution server --port 0
    python -m tasks.s03e04.solution payload --public-url https://example.test
    python -m tasks.s03e04.solution submit --public-url https://example.test --poll
    python -m tasks.s03e04.solution check

The server has no file-serving route and only accepts POST requests on its
generated callback path.  It binds to loopback by default; deployment can
explicitly choose another host after a suitable public tunnel is available.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import secrets
import sys
import time
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ai_devs.api import get_request, post_request
from src.ai_devs.config import HUB_VERIFY_URL, get_api_key


TASK_NAME = "negotiations"
DATA_URL = "https://hub.ag3nts.org/dane/s03e04_csv"
DATA_FILES = ("cities.csv", "connections.csv", "items.csv")
DEFAULT_CALLBACK_PATH_PREFIX = "/api/catalog-"
MAX_REQUEST_BYTES = 16 * 1024
MAX_RESPONSE_BYTES = 500
# Leave room for the JSON object wrapper and UTF-8 quoting.  The Hub's limit
# is described for the response, so bounding the value below 500 keeps the
# complete ``{"output": ...}`` body within the same limit.
MAX_OUTPUT_BYTES = 480
MIN_OUTPUT_BYTES = 4
FLAG_RE = re.compile(r"\{FLG:[^}]+\}")
CODE_RE = re.compile(r"^[A-Z0-9]{6}$")


# Natural-language framing contributes no identifying information.  Technical
# numbers and units are kept because they usually distinguish otherwise very
# similar catalogue rows.
_STOPWORDS = frozenset(
    {
        "a",
        "aby",
        "and",
        "can",
        "chce",
        "chcialbym",
        "chcialabym",
        "czy",
        "dla",
        "do",
        "find",
        "for",
        "from",
        "gdzie",
        "i",
        "jak",
        "jeden",
        "jedna",
        "jednoczesnie",
        "kiedy",
        "kilo",
        "kupic",
        "mi",
        "need",
        "na",
        "najlepiej",
        "of",
        "oferuje",
        "offers",
        "please",
        "potrzebowac",
        "potrzebuje",
        "prosilbym",
        "prosze",
        "przedmiot",
        "przedmiotu",
        "szukam",
        "the",
        "to",
        "towar",
        "want",
        "which",
        "w",
        "where",
        "with",
        "would",
        "z",
        "znajdz",
    }
)


def _ascii(text: str) -> str:
    """Case-fold and remove accents while preserving useful unit symbols."""

    translated = (
        text.replace("Ω", "ohm")
        .replace("ω", "ohm")
        .replace("µ", "u")
        .replace("μ", "u")
        .replace("×", "x")
        .replace("–", "-")
        .replace("—", "-")
    )
    normalised = unicodedata.normalize("NFKD", translated)
    return "".join(char for char in normalised if not unicodedata.combining(char)).casefold()


def _stem(token: str) -> str:
    """Apply a deliberately small suffix normalisation for Polish/English."""

    if len(token) < 4 or token.isdigit():
        return token
    for suffix in (
        "owego",
        "owej",
        "owym",
        "owych",
        "ami",
        "ach",
        "ing",
        "owa",
        "owe",
        "owi",
        "ied",
        "ing",
        "em",
        "om",
        "ie",
        "es",
        "ed",
        "ie",
        "a",
        "u",
        "y",
        "i",
        "s",
    ):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def normalise_tokens(text: str) -> tuple[str, ...]:
    """Turn names and natural-language requests into comparable tokens."""

    value = _ascii(str(text))
    value = re.sub(r"(?<=\d),(?=\d)", ".", value)
    # Make compact forms such as ``0402SMD`` and ``10ohm`` comparable with
    # spaced catalogue names, while leaving ``kohm`` and ``mhz`` intact.
    value = re.sub(r"(?<=\d)(?=[a-z])", " ", value)
    value = re.sub(r"(?<=[a-z])(?=\d)", " ", value)
    raw_tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", value)
    return tuple(
        _stem(token)
        for token in raw_tokens
        if token not in _STOPWORDS and (len(token) > 1 or token.isdigit())
    )


def _csv_rows(text: str) -> list[dict[str, str]]:
    """Parse a UTF-8 CSV response with a BOM-tolerant header."""

    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    return [
        {
            (str(key).strip().lstrip("\ufeff") if key is not None else ""): str(value or "").strip()
            for key, value in row.items()
        }
        for row in reader
    ]


def _required_field(row: Mapping[str, str], *names: str) -> str:
    """Read a required CSV field while tolerating header case and spacing."""

    normalised = {
        str(key).strip().casefold().replace(" ", "_"): value for key, value in row.items()
    }
    for name in names:
        value = normalised.get(name.casefold().replace(" ", "_"))
        if value:
            return value
    raise ValueError(f"CSV row is missing one of: {', '.join(names)}")


@dataclass(frozen=True)
class ItemMatch:
    """One catalogue item selected for a natural-language request."""

    code: str
    name: str
    score: float


@dataclass(frozen=True)
class Catalog:
    """Immutable in-memory view of the three public CSV files."""

    city_names: Mapping[str, str]
    item_names: Mapping[str, str]
    item_cities: Mapping[str, tuple[str, ...]]
    item_tokens: Mapping[str, tuple[str, ...]]

    @classmethod
    def from_csv_texts(
        cls,
        cities_text: str,
        connections_text: str,
        items_text: str,
    ) -> "Catalog":
        city_rows = _csv_rows(cities_text)
        connection_rows = _csv_rows(connections_text)
        item_rows = _csv_rows(items_text)

        city_names: dict[str, str] = {}
        for row in city_rows:
            code = _required_field(row, "code", "citycode")
            if not CODE_RE.fullmatch(code):
                continue
            city_names[code] = _required_field(row, "name", "city")

        item_names: dict[str, str] = {}
        item_tokens: dict[str, tuple[str, ...]] = {}
        for row in item_rows:
            code = _required_field(row, "code", "itemcode")
            name = _required_field(row, "name", "item")
            if not CODE_RE.fullmatch(code):
                continue
            item_names[code] = name
            item_tokens[code] = normalise_tokens(name)

        item_city_codes: dict[str, set[str]] = {code: set() for code in item_names}
        for row in connection_rows:
            item_code = _required_field(row, "itemcode", "item_code")
            city_code = _required_field(row, "citycode", "city_code")
            if item_code in item_names and city_code in city_names:
                item_city_codes[item_code].add(city_code)

        item_cities = {
            item_code: tuple(sorted(city_names[city_code] for city_code in city_codes))
            for item_code, city_codes in item_city_codes.items()
        }
        if not item_names:
            raise ValueError("items.csv contains no valid items")
        return cls(
            city_names=city_names,
            item_names=item_names,
            item_cities=item_cities,
            item_tokens=item_tokens,
        )

    @classmethod
    def from_directory(cls, directory: Path) -> "Catalog":
        """Load the public CSV trio from a local fixture directory."""

        texts = [(directory / filename).read_text(encoding="utf-8") for filename in DATA_FILES]
        return cls.from_csv_texts(*texts)

    @classmethod
    def from_url(cls, base_url: str = DATA_URL) -> "Catalog":
        """Download the public CSV trio without sending Hub credentials."""

        base = base_url.rstrip("/")
        texts: list[str] = []
        for filename in DATA_FILES:
            response = get_request(f"{base}/{filename}", timeout=30)
            texts.append(response.text)
        return cls.from_csv_texts(*texts)

    def _code_from_query(self, query: str) -> str | None:
        """Use an explicit six-character item code when a request contains one."""

        for token in re.findall(r"\b[A-Za-z0-9]{6}\b", query):
            code = token.upper()
            if code in self.item_names:
                return code
        return None

    @staticmethod
    def _token_similarity(query_token: str, item_token: str) -> float:
        if query_token == item_token:
            return 1.0
        if len(query_token) >= 3 and (
            query_token.startswith(item_token) or item_token.startswith(query_token)
        ):
            return 0.85
        ratio = SequenceMatcher(None, query_token, item_token).ratio()
        return ratio if ratio >= 0.82 else 0.0

    def find_item(self, query: str) -> ItemMatch | None:
        """Select the best catalogue row using weighted token evidence."""

        explicit_code = self._code_from_query(query)
        if explicit_code is not None:
            return ItemMatch(explicit_code, self.item_names[explicit_code], 1000.0)

        query_tokens = normalise_tokens(query)
        if not query_tokens:
            return None

        numeric_query = {token for token in query_tokens if any(char.isdigit() for char in token)}
        candidates: list[ItemMatch] = []
        for code, tokens in self.item_tokens.items():
            if not tokens:
                continue
            numeric_item = {token for token in tokens if any(char.isdigit() for char in token)}
            if numeric_query - numeric_item:
                # A request for 48 V should not silently become a 24 V item.
                continue
            total_weight = 0.0
            matched_weight = 0.0
            matched_count = 0
            for query_token in query_tokens:
                weight = 4.0 if any(char.isdigit() for char in query_token) else 1.0
                if len(query_token) <= 2:
                    weight += 0.5
                total_weight += weight
                best = max((self._token_similarity(query_token, token) for token in tokens), default=0.0)
                if best:
                    matched_weight += weight * best
                    matched_count += 1
            if not total_weight or not matched_count:
                continue
            coverage = matched_weight / total_weight
            full_ratio = SequenceMatcher(
                None,
                " ".join(query_tokens),
                " ".join(tokens),
            ).ratio()
            score = coverage * 100 + matched_count * 2 + full_ratio * 8
            candidates.append(ItemMatch(code, self.item_names[code], score))

        if not candidates:
            return None
        candidates.sort(key=lambda match: (-match.score, match.name, match.code))
        winner = candidates[0]
        # A one-word generic query can match hundreds of rows.  Requiring
        # decent token coverage avoids returning a plausible-looking city list
        # for an underspecified request; normal product descriptions score far
        # above this threshold.
        if winner.score < 38:
            return None
        return winner

    @staticmethod
    def _bounded_output(prefix: str, values: Iterable[str]) -> str:
        """Keep output within the Hub's byte limit without splitting UTF-8."""

        values = tuple(values)
        text = prefix + ", ".join(values)
        if len(text.encode("utf-8")) <= MAX_OUTPUT_BYTES:
            return text
        # City names are the useful result, so truncate at a complete city
        # boundary and keep the response comfortably above four bytes.
        chosen: list[str] = []
        for value in values:
            candidate = prefix + ", ".join([*chosen, value])
            if len(candidate.encode("utf-8")) > MAX_OUTPUT_BYTES:
                break
            chosen.append(value)
        result = prefix + ", ".join(chosen)
        return result if len(result.encode("utf-8")) >= MIN_OUTPUT_BYTES else "Brak"

    def answer(self, query: str) -> str:
        """Return a bounded human-readable list of cities for one request."""

        match = self.find_item(query)
        if match is None:
            return "Nie znaleziono pasującego przedmiotu."
        cities = self.item_cities.get(match.code, ())
        if not cities:
            return "Brak miast dla dopasowanego przedmiotu."
        return self._bounded_output("Miasta: ", cities)


TOOL_DESCRIPTION = (
    "Looks up ONE catalogue item from natural-language params. "
    "POST {\"params\": \"complete item name and specifications\"}; response is "
    "{\"output\": \"Miasta: ...\"}. Call once per required item, then intersect city lists. "
    "Include distinguishing numbers and units; send no multiple items in one call."
)


def callback_path() -> str:
    """Generate a per-process path that does not expose any filesystem surface."""

    return f"{DEFAULT_CALLBACK_PATH_PREFIX}{secrets.token_hex(8)}"


def public_tool_url(public_url: str, path: str) -> str:
    """Join a user-supplied public origin to the callback path."""

    origin = public_url.strip().rstrip("/")
    if not re.fullmatch(r"https?://[^?#\s]+", origin):
        raise ValueError("public URL must be an http(s) URL")
    return origin + "/" + path.lstrip("/")


def verification_payload(public_url: str, path: str) -> dict[str, Any]:
    """Build the exact negotiations registration payload without sending it."""

    return {
        "apikey": "<kept-in-config>",
        "task": TASK_NAME,
        "answer": {
            "tools": [
                {
                    "URL": public_tool_url(public_url, path),
                    "description": TOOL_DESCRIPTION,
                }
            ]
        },
    }


def _real_verification_payload(public_url: str, path: str) -> dict[str, Any]:
    """Build the live payload while keeping the API key out of normal output."""

    payload = verification_payload(public_url, path)
    payload["apikey"] = get_api_key()
    return payload


def _extract_flag(value: Any) -> str | None:
    match = FLAG_RE.search(json.dumps(value, ensure_ascii=False, default=str))
    return match.group(0) if match else None


def submit_tools(public_url: str, path: str) -> dict[str, Any]:
    """Register the callback URLs with the Hub (explicit CLI action only)."""

    return post_request(
        HUB_VERIFY_URL,
        _real_verification_payload(public_url, path),
        raise_on_error=False,
    )


def check_verification() -> dict[str, Any]:
    """Ask the Hub for the asynchronous negotiations result."""

    return post_request(
        HUB_VERIFY_URL,
        {
            "apikey": get_api_key(),
            "task": TASK_NAME,
            "answer": {"action": "check"},
        },
        raise_on_error=False,
    )


def poll_verification(attempts: int = 12, interval: float = 5.0) -> dict[str, Any]:
    """Poll check until a flag or a terminal failure is returned."""

    last: dict[str, Any] = {"ok": False, "error": "no check attempted"}
    for attempt in range(attempts):
        if attempt:
            time.sleep(interval)
        last = check_verification()
        flag = _extract_flag(last)
        if flag:
            return last
        if isinstance(last, dict) and last.get("status") in {"failed", "error", "rejected"}:
            return last
    return last


class _CallbackServer(ThreadingHTTPServer):
    """HTTP server carrying only the immutable catalog and exact callback path."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], catalog: Catalog, path: str):
        super().__init__(address, _CallbackHandler)
        self.catalog = catalog
        self.callback_path = path


class _CallbackHandler(BaseHTTPRequestHandler):
    """Accept exactly the Hub callback shape and return an ``output`` string."""

    server: _CallbackServer

    def _send_output(self, output: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps({"output": output}, ensure_ascii=False).encode("utf-8")
        if len(body) > MAX_RESPONSE_BYTES:
            # This should be unreachable because Catalog._bounded_output uses
            # a 480-byte value budget, but keep the transport contract safe if
            # a future handler supplies a different string.
            output = output.encode("utf-8")[: MAX_RESPONSE_BYTES - 32].decode("utf-8", "ignore")
            body = json.dumps({"output": output}, ensure_ascii=False).encode("utf-8")
            while len(body) > MAX_RESPONSE_BYTES and output:
                output = output[:-1]
                body = json.dumps({"output": output}, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != self.server.callback_path:
            self._send_output("Nieznany endpoint.", HTTPStatus.NOT_FOUND)
            return
        raw_length = self.headers.get("Content-Length", "")
        try:
            length = int(raw_length)
        except ValueError:
            self._send_output("Nieprawidłowy Content-Length.", HTTPStatus.BAD_REQUEST)
            return
        if length < 0 or length > MAX_REQUEST_BYTES:
            self._send_output("Zbyt duże żądanie.", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_output("Nieprawidłowy JSON.", HTTPStatus.BAD_REQUEST)
            return
        params = payload.get("params") if isinstance(payload, dict) else None
        if not isinstance(params, str) or not params.strip():
            self._send_output("Pole params musi być tekstem.", HTTPStatus.BAD_REQUEST)
            return
        self._send_output(self.server.catalog.answer(params))

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == self.server.callback_path:
            self._send_output("Callback działa.")
            return
        self._send_output("Nieznany endpoint.", HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        # Keep access logs compact and avoid echoing request bodies.
        sys.stderr.write("[callback] " + (format % args) + "\n")


def serve(catalog: Catalog, host: str, port: int, path: str) -> None:
    """Run the narrow callback server until interrupted."""

    if not path.startswith("/") or "?" in path or "#" in path:
        raise ValueError("callback path must be an absolute path without query or fragment")
    server = _CallbackServer((host, port), catalog, path)
    actual_port = server.server_address[1]
    print(f"Callback listening on http://{host}:{actual_port}{path}")
    print("Use the public origin plus this path in the negotiations payload.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _dry_run() -> int:
    print("Dry run only: no catalog download, callback server, or /verify call was made.")
    print("Use `server`, `payload`, `submit`, or `check` explicitly for the next step.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S03E04 negotiations callback")
    subparsers = parser.add_subparsers(dest="command")

    server_parser = subparsers.add_parser("server", help="serve the catalog callback")
    server_parser.add_argument("--host", default="127.0.0.1")
    server_parser.add_argument("--port", type=int, default=0)
    server_parser.add_argument("--path", default=None, help="fixed callback path; defaults to a random path")
    server_parser.add_argument("--data-dir", type=Path, default=None)
    server_parser.add_argument("--data-url", default=DATA_URL)

    payload_parser = subparsers.add_parser("payload", help="print a dry-run registration payload")
    payload_parser.add_argument("--public-url", required=True)
    payload_parser.add_argument("--path", default="/api/catalog-tool")

    submit_parser = subparsers.add_parser("submit", help="register the callback with /verify")
    submit_parser.add_argument("--public-url", required=True)
    submit_parser.add_argument("--path", default="/api/catalog-tool")
    submit_parser.add_argument("--poll", action="store_true", help="poll the asynchronous result after registration")
    submit_parser.add_argument("--attempts", type=int, default=12)
    submit_parser.add_argument("--interval", type=float, default=5.0)

    subparsers.add_parser("check", help="poll the asynchronous /verify result once")

    args = parser.parse_args(argv)
    if args.command is None:
        return _dry_run()

    if args.command == "server":
        try:
            catalog = Catalog.from_directory(args.data_dir) if args.data_dir else Catalog.from_url(args.data_url)
            serve(catalog, args.host, args.port, args.path or callback_path())
        except Exception as exc:
            print(f"Server failed ({type(exc).__name__}).", file=sys.stderr)
            return 1
        return 0

    if args.command == "payload":
        print(json.dumps(verification_payload(args.public_url, args.path), ensure_ascii=False, indent=2))
        return 0

    if args.command == "submit":
        try:
            result = submit_tools(args.public_url, args.path)
            print(json.dumps(result, ensure_ascii=False, default=str))
            flag = _extract_flag(result)
            if flag:
                print(f"FLAG: {flag}")
                return 0
            if args.poll:
                result = poll_verification(args.attempts, args.interval)
                print(json.dumps(result, ensure_ascii=False, default=str))
                flag = _extract_flag(result)
                if flag:
                    print(f"FLAG: {flag}")
                    return 0
            return 0 if isinstance(result, dict) and not result.get("error") else 1
        except Exception as exc:
            print(f"Verification failed ({type(exc).__name__}).", file=sys.stderr)
            return 1

    if args.command == "check":
        try:
            result = check_verification()
            print(json.dumps(result, ensure_ascii=False, default=str))
            flag = _extract_flag(result)
            if flag:
                print(f"FLAG: {flag}")
                return 0
            return 0 if isinstance(result, dict) and not result.get("error") else 1
        except Exception as exc:
            print(f"Verification check failed ({type(exc).__name__}).", file=sys.stderr)
            return 1

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
