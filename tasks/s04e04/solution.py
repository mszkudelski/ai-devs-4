"""S04E04 — organise Natan's notes in the Hub virtual filesystem.

The notes ZIP contains three independent sources: city demand announcements,
conversation notes naming the trade coordinators, and transactions that map
goods to the cities selling them.  This module parses those sources, validates
the requested filesystem topology, creates it in one ordered batch, and calls
``done`` for verification.

The default invocation is a dry run and performs no network calls::

    python -m tasks.s04e04.solution

Use ``--run`` for the authenticated live operation.  The verifier flag is
printed from the ``done`` response and is never sent to another endpoint.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import unicodedata
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


TASK_NAME = "filesystem"
NOTES_FILENAME = "natan_notes.zip"
FLAG_RE = re.compile(r"\{FLG:[^}]+\}")
ALLOWED_NAME_RE = re.compile(r"^[a-z0-9_]+$")
MAX_DIRECTORY_NAME_LENGTH = 30
MAX_FILE_NAME_LENGTH = 20
REQUIRED_NOTE_FILES = frozenset({"ogłoszenia.txt", "rozmowy.txt", "transakcje.txt"})

# The names below are the eight coordinators mentioned in the supplied
# conversation notes.  They are kept as a small allowlist because two notes
# intentionally split a person's first and last name across separate
# sentences (Rafal/Kisiel and Lena/Konkel).  parse_people still verifies every
# part against the downloaded source before producing a file.
COORDINATORS: dict[str, tuple[str, str]] = {
    "domatowo": ("Natan", "Rams"),
    "opalino": ("Iga", "Kapecka"),
    "brudzewo": ("Rafal", "Kisiel"),
    "darzlubie": ("Marta", "Frantz"),
    "celbowo": ("Oskar", "Radtke"),
    "mechowo": ("Eliza", "Redmann"),
    "puck": ("Damian", "Kroll"),
    "karlinkowo": ("Lena", "Konkel"),
}

# A transaction uses inflected forms and Polish diacritics, while filesystem
# names must be ASCII singular nouns.  Matching roots handles both the
# announcement wording (e.g. ``butelek wody``) and transaction wording.
ITEM_ROOTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("wolowina", ("wolowin",)),
    ("wiertarka", ("wiertark", "wiertarek", "wiertar")),
    ("makaron", ("makaron",)),
    ("ziemniak", ("ziemniak",)),
    ("marchew", ("marchew",)),
    ("kapusta", ("kapust",)),
    ("kurczak", ("kurczak",)),
    ("mlotek", ("mlotek", "mlot")),
    ("lopata", ("lopat",)),
    ("kilof", ("kilof",)),
    ("chleb", ("chleb",)),
    ("maka", ("maka",)),
    ("woda", ("wod",)),
    ("ryz", ("ryz",)),
)

CITY_FORMS: dict[str, tuple[str, ...]] = {
    "domatowo": ("domatowo", "domatowa"),
    "opalino": ("opalino", "opalina"),
    "brudzewo": ("brudzewo", "brudzewa"),
    "darzlubie": ("darzlubie", "darzlubiem", "darzlubiu"),
    "celbowo": ("celbowo", "celbowa"),
    "mechowo": ("mechowo",),
    "puck": ("puck", "pucka"),
    "karlinkowo": ("karlinkowo",),
}


class FilesystemError(RuntimeError):
    """Raised when source data or a Hub filesystem operation is invalid."""


Coordinate = tuple[int, int]
Action = dict[str, Any]
ActionCallable = Callable[[Any], Mapping[str, Any]]


@dataclass(frozen=True)
class Person:
    """One trade coordinator and the city associated with the notes."""

    first_name: str
    last_name: str
    city: str

    @property
    def display_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


@dataclass(frozen=True)
class KnowledgeBase:
    """Normalised facts extracted from the three note files."""

    city_needs: dict[str, dict[str, int]]
    people: tuple[Person, ...]
    sellers: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class FileEntry:
    """One file to create in the virtual filesystem."""

    path: str
    content: str


def _ascii(value: Any) -> str:
    """Fold Polish diacritics and punctuation into an ASCII search slug."""

    text = str(value).strip().casefold().replace("ł", "l").replace("Ł", "L")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _canonical_item(value: Any) -> str | None:
    slug = _ascii(value)
    for canonical, roots in ITEM_ROOTS:
        if any(root in slug for root in roots):
            return canonical
    return None


def _canonical_city(value: Any) -> str:
    slug = _ascii(value)
    for city, forms in CITY_FORMS.items():
        if slug in forms:
            return city
    if slug in COORDINATORS:
        return slug
    raise FilesystemError(f"unknown city in notes: {value!r}")


def _city_in_text(text: str, cities: Sequence[str]) -> str | None:
    slug = _ascii(text)
    candidates: list[tuple[int, str]] = []
    for city in cities:
        for form in CITY_FORMS.get(city, (city,)):
            if re.search(rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])", slug):
                candidates.append((len(form), city))
    if not candidates:
        return None
    return max(candidates)[1]


def parse_notes_zip(payload: bytes) -> dict[str, str]:
    """Read the required UTF-8 text files from Natan's ZIP archive."""

    if not isinstance(payload, (bytes, bytearray)) or not payload:
        raise FilesystemError("notes download was empty")
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise FilesystemError("notes download is not a ZIP archive") from exc
    with archive:
        names = set(archive.namelist())
        missing = REQUIRED_NOTE_FILES - names
        if missing:
            raise FilesystemError(f"notes archive is missing: {sorted(missing)}")
        result: dict[str, str] = {}
        for name in REQUIRED_NOTE_FILES:
            try:
                result[name] = archive.read(name).decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise FilesystemError(f"note {name} is not valid UTF-8") from exc
        return result


def parse_transactions(text: str) -> tuple[list[str], dict[str, set[str]]]:
    """Return ordered city slugs and seller sets from arrow transactions."""

    cities: list[str] = []
    sellers: dict[str, set[str]] = {}

    def remember_city(raw: str) -> str:
        city = _canonical_city(raw)
        if city not in cities:
            cities.append(city)
        return city

    for line in text.splitlines():
        parts = [part.strip() for part in line.split("->")]
        if len(parts) != 3:
            continue
        source_raw, item_raw, target_raw = parts
        source = remember_city(source_raw)
        target = remember_city(target_raw)
        item = _canonical_item(item_raw)
        if item is None:
            raise FilesystemError(f"unknown transaction item: {item_raw!r}")
        sellers.setdefault(item, set()).add(source)
        # Keep the target in the city universe even when it never appears as a
        # seller in the transactions.
        _ = target

    if not cities or not sellers:
        raise FilesystemError("transactions contain no usable city or seller data")
    return cities, sellers


def parse_city_needs(text: str, cities: Sequence[str]) -> dict[str, dict[str, int]]:
    """Extract numeric demand clauses and convert item names to singular ASCII."""

    needs: dict[str, dict[str, int]] = {city: {} for city in cities}
    # ``i``/``oraz`` joins two independent quantity clauses in the notes.
    clause_split = re.compile(r",|\+|\s+i\s+|\s+oraz\s+", re.IGNORECASE)
    number_re = re.compile(r"\b(\d+)\b")

    for line in text.splitlines():
        city = _city_in_text(line, cities)
        if city is None:
            continue
        for clause in clause_split.split(line):
            number_matches = number_re.findall(clause)
            if len(number_matches) != 1:
                continue
            item = _canonical_item(clause)
            if item is None:
                continue
            quantity = int(number_matches[0])
            if quantity <= 0:
                raise FilesystemError(f"non-positive quantity for {city}/{item}")
            if item in needs[city]:
                raise FilesystemError(f"duplicate demand item for {city}: {item}")
            needs[city][item] = quantity

    missing = [city for city, values in needs.items() if not values]
    if missing:
        raise FilesystemError(f"no demand parsed for: {', '.join(missing)}")
    return needs


def parse_people(text: str, cities: Sequence[str]) -> tuple[Person, ...]:
    """Validate coordinator names and associate each one with a city."""

    folded = _ascii(text)
    people: list[Person] = []
    for city in cities:
        names = COORDINATORS.get(city)
        if names is None:
            raise FilesystemError(f"no coordinator mapping for city {city}")
        first, last = names
        if _ascii(first) not in folded or _ascii(last) not in folded:
            raise FilesystemError(f"coordinator for {city} is absent from conversation notes")
        people.append(Person(first, last, city))
    if len(people) != len(cities):
        raise FilesystemError("coordinator count does not match city count")
    return tuple(people)


def parse_knowledge_base(notes: Mapping[str, str]) -> KnowledgeBase:
    """Parse and cross-check all source files into verifier-ready facts."""

    try:
        announcements = notes["ogłoszenia.txt"]
        conversations = notes["rozmowy.txt"]
        transactions = notes["transakcje.txt"]
    except KeyError as exc:
        raise FilesystemError(f"missing note source: {exc.args[0]}") from exc
    cities, seller_sets = parse_transactions(transactions)
    city_needs = parse_city_needs(announcements, cities)
    people = parse_people(conversations, cities)
    sellers = {
        item: tuple(sorted(seller_cities))
        for item, seller_cities in sorted(seller_sets.items())
    }
    return KnowledgeBase(city_needs=city_needs, people=people, sellers=sellers)


def _validate_name(name: str, *, directory: bool) -> str:
    limit = MAX_DIRECTORY_NAME_LENGTH if directory else MAX_FILE_NAME_LENGTH
    if not isinstance(name, str) or not name or len(name) > limit:
        kind = "directory" if directory else "file"
        raise FilesystemError(f"invalid {kind} name length: {name!r}")
    if not ALLOWED_NAME_RE.fullmatch(name):
        kind = "directory" if directory else "file"
        raise FilesystemError(f"invalid {kind} name: {name!r}")
    return name


def _city_display(city: str) -> str:
    return city[:1].upper() + city[1:]


def build_filesystem(kb: KnowledgeBase) -> tuple[FileEntry, ...]:
    """Build and validate the three required directories and their files."""

    entries: list[FileEntry] = []
    used_names = {"miasta", "osoby", "towary"}

    def add_file(directory: str, name: str, content: str) -> None:
        _validate_name(directory, directory=True)
        _validate_name(name, directory=False)
        if name in used_names:
            raise FilesystemError(f"filesystem names must be globally unique: {name}")
        if not isinstance(content, str) or not content:
            raise FilesystemError(f"empty content for /{directory}/{name}")
        entries.append(FileEntry(f"/{directory}/{name}", content))
        used_names.add(name)

    for city, needs in kb.city_needs.items():
        content = json.dumps(needs, ensure_ascii=True, indent=2)
        add_file("miasta", city, content)

    for person in kb.people:
        name = f"{_ascii(person.first_name)}_{_ascii(person.last_name)}"
        city_link = f"[{_city_display(person.city)}](/miasta/{person.city})"
        add_file("osoby", name, f"# {person.display_name}\n\n{city_link}\n")

    for item, seller_cities in kb.sellers.items():
        links = "\n".join(
            f"[{_city_display(city)}](/miasta/{city})" for city in seller_cities
        )
        add_file("towary", item, f"# {item}\n\n{links}\n")

    if len(entries) != len(kb.city_needs) + len(kb.people) + len(kb.sellers):
        raise FilesystemError("filesystem entry count mismatch")

    existing_paths = {entry.path for entry in entries}
    link_re = re.compile(r"\]\((/[^)]+)\)")
    for entry in entries:
        for target in link_re.findall(entry.content):
            if target not in existing_paths:
                raise FilesystemError(f"broken Markdown link in {entry.path}: {target}")
    return tuple(entries)


def build_batch_actions(entries: Sequence[FileEntry]) -> tuple[Action, ...]:
    """Order create operations so all Markdown link targets already exist."""

    actions: list[Action] = [
        {"action": "createDirectory", "path": "/miasta"},
        {"action": "createDirectory", "path": "/osoby"},
        {"action": "createDirectory", "path": "/towary"},
    ]
    city_entries = [entry for entry in entries if entry.path.startswith("/miasta/")]
    people_entries = [entry for entry in entries if entry.path.startswith("/osoby/")]
    goods_entries = [entry for entry in entries if entry.path.startswith("/towary/")]
    for entry in (*city_entries, *people_entries, *goods_entries):
        actions.append(
            {"action": "createFile", "path": entry.path, "content": entry.content}
        )
    return tuple(actions)


def _response_summary(value: Mapping[str, Any]) -> str:
    fields = {
        key: value[key]
        for key in ("http_status", "code", "message", "error")
        if key in value
    }
    return json.dumps(fields or {"type": "unexpected response"}, ensure_ascii=False)


def _response_failed(value: Mapping[str, Any]) -> bool:
    status = value.get("http_status")
    code = value.get("code")
    return (isinstance(status, int) and status >= 400) or (
        isinstance(code, int) and code < 0
    )


class HubClient:
    """Authenticated filesystem API client with an injectable test transport."""

    def __init__(
        self,
        *,
        action_callable: ActionCallable | None = None,
        timeout: int = 60,
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

    def call(self, answer: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Execute one action or a batch and raise on Hub rejection."""

        if isinstance(answer, Mapping):
            if not answer.get("action"):
                raise ValueError("action answer must contain a non-empty action")
            label = str(answer["action"])
            request_answer: Mapping[str, Any] | list[dict[str, Any]] = dict(answer)
        elif isinstance(answer, Sequence) and not isinstance(answer, (str, bytes, bytearray)):
            if not answer or any(not isinstance(item, Mapping) for item in answer):
                raise ValueError("batch answer must contain action mappings")
            label = "batch"
            request_answer = [dict(item) for item in answer]
        else:
            raise ValueError("answer must be an action mapping or a batch sequence")

        if self._action_callable is not None:
            raw_result = self._action_callable(request_answer)
        else:
            self._load_live_dependencies()
            assert self._post_request is not None
            assert self._verify_url is not None
            assert self._api_key is not None
            raw_result = self._post_request(
                self._verify_url,
                {"apikey": self._api_key, "task": TASK_NAME, "answer": request_answer},
                raise_on_error=False,
                timeout=self._timeout,
            )
        if not isinstance(raw_result, Mapping):
            raise FilesystemError(f"{label} returned a non-object response")
        result = dict(raw_result)
        if _response_failed(result):
            raise FilesystemError(f"{label} rejected: {_response_summary(result)}")
        return result


def _extract_flag(value: Any) -> str | None:
    match = FLAG_RE.search(json.dumps(value, ensure_ascii=False, default=str))
    return match.group(0) if match else None


def download_notes() -> bytes:
    """Download the lesson ZIP through the configured Hub host."""

    from src.ai_devs.api import get_request
    from src.ai_devs.config import HUB_BASE_URL

    response = get_request(f"{HUB_BASE_URL.rstrip('/')}/dane/{NOTES_FILENAME}", timeout=60)
    return response.content


def run_operation(
    *,
    reset: bool = True,
    client: HubClient | None = None,
    notes_payload: bytes | None = None,
) -> str:
    """Build the virtual knowledge base and return the ``done`` flag."""

    client = client or HubClient()
    client.call({"action": "help"})
    if reset:
        client.call({"action": "reset"})
    notes = parse_notes_zip(notes_payload if notes_payload is not None else download_notes())
    knowledge_base = parse_knowledge_base(notes)
    entries = build_filesystem(knowledge_base)
    actions = build_batch_actions(entries)
    print(
        f"Parsed {len(knowledge_base.city_needs)} cities, "
        f"{len(knowledge_base.people)} people, {len(knowledge_base.sellers)} goods; "
        f"creating {len(entries)} files."
    )
    client.call(actions)
    for directory in ("/miasta", "/osoby", "/towary"):
        listing = client.call({"action": "listFiles", "path": directory})
        entries_value = listing.get("entries")
        if not isinstance(entries_value, Sequence) or isinstance(
            entries_value, (str, bytes, bytearray)
        ):
            raise FilesystemError(f"listFiles returned no entries for {directory}")
        print(f"{directory}: {len(entries_value)} entries")
    done = client.call({"action": "done"})
    flag = _extract_flag(done)
    if flag is None:
        raise FilesystemError(f"done returned no verifier flag: {_response_summary(done)}")
    return flag


def _dry_run() -> None:
    print("Dry run: no notes download, filesystem mutation, or verifier call made.")
    print("Live mode: python -m tasks.s04e04.solution --run")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Solve AI Devs S04E04 filesystem")
    parser.add_argument(
        "--run",
        action="store_true",
        help="download notes, build the filesystem, and call done on the live Hub",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="keep the current virtual filesystem instead of clearing it",
    )
    args = parser.parse_args(argv)
    if not args.run:
        _dry_run()
        return 0
    try:
        flag = run_operation(reset=not args.no_reset)
    except (FilesystemError, ValueError, OSError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"FLAG: {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
