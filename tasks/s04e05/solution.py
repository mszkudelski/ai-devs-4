"""S04E05 — prepare signed foodwarehouse orders.

The task exposes a self describing action API at the Hub's ``/verify``
endpoint.  This solver downloads the city demand, discovers the destination
and user data through the read only SQLite interface, generates a signature
for every destination, creates one order per city, and verifies the created
orders before calling ``done``.

The default invocation is a dry run and performs no network calls::

    python -m tasks.s04e05.solution

Use ``--run`` for the authorized live operation.  Live mode resets the task
state by default so orders from an interrupted attempt cannot be mistaken for
the current plan.  The verifier flag is printed from ``done`` and is never
submitted anywhere else.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


# Support both ``python -m tasks.s04e05.solution`` and direct execution from
# the repository root without loading credentials during a dry run.
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TASK_NAME = "foodwarehouse"
DATA_FILENAME = "food4cities.json"
FLAG_RE = re.compile(r"\{FLG:[^}]+\}")
SHA1_RE = re.compile(r"^[0-9a-fA-F]{40}$")
ORDER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MAX_CITIES = 100
MAX_ITEMS_PER_CITY = 100
MAX_ITEM_NAME_LENGTH = 80


class FoodwarehouseError(RuntimeError):
    """Raised when task data or a Hub action cannot be validated."""


ActionCallable = Callable[[Mapping[str, Any]], Any]


def _json_text(value: Any) -> str:
    """Render a response for diagnostics without requiring it to be JSON."""

    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def extract_flag(value: Any) -> str | None:
    """Return the first verifier flag in an arbitrary Hub response."""

    match = FLAG_RE.search(_json_text(value))
    return match.group(0) if match else None


def _response_summary(value: Any, limit: int = 700) -> str:
    """Keep error output useful while omitting request data and credentials."""

    if isinstance(value, Mapping):
        safe = {
            key: value[key]
            for key in ("http_status", "code", "message", "error", "tool", "action")
            if key in value
        }
        text = _json_text(safe or {"type": "unexpected_response"})
    else:
        text = json.dumps({"type": type(value).__name__})
    return text if len(text) <= limit else text[:limit] + "..."


def _response_failed(value: Any) -> bool:
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


def _fold(value: Any) -> str:
    """Normalise city names for matching without changing submitted values."""

    text = unicodedata.normalize("NFKD", str(value).strip()).casefold()
    return "".join(char for char in text if not unicodedata.combining(char))


def _positive_int(value: Any, field: str) -> int:
    """Parse an integer field and reject booleans, fractions, and negatives."""

    if isinstance(value, bool):
        raise FoodwarehouseError(f"{field} must be an integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        result = int(value.strip())
    else:
        raise FoodwarehouseError(f"{field} must be an integer")
    if result <= 0:
        raise FoodwarehouseError(f"{field} must be positive")
    return result


def parse_demand(payload: Mapping[str, Any] | str | bytes) -> dict[str, dict[str, int]]:
    """Validate and copy ``food4cities.json`` into city/item quantities."""

    if isinstance(payload, (str, bytes, bytearray)):
        try:
            value = json.loads(payload)
        except (TypeError, ValueError) as exc:
            raise FoodwarehouseError("food4cities.json is not valid JSON") from exc
    else:
        value = payload
    if not isinstance(value, Mapping) or not value:
        raise FoodwarehouseError("food4cities.json must be a non-empty object")
    if len(value) > MAX_CITIES:
        raise FoodwarehouseError(f"food4cities.json contains too many cities: {len(value)}")

    result: dict[str, dict[str, int]] = {}
    seen_cities: set[str] = set()
    for raw_city, raw_items in value.items():
        if not isinstance(raw_city, str) or not raw_city.strip():
            raise FoodwarehouseError("city names must be non-empty strings")
        city = raw_city.strip()
        city_key = _fold(city)
        if city_key in seen_cities:
            raise FoodwarehouseError(f"duplicate city name after normalisation: {city}")
        seen_cities.add(city_key)
        if not isinstance(raw_items, Mapping) or not raw_items:
            raise FoodwarehouseError(f"city {city!r} has no item quantities")
        if len(raw_items) > MAX_ITEMS_PER_CITY:
            raise FoodwarehouseError(f"city {city!r} contains too many items")

        items: dict[str, int] = {}
        for raw_name, raw_quantity in raw_items.items():
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise FoodwarehouseError(f"city {city!r} has an invalid item name")
            name = raw_name.strip()
            if len(name) > MAX_ITEM_NAME_LENGTH or any(
                char in name for char in ("\x00", "\r", "\n")
            ):
                raise FoodwarehouseError(f"city {city!r} has an invalid item name")
            if name in items:
                raise FoodwarehouseError(f"city {city!r} repeats item {name!r}")
            items[name] = _positive_int(raw_quantity, f"quantity for {city}/{name}")
        result[city] = items
    return result


def _rows(value: Any, table: str) -> list[Mapping[str, Any]]:
    """Extract a Hub database ``rows`` array and require object rows."""

    if not isinstance(value, Mapping):
        raise FoodwarehouseError(f"database query for {table} returned a non-object")
    raw_rows = value.get("rows")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes, bytearray)):
        raise FoodwarehouseError(f"database query for {table} returned no rows array")
    result = [row for row in raw_rows if isinstance(row, Mapping)]
    if len(result) != len(raw_rows):
        raise FoodwarehouseError(f"database query for {table} returned a malformed row")
    return result


@dataclass(frozen=True)
class Creator:
    """The minimum user data needed by ``signatureGenerator`` and ``orders``."""

    user_id: int
    login: str
    birthday: str


@dataclass(frozen=True)
class CityDestination:
    """A demand city paired with the numeric destination code."""

    city: str
    destination_id: int
    items: Mapping[str, int]


def parse_creator_rows(response: Mapping[str, Any]) -> tuple[Creator, ...]:
    """Return active users with valid IDs, logins, and birthdays."""

    creators: list[Creator] = []
    seen_ids: set[int] = set()
    for row in _rows(response, "users"):
        raw_id = row.get("user_id")
        if isinstance(raw_id, bool):
            continue
        try:
            user_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if user_id <= 0 or user_id in seen_ids:
            continue
        login = row.get("login")
        birthday = row.get("birthday")
        active = row.get("is_active", 1)
        if not isinstance(login, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,20}", login.strip()):
            continue
        if not isinstance(birthday, str) or not DATE_RE.fullmatch(birthday.strip()):
            continue
        if active not in (1, True, "1", "true", "TRUE"):
            continue
        seen_ids.add(user_id)
        creators.append(Creator(user_id, login.strip(), birthday.strip()))
    creators.sort(key=lambda creator: creator.user_id)
    if not creators:
        raise FoodwarehouseError("users query contained no usable active creator")
    return tuple(creators)


def parse_destinations(
    response: Mapping[str, Any],
    demand: Mapping[str, Mapping[str, int]],
) -> tuple[CityDestination, ...]:
    """Resolve every demand city to exactly one destination database row."""

    by_name: dict[str, int] = {}
    for row in _rows(response, "destinations"):
        raw_id = row.get("destination_id")
        name = row.get("name")
        if isinstance(raw_id, bool) or not isinstance(name, str) or not name.strip():
            continue
        try:
            destination_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if destination_id <= 0:
            continue
        key = _fold(name)
        if key in by_name and by_name[key] != destination_id:
            raise FoodwarehouseError(f"duplicate destination name in database: {name}")
        by_name[key] = destination_id

    destinations: list[CityDestination] = []
    for city, items in demand.items():
        destination_id = by_name.get(_fold(city))
        if destination_id is None:
            raise FoodwarehouseError(f"no destination database row matches city {city!r}")
        destinations.append(CityDestination(city, destination_id, dict(items)))
    return tuple(destinations)


def _sql_string_literal(value: str) -> str:
    """Quote a city name for a read-only SQLite equality query."""

    if not isinstance(value, str) or not value.strip():
        raise FoodwarehouseError("cannot build a destination lookup for an empty city")
    if any(char in value for char in ("\x00", "\r", "\n")):
        raise FoodwarehouseError("city name contains unsupported control characters")
    return "'" + value.replace("'", "''") + "'"


def _nested_mappings(value: Any) -> Sequence[Mapping[str, Any]]:
    """Yield mappings recursively for small, varying Hub response envelopes."""

    found: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        found.append(value)
        for child in value.values():
            found.extend(_nested_mappings(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            found.extend(_nested_mappings(child))
    return found


def _signature_from_response(response: Mapping[str, Any]) -> str:
    """Extract and validate the SHA1 returned by signatureGenerator."""

    for mapping in _nested_mappings(response):
        for key in ("hash", "signature"):
            raw = mapping.get(key)
            if isinstance(raw, str) and SHA1_RE.fullmatch(raw.strip()):
                return raw.strip()
    raise FoodwarehouseError(
        f"signatureGenerator returned no SHA1: {_response_summary(response)}"
    )


def _order_id_from_response(response: Mapping[str, Any]) -> str:
    """Extract a created order ID from current or wrapped response shapes."""

    preferred_keys = ("order_id", "orderId", "id")
    for mapping in _nested_mappings(response):
        for key in preferred_keys:
            raw = mapping.get(key)
            if isinstance(raw, str) and ORDER_ID_RE.fullmatch(raw.strip()):
                return raw.strip()
    raise FoodwarehouseError(f"orders.create returned no valid ID: {_response_summary(response)}")


def _order_list(response: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Extract an order list from ``orders.get`` without trusting extra fields."""

    for mapping in _nested_mappings(response):
        raw_orders = mapping.get("orders")
        if isinstance(raw_orders, Sequence) and not isinstance(
            raw_orders, (str, bytes, bytearray)
        ):
            result = [order for order in raw_orders if isinstance(order, Mapping)]
            if len(result) == len(raw_orders):
                return result
    raise FoodwarehouseError(f"orders.get returned no order list: {_response_summary(response)}")


def _order_items(order: Mapping[str, Any]) -> dict[str, int]:
    """Normalise the order's item array or batch object for exact readback."""

    raw_items = order.get("items")
    result: dict[str, int] = {}
    if isinstance(raw_items, Mapping):
        iterator = raw_items.items()
        for raw_name, raw_quantity in iterator:
            if not isinstance(raw_name, str):
                raise FoodwarehouseError("order contains a non-string item name")
            result[raw_name] = _positive_int(raw_quantity, f"order quantity for {raw_name}")
        return result
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes, bytearray)):
        raise FoodwarehouseError("order contains no item list")
    for raw_item in raw_items:
        if not isinstance(raw_item, Mapping):
            raise FoodwarehouseError("order contains a malformed item")
        name = raw_item.get("name")
        quantity = raw_item.get("items")
        if not isinstance(name, str) or name in result:
            raise FoodwarehouseError("order contains a duplicate or invalid item")
        result[name] = _positive_int(quantity, f"order quantity for {name}")
    return result


def _order_matches(order: Mapping[str, Any], expected: CityDestination, order_id: str) -> bool:
    """Check destination and exact item quantities for one created order."""

    raw_id = order.get("id", order.get("order_id", order.get("orderId")))
    if str(raw_id) != order_id:
        return False
    try:
        destination = int(order.get("destination"))
    except (TypeError, ValueError):
        return False
    return destination == expected.destination_id and _order_items(order) == dict(expected.items)


class HubClient:
    """Authenticated S04E05 client with an injectable transport for tests."""

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

    def call(self, answer: Mapping[str, Any]) -> dict[str, Any]:
        """Execute one action and reject malformed or negative responses."""

        if not isinstance(answer, Mapping) or not isinstance(answer.get("tool"), str):
            raise ValueError("answer must contain a non-empty tool")
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
                {"apikey": self._api_key, "task": TASK_NAME, "answer": request_answer},
                raise_on_error=False,
                timeout=self._timeout,
            )
        if not isinstance(raw_result, Mapping):
            raise FoodwarehouseError(
                f"{request_answer['tool']} returned a non-object response"
            )
        result = dict(raw_result)
        if _response_failed(result):
            raise FoodwarehouseError(
                f"{request_answer['tool']} rejected: {_response_summary(result)}"
            )
        return result


def download_demand() -> dict[str, dict[str, int]]:
    """Download and parse the current city demand from the Hub data endpoint."""

    from src.ai_devs.api import get_request
    from src.ai_devs.config import HUB_BASE_URL

    response = get_request(
        f"{HUB_BASE_URL.rstrip('/')}/dane/{DATA_FILENAME}",
        timeout=60,
    )
    return parse_demand(response.json())


def discover_data(client: HubClient, demand: Mapping[str, Mapping[str, int]]) -> tuple[Creator, tuple[CityDestination, ...]]:
    """Read the database tables needed to authorize every order."""

    tables = client.call({"tool": "database", "query": "show tables"})
    raw_tables = tables.get("tables")
    if not isinstance(raw_tables, Sequence) or isinstance(raw_tables, (str, bytes, bytearray)):
        raise FoodwarehouseError("database show tables returned no table list")
    table_names = {_fold(name) for name in raw_tables if isinstance(name, str)}
    if not {"users", "destinations"}.issubset(table_names):
        raise FoodwarehouseError("database is missing users or destinations table")

    destination_response = client.call(
        {
            "tool": "database",
            "query": "SELECT destination_id, name FROM destinations",
        }
    )
    # The Hub caps an unrestricted SELECT response at 30 rows even when the
    # table is larger.  Fill only the missing demand cities with exact,
    # case-insensitive lookups rather than assuming the first page is complete.
    destination_rows = list(_rows(destination_response, "destinations"))
    known_destination_names = {
        _fold(row.get("name"))
        for row in destination_rows
        if isinstance(row.get("name"), str)
    }
    for city in demand:
        if _fold(city) in known_destination_names:
            continue
        targeted = client.call(
            {
                "tool": "database",
                "query": (
                    "SELECT destination_id, name FROM destinations WHERE "
                    f"LOWER(name) = LOWER({_sql_string_literal(city)})"
                ),
            }
        )
        destination_rows.extend(_rows(targeted, "destinations"))
        known_destination_names.update(
            _fold(row.get("name"))
            for row in _rows(targeted, "destinations")
            if isinstance(row.get("name"), str)
        )
    user_response = client.call(
        {
            "tool": "database",
            "query": (
                "SELECT user_id, login, birthday, is_active FROM users "
                "WHERE user_id IS NOT NULL AND is_active = 1 AND role = 2 "
                "ORDER BY user_id"
            ),
        }
    )
    creators = parse_creator_rows(user_response)
    destinations = parse_destinations({"rows": destination_rows}, demand)
    return creators[0], destinations


def _extract_order_id_for_created(response: Mapping[str, Any]) -> str:
    """Keep order ID extraction separate to make create response checks clear."""

    return _order_id_from_response(response)


def run_operation(
    *,
    reset: bool = True,
    client: HubClient | None = None,
    demand_payload: Mapping[str, Any] | str | bytes | None = None,
) -> str:
    """Create, read back, and verify all required orders; return the flag."""

    live_client = client or HubClient()
    help_response = live_client.call({"tool": "help"})
    if not isinstance(help_response.get("tools"), Sequence):
        raise FoodwarehouseError("help response did not disclose tools")
    if reset:
        live_client.call({"tool": "reset"})

    demand = parse_demand(demand_payload) if demand_payload is not None else download_demand()
    creator, destinations = discover_data(live_client, demand)
    print(
        f"Planning {len(destinations)} orders with creatorID={creator.user_id} "
        f"({creator.login})."
    )

    created: list[tuple[CityDestination, str]] = []
    for index, destination in enumerate(destinations, start=1):
        print(f"[{index}/{len(destinations)}] {destination.city} → {destination.destination_id}")
        signature_response = live_client.call(
            {
                "tool": "signatureGenerator",
                "action": "generate",
                "login": creator.login,
                "birthday": creator.birthday,
                "destination": destination.destination_id,
            }
        )
        signature = _signature_from_response(signature_response)
        create_response = live_client.call(
            {
                "tool": "orders",
                "action": "create",
                "title": f"Dostawa dla {destination.city}",
                "creatorID": creator.user_id,
                "destination": destination.destination_id,
                "signature": signature,
            }
        )
        order_id = _extract_order_id_for_created(create_response)
        live_client.call(
            {
                "tool": "orders",
                "action": "append",
                "id": order_id,
                "items": dict(destination.items),
            }
        )
        created.append((destination, order_id))

    orders_response = live_client.call({"tool": "orders", "action": "get"})
    orders = _order_list(orders_response)
    for destination, order_id in created:
        if not any(_order_matches(order, destination, order_id) for order in orders):
            raise FoodwarehouseError(
                f"readback mismatch for {destination.city} order {order_id}"
            )
    print(f"Read back {len(created)} completed orders with exact item quantities.")

    done_response = live_client.call({"tool": "done"})
    flag = extract_flag(done_response)
    if flag is None:
        raise FoodwarehouseError(
            f"done returned no verifier flag: {_response_summary(done_response)}"
        )
    print(f"FLAG: {flag}")
    return flag


def _dry_run() -> None:
    print("S04E05 foodwarehouse dry run (no network calls or order changes).")
    print("Live mode: python -m tasks.s04e05.solution --run")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Solve AI Devs S04E05 foodwarehouse")
    parser.add_argument(
        "--run",
        action="store_true",
        help="download demand, create signed orders, and call done on the live Hub",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="keep the current warehouse state instead of restoring seeded orders",
    )
    parser.add_argument("--timeout", type=int, default=60, help="per-request timeout in seconds")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not args.run:
        _dry_run()
        return 0
    try:
        run_operation(reset=not args.no_reset, client=HubClient(timeout=args.timeout))
    except (FoodwarehouseError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
