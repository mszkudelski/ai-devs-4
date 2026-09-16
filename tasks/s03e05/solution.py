"""S03E05 — plan a resource-safe route to Skolwin.

The task exposes no fixed data endpoint.  It first requires tool discovery,
then the discovered tools return the current map and vehicle facts.  This
module keeps those calls behind a Hub-host allowlist and computes a route with
a small resource-constrained graph search.  The map and answer are therefore
derived at run time rather than copied from one observed challenge state.

The default invocation is a dry run and makes no network calls::

    python -m tasks.s03e05.solution
    python -m tasks.s03e05.solution --run

Only ``--run`` discovers tools, plans the route, and submits the final answer
to ``/verify``.  API keys remain inside the shared configuration helper.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urljoin, urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ai_devs.api import post_request, send_report
from src.ai_devs.config import HUB_API_URL, HUB_BASE_URL


TASK_NAME = "savethem"
TARGET_CITY = "Skolwin"
TOOLSEARCH_URL = f"{HUB_API_URL}/toolsearch"
MAP_SIZE = 10
INITIAL_FUEL = 10
INITIAL_FOOD = 10
FUEL_SCALE = 10
FOOD_SCALE = 10
FLAG_RE = re.compile(r"\{FLG:[^}]+\}")
ALLOWED_TERRAIN = frozenset({".", "S", "G", "R", "T", "W"})
DIRECTIONS: tuple[tuple[str, int, int], ...] = (
    ("right", 0, 1),
    ("up", -1, 0),
    ("down", 1, 0),
    ("left", 0, -1),
)
SHELL_META_CHARS = frozenset(";|&<>$`\n\r")


class ToolDiscoveryError(RuntimeError):
    """Raised when the Hub does not expose a trusted task tool."""


class MapDataError(ValueError):
    """Raised when the map response violates the 10×10 task contract."""


class NoRouteError(RuntimeError):
    """Raised when no route fits the map, terrain, and resource budgets."""


@dataclass(frozen=True)
class DiscoveredTool:
    """A tool URL returned by the Hub toolsearch endpoint."""

    name: str
    url: str
    description: str = ""


@dataclass(frozen=True)
class TerrainMap:
    """Validated map and its start/goal coordinates, held as row/column pairs."""

    cells: tuple[tuple[str, ...], ...]
    start: tuple[int, int]
    goal: tuple[int, int]
    city_name: str = TARGET_CITY


@dataclass(frozen=True)
class Vehicle:
    """One vehicle record returned by the discovered vehicle tool."""

    name: str
    fuel_per_move: int
    food_per_move: int
    note: str
    blocked_terrain: frozenset[str]
    priority: int

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Vehicle":
        """Parse consumption and terrain hints without trusting prose as code."""

        name = payload.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name.strip()):
            raise ValueError("vehicle response is missing a valid name")
        name = name.strip().lower()
        consumption = payload.get("consumption")
        if not isinstance(consumption, Mapping):
            raise ValueError(f"vehicle {name} is missing consumption data")
        fuel_per_move = _scaled_cost(consumption.get("fuel"), FUEL_SCALE, "fuel", name)
        food_per_move = _scaled_cost(consumption.get("food"), FOOD_SCALE, "food", name)
        note_value = payload.get("note", "")
        note = note_value if isinstance(note_value, str) else str(note_value)
        blocked = _blocked_terrain_from_note(name, note)
        return cls(
            name=name,
            fuel_per_move=fuel_per_move,
            food_per_move=food_per_move,
            note=note,
            blocked_terrain=frozenset(blocked),
            priority=_vehicle_priority(name, note),
        )


@dataclass(frozen=True)
class Route:
    """A verifier-ready route and bounded resource accounting."""

    vehicle: str
    actions: tuple[str, ...]
    fuel_used: int
    food_used: int

    @property
    def answer(self) -> list[str]:
        """Return the exact array shape accepted by the Hub verifier."""

        return [self.vehicle, *self.actions]


def _api_key() -> str:
    """Resolve the Hub key lazily so importing and dry runs stay harmless."""

    from src.ai_devs.config import get_api_key

    return get_api_key()


def _json_text(value: Any) -> str:
    """Render API values for diagnostics without ever including the API key."""

    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _response_summary(value: Any, limit: int = 700) -> str:
    """Keep Hub diagnostics bounded and exclude data fields that may be sensitive."""

    if isinstance(value, Mapping):
        safe = {
            key: value[key]
            for key in ("http_status", "code", "message", "error")
            if key in value
        }
        text = _json_text(safe or {"type": "unexpected_response"})
    else:
        text = f"unexpected response type: {type(value).__name__}"
    return text if len(text) <= limit else text[:limit] + "..."


def _response_is_error(value: Any) -> bool:
    """Recognise HTTP failures and negative task-tool response codes."""

    if not isinstance(value, Mapping):
        return True
    status = value.get("http_status")
    if isinstance(status, int) and status >= 400:
        return True
    code = value.get("code")
    if isinstance(code, int) and code < 0:
        return True
    return bool(value.get("error"))


def _trusted_tool_url(raw_url: Any) -> str:
    """Resolve a discovered relative URL while keeping the key on the Hub host."""

    if not isinstance(raw_url, str) or not raw_url.strip():
        raise ToolDiscoveryError("toolsearch returned an empty URL")
    hub = urlparse(HUB_BASE_URL)
    candidate = urlparse(urljoin(HUB_BASE_URL.rstrip("/") + "/", raw_url.strip().lstrip("/")))
    if candidate.scheme != hub.scheme or candidate.netloc != hub.netloc:
        raise ToolDiscoveryError("toolsearch returned a URL outside the configured Hub")
    if not candidate.path.startswith("/api/"):
        raise ToolDiscoveryError("toolsearch returned a non-API URL")
    if candidate.query or candidate.fragment:
        raise ToolDiscoveryError("toolsearch returned a URL with query or fragment data")
    return candidate.geturl()


def _hub_post(url: str, query: str) -> dict[str, Any]:
    """POST a query to a previously trusted Hub tool."""

    if not isinstance(query, str) or not query.strip() or len(query) > 80:
        raise ValueError("tool query must be a short non-empty English string")
    if any(char in query for char in SHELL_META_CHARS):
        raise ValueError("tool query contains unsupported control characters")
    result = post_request(
        url,
        {"apikey": _api_key(), "query": query.strip()},
        raise_on_error=False,
        timeout=120,
    )
    if not isinstance(result, dict):
        raise ToolDiscoveryError(f"Hub returned a non-object response from {url}")
    return result


def discover_tools() -> tuple[DiscoveredTool, DiscoveredTool]:
    """Discover and validate the map and vehicle tools from toolsearch."""

    response = _hub_post(TOOLSEARCH_URL, "terrain map vehicles")
    if _response_is_error(response):
        raise ToolDiscoveryError(f"toolsearch failed: {_response_summary(response)}")
    raw_tools = response.get("tools")
    if not isinstance(raw_tools, Sequence) or isinstance(raw_tools, (str, bytes, bytearray)):
        raise ToolDiscoveryError("toolsearch response is missing its tools array")

    discovered: list[DiscoveredTool] = []
    seen_urls: set[str] = set()
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, Mapping):
            continue
        name_value = raw_tool.get("name")
        name = name_value.strip() if isinstance(name_value, str) else ""
        description_value = raw_tool.get("description", "")
        description = description_value if isinstance(description_value, str) else str(description_value)
        lowered = f"{name} {description}".casefold()
        if not ("map" in lowered or "vehicle" in lowered or "wehicle" in lowered):
            continue
        url = _trusted_tool_url(raw_tool.get("url"))
        if url in seen_urls:
            continue
        seen_urls.add(url)
        discovered.append(DiscoveredTool(name=name, url=url, description=description))

    map_tools = [tool for tool in discovered if "map" in f"{tool.name} {tool.description}".casefold()]
    vehicle_tools = [
        tool
        for tool in discovered
        if "vehicle" in f"{tool.name} {tool.description}".casefold()
        or "wehicle" in f"{tool.name} {tool.description}".casefold()
    ]
    if not map_tools or not vehicle_tools:
        raise ToolDiscoveryError("toolsearch did not return both map and vehicle tools")
    return map_tools[0], vehicle_tools[0]


def _scaled_cost(value: Any, scale: int, field: str, name: str) -> int:
    """Convert a finite non-negative decimal cost to exact integer units."""

    if isinstance(value, bool):
        raise ValueError(f"vehicle {name} has invalid {field} consumption")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"vehicle {name} has invalid {field} consumption") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"vehicle {name} has invalid {field} consumption")
    scaled = round(number * scale)
    if not math.isclose(number * scale, scaled, abs_tol=1e-7):
        raise ValueError(f"vehicle {name} has unsupported fractional {field} consumption")
    return int(scaled)


def _vehicle_priority(name: str, note: str) -> int:
    """Use only speed statements returned by the vehicle tool for tie-breaking."""

    lowered = note.casefold()
    if "fastest" in lowered:
        return 0
    if "balanced" in lowered or "decent compromise" in lowered:
        return 1
    if name == "walk" or "on foot" in lowered:
        return 3
    return 2


def _blocked_terrain_from_note(name: str, note: str) -> set[str]:
    """Translate explicit vehicle warnings into blocked map cell symbols."""

    if name == "walk":
        return set()
    lowered = note.casefold()
    blocked: set[str] = set()
    # Both current powered-vehicle notes explicitly say water is unsafe.  Keep
    # this conservative and require an explicit warning before blocking a
    # terrain type; unknown future terrain can still be explored by walking.
    if "water" in lowered and any(
        phrase in lowered
        for phrase in ("cannot", "sank", "lost", "serious problem", "immediately")
    ):
        blocked.add("W")
    if any(word in lowered for word in ("rock", "rocks", "stone", "stones")) and any(
        phrase in lowered for phrase in ("cannot", "blocked", "impassable")
    ):
        blocked.add("R")
    if any(word in lowered for word in ("tree", "trees", "forest")) and any(
        phrase in lowered for phrase in ("cannot", "blocked", "impassable")
    ):
        blocked.add("T")
    return blocked


def _map_cells(value: Any) -> tuple[tuple[str, ...], ...]:
    """Validate a 10×10 map array."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise MapDataError("map must be a 2D array")
    rows = list(value)
    if len(rows) != MAP_SIZE:
        raise MapDataError(f"map must contain {MAP_SIZE} rows")
    cells: list[tuple[str, ...]] = []
    for row_number, raw_row in enumerate(rows, start=1):
        if not isinstance(raw_row, Sequence) or isinstance(raw_row, (str, bytes, bytearray)):
            raise MapDataError(f"map row {row_number} must be an array")
        row = tuple(str(cell) for cell in raw_row)
        if len(row) != MAP_SIZE:
            raise MapDataError(f"map row {row_number} must contain {MAP_SIZE} cells")
        unknown = sorted(set(row) - ALLOWED_TERRAIN)
        if unknown:
            raise MapDataError(f"map row {row_number} contains unknown terrain: {unknown}")
        cells.append(row)
    return tuple(cells)


def parse_map_response(value: Mapping[str, Any]) -> TerrainMap:
    """Parse a maps-tool response and locate the start and goal cells."""

    if _response_is_error(value):
        raise MapDataError(f"maps tool failed: {_response_summary(value)}")
    cells = _map_cells(value.get("map"))
    starts = [(row, col) for row, line in enumerate(cells) for col, cell in enumerate(line) if cell == "S"]
    goals = [(row, col) for row, line in enumerate(cells) for col, cell in enumerate(line) if cell == "G"]
    if len(starts) != 1 or len(goals) != 1:
        raise MapDataError("map must contain exactly one S start and one G goal")
    city_name = value.get("cityName", TARGET_CITY)
    if not isinstance(city_name, str) or not city_name.strip():
        city_name = TARGET_CITY
    return TerrainMap(cells=cells, start=starts[0], goal=goals[0], city_name=city_name.strip())


def _map_query(map_tool: DiscoveredTool) -> TerrainMap:
    """Request the current city map using the accepted lowercase city key."""

    responses = [_hub_post(map_tool.url, TARGET_CITY.casefold())]
    if _response_is_error(responses[0]):
        responses.append(_hub_post(map_tool.url, TARGET_CITY))
    for response in responses:
        if not _response_is_error(response):
            return parse_map_response(response)
    raise MapDataError(f"maps tool did not return {TARGET_CITY}: {_response_summary(responses[-1])}")


def _vehicle_names(vehicle_tool: DiscoveredTool) -> tuple[str, ...]:
    """Derive allowed vehicle names from the tool's own error/help response."""

    probe = _hub_post(vehicle_tool.url, "vehicles")
    names: list[str] = []
    if isinstance(probe.get("vehicles"), Sequence) and not isinstance(
        probe.get("vehicles"), (str, bytes, bytearray)
    ):
        names.extend(str(item).strip().lower() for item in probe["vehicles"] if str(item).strip())
    message = probe.get("message")
    if isinstance(message, str):
        match = re.search(r"Allowed values:\s*([^\.]+)", message, re.IGNORECASE)
        if match:
            names.extend(part.strip().lower() for part in match.group(1).split(",") if part.strip())
    # Keep first-seen order but reject malformed names before using them in a
    # query or verifier payload.
    result: list[str] = []
    for name in names:
        if re.fullmatch(r"[A-Za-z0-9_-]+", name) and name not in result:
            result.append(name)
    if not result:
        raise ToolDiscoveryError(f"vehicle tool did not disclose allowed values: {_response_summary(probe)}")
    return tuple(result)


def _vehicle_records(vehicle_tool: DiscoveredTool) -> tuple[Vehicle, ...]:
    """Fetch every vehicle record disclosed by the vehicle tool."""

    records: list[Vehicle] = []
    for name in _vehicle_names(vehicle_tool):
        response = _hub_post(vehicle_tool.url, name)
        if _response_is_error(response):
            raise ToolDiscoveryError(f"vehicle query {name} failed: {_response_summary(response)}")
        records.append(Vehicle.from_payload(response))
    if not records:
        raise ToolDiscoveryError("vehicle tool returned no usable records")
    return tuple(records)


def plan_route(
    terrain_map: TerrainMap,
    vehicles: Sequence[Vehicle],
    fuel_budget: int = INITIAL_FUEL * FUEL_SCALE,
    food_budget: int = INITIAL_FOOD * FOOD_SCALE,
) -> Route:
    """Find the shortest feasible route, allowing one dismount to walking.

    Each state tracks exact tenths of fuel and food, so decimal costs such as
    0.7, 1.6, and 2.5 remain deterministic.  Route length is the primary
    objective because the tool facts expose relative speed but no numeric
    durations; speed wording then breaks ties between equal-length routes.
    """

    if fuel_budget < 0 or food_budget < 0:
        raise ValueError("resource budgets must be non-negative")
    by_name = {vehicle.name: vehicle for vehicle in vehicles}
    if not by_name:
        raise NoRouteError("no vehicles were discovered")
    if "walk" not in by_name:
        raise NoRouteError("walking mode was not returned by the vehicle tool")

    # State: row, col, initial mode, current mode, dismounted, fuel used,
    # food used.  Initial mode remains in the state so the verifier payload can
    # begin with the selected vehicle even after a walk transition.
    State = tuple[int, int, str, str, bool, int, int]
    start_row, start_col = terrain_map.start
    goal = terrain_map.goal
    queue: list[tuple[tuple[int, int, int, int, int], State]] = []
    best: dict[State, tuple[int, int, int, int, int]] = {}
    parents: dict[State, tuple[State | None, str | None]] = {}

    for vehicle in sorted(by_name.values(), key=lambda item: (item.priority, item.name)):
        state: State = (start_row, start_col, vehicle.name, vehicle.name, False, 0, 0)
        cost = (0, 0, vehicle.priority, 0, 0)
        if state not in best or cost < best[state]:
            best[state] = cost
            parents[state] = (None, None)
            heapq.heappush(queue, (cost, state))

    final_state: State | None = None
    while queue:
        cost, state = heapq.heappop(queue)
        if best.get(state) != cost:
            continue
        row, col, initial_mode, current_mode, dismounted, fuel_used, food_used = state
        if (row, col) == goal:
            final_state = state
            break

        current_vehicle = by_name[current_mode]

        # Dismounting is represented by the literal ``dismount`` action in the
        # answer array.  A second vehicle switch is intentionally not allowed;
        # the task describes leaving the selected vehicle and continuing on
        # foot, rather than remounting.
        if current_mode != "walk" and not dismounted:
            switched: State = (row, col, initial_mode, "walk", True, fuel_used, food_used)
            switch_cost = (cost[0], cost[1] + 1, cost[2], cost[3], cost[4])
            if switch_cost < best.get(switched, (10**9,) * 5):
                best[switched] = switch_cost
                parents[switched] = (state, "dismount")
                heapq.heappush(queue, (switch_cost, switched))

        for action, delta_row, delta_col in DIRECTIONS:
            next_row, next_col = row + delta_row, col + delta_col
            if not (0 <= next_row < MAP_SIZE and 0 <= next_col < MAP_SIZE):
                continue
            terrain = terrain_map.cells[next_row][next_col]
            if terrain in current_vehicle.blocked_terrain:
                continue
            next_fuel = fuel_used + current_vehicle.fuel_per_move
            next_food = food_used + current_vehicle.food_per_move
            if next_fuel > fuel_budget or next_food > food_budget:
                continue
            next_state: State = (
                next_row,
                next_col,
                initial_mode,
                current_mode,
                dismounted,
                next_fuel,
                next_food,
            )
            next_cost = (cost[0] + 1, cost[1], cost[2], next_fuel, next_food)
            if next_cost < best.get(next_state, (10**9,) * 5):
                best[next_state] = next_cost
                parents[next_state] = (state, action)
                heapq.heappush(queue, (next_cost, next_state))

    if final_state is None:
        raise NoRouteError("no route reaches the goal within the fuel and food budgets")

    actions: list[str] = []
    state = final_state
    while parents[state][0] is not None:
        previous, action = parents[state]
        assert previous is not None and action is not None
        actions.append(action)
        state = previous
    actions.reverse()
    _, _, initial_mode, _, _, fuel_used, food_used = final_state
    return Route(
        vehicle=initial_mode,
        actions=tuple(actions),
        fuel_used=fuel_used,
        food_used=food_used,
    )


def discover_and_plan() -> tuple[Route, TerrainMap, tuple[Vehicle, ...]]:
    """Discover current tools/data and compute the verifier-ready route."""

    map_tool, vehicle_tool = discover_tools()
    terrain_map = _map_query(map_tool)
    vehicles = _vehicle_records(vehicle_tool)
    route = plan_route(terrain_map, vehicles)
    return route, terrain_map, vehicles


def _extract_flag(value: Any) -> str | None:
    """Extract the live flag returned by the verifier."""

    match = FLAG_RE.search(_json_text(value))
    return match.group(0) if match else None


def _print_dry_run() -> None:
    """Describe the live flow without resolving credentials or contacting Hub."""

    print("Dry run only: no Hub or tool calls were made.")
    print(f"Task: {TASK_NAME}; destination: {TARGET_CITY}; map: {MAP_SIZE}x{MAP_SIZE}")
    print(f"Initial resources: fuel={INITIAL_FUEL}, food={INITIAL_FOOD}")
    print("Live flow: discover tools, fetch map and vehicles, search feasible route, verify answer.")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline dry run or the authorized live plan-and-submit flow."""

    parser = argparse.ArgumentParser(description="S03E05 resource-constrained route solver")
    parser.add_argument(
        "--run",
        action="store_true",
        help="discover live tools/data, compute the route, and submit it",
    )
    args = parser.parse_args(argv)

    if not args.run:
        _print_dry_run()
        return 0

    try:
        route, terrain_map, vehicles = discover_and_plan()
    except Exception as exc:
        print(f"Planner failed ({type(exc).__name__}).", file=sys.stderr)
        return 1

    print(
        f"Map {terrain_map.city_name}: start={terrain_map.start}, goal={terrain_map.goal}; "
        f"vehicles={','.join(vehicle.name for vehicle in vehicles)}"
    )
    print(
        f"Route: {json.dumps(route.answer, ensure_ascii=False)}; "
        f"fuel={route.fuel_used / FUEL_SCALE:g}/{INITIAL_FUEL:g}; "
        f"food={route.food_used / FOOD_SCALE:g}/{INITIAL_FOOD:g}"
    )

    try:
        response = send_report(TASK_NAME, route.answer)
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
