"""S04E03 — Domatowo rescue mission.

The Hub exposes a small stateful search game.  This solution reads the clean
map, sends two four-scout transporters to the two sides of the city, inspects
every cell belonging to a highest (``B3``) block, and calls the helicopter as
soon as a log confirms a survivor.

The default invocation is a dry run and performs no network calls::

    python -m tasks.s04e03.solution

Use ``--run`` for the live operation.  A live run resets the board first so a
partially completed earlier run cannot consume the action-point budget.  The
flag is printed when ``callHelicopter`` succeeds; it is never submitted to a
second endpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


TASK_NAME = "domatowo"
BOARD_SIZE = 11
MAX_ACTION_POINTS = 300
TRANSPORTER_PASSENGERS = 4
MAX_TRANSPORTERS = 2
TRANSPORTER_CREATE_COST = 5 + TRANSPORTER_PASSENGERS * 5
SCOUT_MOVE_COST = 7
INSPECT_COST = 1
FLAG_RE = re.compile(r"\{FLG:[^}]+\}")
COORDINATE_RE = re.compile(r"^([A-Ka-k])(\d{1,2})$")

# The normal Domatowo board has these two road drop points.  The planner still
# validates them against the downloaded map and chooses a nearby road on a
# future board layout.
PREFERRED_DROPS: tuple[str, str] = ("D2", "D9")


class HubApiError(RuntimeError):
    """Raised for an unsuccessful or malformed Hub response."""


class MapError(ValueError):
    """Raised when the Hub returns a map that cannot be planned."""


Coordinate = tuple[int, int]  # zero-based (row, column)
ActionCallable = Callable[[dict[str, Any]], Mapping[str, Any]]


def parse_coordinate(value: str) -> Coordinate:
    """Parse an API coordinate such as ``F6`` into a zero-based pair."""

    if not isinstance(value, str):
        raise ValueError(f"invalid coordinate: {value!r}")
    match = COORDINATE_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"invalid coordinate: {value!r}")
    column = ord(match.group(1).upper()) - ord("A")
    row = int(match.group(2)) - 1
    if not (0 <= column < BOARD_SIZE and 0 <= row < BOARD_SIZE):
        raise ValueError(f"coordinate outside {BOARD_SIZE}x{BOARD_SIZE} board: {value!r}")
    return row, column


def format_coordinate(position: Coordinate) -> str:
    """Render a zero-based pair in the Hub's coordinate format."""

    row, column = position
    if not (0 <= row < BOARD_SIZE and 0 <= column < BOARD_SIZE):
        raise ValueError(f"coordinate outside {BOARD_SIZE}x{BOARD_SIZE} board: {position!r}")
    return f"{chr(ord('A') + column)}{row + 1}"


def manhattan(first: Coordinate, second: Coordinate) -> int:
    """Return the orthogonal distance used by scout movement."""

    return abs(first[0] - second[0]) + abs(first[1] - second[1])


@dataclass(frozen=True)
class CityMap:
    """Validated clean map returned by ``getMap``."""

    grid: tuple[tuple[str, ...], ...]

    @property
    def size(self) -> int:
        return len(self.grid)

    def tile(self, position: Coordinate) -> str:
        return self.grid[position[0]][position[1]]

    def is_road(self, position: Coordinate) -> bool:
        tile = self.tile(position).casefold().replace("_", "")
        return tile in {"road", "ul", "street", "streetroad"}

    def highest_blocks(self) -> tuple[Coordinate, ...]:
        """Return every map cell marked as a three-storey block (symbol B3)."""

        return tuple(
            (row, column)
            for row, line in enumerate(self.grid)
            for column, tile in enumerate(line)
            if tile.casefold() in {"block3", "b3"}
        )

    @classmethod
    def from_response(cls, response: Mapping[str, Any]) -> "CityMap":
        """Validate the nested ``map.grid`` payload from ``getMap``."""

        raw_map = response.get("map")
        if not isinstance(raw_map, Mapping):
            raise MapError("getMap response has no map object")
        raw_grid = raw_map.get("grid")
        if not isinstance(raw_grid, Sequence) or isinstance(raw_grid, (str, bytes, bytearray)):
            raise MapError("getMap response has no grid array")
        rows = list(raw_grid)
        if len(rows) != BOARD_SIZE:
            raise MapError(f"expected an {BOARD_SIZE}x{BOARD_SIZE} map, got {len(rows)} rows")

        grid: list[tuple[str, ...]] = []
        for row_number, raw_row in enumerate(rows, start=1):
            if not isinstance(raw_row, Sequence) or isinstance(raw_row, (str, bytes, bytearray)):
                raise MapError(f"map row {row_number} is not an array")
            row = tuple(str(cell) for cell in raw_row)
            if len(row) != BOARD_SIZE:
                raise MapError(
                    f"map row {row_number} has {len(row)} cells; expected {BOARD_SIZE}"
                )
            grid.append(row)
        return cls(tuple(grid))


@dataclass(frozen=True)
class Scout:
    """A scout identifier and its current board position."""

    identifier: str
    position: Coordinate


@dataclass(frozen=True)
class Route:
    """One scout's ordered target cells."""

    scout: Scout
    targets: tuple[Coordinate, ...]
    distance: int


def _safe_response_summary(response: Mapping[str, Any]) -> str:
    """Keep diagnostics useful without echoing credentials or full payloads."""

    fields = {
        key: response[key]
        for key in ("http_status", "code", "message", "error")
        if key in response
    }
    return json.dumps(fields or {"type": "unexpected response"}, ensure_ascii=False)


class HubClient:
    """Small authenticated client for the Domatowo action protocol.

    ``action_callable`` is intentionally injectable so route and response
    handling can be tested without touching the live board.  In normal live
    mode the shared repository HTTP helper is used and the API key remains in
    the request body only.
    """

    def __init__(
        self,
        *,
        action_callable: ActionCallable | None = None,
        timeout: int = 30,
    ) -> None:
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
        """Execute one action and raise a concise error for rejection."""

        if not isinstance(answer, Mapping) or not answer.get("action"):
            raise ValueError("an action mapping with a non-empty action is required")

        if self._action_callable is not None:
            raw_result = self._action_callable(dict(answer))
        else:
            self._load_live_dependencies()
            assert self._post_request is not None
            assert self._verify_url is not None
            assert self._api_key is not None
            raw_result = self._post_request(
                self._verify_url,
                {"apikey": self._api_key, "task": TASK_NAME, "answer": dict(answer)},
                raise_on_error=False,
                timeout=self._timeout,
            )

        if not isinstance(raw_result, Mapping):
            raise HubApiError(f"Hub returned a non-object response: {type(raw_result).__name__}")
        result = dict(raw_result)
        status = result.get("http_status")
        code = result.get("code")
        if isinstance(status, int) and status >= 400:
            raise HubApiError(f"{answer['action']} rejected: {_safe_response_summary(result)}")
        if isinstance(code, int) and code < 0:
            raise HubApiError(f"{answer['action']} rejected: {_safe_response_summary(result)}")
        return result


def _map_from_hub(client: HubClient) -> CityMap:
    return CityMap.from_response(client.call({"action": "getMap"}))


def _candidate_positions(client: HubClient, city_map: CityMap) -> tuple[Coordinate, ...]:
    """Find all B3 cells, with a symbol-search fallback for future payloads."""

    candidates = city_map.highest_blocks()
    if candidates:
        return candidates

    response = client.call({"action": "searchSymbol", "symbol": "B3"})
    found = response.get("found")
    if not isinstance(found, Sequence) or isinstance(found, (str, bytes, bytearray)):
        raise MapError("map contains no B3 cells and symbol search returned no fields")
    parsed: list[Coordinate] = []
    for item in found:
        if not isinstance(item, Mapping) or item.get("symbol") != "B3":
            continue
        raw_position = item.get("position")
        if isinstance(raw_position, str):
            parsed.append(parse_coordinate(raw_position))
    if not parsed:
        raise MapError("could not find any highest-block cells to inspect")
    return tuple(dict.fromkeys(parsed))


def _connected_components(cells: Sequence[Coordinate]) -> list[list[Coordinate]]:
    """Group orthogonally adjacent cells for route/drop-point selection."""

    remaining = set(cells)
    components: list[list[Coordinate]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = [start]
        queue = [start]
        while queue:
            row, column = queue.pop()
            for neighbour in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    queue.append(neighbour)
                    component.append(neighbour)
        components.append(sorted(component))
    return components


def _road_distance(city_map: CityMap, start: Coordinate, goal: Coordinate) -> int | None:
    """Find a shortest road-only transporter path length."""

    if not city_map.is_road(start) or not city_map.is_road(goal):
        return None
    distances = {start: 0}
    queue = [start]
    while queue:
        current = queue.pop(0)
        if current == goal:
            return distances[current]
        row, column = current
        for neighbour in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            if (
                0 <= neighbour[0] < city_map.size
                and 0 <= neighbour[1] < city_map.size
                and city_map.is_road(neighbour)
                and neighbour not in distances
            ):
                distances[neighbour] = distances[current] + 1
                queue.append(neighbour)
    return None


def _nearest_road(
    city_map: CityMap,
    targets: Sequence[Coordinate],
    *,
    origin: Coordinate,
    preferred: str,
) -> Coordinate:
    """Choose a road near a target group, retaining the known good drops."""

    preferred_position = parse_coordinate(preferred)
    if city_map.is_road(preferred_position) and _road_distance(city_map, origin, preferred_position) is not None:
        return preferred_position

    roads = [
        (row, column)
        for row in range(city_map.size)
        for column in range(city_map.size)
        if city_map.is_road((row, column))
    ]
    if not roads:
        raise MapError("map contains no road on which to deploy a transporter")
    reachable = [road for road in roads if _road_distance(city_map, origin, road) is not None]
    if not reachable:
        raise MapError("transporter spawn is disconnected from every map road")
    return min(
        reachable,
        key=lambda road: (
            min(manhattan(road, target) for target in targets),
            _road_distance(city_map, origin, road) or math.inf,
            road,
        ),
    )


def choose_drop_points(city_map: CityMap, candidates: Sequence[Coordinate]) -> tuple[str, str]:
    """Choose top/bottom road staging points for the two transporters."""

    if not candidates:
        raise MapError("no candidate fields")
    midpoint = city_map.size // 2
    top = [position for position in candidates if position[0] < midpoint]
    bottom = [position for position in candidates if position[0] >= midpoint]
    if not top or not bottom:
        ordered = sorted(candidates)
        split = max(1, len(ordered) // 2)
        top, bottom = ordered[:split], ordered[split:] or ordered[:split]

    spawn = parse_coordinate("A6")
    second_spawn = parse_coordinate("B6")
    return (
        format_coordinate(
            _nearest_road(
                city_map,
                top,
                origin=spawn,
                preferred=PREFERRED_DROPS[0],
            )
        ),
        format_coordinate(
            _nearest_road(
                city_map,
                bottom,
                origin=second_spawn,
                preferred=PREFERRED_DROPS[1],
            )
        ),
    )


def planned_action_points(
    city_map: CityMap,
    drops: Sequence[str],
    routes: Sequence[Route],
    inspections: int,
) -> int:
    """Estimate the complete two-transporter operation cost before searching."""

    if len(drops) != MAX_TRANSPORTERS:
        raise MapError(f"expected {MAX_TRANSPORTERS} transporter drop points")
    transporter_steps = 0
    for origin_name, drop in zip(("A6", "B6"), drops):
        steps = _road_distance(city_map, parse_coordinate(origin_name), parse_coordinate(drop))
        if steps is None:
            raise MapError(f"transporter cannot reach road drop {drop}")
        transporter_steps += steps
    scout_steps = sum(route.distance for route in routes)
    return (
        MAX_TRANSPORTERS * TRANSPORTER_CREATE_COST
        + transporter_steps
        + scout_steps * SCOUT_MOVE_COST
        + inspections * INSPECT_COST
    )


def _all_routes_for_start(
    start: Coordinate,
    targets: Sequence[Coordinate],
) -> tuple[list[int], list[dict[int, int | None]]]:
    """Held–Karp costs and predecessor pointers for every target subset."""

    count = len(targets)
    full_size = 1 << count
    costs = [[math.inf] * count for _ in range(full_size)]
    previous: list[dict[int, int | None]] = [dict() for _ in range(full_size)]
    for last in range(count):
        mask = 1 << last
        costs[mask][last] = manhattan(start, targets[last])
        previous[mask][last] = None

    for mask in range(1, full_size):
        remaining_last = mask
        while remaining_last:
            bit = remaining_last & -remaining_last
            last = bit.bit_length() - 1
            prior_mask = mask ^ bit
            if prior_mask:
                best_cost = math.inf
                best_previous: int | None = None
                prior_bits = prior_mask
                while prior_bits:
                    prior_bit = prior_bits & -prior_bits
                    prior_last = prior_bit.bit_length() - 1
                    candidate = costs[prior_mask][prior_last] + manhattan(
                        targets[prior_last], targets[last]
                    )
                    if candidate < best_cost:
                        best_cost = candidate
                        best_previous = prior_last
                    prior_bits ^= prior_bit
                costs[mask][last] = best_cost
                previous[mask][last] = best_previous
            remaining_last ^= bit

    route_costs = [0] * full_size
    route_last: list[int | None] = [None] * full_size
    for mask in range(1, full_size):
        last = min(range(count), key=lambda index: costs[mask][index] if mask & (1 << index) else math.inf)
        route_costs[mask] = int(costs[mask][last])
        route_last[mask] = last
    # Keep the final target pointer in a sentinel dictionary keyed by -1.
    for mask, last in enumerate(route_last):
        previous[mask][-1] = last
    return route_costs, previous


def _reconstruct_route(
    mask: int,
    targets: Sequence[Coordinate],
    previous: list[dict[int, int | None]],
) -> tuple[Coordinate, ...]:
    """Reconstruct the ordered target sequence for one subset."""

    if not mask:
        return ()
    last = previous[mask][-1]
    if last is None:
        raise RuntimeError("route predecessor is missing")
    route: list[Coordinate] = []
    current_mask = mask
    current_last: int | None = last
    while current_last is not None:
        route.append(targets[current_last])
        bit = 1 << current_last
        prior_mask = current_mask ^ bit
        current_last = previous[current_mask].get(current_last)
        current_mask = prior_mask
    route.reverse()
    return tuple(route)


def plan_routes(scouts: Sequence[Scout], targets: Sequence[Coordinate]) -> tuple[Route, ...]:
    """Minimize total scout movement while covering every target exactly once.

    The target groups contain at most ten fields on the current board.  Exact
    subset dynamic programming is therefore small, deterministic, and avoids
    a greedy assignment accidentally exceeding the action-point budget.
    """

    all_scouts = list(scouts)
    target_list = list(dict.fromkeys(targets))
    if not target_list:
        return tuple(Route(scout, (), 0) for scout in all_scouts)
    if not all_scouts:
        raise MapError("cannot plan target fields without scouts")

    route_count = min(len(all_scouts), len(target_list))
    active_scouts = all_scouts[:route_count]
    count = len(target_list)
    subset_size = 1 << count
    route_costs: list[list[int]] = []
    predecessors: list[list[dict[int, int | None]]] = []
    for scout in active_scouts:
        costs, previous = _all_routes_for_start(scout.position, target_list)
        route_costs.append(costs)
        predecessors.append(previous)

    partition_cost = [[math.inf] * subset_size for _ in range(route_count + 1)]
    partition_choice: list[list[int | None]] = [
        [None] * subset_size for _ in range(route_count + 1)
    ]
    partition_cost[0][0] = 0
    for scout_index in range(1, route_count + 1):
        for mask in range(subset_size):
            submask = mask
            while True:
                prior_mask = mask ^ submask
                candidate = partition_cost[scout_index - 1][prior_mask] + route_costs[scout_index - 1][submask]
                if candidate < partition_cost[scout_index][mask]:
                    partition_cost[scout_index][mask] = candidate
                    partition_choice[scout_index][mask] = submask
                if submask == 0:
                    break
                submask = (submask - 1) & mask

    full_mask = subset_size - 1
    if not math.isfinite(partition_cost[route_count][full_mask]):
        raise MapError("could not assign all candidate fields to scouts")

    assigned_masks = [0] * route_count
    mask = full_mask
    for scout_index in range(route_count, 0, -1):
        chosen = partition_choice[scout_index][mask]
        if chosen is None:
            raise RuntimeError("route partition predecessor is missing")
        assigned_masks[scout_index - 1] = chosen
        mask ^= chosen

    routes: list[Route] = []
    for scout_index, (scout, assigned) in enumerate(zip(active_scouts, assigned_masks)):
        sequence = _reconstruct_route(assigned, target_list, predecessors[scout_index])
        routes.append(Route(scout, sequence, route_costs[scout_index][assigned]))
    # Preserve any scouts beyond the number of target fields as idle routes.
    routes.extend(Route(scout, (), 0) for scout in all_scouts[route_count:])
    return tuple(routes)


def _extract_scouts(response: Mapping[str, Any]) -> tuple[Scout, ...]:
    """Parse dismount output, retaining only valid scout id/coordinate pairs."""

    raw_spawned = response.get("spawned")
    if not isinstance(raw_spawned, Sequence) or isinstance(raw_spawned, (str, bytes, bytearray)):
        raise HubApiError("dismount response has no spawned scout list")
    scouts: list[Scout] = []
    for item in raw_spawned:
        if not isinstance(item, Mapping):
            continue
        identifier = item.get("scout")
        where = item.get("where")
        if isinstance(identifier, str) and isinstance(where, str):
            scouts.append(Scout(identifier, parse_coordinate(where)))
    if not scouts:
        raise HubApiError("dismount response contains no usable scouts")
    return tuple(scouts)


def _extract_flag(value: Any) -> str | None:
    match = FLAG_RE.search(json.dumps(value, ensure_ascii=False, default=str))
    return match.group(0) if match else None


def _normalise_text(value: Any) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value).casefold())
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def _confirms_survivor(entry: Mapping[str, Any]) -> bool:
    """Recognize a positive inspect log without mistaking a negative clue."""

    for key in ("found", "human", "survivor", "person", "isHuman", "targetFound"):
        if entry.get(key) is True:
            return True
    text = _normalise_text(entry.get("msg", entry.get("message", "")))
    if any(
        phrase in text
        for phrase in (
            "brak celu",
            "brak osoby",
            "brak czlowieka",
            "nie znaleziono",
            "nie znalezlismy",
            "nie odnaleziono",
            "nie ma zadnej osoby",
            "nie ma zadnego czlowieka",
            "nikogo",
            "pust",
        )
    ):
        return False
    return any(
        phrase in text
        for phrase in (
            "partyzant",
            "czlowiek",
            "osob",
            "mezczyzna",
            "kobieta",
            "dziecko",
            "mam go",
            "mam ja",
            "to on",
            "to ona",
            "ranny",
            "ranna",
            "zywy",
            "odnalaz",
            "znalaz",
            "survivor",
            "human",
        )
    )


def _survivor_field(log_response: Mapping[str, Any], fallback: Coordinate) -> Coordinate | None:
    raw_logs = log_response.get("logs")
    if not isinstance(raw_logs, Sequence) or isinstance(raw_logs, (str, bytes, bytearray)):
        return None
    for raw_entry in reversed(raw_logs):
        if not isinstance(raw_entry, Mapping) or not _confirms_survivor(raw_entry):
            continue
        field = raw_entry.get("field")
        if isinstance(field, str):
            try:
                return parse_coordinate(field)
            except ValueError:
                pass
        return fallback
    return None


def _do_move_and_inspect(
    client: HubClient,
    scout: Scout,
    target: Coordinate,
) -> tuple[Coordinate, str | None]:
    """Move one scout, inspect its field, and return a confirmed survivor field."""

    if scout.position != target:
        client.call(
            {
                "action": "move",
                "object": scout.identifier,
                "where": format_coordinate(target),
            }
        )
    client.call({"action": "inspect", "object": scout.identifier})
    logs = client.call({"action": "getLogs"})
    found = _survivor_field(logs, target)
    return target, format_coordinate(found) if found is not None else None


def _create_and_dismount(client: HubClient, drop: str) -> tuple[str, tuple[Scout, ...]]:
    created = client.call(
        {
            "action": "create",
            "type": "transporter",
            "passengers": TRANSPORTER_PASSENGERS,
        }
    )
    transporter = created.get("object")
    if not isinstance(transporter, str) or not transporter:
        raise HubApiError("create response has no transporter identifier")
    client.call({"action": "move", "object": transporter, "where": drop})
    dismounted = client.call(
        {
            "action": "dismount",
            "object": transporter,
            "passengers": TRANSPORTER_PASSENGERS,
        }
    )
    return transporter, _extract_scouts(dismounted)


def _split_targets(candidates: Sequence[Coordinate], city_map: CityMap) -> tuple[list[Coordinate], list[Coordinate]]:
    """Split the fixed top and lower B3 clusters while tolerating future maps."""

    midpoint = city_map.size // 2
    top = sorted(position for position in candidates if position[0] < midpoint)
    bottom = sorted(position for position in candidates if position[0] >= midpoint)
    if top and bottom:
        return top, bottom
    ordered = sorted(candidates)
    split = max(1, len(ordered) // 2)
    return ordered[:split], ordered[split:] or ordered[:split]


def run_operation(*, reset: bool = True, client: HubClient | None = None) -> str:
    """Run the complete live rescue operation and return the verifier flag."""

    client = client or HubClient()
    if reset:
        client.call({"action": "reset"})

    city_map = _map_from_hub(client)
    candidates = _candidate_positions(client, city_map)
    top_targets, bottom_targets = _split_targets(candidates, city_map)
    top_drop, bottom_drop = choose_drop_points(city_map, candidates)
    print(
        f"Map: {city_map.size}x{city_map.size}; {len(candidates)} B3 fields; "
        f"drops {top_drop}/{bottom_drop}"
    )

    _, top_scouts = _create_and_dismount(client, top_drop)
    _, bottom_scouts = _create_and_dismount(client, bottom_drop)
    routes = plan_routes(top_scouts, top_targets) + plan_routes(bottom_scouts, bottom_targets)
    estimated_points = planned_action_points(
        city_map,
        (top_drop, bottom_drop),
        routes,
        len(candidates),
    )
    print(
        "Planned scout movement: "
        f"{sum(route.distance for route in routes)} fields, "
        f"{len(candidates)} inspections, {estimated_points} points"
    )
    if estimated_points > MAX_ACTION_POINTS:
        raise HubApiError(
            f"planned operation needs {estimated_points} points; "
            f"budget is {MAX_ACTION_POINTS}"
        )

    for route in routes:
        for target in route.targets:
            _, found_field = _do_move_and_inspect(client, route.scout, target)
            if found_field is None:
                continue
            helicopter = client.call({"action": "callHelicopter", "destination": found_field})
            flag = _extract_flag(helicopter)
            if flag is None:
                raise HubApiError(
                    "helicopter call was accepted but returned no verifier flag: "
                    f"{_safe_response_summary(helicopter)}"
                )
            print(f"Survivor confirmed at {found_field}; helicopter called.")
            return flag

    raise HubApiError("all B3 fields were inspected but no survivor was confirmed")


def _dry_run() -> None:
    print("Dry run: no Hub calls made and no board state changed.")
    print("Live mode: python -m tasks.s04e03.solution --run")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Solve AI Devs S04E03 Domatowo")
    parser.add_argument(
        "--run",
        action="store_true",
        help="run the authenticated rescue operation against the live Hub",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="resume the current board instead of resetting it (live mode only)",
    )
    args = parser.parse_args(argv)
    if not args.run:
        _dry_run()
        return 0

    try:
        flag = run_operation(reset=not args.no_reset)
    except (HubApiError, MapError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"FLAG: {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
