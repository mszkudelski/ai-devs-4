"""S04E02 — schedule a wind turbine within the Hub service window.

The task API exposes a small, asynchronous, self describing protocol.  The
runner starts the service window, asks for the API help and turbine reports,
derives the schedule from those reports, signs every schedule point in
parallel, and submits one batch configuration followed by ``done``.

The default invocation is deliberately a dry run and performs no network
calls::

    python -m tasks.s04e02.solution
    python -m tasks.s04e02.solution --run

``--run`` is the only mode that sends requests to the Hub.  The Hub flag is
printed when ``done`` returns it; this module never submits the flag anywhere.
The parser accepts the small schema used by the current Hub and is tolerant of
the wrapper keys used by older task responses, while refusing to invent a
schedule when a required turbine limit cannot be derived.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import math
import os
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import requests


TASK_NAME = "windpower"
FLAG_RE = re.compile(r"\{FLG:[^}]+\}")
QUEUE_CONFIRMATION_CODES = frozenset({14, 21, 31, 41})
NO_RESULT_CODE = 11
DEFAULT_SERVICE_SECONDS = 39.0
DEFAULT_POLL_INTERVAL = 0.12
DEFAULT_HTTP_TIMEOUT = (2.5, 7.0)
# The Hub can queue three unlock jobs reliably at once.  Larger bursts may
# acknowledge every request but leave one or more signatures unprocessed.
UNLOCK_BATCH_SIZE = 3


class WindpowerError(RuntimeError):
    """Raised when the Hub response cannot be interpreted safely."""


@dataclass(frozen=True)
class WeatherPoint:
    """One hourly weather observation used by the schedule planner."""

    datetime: str
    wind_ms: float
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class SchedulePoint:
    """One turbine setting before the Hub generates its unlock code."""

    datetime: str
    pitch_angle: float
    turbine_mode: str
    wind_ms: float


@dataclass(frozen=True)
class TurbineFacts:
    """Limits and curves extracted from the turbine report/documentation."""

    maximum_wind_ms: float
    cut_in_wind_ms: float | None
    optimal_pitch_angle: float | None
    pitch_curve: tuple[tuple[float, float], ...]
    power_curve: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class PlantFacts:
    """Power requirement extracted from the power-plant report."""

    required_power: float | None


def _number(value: Any) -> float | None:
    """Return a finite number, accepting numeric strings and unit suffixes."""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if not isinstance(value, str):
        return None
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", value.replace(" ", ""))
    if not match:
        return None
    try:
        result = float(match.group(0).replace(",", "."))
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _key(value: Any) -> str:
    """Normalise API field names for comparisons."""

    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def _json_value(value: Any) -> Any:
    """Decode a JSON string while preserving ordinary text responses."""

    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _walk(value: Any) -> Iterable[Any]:
    """Yield a value and all nested JSON containers."""

    value = _json_value(value)
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk(child)


def _mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    for node in _walk(value):
        if isinstance(node, Mapping):
            yield node


def _field(mapping: Mapping[str, Any], *names: str) -> Any:
    """Get a field by normalised name, including common suffix variants."""

    wanted = {_key(name) for name in names}
    for name, value in mapping.items():
        normalised = _key(name)
        if normalised in wanted:
            return value
    return None


def _field_name(mapping: Mapping[str, Any], predicates: Sequence[str]) -> str | None:
    """Return the first key whose normalised name contains a predicate."""

    normalised_predicates = tuple(_key(item) for item in predicates)
    for name in mapping:
        candidate = _key(name)
        if any(predicate in candidate for predicate in normalised_predicates):
            return str(name)
    return None


def _unwrap_result(value: Any) -> Any:
    """Unwrap transport/result envelopes without losing the original value."""

    value = _json_value(value)
    if not isinstance(value, Mapping):
        return value
    for name in ("result", "data", "payload", "response", "output"):
        child = _field(value, name)
        if child is not None and child is not value:
            # Keep the outer sourceFunction/code available to callers that
            # need it; data parsers can still see the nested value separately.
            return child
    return value


def _response_code(value: Any) -> int | None:
    if not isinstance(value, Mapping):
        return None
    raw = _field(value, "code", "statusCode")
    if isinstance(raw, str):
        try:
            return int(raw.strip())
        except ValueError:
            return None
    return raw if isinstance(raw, int) and not isinstance(raw, bool) else None


def _source_function(value: Any) -> str | None:
    for node in _mappings(value):
        raw = _field(node, "sourceFunction", "source", "function")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _extract_flag(value: Any) -> str | None:
    match = FLAG_RE.search(_as_text(value))
    return match.group(0) if match else None


def _datetime_from_value(value: Any) -> dt.datetime | None:
    """Parse ISO/date/hour values into a naive UTC-like datetime."""

    if isinstance(value, dt.datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Weather APIs occasionally expose Unix seconds.  Treat implausible
        # small values as non-dates rather than converting them silently.
        if value > 10**9:
            try:
                return dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc).replace(tzinfo=None)
            except (OverflowError, OSError, ValueError):
                return None
        return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.isdigit() and len(text) >= 10:
        try:
            return dt.datetime.fromtimestamp(float(text), tz=dt.timezone.utc).replace(tzinfo=None)
        except (OverflowError, OSError, ValueError):
            return None
    candidate = text.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(candidate)
        return parsed.replace(tzinfo=None)
    except ValueError:
        pass
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%d.%m.%Y %H:%M:%S",
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y",
    ):
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _normalise_hour(value: Any) -> str | None:
    """Normalise an hour field to ``HH:00:00``."""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        hour = int(value)
        if 0 <= hour <= 23:
            return f"{hour:02d}:00:00"
        return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    match = re.search(r"(?:^|[ T])([01]?\d|2[0-3])(?::\d{1,2})?(?::\d{1,2})?(?:Z|$)", text)
    if match:
        return f"{int(match.group(1)):02d}:00:00"
    if text.isdigit() and 0 <= int(text) <= 23:
        return f"{int(text):02d}:00:00"
    return None


def normalise_datetime(value: Any, hour: Any = None) -> str | None:
    """Return the Hub schedule format ``YYYY-MM-DD HH:00:00``."""

    parsed = _datetime_from_value(value)
    if parsed is None and isinstance(value, str):
        # A separate date and hour is common in compact forecast rows.
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", value)
        if date_match:
            parsed = _datetime_from_value(date_match.group(0))
    if parsed is None:
        return None
    if hour is not None:
        hour_text = _normalise_hour(hour)
        if hour_text:
            parsed = parsed.replace(hour=int(hour_text[:2]), minute=0, second=0, microsecond=0)
    return parsed.strftime("%Y-%m-%d %H:00:00")


def _row_datetime(row: Mapping[str, Any]) -> str | None:
    """Read combined or separate date/time fields from a forecast row."""

    combined = _field(row, "datetime", "dateTime", "timestamp", "validTime", "validAt", "time")
    date_value = _field(row, "date", "day", "forecastDate", "validDate")
    hour_value = _field(row, "hour", "forecastHour", "validHour", "timeHour")

    if combined is not None:
        result = normalise_datetime(combined)
        if result:
            return result
    if date_value is not None:
        return normalise_datetime(date_value, hour_value)
    # A row may use ``time`` as an hour while date is inherited at a parent;
    # this case is handled by the array parser below when possible.
    return None


def _wind_value(row: Mapping[str, Any]) -> float | None:
    """Extract wind speed while preferring explicit wind fields."""

    candidates: list[tuple[int, float]] = []
    for name, raw in row.items():
        normalised = _key(name)
        value = _number(raw)
        if value is None:
            continue
        score = -1
        if "wind" in normalised or "wiatr" in normalised:
            score = 100
        elif normalised in {"speed", "windspeed", "velocity"}:
            score = 90
        elif normalised in {"value", "reading", "measurement"}:
            score = 25
        if "direction" in normalised or "gust" in normalised or "angle" in normalised:
            score = -1
        if score >= 0:
            candidates.append((score, value))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _forecast_rows(value: Any) -> list[WeatherPoint]:
    """Extract hourly weather rows from current and legacy response shapes."""

    points: list[WeatherPoint] = []
    seen: set[tuple[str, float]] = set()
    for row in _mappings(value):
        datetime_text = _row_datetime(row)
        wind = _wind_value(row)
        if datetime_text is None or wind is None:
            continue
        marker = (datetime_text, round(wind, 8))
        if marker in seen:
            continue
        seen.add(marker)
        points.append(WeatherPoint(datetime_text, wind, row))

    # Handle column-oriented arrays such as {"dates": [...], "windMs": [...]}.
    for node in _mappings(value):
        date_values = None
        wind_values = None
        for name, raw in node.items():
            normalised = _key(name)
            if isinstance(raw, (list, tuple)):
                if any(token in normalised for token in ("date", "time", "timestamp")):
                    date_values = raw
                elif ("wind" in normalised or "wiatr" in normalised) and raw:
                    wind_values = raw
        if date_values is None or wind_values is None:
            continue
        for date_value, wind_value in zip(date_values, wind_values):
            datetime_text = normalise_datetime(date_value)
            wind = _number(wind_value)
            if datetime_text is None or wind is None:
                continue
            marker = (datetime_text, round(wind, 8))
            if marker not in seen:
                seen.add(marker)
                points.append(WeatherPoint(datetime_text, wind, node))

    points.sort(key=lambda item: item.datetime)
    return points


def extract_weather_points(value: Any) -> list[WeatherPoint]:
    """Public weather parser used by the planner and local checks."""

    points = _forecast_rows(value)
    if not points:
        raise WindpowerError("weather response contains no dated wind observations")
    return points


def _numeric_fields(value: Any) -> Iterable[tuple[str, float, Mapping[str, Any]]]:
    for node in _mappings(value):
        for name, raw in node.items():
            number = _number(raw)
            if number is not None:
                yield _key(name), number, node


def _find_named_number(value: Any, names: Sequence[str], contains: Sequence[str] = ()) -> float | None:
    exact = {_key(name) for name in names}
    contains_keys = tuple(_key(name) for name in contains)
    # Exact names win over broad textual matches.
    for name, number, _ in _numeric_fields(value):
        if name in exact:
            return number
    for name, number, _ in _numeric_fields(value):
        if any(token in name for token in contains_keys):
            return number
    return None


def _curve(value: Any, x_names: Sequence[str], y_names: Sequence[str]) -> tuple[tuple[float, float], ...]:
    """Extract numeric (wind, output/angle) pairs from table-like JSON."""

    x_exact = {_key(name) for name in x_names}
    y_exact = {_key(name) for name in y_names}
    rows: list[tuple[float, float]] = []
    for row in _mappings(value):
        x: float | None = None
        y: float | None = None
        for name, raw in row.items():
            normalised = _key(name)
            number = _number(raw)
            if number is None:
                continue
            if normalised in x_exact or ("wind" in normalised and "direction" not in normalised):
                x = number
            if normalised in y_exact or any(token in normalised for token in ("power", "output", "pitch", "angle")):
                y = number
        if x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
            rows.append((x, y))
    return tuple(sorted(set(rows)))


def _extract_max_wind(turbine: Any, documentation: Any) -> float | None:
    names = (
        "maximumWindSpeed",
        "maxWindSpeed",
        "maxWind",
        "maximumWind",
        "windResistance",
        "maxResistance",
        "cutOutWindSpeed",
        "cutoutWindSpeed",
        "cutoffWindMs",
        "cutoffWindSpeed",
        "maxSafeWindSpeed",
        "maximumSafeWind",
    )
    value = _find_named_number(turbine, names)
    if value is None:
        value = _find_named_number(documentation, names)
    if value is not None:
        return value
    text = f"{_as_text(turbine)}\n{_as_text(documentation)}"
    patterns = (
        r"(?:maximum|max(?:imum)?|cut[- ]?out|resistance|wytrzym[a-z]*)[^\d]{0,100}(\d+(?:[.,]\d+)?)\s*(?:m\s*/?\s*s|mps|ms\b)",
        r"(\d+(?:[.,]\d+)?)\s*(?:m\s*/?\s*s|mps|ms\b)[^\n]{0,80}(?:maximum|max|cut[- ]?out|resistance|wytrzym[a-z]*)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _number(match.group(1))
    return None


def _extract_cut_in(turbine: Any, documentation: Any) -> float | None:
    names = (
        "cutInWindSpeed",
        "cutinWindSpeed",
        "minimumWindSpeed",
        "minWindSpeed",
        "startWindSpeed",
        "minOperationalWindMs",
        "minimumOperationalWindMs",
    )
    return _find_named_number(turbine, names) or _find_named_number(documentation, names)


def _extract_pitch_curve(turbine: Any, documentation: Any) -> tuple[tuple[float, float], ...]:
    return _curve(
        (turbine, documentation),
        ("wind", "windSpeed", "windMs", "speed"),
        ("pitch", "pitchAngle", "optimalPitch", "angle"),
    )


def _extract_power_curve(turbine: Any, documentation: Any) -> tuple[tuple[float, float], ...]:
    return _curve(
        (turbine, documentation),
        ("wind", "windSpeed", "windMs", "speed"),
        ("power", "powerOutput", "output", "generation", "production"),
    )


def _extract_optimal_pitch(turbine: Any, documentation: Any) -> float | None:
    names = (
        "optimalPitchAngle",
        "recommendedPitchAngle",
        "productionPitchAngle",
        "pitchAngle",
        "pitchAngleDeg",
        "pitch",
    )
    value = _find_named_number(turbine, names)
    if value is None:
        value = _find_named_number(documentation, names)
    return value


def extract_turbine_facts(turbine: Any, documentation: Any) -> TurbineFacts:
    """Extract safety and production facts from the two turbine sources."""

    maximum = _extract_max_wind(turbine, documentation)
    if maximum is None or maximum <= 0:
        raise WindpowerError("turbine documentation contains no positive maximum wind speed")
    return TurbineFacts(
        maximum_wind_ms=maximum,
        cut_in_wind_ms=_extract_cut_in(turbine, documentation),
        optimal_pitch_angle=_extract_optimal_pitch(turbine, documentation),
        pitch_curve=_extract_pitch_curve(turbine, documentation),
        power_curve=_extract_power_curve(turbine, documentation),
    )


def _extract_required_power(powerplant: Any) -> float | None:
    names = (
        "requiredPower",
        "powerRequired",
        "missingPower",
        "powerDeficit",
        "deficit",
        "energyDeficit",
        "requiredEnergy",
        "demand",
        "neededPower",
        "powerNeed",
    )
    value = _find_named_number(powerplant, names)
    if value is not None and value >= 0:
        return value
    # Some versions return a short sentence instead of an object.
    text = _as_text(powerplant)
    match = re.search(
        r"(?:required|missing|deficit|needed|demand|zapotrzeb|brakuj[a-z]*)[^\d]{0,80}(\d+(?:[.,]\d+)?)\s*(?:kw|mw|w|kwh|mwh)?",
        text,
        re.IGNORECASE,
    )
    return _number(match.group(1)) if match else None


def extract_plant_facts(powerplant: Any) -> PlantFacts:
    return PlantFacts(required_power=_extract_required_power(powerplant))


def _interpolated(curve: Sequence[tuple[float, float]], x: float) -> float | None:
    if not curve:
        return None
    ordered = sorted(curve)
    if x <= ordered[0][0]:
        return ordered[0][1]
    if x >= ordered[-1][0]:
        return ordered[-1][1]
    for (left_x, left_y), (right_x, right_y) in zip(ordered, ordered[1:]):
        if left_x <= x <= right_x:
            if right_x == left_x:
                return right_y
            fraction = (x - left_x) / (right_x - left_x)
            return left_y + fraction * (right_y - left_y)
    return None


def _production_pitch(facts: TurbineFacts, wind_ms: float) -> float:
    if facts.pitch_curve:
        # A curve row with the closest wind speed is safer than interpolating
        # between angles when the API describes discrete operating modes.
        return min(facts.pitch_curve, key=lambda item: abs(item[0] - wind_ms))[1]
    if facts.optimal_pitch_angle is not None:
        return facts.optimal_pitch_angle
    raise WindpowerError("turbine documentation contains no production pitch angle")


def _has_enough_power(facts: TurbineFacts, plant: PlantFacts, point: WeatherPoint) -> bool:
    if plant.required_power is None:
        return True
    available = _interpolated(facts.power_curve, point.wind_ms)
    if available is None:
        # If the Hub only exposes a demand and no power curve, the first safe
        # point is the only defensible answer; the verifier will still reject
        # an impossible point rather than this code inventing a capacity.
        return True
    return available >= plant.required_power


def plan_schedule(
    weather: Any,
    turbine: Any,
    powerplant: Any,
    documentation: Any,
) -> list[SchedulePoint]:
    """Build all storm points plus the first safe point meeting demand.

    Every forecast hour above the documented wind resistance is represented.
    This is intentional: the rotor automatically returns to its normal
    position roughly one hour later, so a long storm needs repeated idle
    settings.  The production point is selected chronologically and never
    replaces a storm setting.
    """

    weather_points = extract_weather_points(weather)
    turbine_facts = extract_turbine_facts(turbine, documentation)
    plant_facts = extract_plant_facts(powerplant)

    storms = [
        SchedulePoint(point.datetime, 90.0, "idle", point.wind_ms)
        for point in weather_points
        if point.wind_ms > turbine_facts.maximum_wind_ms
    ]
    storm_datetimes = {point.datetime for point in storms}
    cut_in = turbine_facts.cut_in_wind_ms
    safe_candidates = [
        point
        for point in weather_points
        if point.datetime not in storm_datetimes
        and point.wind_ms <= turbine_facts.maximum_wind_ms
        and (cut_in is None or point.wind_ms >= cut_in)
        and _has_enough_power(turbine_facts, plant_facts, point)
    ]
    if not safe_candidates:
        raise WindpowerError("forecast contains no safe hour capable of producing the required power")
    production = safe_candidates[0]
    production_point = SchedulePoint(
        production.datetime,
        _production_pitch(turbine_facts, production.wind_ms),
        "production",
        production.wind_ms,
    )

    # Keep chronological order for reproducible diagnostics and API payloads.
    all_points = storms + [production_point]
    all_points.sort(key=lambda item: (item.datetime, item.turbine_mode != "idle"))
    return all_points


def _action_node(help_response: Any, action: str) -> Mapping[str, Any] | None:
    """Find an action specification in structured or nested help output."""

    for node in _mappings(help_response):
        actions = _field(node, "actions", "endpoints", "operations")
        if isinstance(actions, Mapping):
            candidate = _field(actions, action)
            if isinstance(candidate, Mapping):
                return candidate
        elif isinstance(actions, Sequence) and not isinstance(actions, (str, bytes, bytearray)):
            for item in actions:
                if isinstance(item, Mapping) and str(_field(item, "action", "name", "id") or "") == action:
                    return item
        name = _field(node, "action", "name", "id")
        if isinstance(name, str) and name == action:
            return node
    return None


def _required_params(help_response: Any, action: str, fallback: Sequence[str]) -> list[str]:
    node = _action_node(help_response, action)
    if not node:
        return list(fallback)
    value = _field(node, "required", "requiredParams", "requiredParameters", "requires")
    if isinstance(value, Mapping):
        value = list(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        names = [str(item) for item in value if isinstance(item, (str, int, float))]
        if names:
            return names
    # A few help responses describe params as a list of objects with a
    # required flag; preserve their order.
    params = _field(node, "params", "parameters", "arguments", "input")
    if isinstance(params, Sequence) and not isinstance(params, (str, bytes, bytearray)):
        names = []
        for item in params:
            if isinstance(item, Mapping) and _field(item, "required") is not False:
                name = _field(item, "name", "key", "param")
                if name is not None:
                    names.append(str(name))
        if names:
            return names
    return list(fallback)


def _param_values(help_response: Any, action: str) -> list[str]:
    node = _action_node(help_response, action)
    if not node:
        return []
    raw = _field(node, "paramValues", "param_values", "values", "enum")
    if isinstance(raw, Mapping):
        raw = _field(raw, "param", "values")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    return [str(item) for item in raw if isinstance(item, (str, int, float))]


def data_params_from_help(help_response: Any) -> list[str]:
    """Return data params in help order, with current-Hub fallback names."""

    values = _param_values(help_response, "get")
    expected = ("weather", "turbinecheck", "powerplantcheck", "documentation")
    chosen = [name for name in expected if name in values]
    if chosen:
        return chosen
    # The lesson's API has kept these names stable; use the fallback only when
    # a malformed/legacy help response omits enum metadata.
    return list(expected)


def _signed_params(value: Any) -> Mapping[str, Any] | None:
    for node in _mappings(value):
        raw = _field(node, "signedParams", "signed_parameters", "params")
        if isinstance(raw, Mapping):
            if _field(raw, "startDate", "startHour") is not None:
                return raw
    return None


def _unlock_code(value: Any) -> str | None:
    for node in _mappings(value):
        for name in ("unlockCode", "unlock_code", "signature", "signedCode"):
            raw = _field(node, name)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
    return None


def _source_matches(value: Any, expected: str) -> bool:
    source = _source_function(value)
    return source == expected


class HubClient:
    """Small HTTP client with bounded retries and no global result assumptions."""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        *,
        timeout: tuple[float, float] = DEFAULT_HTTP_TIMEOUT,
        deadline: float | None = None,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.deadline = deadline
        self._request_lock = threading.Lock()

    def call(self, action: str, **params: Any) -> Any:
        """Send one action and return decoded JSON or text."""

        payload = {"apikey": self.api_key, "task": TASK_NAME, "answer": {"action": action, **params}}
        attempts = 0
        while True:
            if self.deadline is not None and time.monotonic() >= self.deadline:
                raise WindpowerError(f"service window expired before action {action}")
            try:
                response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempts >= 1:
                    raise WindpowerError(f"Hub request failed for {action}: {type(exc).__name__}") from exc
                attempts += 1
                continue
            if response.status_code in (429, 502, 503, 504) and attempts < 2:
                attempts += 1
                retry_after = response.headers.get("Retry-After")
                delay = min(float(retry_after), 0.5) if retry_after and retry_after.replace(".", "", 1).isdigit() else 0.15
                if self.deadline is not None:
                    delay = min(delay, max(0.0, self.deadline - time.monotonic()))
                if delay:
                    time.sleep(delay)
                continue
            text = response.text
            try:
                result = response.json()
            except ValueError:
                result = text
            if not response.ok:
                detail = _as_text(result)[:300]
                raise WindpowerError(f"Hub returned HTTP {response.status_code} for {action}: {detail}")
            flag = _extract_flag(result)
            if flag:
                # Returning the response lets the caller decide whether this
                # is a final or incidental flag; no external submission occurs.
                return result
            return result


class ResultCollector:
    """Consume the Hub's global result queue without losing out-of-order jobs."""

    def __init__(self, client: HubClient, *, poll_interval: float = DEFAULT_POLL_INTERVAL) -> None:
        self.client = client
        self.poll_interval = poll_interval
        self.pool: list[Any] = []
        self._lock = threading.Lock()

    def _take_from_pool(self, predicate: Callable[[Any], bool]) -> Any | None:
        for index, value in enumerate(self.pool):
            if predicate(value):
                return self.pool.pop(index)
        return None

    def collect(
        self,
        predicates: Mapping[str, Callable[[Any], bool]],
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Collect one result for each named predicate, pooling other results."""

        deadline = min(time.monotonic() + timeout_seconds, self.client.deadline or float("inf"))
        pending = dict(predicates)
        collected: dict[str, Any] = {}
        while pending and time.monotonic() < deadline:
            for name in list(pending):
                pooled = self._take_from_pool(pending[name])
                if pooled is not None:
                    collected[name] = pooled
                    del pending[name]
            if not pending:
                break

            # Serialise getResult calls.  The result queue is global, and two
            # simultaneous consumers can otherwise consume each other's item
            # between the predicate check and pool insertion.
            with self._lock:
                for name in list(pending):
                    pooled = self._take_from_pool(pending[name])
                    if pooled is not None:
                        collected[name] = pooled
                        del pending[name]
                if not pending:
                    continue
                result = self.client.call("getResult")
                if _response_code(result) == NO_RESULT_CODE:
                    pass
                else:
                    matched_name = next((name for name, predicate in pending.items() if predicate(result)), None)
                    if matched_name is None:
                        self.pool.append(result)
                    else:
                        collected[matched_name] = result
                        del pending[matched_name]
            if pending:
                time.sleep(min(self.poll_interval, max(0.0, deadline - time.monotonic())))
        if pending:
            raise WindpowerError(f"timed out waiting for results: {', '.join(pending)}")
        return collected


def _parallel_calls(client: HubClient, calls: Sequence[tuple[str, Mapping[str, Any]]]) -> list[Any]:
    """Submit independent actions concurrently and preserve input order."""

    if not calls:
        return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(calls)) as executor:
        futures = [executor.submit(client.call, action, **dict(params)) for action, params in calls]
        return [future.result() for future in futures]


def _unlock_predicate(point: SchedulePoint) -> Callable[[Any], bool]:
    """Match one unlock result by its signed date and hour."""

    date_text, hour_text = point.datetime.split(" ", 1)

    def matches(result: Any) -> bool:
        if not _source_matches(result, "unlockCodeGenerator"):
            return False
        signed = _signed_params(result)
        if signed is None:
            return False
        signed_date = _field(signed, "startDate", "date")
        signed_hour = _field(signed, "startHour", "hour")
        return str(signed_date) == date_text and str(signed_hour) == hour_text

    return matches


def _endpoint_and_key() -> tuple[str, str]:
    """Read live credentials lazily so dry runs remain harmless."""

    from src.ai_devs.config import HUB_VERIFY_URL, get_api_key

    return HUB_VERIFY_URL, get_api_key()


def build_config_payload(points: Sequence[SchedulePoint], unlock_results: Sequence[Any]) -> dict[str, Any]:
    """Match unlock results by signed date/hour and build batch config."""

    by_datetime: dict[str, str] = {}
    for result in unlock_results:
        code = _unlock_code(result)
        signed = _signed_params(result)
        if code is None or signed is None:
            continue
        date_value = _field(signed, "startDate", "date")
        hour_value = _field(signed, "startHour", "hour")
        datetime_text = normalise_datetime(date_value, hour_value)
        if datetime_text:
            by_datetime[datetime_text] = code
    configs: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for point in points:
        code = by_datetime.get(point.datetime)
        if code is None:
            missing.append(point.datetime)
            continue
        configs[point.datetime] = {
            "pitchAngle": point.pitch_angle,
            "turbineMode": point.turbine_mode,
            "unlockCode": code,
        }
    if missing:
        raise WindpowerError(f"missing unlock result for: {', '.join(missing)}")
    return configs


def run_live(*, service_seconds: float = DEFAULT_SERVICE_SECONDS) -> str:
    """Execute the complete windpower protocol and return the Hub flag."""

    endpoint, api_key = _endpoint_and_key()
    started_at = time.monotonic()
    deadline = started_at + service_seconds
    client = HubClient(api_key, endpoint, deadline=deadline)

    print("Starting windpower service window")
    start_response = client.call("start")
    # The current Hub acknowledges ``start`` with code 60.  Other successful
    # versions use 0 or omit the code, while queued actions use the documented
    # positive confirmation codes.
    if _response_code(start_response) not in (None, 0, 60) and _response_code(start_response) not in QUEUE_CONFIRMATION_CODES:
        raise WindpowerError(f"unexpected start response code: {_response_code(start_response)}")

    # Help and documentation are independent and are fetched together.  The
    # help response determines the exact unlock fields and get enum values.
    help_response, documentation = _parallel_calls(
        client,
        (("help", {}), ("get", {"param": "documentation"})),
    )
    data_params = data_params_from_help(help_response)
    data_calls = [("get", {"param": param}) for param in data_params if param != "documentation"]
    data_responses = _parallel_calls(client, data_calls)
    data_by_source = {
        param: response
        for param, response in zip((param for param in data_params if param != "documentation"), data_responses)
    }
    weather = data_by_source.get("weather")
    turbine = data_by_source.get("turbinecheck")
    powerplant = data_by_source.get("powerplantcheck")
    if weather is None or turbine is None or powerplant is None:
        raise WindpowerError("help response did not expose weather, turbinecheck and powerplantcheck data")

    collector = ResultCollector(client)
    predicates = {
        param: (lambda result, expected=param: _source_matches(result, expected))
        for param in data_by_source
    }
    async_data = collector.collect(predicates, timeout_seconds=max(0.1, deadline - time.monotonic() - 1.5))
    weather = async_data.get("weather", weather)
    turbine = async_data.get("turbinecheck", turbine)
    powerplant = async_data.get("powerplantcheck", powerplant)
    points = plan_schedule(weather, turbine, powerplant, documentation)
    print(f"Planned {len(points)} configuration points ({sum(p.turbine_mode == 'idle' for p in points)} storm points)")

    unlock_required = _required_params(
        help_response,
        "unlockCodeGenerator",
        ("startDate", "startHour", "pitchAngle", "turbineMode", "windMs"),
    )
    lookup: dict[str, Any] = {}
    unlock_calls: list[tuple[str, Mapping[str, Any]]] = []
    for point in points:
        date_text, hour_text = point.datetime.split(" ", 1)
        lookup.update(
            {
                "startDate": date_text,
                "startHour": hour_text,
                "pitchAngle": point.pitch_angle,
                "turbineMode": point.turbine_mode,
                "windMs": point.wind_ms,
            }
        )
        params = {name: lookup[name] for name in unlock_required if name in lookup}
        if len(params) != len(unlock_required):
            missing = [name for name in unlock_required if name not in params]
            raise WindpowerError(f"cannot build unlock request; unsupported fields: {', '.join(missing)}")
        unlock_calls.append(("unlockCodeGenerator", params))
    # Keep the burst bounded.  In live probing the Hub acknowledged a burst
    # of four requests but only completed three of them; collect each small
    # batch before enqueueing the next one so every signature is retained.
    unlock_results: list[Any] = []
    for offset in range(0, len(points), UNLOCK_BATCH_SIZE):
        batch_points = points[offset : offset + UNLOCK_BATCH_SIZE]
        batch_calls = unlock_calls[offset : offset + UNLOCK_BATCH_SIZE]
        _parallel_calls(client, batch_calls)
        batch_predicates = {
            point.datetime: _unlock_predicate(point)
            for point in batch_points
        }
        batch_results = collector.collect(
            batch_predicates,
            timeout_seconds=max(0.1, deadline - time.monotonic() - 0.8),
        )
        unlock_results.extend(batch_results.values())

    configs = build_config_payload(points, unlock_results)
    client.call("config", configs=configs)

    # The initial get(turbinecheck) is the documented turbine test action. It
    # is deliberately consumed before configuration; the Hub permits each
    # generated report to be read only once.
    done_response = client.call("done")
    flag = _extract_flag(done_response)
    if flag is None:
        raise WindpowerError("done completed without a Hub flag")
    print(f"FLAG: {flag}")
    return flag


def _load_fixture(path: str) -> tuple[Any, Any, Any, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise WindpowerError("fixture root must be an object")
    required = ("weather", "turbine", "powerplant", "documentation")
    missing = [name for name in required if name not in value]
    if missing:
        raise WindpowerError(f"fixture missing: {', '.join(missing)}")
    return tuple(value[name] for name in required)  # type: ignore[return-value]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Solve the AI_devs S04E02 windpower task")
    parser.add_argument("--run", action="store_true", help="run against the live Hub API")
    parser.add_argument("--plan", metavar="JSON", help="plan a local fixture without network calls")
    args = parser.parse_args(argv)
    try:
        if args.plan:
            weather, turbine, powerplant, documentation = _load_fixture(args.plan)
            points = plan_schedule(weather, turbine, powerplant, documentation)
            print(json.dumps([point.__dict__ for point in points], ensure_ascii=False, indent=2))
            return 0
        if not args.run:
            print("Dry run: no network calls. Use --run to execute windpower against the Hub.")
            return 0
        run_live()
        return 0
    except (WindpowerError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
