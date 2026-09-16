"""S02E05 — program the drone for the ``drone`` task.

The dam sector is obtained from the task map with a vision model.  The
documentation is then given to a small text agent which submits a minimal
instruction sequence and uses the Hub's validation feedback to correct it.

The default invocation is a dry run and makes no network calls::

    python -m tasks.s02e05.solution
    python -m tasks.s02e05.solution --run
    python -m tasks.s02e05.solution --run --direct  # no LLM provider required

The live run prints a flag only when the Hub returns one.  Credentials remain
inside the shared configuration and HTTP helpers.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import struct
import sys
import zlib
from html.parser import HTMLParser
from typing import Any, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ai_devs import LLMService, Tool, get_hub_data, post_request, run_agent
from src.ai_devs.config import (
    HUB_VERIFY_URL,
    get_ai_gateway_api_key,
    get_ai_gateway_base_url,
    get_api_key,
    get_openai_api_key,
    get_openai_base_url,
)


TASK_NAME = "drone"
DOC_URL = "https://hub.ag3nts.org/dane/drone.html"
MAP_FILENAME = "drone.png"
TARGET_OBJECT_ID = "PWR6132PL"
DEFAULT_GATEWAY_VISION_MODEL = "gemini-2.5-flash"
DEFAULT_GATEWAY_TEXT_MODEL = "gpt-4.1-mini"
DEFAULT_OPENAI_VISION_MODEL = "gpt-4o"
DEFAULT_OPENAI_TEXT_MODEL = "gpt-4.1-mini"

_FLAG_RE = re.compile(r"\{FLG:[^}]+\}")
_MARKER_RE = re.compile(
    r"\bDAM\s*[:=]\s*\(?\s*(?P<column>\d+)\s*[,x;]\s*(?P<row>\d+)\s*\)?",
    re.IGNORECASE,
)
_COLUMN_ROW_RE = re.compile(
    r"(?:column|columna|col)\s*[:=]?\s*(?P<column>\d+)"
    r"[^\d]{1,80}?"
    r"(?:row|wiersz)\s*[:=]?\s*(?P<row>\d+)",
    re.IGNORECASE,
)
_GRID_RE = re.compile(
    r"\bGRID\s*[:=]\s*(?P<columns>\d+)\s*[x×]\s*(?P<rows>\d+)",
    re.IGNORECASE,
)


class _VisibleTextParser(HTMLParser):
    """Extract visible documentation text without pulling HTML/CSS noise into prompts."""

    _IGNORED_TAGS = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self._ignored_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._IGNORED_TAGS:
            self._ignored_depth += 1
        elif not self._ignored_depth and tag.lower() in {"p", "li", "tr", "h1", "h2", "h3", "pre"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
        elif not self._ignored_depth and tag.lower() in {"p", "li", "tr", "h1", "h2", "h3", "pre"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)


def _visible_text(document: str) -> str:
    """Return readable text from HTML while preserving code examples."""

    parser = _VisibleTextParser()
    parser.feed(document)
    text = html.unescape("".join(parser.parts))
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _extract_flag(value: Any) -> str | None:
    """Extract a Hub flag from a response without manufacturing one."""

    match = _FLAG_RE.search(json.dumps(value, ensure_ascii=False, default=str))
    return match.group(0) if match else None


def _configured(value: str | None) -> bool:
    """Return whether an environment value is usable rather than a template."""

    if not value:
        return False
    lowered = value.strip().lower()
    return not any(marker in lowered for marker in ("your-", ".example", "replace-me", "tutaj"))


def _resolve_provider(provider: str) -> str:
    """Resolve ``auto`` without attempting the repository's placeholder Gateway URL."""

    if provider not in {"auto", "gateway", "openai"}:
        raise ValueError(f"unsupported LLM provider: {provider}")
    if provider != "auto":
        key = get_ai_gateway_api_key() if provider == "gateway" else get_openai_api_key()
        base_url = get_ai_gateway_base_url() if provider == "gateway" else get_openai_base_url()
        if not (_configured(key) and _configured(base_url)):
            raise RuntimeError(f"provider '{provider}' is not configured")
        return provider

    if _configured(os.getenv("AI_GATEWAY_KEY")) and _configured(os.getenv("AI_GATEWAY_BASE_URL")):
        return "gateway"
    if _configured(os.getenv("OPENAI_API_KEY")) and _configured(os.getenv("OPENAI_BASE_URL")):
        return "openai"
    raise RuntimeError("no configured LLM provider (set AI Gateway or OpenAI credentials)")


def _provider_connection(provider: str) -> tuple[str, str]:
    """Return API key and base URL for an already-resolved provider."""

    if provider == "gateway":
        return get_ai_gateway_api_key(), get_ai_gateway_base_url()
    return get_openai_api_key(), get_openai_base_url()


def _feedback_text(value: Any) -> str:
    """Return bounded Hub feedback for the agent while avoiding noisy payloads."""

    if isinstance(value, dict):
        for key in ("error", "message", "feedback", "hint", "detail"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip()[:800]
    return json.dumps(value, ensure_ascii=False, default=str)[:800]


def _parse_map_analysis(analysis: str) -> tuple[int, int, tuple[int, int]]:
    """Read ``GRID=CxR; DAM=column,row`` from the vision model response."""

    grid_match = _GRID_RE.search(analysis)
    if grid_match:
        columns = int(grid_match.group("columns"))
        rows = int(grid_match.group("rows"))
    else:
        # Keep the parser useful if the model omits the explicit marker but
        # still states the grid dimensions in prose.
        dimensions = re.search(r"\b(\d+)\s*[x×]\s*(\d+)\s*(?:grid|sectors?)?\b", analysis, re.IGNORECASE)
        if not dimensions:
            raise ValueError("vision response did not include grid dimensions")
        columns, rows = int(dimensions.group(1)), int(dimensions.group(2))

    marker_match = _MARKER_RE.search(analysis)
    if marker_match:
        column, row = int(marker_match.group("column")), int(marker_match.group("row"))
    else:
        prose_match = _COLUMN_ROW_RE.search(analysis)
        if not prose_match:
            raise ValueError("vision response did not include a dam sector")
        column, row = int(prose_match.group("column")), int(prose_match.group("row"))

    if columns < 1 or rows < 1:
        raise ValueError("vision response contained invalid grid dimensions")
    if not (1 <= column <= columns and 1 <= row <= rows):
        raise ValueError(
            f"dam sector {column},{row} is outside the {columns}x{rows} grid"
        )
    return columns, rows, (column, row)


def _image_message(image_bytes: bytes, content_type: str, prompt: str) -> list[dict]:
    """Build an OpenAI-compatible vision message using bytes, never a key-bearing URL."""

    encoded = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{content_type};base64,{encoded}"
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]


def _paeth(left: int, above: int, upper_left: int) -> int:
    """PNG's Paeth predictor for the dependency-free map fallback."""

    estimate = left + above - upper_left
    distances = (
        abs(estimate - left),
        abs(estimate - above),
        abs(estimate - upper_left),
    )
    return (left, above, upper_left)[distances.index(min(distances))]


def _decode_png_rgb(image_bytes: bytes) -> tuple[int, int, list[list[tuple[int, int, int]]]]:
    """Decode the 8-bit RGB PNG used by the task with only the stdlib."""

    signature = b"\x89PNG\r\n\x1a\n"
    if not image_bytes.startswith(signature):
        raise ValueError("drone map is not a PNG image")

    position = len(signature)
    width = height = bit_depth = color_type = interlace = None
    compressed = bytearray()
    while position + 12 <= len(image_bytes):
        length = struct.unpack(">I", image_bytes[position : position + 4])[0]
        chunk_type = image_bytes[position + 4 : position + 8]
        chunk = image_bytes[position + 8 : position + 8 + length]
        position += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
        elif chunk_type == b"IDAT":
            compressed.extend(chunk)
        elif chunk_type == b"IEND":
            break

    if (width, height, bit_depth, color_type, interlace) != (width, height, 8, 2, 0):
        raise ValueError("unsupported drone map PNG format")
    assert width is not None and height is not None

    row_bytes = width * 3
    raw = zlib.decompress(bytes(compressed))
    expected = height * (row_bytes + 1)
    if len(raw) != expected:
        raise ValueError("drone map PNG data is truncated")

    rows: list[list[tuple[int, int, int]]] = []
    previous = bytearray(row_bytes)
    offset = 0
    for _ in range(height):
        filter_type = raw[offset]
        encoded = raw[offset + 1 : offset + 1 + row_bytes]
        offset += row_bytes + 1
        current = bytearray(row_bytes)
        for index, value in enumerate(encoded):
            left = current[index - 3] if index >= 3 else 0
            above = previous[index]
            upper_left = previous[index - 3] if index >= 3 else 0
            if filter_type == 0:
                prediction = 0
            elif filter_type == 1:
                prediction = left
            elif filter_type == 2:
                prediction = above
            elif filter_type == 3:
                prediction = (left + above) // 2
            elif filter_type == 4:
                prediction = _paeth(left, above, upper_left)
            else:
                raise ValueError(f"unsupported PNG filter {filter_type}")
            current[index] = (value + prediction) & 0xFF
        rows.append([tuple(current[index : index + 3]) for index in range(0, row_bytes, 3)])
        previous = current
    return width, height, rows


def _red_grid_groups(rows: list[list[tuple[int, int, int]]], *, axis: int) -> list[int]:
    """Find centers of near-red grid lines along x (axis 0) or y (axis 1)."""

    height = len(rows)
    width = len(rows[0])
    count = width if axis == 0 else height
    line_scores: list[float] = []
    for coordinate in range(count):
        red_pixels = 0
        samples = height if axis == 0 else width
        for other in range(samples):
            x, y = (coordinate, other) if axis == 0 else (other, coordinate)
            red, green, blue = rows[y][x]
            if red >= 170 and red > green * 1.7 and red > blue * 1.7:
                red_pixels += 1
        line_scores.append(red_pixels / samples)

    groups: list[list[int]] = []
    for coordinate, score in enumerate(line_scores):
        if score < 0.55:
            continue
        if groups and coordinate == groups[-1][-1] + 1:
            groups[-1].append(coordinate)
        else:
            groups.append([coordinate])
    if len(groups) < 2:
        raise ValueError("could not detect the map grid")
    return [sum(group) // len(group) for group in groups]


def _pixel_map_analysis(image_bytes: bytes) -> tuple[int, int, tuple[int, int], str]:
    """Locate the highest-intensity water sector when no vision API is available."""

    width, height, rows = _decode_png_rgb(image_bytes)
    vertical = _red_grid_groups(rows, axis=0)
    horizontal = _red_grid_groups(rows, axis=1)
    # The first and last detected lines are the image border.  The lines in
    # between are the actual sector boundaries; using every midpoint here
    # would create two spurious half-sectors around the outer border.
    x_edges = [0, *vertical[1:-1], width]
    y_edges = [0, *horizontal[1:-1], height]

    best: tuple[float, int, int] | None = None
    for row_index, (top, bottom) in enumerate(zip(y_edges, y_edges[1:]), start=1):
        for column_index, (left, right) in enumerate(zip(x_edges, x_edges[1:]), start=1):
            # Exclude the red border and sample at a modest stride so the
            # fallback remains quick even if the map is served at high DPI.
            margin = 10
            score = 0.0
            pixels = 0
            for y in range(top + margin, max(top + margin, bottom - margin), 4):
                for x in range(left + margin, max(left + margin, right - margin), 4):
                    red, green, blue = rows[y][x]
                    if blue > red + 15 and green > red + 5:
                        score += (blue - red) + (green - red)
                    pixels += 1
            normalised = score / max(1, pixels)
            if best is None or normalised > best[0]:
                best = (normalised, column_index, row_index)

    if best is None:
        raise ValueError("could not score map sectors")
    _, column, row = best
    columns, rows_count = len(x_edges) - 1, len(y_edges) - 1
    return columns, rows_count, (column, row), (
        f"GRID={columns}x{rows_count}; DAM={column},{row}; "
        "selected by intensified blue-green water pixels"
    )


def analyze_map(service: LLMService) -> tuple[int, int, tuple[int, int], str]:
    """Download and inspect the map, returning dimensions, sector, and raw analysis."""

    response = get_hub_data(MAP_FILENAME)
    image_bytes = response.content
    content_type = response.headers.get("Content-Type", "image/png").split(";", 1)[0]
    try:
        analysis = service.chat(
            _image_message(
                image_bytes,
                content_type,
                (
                    "Analyze this aerial map divided by red grid lines. Count the columns "
                    "from left to right and rows from top to bottom. The dam is the sector "
                    "where the water color was intentionally intensified. Coordinates are "
                    "1-indexed. Return a machine-readable first line exactly in the form "
                    "GRID=<columns>x<rows>; DAM=<column>,<row>, then briefly explain the "
                    "visual evidence. Do not use zero-based coordinates."
                ),
            ),
            temperature=0.0,
        )
        columns, rows, sector = _parse_map_analysis(analysis)
    except Exception as exc:
        print(f"Vision model unavailable ({type(exc).__name__}); using local map analysis.")
        columns, rows, sector, analysis = _pixel_map_analysis(image_bytes)
    return columns, rows, sector, analysis


def _submit_instructions(instructions: Sequence[str]) -> dict[str, Any]:
    """Submit one candidate instruction sequence through the stable HTTP helper."""

    if isinstance(instructions, (str, bytes, bytearray)):
        return {"error": "instructions must be a JSON array of strings"}
    candidate = [str(item).strip() for item in instructions]
    if not candidate or any(not item for item in candidate):
        return {"error": "instructions must contain at least one non-empty string"}

    payload = {
        "apikey": get_api_key(),
        "task": TASK_NAME,
        "answer": {"instructions": candidate},
    }
    result = post_request(HUB_VERIFY_URL, payload, raise_on_error=False)
    return result if isinstance(result, dict) else {"response": result}


def _deterministic_instructions(sector: tuple[int, int]) -> list[str]:
    """Build the minimal mission directly from the documented prerequisites."""

    column, row = sector
    return [
        f"setDestinationObject({TARGET_OBJECT_ID})",
        f"set({column},{row})",
        "set(50m)",
        "set(100%)",
        "set(engineON)",
        "set(destroy)",
        "set(return)",
        "flyToLocation",
    ]


def _make_submit_tool(state: dict[str, Any]) -> Tool:
    """Create a tool that records the last live response and any returned flag."""

    def callback(instructions: Sequence[str]) -> dict[str, Any]:
        response = _submit_instructions(instructions)
        state["last_response"] = response
        flag = _extract_flag(response)
        if flag:
            state["flag"] = flag
        return response

    return Tool(
        name="submit_instructions",
        description=(
            "Submit a candidate drone instruction array to the Hub. The response "
            "contains precise validation feedback; correct the sequence and retry "
            "when rejected. Stop immediately when it contains {FLG:...}."
        ),
        parameters={
            "type": "object",
            "properties": {
                "instructions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered drone instruction strings from the API documentation",
                }
            },
            "required": ["instructions"],
        },
        callback=callback,
    )


def run_live(
    *,
    max_iterations: int = 8,
    provider: str = "auto",
    vision_model: str | None = None,
    text_model: str | None = None,
    direct: bool = False,
) -> str | None:
    """Perform map analysis, documentation-guided submission, and return the live flag."""

    if direct:
        image_bytes = get_hub_data(MAP_FILENAME).content
        columns, rows, sector, _ = _pixel_map_analysis(image_bytes)
        print(f"Map grid: {columns} columns x {rows} rows")
        print(f"Locally selected dam sector: column {sector[0]}, row {sector[1]}")
        return _extract_flag(_submit_instructions(_deterministic_instructions(sector)))

    resolved_provider = _resolve_provider(provider)
    if vision_model is None:
        vision_model = (
            DEFAULT_GATEWAY_VISION_MODEL
            if resolved_provider == "gateway"
            else DEFAULT_OPENAI_VISION_MODEL
        )
    if text_model is None:
        text_model = (
            DEFAULT_GATEWAY_TEXT_MODEL
            if resolved_provider == "gateway"
            else DEFAULT_OPENAI_TEXT_MODEL
        )
    api_key, base_url = _provider_connection(resolved_provider)

    vision_service = LLMService(provider=resolved_provider, model=vision_model)
    columns, rows, sector, map_analysis = analyze_map(vision_service)
    print(f"Map grid: {columns} columns x {rows} rows")
    print(f"Vision-selected dam sector: column {sector[0]}, row {sector[1]}")

    from src.ai_devs import get_request

    documentation = _visible_text(get_request(DOC_URL).text)
    state: dict[str, Any] = {}
    submit_tool = _make_submit_tool(state)
    system_prompt = f"""\
You program the DRN-BMB7 drone for the AI DEVS task {TASK_NAME}.

The map was analyzed by a vision model. It reported a {columns}x{rows} grid and
the dam sector at column {sector[0]}, row {sector[1]} (coordinates are 1-indexed).
The requested object identifier is {TARGET_OBJECT_ID}.

Use only methods and syntax supported by the documentation below. Build the
smallest valid sequence that sets the required destination object, dam sector,
flight height, destruction mission, return mission, and starts the flight.
The Hub may reject a candidate with a precise error; use that feedback to fix
the candidate and call submit_instructions again. Do not invent a flag. Stop
as soon as a Hub response contains {{FLG:...}}.

DRONE API DOCUMENTATION:
{documentation}
"""
    user_message = (
        "Prepare and submit the drone mission now. The raw vision analysis was:\n"
        f"{map_analysis[:1200]}"
    )
    try:
        run_agent(
            system_prompt=system_prompt,
            user_message=user_message,
            tools=[submit_tool],
            model=text_model,
            max_iterations=max(1, max_iterations),
            max_tokens=4096,
            verbose=True,
            api_key=api_key,
            base_url=base_url,
        )
    except Exception as exc:
        # A configured model can still be unavailable because of quota or a
        # transient provider error.  The required instruction sequence is
        # fully determined by the fetched documentation, so keep the live
        # task actionable with the deterministic candidate below.
        print(f"Text agent unavailable ({type(exc).__name__}); using documented sequence.")

    if not state.get("flag"):
        response = _submit_instructions(_deterministic_instructions(sector))
        state["last_response"] = response
        flag = _extract_flag(response)
        if flag:
            state["flag"] = flag
    return state.get("flag")


def main() -> int:
    parser = argparse.ArgumentParser(description="AI DEVS S02E05 drone solver")
    parser.add_argument("--run", action="store_true", help="allow live map, docs, and Hub requests")
    parser.add_argument("--direct", action="store_true", help="use local map analysis and the documented instruction sequence without an LLM")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument(
        "--provider",
        choices=("auto", "gateway", "openai"),
        default=os.getenv("DRONE_LLM_PROVIDER", "auto"),
        help="LLM provider (auto prefers a configured AI Gateway, then OpenAI)",
    )
    parser.add_argument("--vision-model", default=os.getenv("DRONE_VISION_MODEL"))
    parser.add_argument("--text-model", default=os.getenv("DRONE_TEXT_MODEL"))
    args = parser.parse_args()

    if not args.run:
        print("Dry run only: no Hub or LLM request was made.")
        print("Expected map coordinate format: GRID=<columns>x<rows>; DAM=<column>,<row>")
        print("Use --run to analyze drone.png and submit the documented mission.")
        return 0

    try:
        flag = run_live(
            max_iterations=args.max_iterations,
            provider=args.provider,
            vision_model=args.vision_model,
            text_model=args.text_model,
            direct=args.direct,
        )
    except Exception as exc:
        print(f"S02E05 failed ({type(exc).__name__}); check Hub and AI Gateway configuration.", file=sys.stderr)
        return 1

    if flag:
        print(f"FLAG: {flag}")
        return 0
    print("Hub did not return a flag within the agent iteration limit.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
