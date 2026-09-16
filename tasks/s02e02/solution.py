"""S02E02 — solve the 3×3 electricity board.

The Hub exposes the current board as ``electricity.json``.  Each matrix value
is a tile state; the immutable solved board is shown by
``https://hub.ag3nts.org/i/solved_electricity.png``.  A rotation request turns
one tile 90 degrees clockwise, so a tile requiring three turns is submitted
three times.

The default invocation is a dry run and does not contact the Hub::

    python -m tasks.s02e02.solution
    python -m tasks.s02e02.solution --run

The planner is deterministic and keeps the API key inside the shared Hub
helpers.  It reports the returned flag when the final live rotation is
accepted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Sequence
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ai_devs import get_hub_data, send_report


TASK_NAME = "electricity"
BOARD_FILENAME = "electricity.json"
SOLVED_IMAGE_URL = "https://hub.ag3nts.org/i/solved_electricity.png"
BOARD_SIZE = 3

# Directions are ordered clockwise.  The integer codes are the tile states
# used by electricity.json.  The mapping was derived by pairing the live JSON
# states with the black wire arms in electricity.png and cross-checking them
# against the solved image.  A value identifies one current orientation;
# rotate a mask to compare it with the target cell rather than treating the
# numbers as rotation counts.
_VALUE_TO_MASK: dict[int, frozenset[str]] = {
    0: frozenset("UR"),
    1: frozenset("UD"),
    2: frozenset("URD"),
    3: frozenset("RD"),
    4: frozenset("RL"),
    5: frozenset("RDL"),
    6: frozenset("DL"),
    7: frozenset("UL"),
    8: frozenset("UDL"),
    9: frozenset("URL"),
}

# The target image is fixed by the task.  These are the nine black wire arms
# read from its 3×3 grid, in row × column order (U/R/D/L directions).
_TARGET_MASKS: tuple[tuple[frozenset[str], ...], ...] = (
    (frozenset("RD"), frozenset("RDL"), frozenset("RL")),
    (frozenset("UD"), frozenset("URD"), frozenset("RDL")),
    (frozenset("URL"), frozenset("UL"), frozenset("UR")),
)

_CLOCKWISE = {"U": "R", "R": "D", "D": "L", "L": "U"}
_FLAG_RE = re.compile(r"\{FLG:[^}]+\}")


def rotate_mask(mask: frozenset[str], turns: int = 1) -> frozenset[str]:
    """Rotate a tile mask clockwise by ``turns`` quarter turns."""

    result = set(mask)
    for _ in range(turns % 4):
        result = {_CLOCKWISE[direction] for direction in result}
    return frozenset(result)


def _coerce_board(value: Any) -> list[list[int]]:
    """Extract and validate a rectangular integer matrix from Hub JSON."""

    # The current Hub response is the matrix itself.  Accept a small set of
    # obvious wrappers so a harmless API envelope does not break the solver.
    if isinstance(value, dict):
        for key in ("board", "grid", "matrix", "data", "electricity"):
            if key in value:
                return _coerce_board(value[key])
        raise ValueError("electricity.json did not contain a board matrix")

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("electricity.json must be a 3x3 integer matrix")
    rows = list(value)
    if len(rows) != BOARD_SIZE:
        raise ValueError(f"expected {BOARD_SIZE} board rows, received {len(rows)}")

    board: list[list[int]] = []
    for row_index, row in enumerate(rows, start=1):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise ValueError(f"board row {row_index} is not a sequence")
        cells = list(row)
        if len(cells) != BOARD_SIZE:
            raise ValueError(
                f"expected {BOARD_SIZE} cells in row {row_index}, received {len(cells)}"
            )
        parsed: list[int] = []
        for col_index, cell in enumerate(cells, start=1):
            # bool is an int subclass but is not a valid tile state.
            if isinstance(cell, bool) or not isinstance(cell, int):
                raise ValueError(f"cell {row_index}x{col_index} is not an integer")
            if cell not in _VALUE_TO_MASK:
                raise ValueError(f"cell {row_index}x{col_index} has unknown tile state {cell}")
            parsed.append(cell)
        board.append(parsed)
    return board


def parse_board_response(response: Any) -> list[list[int]]:
    """Read the matrix from a requests response or an already-decoded value."""

    payload = response
    json_method = getattr(response, "json", None)
    if callable(json_method):
        payload = json_method()
    return _coerce_board(payload)


def plan_rotations(
    board: Sequence[Sequence[int]],
    target_masks: Sequence[Sequence[frozenset[str]]] = _TARGET_MASKS,
) -> list[str]:
    """Return row×column IDs, one per clockwise quarter-turn required.

    The order is row-major and stable.  Repeating an ID is intentional: the
    Hub API accepts one 90-degree rotation per report.
    """

    current = _coerce_board(board)
    if len(target_masks) != BOARD_SIZE or any(len(row) != BOARD_SIZE for row in target_masks):
        raise ValueError("target masks must be a 3x3 matrix")

    plan: list[str] = []
    for row_index, row in enumerate(current):
        for col_index, value in enumerate(row):
            initial = _VALUE_TO_MASK[value]
            target = frozenset(target_masks[row_index][col_index])
            turns = next(
                (
                    count
                    for count in range(4)
                    if rotate_mask(initial, count) == target
                ),
                None,
            )
            if turns is None:
                raise ValueError(
                    f"tile {row_index + 1}x{col_index + 1} state {value} "
                    "cannot reach the target orientation"
                )
            plan.extend([f"{row_index + 1}x{col_index + 1}"] * turns)
    return plan


def _extract_flag(value: Any) -> str | None:
    """Extract a Hub flag without manufacturing one in dry-run output."""

    match = _FLAG_RE.search(json.dumps(value, ensure_ascii=False, default=str))
    return match.group(0) if match else None


def _download_board() -> list[list[int]]:
    response = get_hub_data(BOARD_FILENAME)
    return parse_board_response(response)


def _reset_board() -> None:
    """Ask the Hub for a fresh board through the shared data helper."""

    # The image endpoint supports the documented reset query.  Fetching JSON
    # afterwards still uses the task's required electricity.json endpoint.
    # Keep the API key inside get_hub_data rather than constructing a URL here.
    get_hub_data("electricity.png?reset=1")


def _run(board: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Apply a computed plan sequentially and return bounded run metadata."""

    plan = plan_rotations(board)
    completed: list[str] = []
    for coordinate in plan:
        try:
            response = send_report(TASK_NAME, {"rotate": coordinate})
        except Exception as exc:
            return {
                "status": "error",
                "completed": completed,
                "error": f"rotation {coordinate} failed ({type(exc).__name__})",
            }
        completed.append(coordinate)
        flag = _extract_flag(response)
        if flag:
            return {
                "status": "success",
                "completed": completed,
                "flag": flag,
                "hub_response": response,
            }

    return {
        "status": "solved-without-flag" if not plan else "error",
        "completed": completed,
        "error": "Hub returned no flag after the planned rotations.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="S02E02 deterministic electricity solver")
    parser.add_argument("--run", action="store_true", help="allow live Hub download and rotations")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="request a fresh board before downloading electricity.json",
    )
    args = parser.parse_args()

    if not args.run:
        fixture = [[0, 2, 1], [4, 5, 5], [8, 7, 0]]
        print("Dry run only: no Hub calls were made.")
        print(f"Fixture plan: {', '.join(plan_rotations(fixture))}")
        print(f"Target source: {SOLVED_IMAGE_URL}")
        print("Use --run to download electricity.json and apply the rotations.")
        return 0

    try:
        if args.reset:
            _reset_board()
        board = _download_board()
        plan = plan_rotations(board)
        print(f"Current board: {board}")
        print(f"Rotation plan ({len(plan)} quarter-turns): {', '.join(plan) or 'already solved'}")
        result = _run(board)
    except Exception as exc:
        # Exception text from requests can include the credential-bearing data
        # URL.  Keep normal output useful without echoing that URL.
        print(f"Solver failed ({type(exc).__name__}).", file=sys.stderr)
        return 1

    print(f"Run result: {json.dumps(result, ensure_ascii=False, default=str)}")
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
