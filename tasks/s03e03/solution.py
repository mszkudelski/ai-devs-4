"""S03E03 — guide a robot through a moving reactor.

The reactor API returns a 7×5 board and the position/direction of every
two-cell block after each command.  This solver predicts one block tick before
choosing the next command: move right when the next cell is safe, wait when it
is not, and move left when waiting would leave the robot in danger.

The default invocation is an offline dry run and performs no Hub calls::

    python -m tasks.s03e03.solution
    python -m tasks.s03e03.solution --run

``--run`` sends ``start`` first and then sends one command at a time to
``/verify`` until the goal is reached.  The returned Hub flag is printed
locally and is never sent anywhere else.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ai_devs.api import post_request
from src.ai_devs.config import HUB_VERIFY_URL, get_api_key


TASK_NAME = "reactor"
BOARD_WIDTH = 7
BOARD_HEIGHT = 5
GOAL_COLUMN = BOARD_WIDTH
COMMANDS = ("start", "reset", "left", "wait", "right")
MOVEMENT_COMMANDS = ("right", "wait", "left")
DEFAULT_MAX_STEPS = 50
FLAG_RE = re.compile(r"\{FLG:[^}]+\}")


class ReactorStateError(ValueError):
    """Raised when a response does not match the reactor API contract."""


class ReactorAPIError(RuntimeError):
    """Raised when a command response contains no usable board state."""

    def __init__(self, message: str, response: Any = None) -> None:
        super().__init__(message)
        self.response = response


class NoSafeActionError(RuntimeError):
    """Raised when all legal movement commands are predicted to collide."""


@dataclass(frozen=True)
class Block:
    """A reactor block using the API's one-based coordinates."""

    column: int
    top_row: int
    bottom_row: int
    direction: str

    def __post_init__(self) -> None:
        if not 1 <= self.column <= BOARD_WIDTH:
            raise ReactorStateError(f"block column outside board: {self.column}")
        if not (1 <= self.top_row < self.bottom_row <= BOARD_HEIGHT):
            raise ReactorStateError(
                f"block rows outside board: {self.top_row}-{self.bottom_row}"
            )
        if self.bottom_row - self.top_row != 1:
            raise ReactorStateError("each block must occupy exactly two rows")
        if self.direction not in {"up", "down"}:
            raise ReactorStateError(f"unknown block direction: {self.direction!r}")

    def occupies(self, row: int) -> bool:
        """Return whether this block occupies a given row."""

        return self.top_row <= row <= self.bottom_row

    def advance(self) -> "Block":
        """Advance one tick, reversing at the top/bottom edge."""

        if self.direction == "down":
            if self.bottom_row == BOARD_HEIGHT:
                return Block(self.column, self.top_row - 1, self.bottom_row - 1, "up")
            return Block(self.column, self.top_row + 1, self.bottom_row + 1, "down")
        if self.top_row == 1:
            return Block(self.column, self.top_row + 1, self.bottom_row + 1, "down")
        return Block(self.column, self.top_row - 1, self.bottom_row - 1, "up")


@dataclass(frozen=True)
class ReactorState:
    """Validated state returned by one command."""

    board: tuple[tuple[str, ...], ...]
    player_col: int
    player_row: int
    goal_col: int
    goal_row: int
    blocks: tuple[Block, ...]
    reached_goal: bool = False
    message: str = ""
    code: int | None = None

    def __post_init__(self) -> None:
        if len(self.board) != BOARD_HEIGHT or any(
            len(row) != BOARD_WIDTH for row in self.board
        ):
            raise ReactorStateError("board must be 7 columns by 5 rows")
        for name, column, row in (
            ("player", self.player_col, self.player_row),
            ("goal", self.goal_col, self.goal_row),
        ):
            if not 1 <= column <= BOARD_WIDTH or not 1 <= row <= BOARD_HEIGHT:
                raise ReactorStateError(f"{name} position outside board: {column},{row}")


@dataclass(frozen=True)
class ActionSafety:
    """One-step safety result for a movement command."""

    command: str
    target_col: int
    safe: bool
    reasons: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        return "SAFE" if self.safe else "DANGER"


def _payload(response: Any) -> Mapping[str, Any]:
    """Decode a requests response or accept an already-decoded JSON object."""

    value = response
    json_method = getattr(response, "json", None)
    if callable(json_method):
        try:
            value = json_method()
        except Exception as exc:
            raise ReactorStateError("response did not contain valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ReactorStateError("response must be a JSON object")
    return value


def _board(value: Any) -> tuple[tuple[str, ...], ...]:
    """Validate the API's 2D board array."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ReactorStateError("board must be a 2D array")
    rows = list(value)
    if len(rows) != BOARD_HEIGHT:
        raise ReactorStateError(f"board must contain {BOARD_HEIGHT} rows")

    result: list[tuple[str, ...]] = []
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise ReactorStateError(f"board row {row_number} must be an array")
        cells = tuple(str(cell) for cell in row)
        if len(cells) != BOARD_WIDTH:
            raise ReactorStateError(f"board row {row_number} must contain {BOARD_WIDTH} cells")
        if any(cell not in {".", "B", "P", "G"} for cell in cells):
            raise ReactorStateError(f"board row {row_number} contains an unknown cell")
        result.append(cells)
    return tuple(result)


def _position(value: Any, field: str) -> tuple[int, int]:
    """Read the API's one-based ``{"col": ..., "row": ...}`` position."""

    if not isinstance(value, Mapping):
        raise ReactorStateError(f"{field} must be an object")
    column, row = value.get("col"), value.get("row")
    if isinstance(column, bool) or not isinstance(column, int):
        raise ReactorStateError(f"{field}.col must be an integer")
    if isinstance(row, bool) or not isinstance(row, int):
        raise ReactorStateError(f"{field}.row must be an integer")
    return column, row


def _blocks(value: Any) -> tuple[Block, ...]:
    """Read the API's list of two-cell blocks."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ReactorStateError("blocks must be an array")
    result: list[Block] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            raise ReactorStateError(f"block {index} must be an object")
        fields = ("col", "top_row", "bottom_row", "direction")
        if any(field not in item for field in fields):
            raise ReactorStateError(f"block {index} is missing a required field")
        result.append(
            Block(
                column=item["col"],
                top_row=item["top_row"],
                bottom_row=item["bottom_row"],
                direction=item["direction"],
            )
        )
    return tuple(result)


def parse_state(response: Any) -> ReactorState:
    """Parse the documented response returned by ``/verify``."""

    value = _payload(response)
    board = _board(value.get("board"))
    player_col, player_row = _position(value.get("player"), "player")
    goal_col, goal_row = _position(value.get("goal"), "goal")
    reached_goal = value.get("reached_goal")
    if not isinstance(reached_goal, bool):
        raise ReactorStateError("reached_goal must be a boolean")
    message = value.get("message", "")
    if not isinstance(message, str):
        message = str(message)
    code = value.get("code")
    if code is not None and (isinstance(code, bool) or not isinstance(code, int)):
        code = None
    return ReactorState(
        board=board,
        player_col=player_col,
        player_row=player_row,
        goal_col=goal_col,
        goal_row=goal_row,
        blocks=_blocks(value.get("blocks")),
        reached_goal=reached_goal,
        message=message,
        code=code,
    )


def predict_blocks(state: ReactorState) -> tuple[Block, ...]:
    """Return block positions after the next command tick."""

    return tuple(block.advance() for block in state.blocks)


def evaluate_actions(state: ReactorState) -> dict[str, ActionSafety]:
    """Predict collisions for all three movement commands.

    Blocks advance once for every command, including ``wait``.  A candidate is
    dangerous when its target column contains an advanced block on row five.
    """

    advanced = predict_blocks(state)
    deltas = {"right": 1, "wait": 0, "left": -1}
    actions: dict[str, ActionSafety] = {}
    for command in MOVEMENT_COMMANDS:
        target_col = state.player_col + deltas[command]
        reasons: list[str] = []
        if not 1 <= target_col <= BOARD_WIDTH:
            reasons.append("target column is outside the board")
        for block in advanced:
            if block.column == target_col and block.occupies(BOARD_HEIGHT):
                reasons.append(f"column {target_col} is occupied on row {BOARD_HEIGHT}")
        actions[command] = ActionSafety(
            command=command,
            target_col=target_col,
            safe=not reasons,
            reasons=tuple(reasons),
        )
    return actions


def safety_statuses(state: ReactorState) -> dict[str, str]:
    """Return ``SAFE``/``DANGER`` labels for diagnostics."""

    return {command: action.status for command, action in evaluate_actions(state).items()}


def choose_command(
    state: ReactorState,
    analysis: Mapping[str, ActionSafety] | None = None,
) -> str:
    """Prefer safe progress, then a safe wait, then a safe escape left."""

    actions = analysis or evaluate_actions(state)
    for command in MOVEMENT_COMMANDS:
        action = actions.get(command)
        if action is not None and action.safe:
            return command
    details = "; ".join(
        f"{command}: {', '.join(actions[command].reasons) or 'danger'}"
        for command in MOVEMENT_COMMANDS
        if command in actions
    )
    raise NoSafeActionError(f"no safe command is available ({details})")


def _render_board(
    state: ReactorState, player_col: int, blocks: Sequence[Block]
) -> tuple[tuple[str, ...], ...]:
    """Render a simulated state for offline checks."""

    board = [["." for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
    for block in blocks:
        for row in (block.top_row, block.bottom_row):
            board[row - 1][block.column - 1] = "B"
    board[state.goal_row - 1][state.goal_col - 1] = "G"
    board[state.player_row - 1][player_col - 1] = "P"
    return tuple(tuple(row) for row in board)


def simulate_command(state: ReactorState, command: str) -> ReactorState:
    """Apply one safe movement command to a local fixture."""

    if command not in MOVEMENT_COMMANDS:
        raise ValueError(f"cannot simulate command {command!r}")
    action = evaluate_actions(state)[command]
    if not action.safe:
        raise NoSafeActionError(f"cannot simulate dangerous command {command!r}")
    blocks = predict_blocks(state)
    return ReactorState(
        board=_render_board(state, action.target_col, blocks),
        player_col=action.target_col,
        player_row=state.player_row,
        goal_col=state.goal_col,
        goal_row=state.goal_row,
        blocks=blocks,
        reached_goal=(action.target_col, state.player_row) == (state.goal_col, state.goal_row),
        message=f"Player moved {command}.",
        code=100,
    )


def plan_commands(state: ReactorState, max_steps: int = DEFAULT_MAX_STEPS) -> list[str]:
    """Build an offline deterministic plan from a validated initial state."""

    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    current = state
    commands: list[str] = []
    while not current.reached_goal:
        if len(commands) >= max_steps:
            raise NoSafeActionError("plan exceeded max_steps")
        command = choose_command(current)
        commands.append(command)
        current = simulate_command(current, command)
    return commands


def extract_flag(value: Any) -> str | None:
    """Extract a returned Hub flag without inventing one for dry runs."""

    match = FLAG_RE.search(json.dumps(value, ensure_ascii=False, default=str))
    return match.group(0) if match else None


def send_command(command: str, *, timeout: float = 120) -> tuple[ReactorState, Any]:
    """Send one command through the shared Hub helper and parse its response."""

    if command not in COMMANDS:
        raise ValueError(f"unsupported reactor command: {command!r}")
    payload = {
        "apikey": get_api_key(),
        "task": TASK_NAME,
        "answer": {"command": command},
    }
    response: Any = None
    try:
        response = post_request(
            HUB_VERIFY_URL,
            payload,
            raise_on_error=False,
            timeout=timeout,
        )
        return parse_state(response), response
    except ReactorStateError as exc:
        detail = ""
        if isinstance(response, Mapping):
            summary = {
                key: response[key]
                for key in ("http_status", "code", "message", "error", "ok")
                if key in response
            }
            if summary:
                detail = f": {json.dumps(summary, ensure_ascii=False, default=str)[:300]}"
        raise ReactorAPIError(
            f"command {command!r} returned no usable state{detail}", response
        ) from exc
    except Exception as exc:
        raise ReactorAPIError(f"command {command!r} failed ({type(exc).__name__})") from exc


def run_live(*, max_steps: int = DEFAULT_MAX_STEPS, timeout: float = 120) -> dict[str, Any]:
    """Run the guarded command loop against the live Hub."""

    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    commands: list[str] = []
    try:
        state, response = send_command("start", timeout=timeout)
        commands.append("start")
        flag = extract_flag(response)
        if flag:
            return {"status": "success", "steps": 1, "flag": flag, "commands": commands}
        if state.reached_goal:
            return {"status": "solved-without-flag", "steps": 1, "commands": commands}

        for _ in range(max_steps):
            command = choose_command(state)
            commands.append(command)
            state, response = send_command(command, timeout=timeout)
            flag = extract_flag(response)
            if flag:
                return {
                    "status": "success",
                    "steps": len(commands),
                    "flag": flag,
                    "commands": commands,
                }
            if state.reached_goal:
                return {
                    "status": "solved-without-flag",
                    "steps": len(commands),
                    "commands": commands,
                }
    except ReactorAPIError as exc:
        flag = extract_flag(exc.response)
        if flag:
            return {
                "status": "success",
                "steps": len(commands),
                "flag": flag,
                "commands": commands,
            }
        return {"status": "error", "steps": len(commands), "commands": commands, "error": str(exc)}
    except (NoSafeActionError, ReactorStateError) as exc:
        return {"status": "error", "steps": len(commands), "commands": commands, "error": str(exc)}
    return {
        "status": "max-steps",
        "steps": len(commands),
        "commands": commands,
        "error": f"goal was not reached within {max_steps} movement steps",
    }


def _fixture_state() -> ReactorState:
    """Return a lesson-shaped board used by the offline dry run."""

    return parse_state(
        {
            "code": 100,
            "message": "Fixture state.",
            "board": [
                [".", "B", ".", ".", ".", ".", "."],
                [".", "B", ".", ".", "B", ".", "."],
                [".", ".", "B", "B", "B", "B", "."],
                [".", ".", "B", "B", ".", "B", "."],
                ["P", ".", ".", ".", ".", ".", "G"],
            ],
            "player": {"col": 1, "row": 5},
            "goal": {"col": 7, "row": 5},
            "blocks": [
                {"col": 2, "top_row": 1, "bottom_row": 2, "direction": "down"},
                {"col": 3, "top_row": 3, "bottom_row": 4, "direction": "up"},
                {"col": 4, "top_row": 3, "bottom_row": 4, "direction": "down"},
                {"col": 5, "top_row": 2, "bottom_row": 3, "direction": "down"},
                {"col": 6, "top_row": 3, "bottom_row": 4, "direction": "up"},
            ],
            "reached_goal": False,
        }
    )


def render_board(state: ReactorState) -> str:
    """Render a board as plain text for dry-run diagnostics."""

    return "\n".join(" ".join(row) for row in state.board)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline preview or the explicitly requested live game."""

    parser = argparse.ArgumentParser(description="S03E03 deterministic reactor solver")
    parser.add_argument(
        "--run", action="store_true", help="send guarded commands to the live Hub"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"maximum movement commands after start (default: {DEFAULT_MAX_STEPS})",
    )
    args = parser.parse_args(argv)

    if not args.run:
        fixture = _fixture_state()
        plan = plan_commands(fixture, args.max_steps)
        print("Dry run only: no Hub calls were made.")
        print(f"Fixture board:\n{render_board(fixture)}")
        print(f"Initial safety: {json.dumps(safety_statuses(fixture))}")
        print(f"Planned commands ({len(plan)}): {' '.join(plan)}")
        print("Use --run to send start and the guarded movement commands to the Hub.")
        return 0

    result = run_live(max_steps=args.max_steps)
    print(f"Run result: {json.dumps(result, ensure_ascii=False)}")
    if result.get("flag"):
        print(f"Flag: {result['flag']}")
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
