"""Reusable Function Calling agent loop.

Provides a generic agent runner that connects an LLM to a set of tools.
The LLM decides which tools to call and with what arguments; tool callbacks
execute the actual logic (including any secret handling like API keys).

The agent NEVER sees API keys or secrets — those are encapsulated in callbacks.

Usage:
    from src.ai_devs.agent import Tool, run_agent

    tools = [
        Tool(
            name="get_weather",
            description="Get current weather for a city.",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            callback=lambda city: {"temp": 22, "unit": "C"},
        ),
    ]
    result = run_agent("You are a helpful assistant.", "What's the weather?", tools)
"""

import json
from typing import Callable, Any, Optional

from openai import OpenAI

from .config import get_open_router_api_key, get_open_router_base_url


class Tool:
    """A tool that an LLM agent can invoke via Function Calling."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,
        callback: Callable[..., Any],
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.callback = callback

    @property
    def openai_schema(self) -> dict:
        """Return the tool definition in OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, **kwargs) -> str:
        """Run the callback and return a JSON string (or error)."""
        try:
            result = self.callback(**kwargs)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})


def run_agent_turn(
    messages: list[dict],
    tools: list[Tool],
    model: str = "gpt-4.1-mini",
    max_iterations: int = 5,
    max_tokens: int = 2048,
    verbose: bool = True,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> tuple[str, list[dict]]:
    """Run one agent turn over an existing messages list.

    The first element of `messages` MUST be the system message; the last MUST
    be the new user message. The input list is shallow-copied (not mutated).

    Args:
        messages: Pre-built conversation history ending with the user message
            that triggers this turn.
        tools: Available tools the agent may call.
        model: LLM model identifier (OpenRouter-compatible).
        max_iterations: Safety cap on the number of LLM round-trips this turn.
        max_tokens: Max tokens per LLM response.
        verbose: Print progress to stdout.
        api_key: Override the default OpenRouter API key.
        base_url: Override the default OpenRouter base URL.

    Returns:
        (final_text, updated_messages) — the agent's final reply and the new
        message history including all assistant + tool messages from this turn.
    """
    client = OpenAI(
        api_key=api_key or get_open_router_api_key(),
        base_url=base_url or get_open_router_base_url(),
    )

    tool_map: dict[str, Tool] = {t.name: t for t in tools}
    openai_tools = [t.openai_schema for t in tools]

    messages = list(messages)

    for i in range(1, max_iterations + 1):
        if verbose:
            print(f"\n--- Iteration {i}/{max_iterations} ---")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools if openai_tools else None,
            max_tokens=max_tokens,
        )

        message = response.choices[0].message

        # ── No tool calls → final answer ──
        if not message.tool_calls:
            final = message.content or ""
            if verbose:
                print(f"Final response: {final[:300]}")
            messages.append({"role": "assistant", "content": final})
            return final, messages

        # ── Record assistant message with tool calls ──
        assistant_msg: dict = {"role": "assistant", "content": message.content or ""}
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]
        messages.append(assistant_msg)

        # ── Execute each tool call ──
        for tc in message.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            if verbose:
                args_preview = json.dumps(fn_args, ensure_ascii=False)[:150]
                print(f"  → {fn_name}({args_preview})")

            tool = tool_map.get(fn_name)
            if tool:
                result = tool.execute(**fn_args)
            else:
                result = json.dumps(
                    {"error": f"Unknown tool '{fn_name}'. Available: {list(tool_map)}"}
                )

            if verbose:
                print(f"  ← {result[:200]}{'...' if len(result) > 200 else ''}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # Exhausted iterations
    if verbose:
        print("Agent reached max iterations without a final response.")
    return (
        "ERROR: Agent reached the maximum iteration limit without producing a final answer.",
        messages,
    )


def run_agent(
    system_prompt: str,
    user_message: str,
    tools: list[Tool],
    model: str = "gpt-4.1-mini",
    max_iterations: int = 15,
    max_tokens: int = 4096,
    verbose: bool = True,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """Run a one-shot Function Calling agent.

    Builds a fresh `[system, user]` message list and runs the tool loop until
    the model produces a final text response or the iteration limit is reached.
    Returns only the final text — see `run_agent_turn` for multi-turn use.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    final, _ = run_agent_turn(
        messages=messages,
        tools=tools,
        model=model,
        max_iterations=max_iterations,
        max_tokens=max_tokens,
        verbose=verbose,
        api_key=api_key,
        base_url=base_url,
    )
    return final
