---
name: design-agent-tools
description: Guidelines and checklist for designing Function Calling tools for LLM agents. Covers granularity, data completeness, secret encapsulation, error handling, and schema design. Use this skill when building tools for an agent, creating function calling tools, or designing agent capabilities.
---

# Design Agent Tools

Best practices for designing Function Calling tools, learned from building AI Devs agents.

## Core Principles

### 1. Match tool granularity to agent decision points

The agent should call a tool when it needs to **make a decision**, not to perform a mechanical sub-step.

**Bad:** `calculate_distance(lat1, lon1, lat2, lon2)` — forces O(N×M) calls for N locations × M plants.

**Good:** `find_nearest_plant(name, surname)` — one call per suspect, returns the closest match.

**Rule of thumb:** If the agent will always loop over items with the same tool in a predictable pattern, that loop belongs **inside the callback**.

### 2. Ensure data completeness in tool outputs

LLMs hallucinate missing data rather than reporting gaps. If an API returns city names but no coordinates, the agent will **invent** plausible coordinates silently.

**Fix:** Enrich data inside the callback before returning. Add coordinates from a lookup, parse dates into usable formats, etc. The agent should never need to "fill in" structural gaps.

### 3. Encapsulate secrets in callbacks

API keys, tokens, and auth headers go **inside** callbacks. The tool schema and description must never mention credentials.

```python
# Good — key is invisible to the agent
def _get_locations(name: str, surname: str) -> dict:
    payload = {"apikey": get_api_key(), "name": name, "surname": surname}
    return post_request(url, payload)

# Bad — key exposed in schema
# parameters: {"apikey": {"type": "string"}, "name": ...}
```

### 4. Descriptions must match actual return data

If the tool description says "returns coordinates", the response MUST contain coordinates. Mismatched descriptions cause the agent to expect fields that don't exist, leading to hallucinated values.

**Before writing the description:** Call the actual API, inspect the response, then write the description to match.

### 5. Return errors as data, not exceptions

Use `raise_on_error=False` (or equivalent) so error bodies flow back to the agent as tool results. The agent can then read error messages and self-correct.

```python
return post_request(url, payload, raise_on_error=False)
# Returns: {"http_status": 400, "code": -910, "message": "Incorrect person..."}
```

### 6. Use precise JSON Schema types

- Use `"type": "integer"` when the API expects integers (birthYear, accessLevel)
- Use `"type": "number"` only for floats (coordinates, distances)
- Use `"type": "string"` with `"description"` showing the expected format (e.g. `"PWR1234PL"`)

### 7. Prefer batch tools over atomic ones

| Atomic (avoid) | Batch (prefer) |
|----------------|----------------|
| `calculate_distance(p1, p2)` | `find_nearest_plant(person)` |
| `get_single_tag(text)` | `tag_all_candidates(candidates)` |
| `check_one_location(lat, lon)` | `get_person_locations(name)` |

Keep atomic tools available as fallbacks for ad-hoc exploration, but provide batch alternatives for the main workflow.

## Tool Definition Checklist

For each tool, verify:

- [ ] **Name** is a verb phrase describing the action (`find_nearest_plant`, not `plant_distance`)
- [ ] **Description** accurately describes what is returned (not what you wish it returned)
- [ ] **Parameters** use correct JSON Schema types with helpful descriptions
- [ ] **Required fields** are listed — don't make the agent guess
- [ ] **Callback** handles auth internally — no secrets in schema
- [ ] **Callback** enriches data if the raw API response has gaps
- [ ] **Callback** returns error bodies as data (not raising exceptions)
- [ ] **Granularity** matches a single agent decision point

## Anti-Patterns

1. **The O(N²) trap:** Atomic tools that force the agent into nested loops. The agent has limited iterations and will run out before finishing.
2. **The hallucination gap:** Tool output missing a field the agent needs. It won't ask — it'll invent.
3. **The silent failure:** Exceptions that turn into generic "tool error" messages. Always return the actual error body.
4. **The description lie:** Saying "returns coordinates" when it returns city names. The agent trusts descriptions.
5. **The key leak:** Putting API keys in tool parameters. The agent will echo them in its reasoning.
