---
name: hub-api-patterns
description: Common patterns and gotchas for the AI Devs Hub API (hub.ag3nts.org). Covers authentication, endpoints, error codes, rate limiting, data enrichment, and answer pool constraints. Use this skill when working with the Hub API, submitting reports, fetching hub data, or debugging Hub errors.
---

# Hub API Patterns

Documented patterns for interacting with the AI Devs Hub API, based on real task experience.

## Endpoints

All endpoints derive from `HUB_BASE_URL` (set in `.env`, accessed via `config.py`):

| Endpoint | Purpose | Import |
|----------|---------|--------|
| `{base}/verify` | Submit task answers | `HUB_VERIFY_URL` from config |
| `{base}/data/{apikey}/{file}` | Fetch task data files | `get_hub_data(filename)` |
| `{base}/api/location` | Person GPS sightings | `HUB_API_URL` from config |
| `{base}/api/accesslevel` | Person access level | `HUB_API_URL` from config |

## Authentication

All Hub requests use API key **in the POST body**, not in headers:

```python
payload = {
    "apikey": get_api_key(),  # from config.py
    "name": "...",
    "surname": "..."
}
result = post_request(f"{HUB_API_URL}/location", payload)
```

## Submitting Reports

Use `send_report()` for simple submissions:

```python
from src.ai_devs import send_report
send_report("task_name", answer_value)
```

For agent callbacks where you need error bodies returned (not raised):

```python
payload = {
    "apikey": get_api_key(),
    "task": "findhim",
    "answer": {"name": "...", "accessLevel": 7, ...}
}
return post_request(HUB_VERIFY_URL, payload, raise_on_error=False)
```

## Known Error Codes

| Code | Message | Meaning |
|------|---------|---------|
| -910 | Incorrect person identification | Wrong name, surname, access level, or plant code |
| -920 | Field must match format | Value doesn't match expected pattern (e.g. `PWR0000PL`) |
| 0 | `{FLG:...}` | Success — the flag is in the message |

**Important:** Error `-910` does NOT tell you which field is wrong. The agent must systematically re-examine all fields.

## Rate Limiting

The Hub returns HTTP 429 when rate-limited. The shared `post_request()` in `api.py` includes retry with exponential backoff (3s, 6s, 12s). No extra handling needed in task code.

## Data Enrichment Gotchas

Hub data files often lack fields you'd expect:

| File | What's missing | Fix |
|------|---------------|-----|
| `findhim_locations.json` | Plant coordinates (only city names) | Geocode using `POLISH_CITY_COORDS` from `geo.py` |
| `people.csv` | Job category tags | Run through LLM tagging (see S01E01 `tag_jobs`) |

**Always inspect the actual API response** before writing tool descriptions. Don't assume fields exist.

## Answer Pool Constraints

Some tasks only accept answers from a specific pool:

- **S01E02 "findhim"**: Only accepts names that were in the S01E01 transport-tagged result set
- If Hub says "Unknown name", your suspect list is too broad — re-derive from the prior task

When building tools, import and reuse prior task logic:

```python
from tasks.s01e01.solution import parse_csv, filter_people, tag_jobs
```

Ensure `tasks/__init__.py` and `tasks/s01e01/__init__.py` exist for cross-task imports.

## Fetching Data Files

Use the shared utility — don't build raw URLs:

```python
from src.ai_devs import get_hub_data

resp = get_hub_data("people.csv")     # returns requests.Response
text = resp.text                       # for CSV
data = resp.json()                     # for JSON
```

## Checklist for Hub Interactions

- [ ] Using `get_api_key()` from config (not hardcoded)
- [ ] Using `HUB_API_URL` / `HUB_VERIFY_URL` from config (not hardcoded URLs)
- [ ] Using `post_request()` or `get_hub_data()` (not raw requests)
- [ ] Agent tool callbacks use `raise_on_error=False` for submit endpoints
- [ ] Data enrichment done in callbacks when API responses lack expected fields
- [ ] Cross-task imports have `__init__.py` files in place
