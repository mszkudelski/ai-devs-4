# AI_devs 4 — Task Solutions

Python solutions for [AI_devs 4: Builders](https://hub.ag3nts.org/) tasks.

## Setup

1. **Clone & enter the project:**
   ```bash
   cd ai-devs-4
   ```

2. **Install dependencies:**
   ```bash
   python3 -m pip install -r requirements.txt
   ```

3. **Configure `.env`:**
   ```dotenv
   AI_DEVS_API_KEY=your-key-from-hub
   OPEN_ROUTER_API_KEY=your-openrouter-key
   ```
   - `AI_DEVS_API_KEY` — get it from [hub.ag3nts.org](https://hub.ag3nts.org/)
   - `OPEN_ROUTER_API_KEY` — get it from [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys)

## Running Tasks

From the project root:

```bash
python3 -m tasks.<task_dir>.solution
```

For example:

```bash
python3 -m tasks.s01e01.solution
```

## Project Structure

```
ai-devs-4/
├── .env                    # API keys (not committed)
├── requirements.txt        # Python dependencies
├── src/
│   └── ai_devs/            # Reusable tools
│       ├── config.py        # Env vars, URLs
│       ├── api.py           # HTTP helpers, send_report(), get_hub_data()
│       └── openai_service.py # OpenRouterService (chat, structured output)
└── tasks/
    └── s01e01/
        └── solution.py      # Task solution
```

### Shared Tools (`src/ai_devs/`)

| Module | Key exports |
|--------|-------------|
| `config` | `get_api_key()`, `get_open_router_api_key()` |
| `api` | `send_report(task, answer)`, `get_hub_data(filename)`, `post_request()`, `get_request()` |
| `openai_service` | `OpenRouterService` — `.chat()`, `.structured_output()`, `.simple_query()` |

### Usage in a task

```python
from src.ai_devs import OpenRouterService, send_report, get_hub_data

service = OpenRouterService()

# Simple query
answer = service.simple_query("What is 2+2?")

# Structured output with Pydantic
from pydantic import BaseModel
class MySchema(BaseModel):
    value: int

result = service.structured_output(messages=[...], response_schema=MySchema)

# Submit answer
send_report("task-name", answer)
```
