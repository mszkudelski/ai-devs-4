# AI_devs 4 — Task Solutions

Python solutions for AI_devs 4: Builders tasks.

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
   - `AI_DEVS_API_KEY` — hub API key
   - `OPEN_ROUTER_API_KEY` — OpenRouter API key

## AI Gateway — Available Models

Use these exact names as `model=` when `provider="gateway"` or passing `base_url=get_ai_gateway_base_url()`:

| Model | Provider | Mode | Max Input | Max Output |
|---|---|---|---|---|
| `gemini-2.5-flash` | Vertex AI | chat | 1049K | 66K |
| `gemini-2.5-flash-lite` | Vertex AI | chat | 1049K | 66K |
| `gemini-3-flash-preview` | Vertex AI | chat | 1049K | 66K |
| `gemini-3.1-flash-lite-preview` | Vertex AI | chat | 1049K | 66K |
| `gemini-3.1-pro-preview` | Vertex AI | chat | 1049K | 66K |
| `gpt-4.1-mini` | Azure | chat | 1048K | 33K |
| `gpt-4.1-mini-batch` | Azure | chat | N/A | N/A |
| `gpt-4o-mini` | Azure | chat | 128K | 16K |
| `gpt-4o-mini-batch` | Azure | chat | N/A | N/A |
| `gpt-5-chat` | Azure | chat | 128K | 16K |
| `gpt-5-mini` | Azure | chat | 272K | 128K |
| `gpt-5.1` | Azure | chat | 272K | 128K |
| `gpt-5.1-codex-mini` | Azure | responses | 272K | 128K |
| `gpt-5.2-chat` | Azure | chat | 128K | 16K |
| `gpt-5.3-chat` | Azure | chat | 128K | 16K |
| `text-embedding-3-small` | Azure | embedding | 8K | — |

**Recommended:** `gpt-4.1-mini` for agents/function-calling, `gemini-2.5-flash` for vision.

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
