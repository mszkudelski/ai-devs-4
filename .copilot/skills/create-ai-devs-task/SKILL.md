---
name: create-ai-devs-task
description: Scaffold a new AI Devs task solution. Creates the task directory, entry file, and __init__.py following project conventions. Use this skill when asked to create a new task, scaffold a solution, or start a new AI Devs exercise.
---

# Create AI Devs Task

Scaffold a new task solution in the `ai-devs-4` Python project.

## Steps

### 1. Determine task identifier

Extract the season/episode from the user's request (e.g. `s01e03`, `s02e01`). The format is always `s##e##`.

### 2. Check existing tasks

Before creating, check if `tasks/{task_id}/` already exists. If it does, ask the user before overwriting.

### 3. Read the lesson notes (if available)

Check `ai-dev-lessons/4/` for a matching lesson file. Read it to understand the task requirements. This helps tailor the solution scaffold.

### 4. Check existing shared utilities

Before writing any utility code, review what's already available in `src/ai_devs/`:

- `config.py` — `get_api_key()`, `HUB_BASE_URL`, `HUB_VERIFY_URL`, `HUB_DATA_URL`, `HUB_API_URL`
- `api.py` — `post_request()`, `get_request()`, `get_hub_data()`, `send_report()`
- `openai_service.py` — `LLMService` (providers: openrouter, gateway, openai)
- `agent.py` — `Tool`, `run_agent()` for Function Calling agents
- `geo.py` — `haversine_distance()`, `POLISH_CITY_COORDS`

Do NOT recreate these utilities in the task file.

### 5. Create the files

Create these files:

**`tasks/{task_id}/__init__.py`** — Empty file for cross-task imports.

**`tasks/{task_id}/solution.py`** — Main entry point following this template:

```python
"""S##E## — {Task Name}.

{Brief description of what the task does.}

Usage:
    python -m tasks.{task_id}.solution
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ai_devs import get_api_key, get_hub_data, post_request, send_report, LLMService
from src.ai_devs.config import HUB_API_URL


def main():
    # TODO: Implement task logic
    pass


if __name__ == "__main__":
    main()
```

### 6. Import patterns

Always use these import conventions:

```python
# Absolute imports from project root
from src.ai_devs import get_api_key, post_request, send_report, LLMService
from src.ai_devs.config import HUB_VERIFY_URL, HUB_API_URL

# Cross-task imports (when reusing prior task logic)
from tasks.s01e01.solution import some_function
```

### 7. Run command

Tell the user how to run the task:

```bash
cd /Users/marek.szkudelski/VSCode/ai-devs-4
python3 -m tasks.{task_id}.solution
```

## Checklist

- [ ] `tasks/{task_id}/__init__.py` exists
- [ ] `tasks/{task_id}/solution.py` follows the template
- [ ] No hardcoded URLs (use config.py)
- [ ] No hardcoded API keys (use `get_api_key()`)
- [ ] Shared utilities imported, not recreated
- [ ] `if __name__ == "__main__": main()` pattern used
