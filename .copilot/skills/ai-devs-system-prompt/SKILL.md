---
name: ai-devs-system-prompt
description: Guidelines for crafting effective system prompts for Function Calling agents in AI Devs tasks. Covers workflow structure, error handling rules, descriptive vs prescriptive style, and length constraints. Use this skill when writing a system prompt for an agent, designing agent instructions, or fixing agent behavior through prompt changes.
---

# AI Devs System Prompt Design

Guidelines for writing system prompts that make Function Calling agents effective and focused.

## Core Principles

### 1. Be descriptive, not prescriptive

Describe the **mission and context**, not the tool names. Let the agent discover tools from their schemas.

**Bad:**
```
Use get_suspects to get the list, then call find_nearest_plant for each one,
then call get_access_level, then call submit_report.
```

**Good:**
```
A group of suspects has been identified. One was recently observed near a
nuclear power plant. Determine which suspect it is, verify their access level,
and file a report.
```

### 2. Include a numbered workflow

Vague prompts lead to scattered behavior. Explicit steps focus the agent dramatically.

```
Workflow:
1. Get the suspect list.
2. For each suspect, find the nearest power plant to their observed locations.
3. Identify the suspect with the smallest distance.
4. Get that suspect's access level.
5. Submit the report with name, access level, and plant code.
```

**Why it works:** The agent treats numbered steps as a checklist. Without them, it may skip ahead (submitting before checking all candidates) or loop endlessly (rechecking data it already has).

### 3. Add explicit rules for common failure modes

Always include these rules (adapted to the task):

```
Rules:
- Always use the available tools — never guess coordinates or distances.
- Work efficiently: check all candidates before drawing conclusions.
- If a report submission fails, carefully read the error message — it
  contains hints about what went wrong. Use those hints to re-examine
  your data and try a corrected submission. Never retry with the same arguments.
```

### 4. Keep it under ~200 words

Longer prompts dilute focus, especially for smaller models (gpt-4.1-mini). Every sentence should earn its place.

| Section | Target length |
|---------|---------------|
| Mission context | 2-3 sentences |
| Workflow steps | 4-6 numbered items |
| Rules | 3-5 bullet points |
| **Total** | **~150-200 words** |

## Template

```python
SYSTEM_PROMPT = """\
You are an investigative agent solving {brief context}.

{2-3 sentences describing the situation and goal, without naming tools.}

You have access to {abstract description of tool categories: databases, APIs,
calculators, reporting endpoints}.

Workflow:
1. {First data gathering step}
2. {Main analysis step — often "for each X, do Y"}
3. {Decision step — "identify the one that..."}
4. {Verification step — get additional details}
5. {Submission step}

Rules:
- Always use the available tools — never guess or fabricate data.
- Work efficiently: {task-specific efficiency hint}.
- If a submission fails, read the error message carefully and adjust. \
Never retry with identical arguments.
"""
```

## Anti-Patterns

1. **Tool name dropping:** Mentioning tool names in the prompt. The agent should discover them from schemas. Naming tools makes the prompt brittle if tools change.

2. **No workflow:** "Find the answer and submit it." Too vague — the agent wanders.

3. **No error rules:** Without "never retry same args", the agent will submit the same wrong answer 5+ times.

4. **Over-constraining:** "You MUST call get_suspects first, then EXACTLY 5 calls to find_nearest_plant." Too rigid — breaks if the suspect count changes.

5. **Wall of text:** 500+ word prompts with background lore. The agent stops reading. Keep it focused.

## Iteration Tips

If the agent misbehaves, diagnose before editing the prompt:

| Behavior | Likely cause | Fix |
|----------|-------------|-----|
| Submits before checking all candidates | No "check all first" rule | Add to workflow: "check ALL suspects before concluding" |
| Retries same failed submission | No retry rule | Add: "Never retry with identical arguments" |
| Calls tools in wrong order | No workflow steps | Add numbered workflow |
| Ignores some tools | Too many tools / unclear descriptions | Improve tool descriptions, reduce tool count |
| Runs out of iterations | Tools too granular | Replace atomic tools with batch tools (see design-agent-tools skill) |
