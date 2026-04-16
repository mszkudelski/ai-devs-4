---
name: analyze-agent-trace
description: Structured analysis of Function Calling agent iteration traces. Identifies wasted iterations, hallucinations, repeated calls, and inefficient patterns. Use this skill when debugging an agent run, analyzing why an agent failed, or reviewing agent output from the terminal.
---

# Analyze Agent Trace

Structured framework for analyzing Function Calling agent iteration logs. Apply this when the user shares terminal output from an agent run (typically via `terminal_selection`).

## Analysis Steps

### 1. Parse the trace

Extract from each iteration:
- Iteration number and max
- Tool name and arguments
- Tool response (or error)
- Whether tools were called in parallel (multiple `→` in same iteration)

### 2. Budget analysis

Calculate:
- **Total iterations used** vs max allowed
- **Useful iterations**: data fetching, access level checks, final submission
- **Wasted iterations**: redundant fetches, failed submissions with same args, unnecessary distance calcs
- **Utilization rate**: useful / total (target: >70%)

### 3. Parallelization check

The agent can call multiple tools in one iteration. Check:
- Were independent data fetches batched? (e.g. 5 `get_person_locations` in one iteration = good)
- Were sequential calls made when parallel was possible? (e.g. 5 `get_access_level` one per iteration = bad)

### 4. Repeated call detection

Flag any tool called with **identical arguments** more than once:
- Same data fetched multiple times → needs caching or prompt fix ("you have perfect memory")
- Same submission retried → needs stronger "never retry same args" instruction

### 5. Hallucination detection

Compare values the agent uses as tool arguments against values actually present in prior tool responses:
- Coordinates not in any response → hallucinated (common when API lacks lat/lon)
- Names not in suspect list → wrong data source
- Plant codes that don't match any response → fabricated

### 6. Decision quality

- Did the agent act on strong signals? (e.g. 2.63 km distance is obviously close — did it pursue that suspect?)
- Did it exhaustively check all candidates before concluding?
- Did it submit prematurely before gathering all evidence?

### 7. Error loop detection

When submissions fail:
- Does the error message give actionable feedback?
- Did the agent change its approach after the error?
- Did it retry with identical args? (= prompt/description needs fixing)

## Output Format

Present findings as:

```
## Agent Trace Analysis

**Summary:** X/Y iterations used, Z wasted. {Succeeded|Failed}.

### Issues (by impact)

| # | Issue | Iterations wasted | Fix |
|---|-------|-------------------|-----|
| 1 | ... | N | ... |

### What went well
- ...

### Recommended changes
1. ...
```

## Common Fixes

| Problem | Fix |
|---------|-----|
| Too many atomic tool calls | Replace with batch tool (e.g. `find_nearest_plant`) |
| Agent re-fetches same data | Add to prompt: "You have perfect memory — never re-fetch" |
| Agent retries same submission | Strengthen: "Never retry with identical arguments" |
| Agent hallucinates coordinates | Enrich data in tool callback before returning |
| Agent doesn't act on strong signals | Add workflow steps to prompt: "If distance < 5km, that's your match" |
| Sequential calls that could be parallel | Model limitation — try a stronger model or reduce tool count |
