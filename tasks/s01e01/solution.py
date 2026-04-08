"""S01E01 — People task.

Download people.csv, filter by criteria, tag jobs via LLM structured output,
and submit the answer to the hub.

Usage:
    python -m tasks.s01e01.solution
"""

import csv
import io
from typing import Optional
from pydantic import BaseModel, Field

# Allow running as module from project root
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ai_devs import OpenRouterService, send_report, get_hub_data


# ── Pydantic schemas for structured output ──────────────────────────

AVAILABLE_TAGS = ["IT", "transport", "edukacja", "medycyna",
                  "praca z ludźmi", "praca z pojazdami", "praca fizyczna"]


class PersonTags(BaseModel):
    """Tags assigned to a single person's job."""
    index: int = Field(description="The index number of the person in the input list")
    tags: list[str] = Field(
        description="List of tags from the allowed set that match this person's job description"
    )


class BatchTaggingResult(BaseModel):
    """Result of batch-tagging multiple job descriptions at once."""
    results: list[PersonTags] = Field(
        description="A list of tagging results, one per person from the input"
    )


# ── Helpers ─────────────────────────────────────────────────────────

def parse_csv(text: str) -> list[dict]:
    """Parse CSV text into a list of dicts."""
    reader = csv.DictReader(io.StringIO(text))
    return [row for row in reader]


def filter_people(people: list[dict]) -> list[dict]:
    """Filter: males, born in Grudziądz, aged 20-40 in 2026."""
    filtered = []
    for p in people:
        gender = p.get("gender", "").strip()
        birth_place = p.get("birthPlace", "").strip()
        birth_date = p.get("birthDate", "").strip()
        try:
            born_year = int(birth_date.split("-")[0])
        except (ValueError, IndexError):
            continue

        age_in_2026 = 2026 - born_year
        if gender == "M" and birth_place == "Grudziądz" and 20 <= age_in_2026 <= 40:
            filtered.append(p)
    return filtered


def build_tagging_prompt(people: list[dict]) -> str:
    """Build a prompt listing job descriptions for batch tagging."""
    lines = []
    for i, p in enumerate(people):
        job = p.get("job", "unknown")
        lines.append(f"{i}. {job}")
    return "\n".join(lines)


SYSTEM_PROMPT = """\
You are a job classification assistant. You receive a numbered list of job descriptions.
For each job, assign one or more tags from the following set:

- IT — information technology, software development, system administration, data analysis, programming
- transport — transportation, logistics, shipping, driving vehicles, fleet management, couriering
- edukacja — education, teaching, tutoring, academic work, training
- medycyna — medicine, healthcare, pharmacy, nursing, medical diagnostics
- praca z ludźmi — customer service, sales, management, HR, social work, consulting
- praca z pojazdami — vehicle maintenance, mechanic work, automotive, operating machinery
- praca fizyczna — manual labor, construction, warehouse work, cleaning, physical tasks

Rules:
- A person can have multiple tags if their job spans several categories.
- Use ONLY tags from the list above (exact spelling).
- If unsure, pick the closest matching tag(s).
- Return results for ALL people in the input list.
"""


def tag_jobs(people: list[dict], service: OpenRouterService) -> list[list[str]]:
    """Use LLM structured output to tag job descriptions in batch."""
    prompt = build_tagging_prompt(people)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    result: BatchTaggingResult = service.structured_output(
        messages=messages,
        response_schema=BatchTaggingResult,
        max_tokens=4000,
    )
    # Build index→tags map
    tag_map = {r.index: r.tags for r in result.results}
    return [tag_map.get(i, []) for i in range(len(people))]


def build_answer(people: list[dict], tags_list: list[list[str]]) -> list[dict]:
    """Build the answer payload: only people with 'transport' tag."""
    answer = []
    for person, tags in zip(people, tags_list):
        if "transport" in tags:
            birth_year = int(person["birthDate"].split("-")[0])
            answer.append({
                "name": person["name"].strip(),
                "surname": person["surname"].strip(),
                "gender": person["gender"].strip(),
                "born": birth_year,
                "city": person["birthPlace"].strip(),
                "tags": tags,
            })
    return answer


# ── Main ────────────────────────────────────────────────────────────

def main():
    # 1. Download CSV
    response = get_hub_data("people.csv")
    csv_text = response.text
    people_all = parse_csv(csv_text)
    print(f"Total people in CSV: {len(people_all)}")

    # 2. Filter by criteria
    candidates = filter_people(people_all)
    print(f"After filtering (male, Grudziądz, age 20-40): {candidates}")
    print(f"Candidates count: {len(candidates)}")

    if not candidates:
        print("No candidates found matching criteria. Check the CSV data.")
        return

    # 3. Tag jobs via LLM
    service = OpenRouterService()
    tags_list = tag_jobs(candidates, service)
    for person, tags in zip(candidates, tags_list):
        print(f"  {person['name']} {person['surname']} — {person.get('job', '?')} → {tags}")

    # 4. Build answer (only transport-tagged)
    answer = build_answer(candidates, tags_list)
    print(f"\nFinal answer ({len(answer)} people with 'transport' tag):")
    for a in answer:
        print(f"  {a['name']} {a['surname']} — tags: {a['tags']}")

    # 5. Submit
    result = send_report("people", answer)
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
