"""S01E02 — Find Him task.

Uses a Function Calling agent to autonomously:
1. Retrieve the suspect list (from S01E01 criteria)
2. Retrieve nuclear power-plant locations
3. Check each suspect's observed GPS coordinates
4. Find who was closest to a power plant
5. Retrieve that person's access level
6. Submit the report

Usage:
    python -m tasks.s01e02.solution
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ai_devs import get_api_key, get_hub_data, post_request, send_report, LLMService, parse_csv
from src.ai_devs.config import HUB_VERIFY_URL, HUB_API_URL
from src.ai_devs.agent import Tool, run_agent
from src.ai_devs.geo import find_nearest_facility, POLISH_CITY_COORDS
from tasks.s01e01.solution import filter_people, tag_jobs


# ── Tool callbacks ──────────────────────────────────────────────────
# All API keys are handled INSIDE callbacks — the LLM agent never sees them.


def _get_suspects() -> list[dict]:
    """Get the S01E01 suspect pool: males from Grudziądz, aged 20-40, with transport-related jobs."""
    resp = get_hub_data("people.csv")
    all_people = parse_csv(resp.text)
    candidates = filter_people(all_people)

    service = LLMService(provider="gateway")
    tags_list = tag_jobs(candidates, service)

    # Keep only transport-tagged (the S01E01 answer pool)
    suspects = []
    for person, tags in zip(candidates, tags_list):
        if "transport" in tags:
            birth_year = int(person["birthDate"].split("-")[0])
            suspects.append({
                "name": person["name"].strip(),
                "surname": person["surname"].strip(),
                "birthYear": birth_year,
            })
    return suspects


def _get_power_plants() -> dict:
    """Fetch nuclear power-plant locations JSON from Hub, enriched with coordinates."""
    resp = get_hub_data("findhim_locations.json")
    data = resp.json()
    plants = data.get("power_plants", data)
    for city, info in plants.items():
        coords = POLISH_CITY_COORDS.get(city)
        if coords:
            info["latitude"] = coords[0]
            info["longitude"] = coords[1]
        else:
            info["latitude"] = None
            info["longitude"] = None
            print(f"WARNING: No coordinates for city '{city}' — add it to POLISH_CITY_COORDS")
    return data


def _get_person_locations(name: str, surname: str) -> dict:
    """Query Hub API for recent GPS sightings of a person."""
    payload = {"apikey": get_api_key(), "name": name, "surname": surname}
    return post_request(f"{HUB_API_URL}/location", payload)


def _get_access_level(name: str, surname: str, birthYear: int) -> dict:
    """Query Hub API for a person's system access level."""
    payload = {
        "apikey": get_api_key(),
        "name": name,
        "surname": surname,
        "birthYear": int(birthYear),
    }
    return post_request(f"{HUB_API_URL}/accesslevel", payload)


def _find_nearest_plant(name: str, surname: str) -> dict:
    """Find the nearest active power plant to any of a person's observed locations."""
    payload = {"apikey": get_api_key(), "name": name, "surname": surname}
    locations = post_request(f"{HUB_API_URL}/location", payload)
    if not locations:
        return {"error": f"No locations found for {name} {surname}"}

    plants_data = _get_power_plants()
    plants = plants_data.get("power_plants", plants_data)

    result = find_nearest_facility(
        person_coords=[(loc["latitude"], loc["longitude"]) for loc in locations],
        facilities=plants,
        active_key="is_active",
    )
    if not result:
        return {"error": "No active plants with coordinates found"}

    plant_info = plants[result["facility_id"]]
    return {**result, "city": result["facility_id"], "plant_code": plant_info["code"]}


def _submit_report(name: str, surname: str, accessLevel: int, powerPlant: str) -> dict:
    """Submit the final findhim answer to the Hub."""
    answer = {
        "name": name,
        "surname": surname,
        "accessLevel": int(accessLevel),
        "powerPlant": powerPlant,
    }
    payload = {
        "apikey": get_api_key(),
        "task": "findhim",
        "answer": answer,
    }
    return post_request(HUB_VERIFY_URL, payload, raise_on_error=False)


# ── Tool definitions (schemas + callbacks) ──────────────────────────

TOOLS = [
    Tool(
        name="get_suspects",
        description=(
            "Get the list of suspects from the people database. "
            "Returns name, surname, and birthYear for each suspect. "
            "These are males born in Grudziądz, aged 20-40."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        callback=_get_suspects,
    ),
    Tool(
        name="get_power_plants",
        description=(
            "Get the list of nuclear power plants with their city names, codes, "
            "power output, active status, and geographic coordinates "
            "(latitude, longitude). Use these coordinates for distance calculations."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        callback=_get_power_plants,
    ),
    Tool(
        name="get_person_locations",
        description=(
            "Get recent GPS coordinates where a specific person was observed. "
            "Returns a list of latitude/longitude pairs."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's first name"},
                "surname": {"type": "string", "description": "Person's surname"},
            },
            "required": ["name", "surname"],
        },
        callback=_get_person_locations,
    ),
    Tool(
        name="find_nearest_plant",
        description=(
            "Find the nearest active nuclear power plant to a person's observed locations. "
            "Internally fetches the person's GPS sightings and all plant coordinates, "
            "then returns the single closest match with city, plant code, coordinates, "
            "and distance in km. Call this once per suspect to efficiently compare all "
            "their locations against all plants."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's first name"},
                "surname": {"type": "string", "description": "Person's surname"},
            },
            "required": ["name", "surname"],
        },
        callback=_find_nearest_plant,
    ),
    Tool(
        name="get_access_level",
        description=(
            "Get the system access level for a person. "
            "Requires name, surname, and birthYear (integer, e.g. 1987)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's first name"},
                "surname": {"type": "string", "description": "Person's surname"},
                "birthYear": {
                    "type": "integer",
                    "description": "Year of birth as integer, e.g. 1987",
                },
            },
            "required": ["name", "surname", "birthYear"],
        },
        callback=_get_access_level,
    ),
    Tool(
        name="submit_report",
        description=(
            "Submit the final report identifying the suspect found near a power plant. "
            "The response may contain a success flag or an error with a hint about "
            "what is wrong (e.g. wrong name, wrong plant code). If the submission "
            "fails, READ the error message carefully — it tells you what to fix. "
            "Then go back, re-examine your data, and try again with corrected values. "
            "Do NOT retry with the same arguments."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Suspect's first name"},
                "surname": {"type": "string", "description": "Suspect's surname"},
                "accessLevel": {
                    "type": "integer",
                    "description": "The suspect's access level (integer)",
                },
                "powerPlant": {
                    "type": "string",
                    "description": "Power plant code, e.g. PWR1234PL",
                },
            },
            "required": ["name", "surname", "accessLevel", "powerPlant"],
        },
        callback=_submit_report,
    ),
]


# ── System prompt ───────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an investigative agent solving a surveillance case.

A group of suspects has been identified in a previous investigation. One of \
them was recently observed near a nuclear power plant. Your mission is to \
determine which suspect it is, verify their system access level, and file \
a report with your findings.

You have access to databases of suspects and power-plant locations, APIs to \
look up where people were recently seen, a tool to find the nearest plant to \
a person across all their sightings, and a reporting endpoint.

Workflow:
1. Get the suspect list.
2. For each suspect, find the nearest power plant to their observed locations.
3. Identify the suspect with the smallest distance — they are the one near a plant.
4. Get that suspect's access level.
5. Submit the report with the suspect's name, access level, and plant code.

Rules:
- Always use the available tools — never guess coordinates or distances.
- Work efficiently: check all suspects before drawing conclusions.
- If a report submission fails, carefully read the error message — it contains \
hints about what went wrong. Use those hints to re-examine your data and try \
a corrected submission. Never retry with the same arguments.
"""


# ── Main ────────────────────────────────────────────────────────────

def main():
    result = run_agent(
        system_prompt=SYSTEM_PROMPT,
        user_message="Find the suspect located near a nuclear power plant and submit the report.",
        tools=TOOLS,
        model="gpt-4.1-mini",
        max_iterations=20,
        max_tokens=2048,
    )
    print(f"\n{'='*60}")
    print(f"Agent finished. Final output:\n{result}")


if __name__ == "__main__":
    main()
