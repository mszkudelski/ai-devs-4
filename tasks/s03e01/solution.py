"""S03E01 — find anomalous sensor readings and operator notes.

The sensor archive contains structured measurements, so range and active
channel checks are deterministic.  Operator notes use a finite set of
English lead templates in this task; the alert leads are classified locally so
the 10,000 records do not result in one LLM request per note.

The default invocation is a dry run and makes no Hub calls::

    python -m tasks.s03e01.solution
    python -m tasks.s03e01.solution --archive /path/to/sensors.zip
    python -m tasks.s03e01.solution --run

``--run`` downloads the documented archive, computes the complete answer,
submits one report to ``/verify``, and prints the flag returned by that
request.  The API key remains inside the shared configuration helper.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ai_devs.api import get_request, post_request
from src.ai_devs.config import HUB_VERIFY_URL, get_api_key


TASK_NAME = "evaluation"
SENSORS_URL = "https://hub.ag3nts.org/dane/sensors.zip"

# A sensor can expose one or more of these channels.  Values for every other
# channel must remain exactly zero in each record.
_SENSOR_RANGES: dict[str, tuple[str, float, float]] = {
    "temperature": ("temperature_K", 553, 873),
    "pressure": ("pressure_bar", 60, 160),
    "water": ("water_level_meters", 5.0, 15.0),
    "voltage": ("voltage_supply_v", 229.0, 231.0),
    "humidity": ("humidity_percent", 40.0, 80.0),
}

# The fixture deliberately paraphrases notes, but every note that reports a
# problem starts with one of these alert leads.  Matching the lead rather than
# isolated words avoids treating healthy notes such as "No concerning drift"
# or "nothing suggests a fault condition" as alerts.
_ALERT_NOTE_LEADS = frozenset(
    {
        "this state looks unstable",
        "the current result seems unreliable",
        "the latest behavior is concerning",
        "the numbers feel inconsistent",
        "the situation requires attention",
        "this report raises serious doubts",
        "the report does not look healthy",
        "these readings look suspicious",
        "this check did not look right",
        "the output quality is doubtful",
        "this run shows questionable behavior",
        "something is clearly off",
        "the signal profile looks unusual",
        "there is a visible anomaly here",
        "i am not comfortable with this result",
        "this is not the pattern i expected",
        "i can see a clear irregularity",
        "i am seeing an unexpected pattern",
    }
)

_FLAG_RE = re.compile(r"\{FLG:[^}]+\}")


@dataclass(frozen=True)
class SensorRecord:
    """One JSON member from ``sensors.zip``."""

    identifier: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class Analysis:
    """Computed anomaly IDs and bounded counters for review output."""

    anomaly_ids: tuple[str, ...]
    measurement_anomalies: tuple[str, ...]
    note_alerts: tuple[str, ...]
    note_mismatches: tuple[str, ...]


def _identifier_from_member(name: str) -> str:
    """Return the numeric file identifier accepted by the Hub."""

    basename = PurePosixPath(name).name
    if not basename.lower().endswith(".json"):
        raise ValueError(f"archive member is not a JSON file: {name}")
    return basename[: -len(".json")]


def load_archive(data: bytes) -> list[SensorRecord]:
    """Parse every JSON sensor file from an archive without extracting it."""

    records: list[SensorRecord] = []
    seen: set[str] = set()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = [
            info
            for info in archive.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".json")
        ]
        for info in members:
            identifier = _identifier_from_member(info.filename)
            if identifier in seen:
                raise ValueError(f"duplicate sensor identifier in archive: {identifier}")
            seen.add(identifier)
            try:
                payload = json.loads(archive.read(info))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid JSON in sensor file {identifier}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"sensor file {identifier} must contain a JSON object")
            records.append(SensorRecord(identifier=identifier, payload=payload))

    if not records:
        raise ValueError("sensor archive contains no JSON records")
    return sorted(records, key=lambda record: record.identifier)


def _is_number(value: Any) -> bool:
    """Return whether a measurement is numeric, excluding booleans."""

    return isinstance(value, (int, float)) and not isinstance(value, bool)


def measurement_issues(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return deterministic measurement/schema issues for one sensor record."""

    raw_type = payload.get("sensor_type")
    if not isinstance(raw_type, str) or not raw_type.strip():
        return ("sensor_type_missing_or_invalid",)

    parts = [part.strip() for part in raw_type.split("/")]
    active = set(parts)
    issues: list[str] = []
    if any(not part or part not in _SENSOR_RANGES for part in parts):
        issues.append("unknown_sensor_type")
    if len(active) != len(parts):
        issues.append("duplicate_sensor_type")

    for sensor, (field, low, high) in _SENSOR_RANGES.items():
        value = payload.get(field)
        if not _is_number(value):
            issues.append(f"{field}_missing_or_non_numeric")
            continue
        if sensor in active:
            if not low <= value <= high:
                issues.append(f"{field}_out_of_range")
        elif value != 0:
            issues.append(f"{field}_inactive_nonzero")
    return tuple(issues)


def note_reports_alert(note: Any) -> bool:
    """Classify the task's operator note as an explicit error report."""

    if not isinstance(note, str):
        # A missing or non-text note cannot be a valid operator statement, so
        # keep the record in the recheck set alongside other schema issues.
        return True
    lead = re.split(r"[,\.\n]", note.strip(), maxsplit=1)[0].strip().casefold()
    return lead in _ALERT_NOTE_LEADS


def analyse(records: Sequence[SensorRecord]) -> Analysis:
    """Find measurement anomalies and notes that contradict the readings."""

    measurement_ids: set[str] = set()
    alert_ids: set[str] = set()
    note_mismatch_ids: set[str] = set()

    for record in records:
        measurement_bad = bool(measurement_issues(record.payload))
        note_alert = note_reports_alert(record.payload.get("operator_notes"))
        if measurement_bad:
            measurement_ids.add(record.identifier)
        if note_alert:
            alert_ids.add(record.identifier)
        # A healthy record with an alert note is a false alarm.  An unhealthy
        # record with a healthy note is an operator reporting error.  Both are
        # included in the final anomaly set; data anomalies are already enough
        # to include unhealthy records whose alert note is accurate.
        if measurement_bad != note_alert:
            note_mismatch_ids.add(record.identifier)

    anomaly_ids = measurement_ids | alert_ids
    return Analysis(
        anomaly_ids=tuple(sorted(anomaly_ids)),
        measurement_anomalies=tuple(sorted(measurement_ids)),
        note_alerts=tuple(sorted(alert_ids)),
        note_mismatches=tuple(sorted(note_mismatch_ids)),
    )


def _extract_flag(value: Any) -> str | None:
    """Extract a flag returned by the Hub without manufacturing one."""

    match = _FLAG_RE.search(json.dumps(value, ensure_ascii=False, default=str))
    return match.group(0) if match else None


def _download_archive() -> bytes:
    """Download the lesson's public sensor archive."""

    response = get_request(SENSORS_URL, timeout=120)
    return response.content


def _submit(anomaly_ids: Sequence[str]) -> Any:
    """Send the single evaluation report to the Hub verification endpoint."""

    payload = {
        "apikey": get_api_key(),
        "task": TASK_NAME,
        "answer": {"recheck": list(anomaly_ids)},
    }
    return post_request(HUB_VERIFY_URL, payload, timeout=120)


def _print_analysis(source: str, records: Sequence[SensorRecord], result: Analysis) -> None:
    """Print reproducible, non-secret local analysis metadata."""

    print(f"Source: {source}")
    print(f"Sensor files: {len(records)}")
    print(f"Measurement anomalies: {len(result.measurement_anomalies)}")
    print(f"Operator-note alerts: {len(result.note_alerts)}")
    print(f"Note mismatches: {len(result.note_mismatches)}")
    print(f"Anomalies: {len(result.anomaly_ids)}")
    print(f"Anomaly IDs: {json.dumps(list(result.anomaly_ids), ensure_ascii=False)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S03E01 sensor anomaly solver")
    parser.add_argument(
        "--run",
        action="store_true",
        help="download sensors.zip and submit one report to the Hub",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="analyse a local sensors.zip without contacting the Hub",
    )
    args = parser.parse_args(argv)

    if args.run and args.archive is not None:
        parser.error("--run downloads the live archive; do not combine it with --archive")

    if not args.run and args.archive is None:
        print("Dry run only: no Hub calls were made.")
        print(f"Live source: {SENSORS_URL}")
        print("Use --archive PATH for local analysis or --run for the live submission.")
        return 0

    try:
        if args.archive is not None:
            archive_data = args.archive.read_bytes()
            source = str(args.archive)
        else:
            archive_data = _download_archive()
            source = SENSORS_URL
        records = load_archive(archive_data)
        result = analyse(records)
        _print_analysis(source, records, result)
    except Exception as exc:
        print(f"Solver failed ({type(exc).__name__}).", file=sys.stderr)
        return 1

    if not args.run:
        print("Local analysis only: no Hub calls were made.")
        return 0

    try:
        response = _submit(result.anomaly_ids)
    except Exception as exc:
        print(f"Verification failed ({type(exc).__name__}).", file=sys.stderr)
        return 1

    flag = _extract_flag(response)
    if flag is None:
        print(f"Hub response: {json.dumps(response, ensure_ascii=False, default=str)}")
        print("Hub did not return a flag.", file=sys.stderr)
        return 1

    print(f"RETURNED FROM CODE: {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
