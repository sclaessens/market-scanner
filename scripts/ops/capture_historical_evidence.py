from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORY_DIR = PROJECT_ROOT / "data" / "history"

PIPELINE_RUNS_FILE = HISTORY_DIR / "pipeline_runs.csv"
PIPELINE_ARTIFACTS_FILE = HISTORY_DIR / "pipeline_artifacts.csv"
DECISION_REPORTING_OBSERVATIONS_FILE = HISTORY_DIR / "decision_reporting_observations.csv"

DECISION_ARTIFACT_PATH = "data/processed/final_decisions.csv"
REPORTING_ARTIFACT_PATH = "data/processed/reporting_dashboard_data.csv"

OBSERVED_ARTIFACTS = [
    ("scanner", "processed_output", "data/processed/scanner_ranked.csv"),
    ("validation", "processed_output", "data/processed/validation_layer.csv"),
    ("context", "processed_output", "data/processed/context_strength.csv"),
    ("fundamental", "processed_output", "data/processed/fundamental_quality.csv"),
    ("timing_state", "processed_output", "data/processed/timing_state_layer.csv"),
    ("portfolio_intelligence", "processed_output", "data/processed/portfolio_intelligence.csv"),
    ("decision_engine", "processed_output", DECISION_ARTIFACT_PATH),
    ("stability", "processed_output", "data/processed/stability_state.csv"),
    ("reporting", "processed_output", REPORTING_ARTIFACT_PATH),
    ("validation", "log", "data/logs/validation_layer_log.csv"),
    ("context", "log", "data/logs/context_layer_log.csv"),
    ("fundamental", "log", "data/logs/fundamental_layer_log.csv"),
    ("timing_state", "log", "data/logs/timing_state_layer_log.csv"),
    ("portfolio_intelligence", "log", "data/logs/portfolio_intelligence_log.csv"),
    ("decision_engine", "log", "data/logs/decision_engine_log.csv"),
    ("stability", "log", "data/logs/stability_layer_log.csv"),
    ("reporting", "log", "data/logs/reporting_layer_log.csv"),
    ("scanner", "log", "data/logs/scans_log.csv"),
    ("scanner", "diagnostic_log", "data/logs/failed_tickers.csv"),
]

PIPELINE_RUN_COLUMNS = [
    "run_id",
    "captured_at",
    "capture_status",
    "artifact_count",
    "missing_artifact_count",
    "decision_row_count",
    "reporting_row_count",
    "decision_reporting_linkage_status",
    "decision_reporting_observation_count",
    "diagnostic_notes",
]

PIPELINE_ARTIFACT_COLUMNS = [
    "run_id",
    "captured_at",
    "pipeline_step",
    "artifact_role",
    "artifact_path",
    "artifact_exists",
    "row_count",
    "file_size_bytes",
    "modified_at",
    "content_hash",
    "diagnostic_notes",
]

DECISION_REPORTING_OBSERVATION_COLUMNS = [
    "run_id",
    "captured_at",
    "ticker",
    "date",
    "decision_row_index",
    "reporting_row_index",
    "decision_artifact_path",
    "reporting_artifact_path",
    "decision_input_row_hash",
    "decision_row_hash",
    "reporting_source_row_identity",
    "reporting_source_row_index",
    "reporting_source_input_row_hash",
    "reporting_represented_flag",
    "source_final_action",
    "source_allocation_decision",
    "source_execution_decision",
    "diagnostic_note",
]


@dataclass(frozen=True)
class HistoricalCaptureResult:
    run_id: str
    captured_at: str
    pipeline_runs_file: Path
    pipeline_artifacts_file: Path
    decision_reporting_observations_file: Path
    artifact_rows: int
    observation_rows: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.upper() in {"", "NAN", "NONE", "NULL"}:
        return ""
    return text


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return [], []
        rows = [{key: normalize_cell(value) for key, value in row.items()} for row in reader]
    return rows, list(reader.fieldnames)


def safe_row_count(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            return str(sum(1 for _ in reader))
    except (csv.Error, UnicodeDecodeError, OSError):
        return ""


def short_file_hash(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def row_hash(row: dict[str, str]) -> str:
    encoded = json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def modified_at(path: Path) -> str:
    if not path.exists():
        return ""
    return (
        datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def artifact_metadata(project_root: Path, captured_at: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for pipeline_step, artifact_role, artifact_path in OBSERVED_ARTIFACTS:
        path = project_root / artifact_path
        exists = path.exists()
        row = {
            "run_id": "",
            "captured_at": captured_at,
            "pipeline_step": pipeline_step,
            "artifact_role": artifact_role,
            "artifact_path": artifact_path,
            "artifact_exists": str(exists),
            "row_count": safe_row_count(path),
            "file_size_bytes": str(path.stat().st_size) if exists else "",
            "modified_at": modified_at(path),
            "content_hash": short_file_hash(path),
            "diagnostic_notes": "ARTIFACT_OBSERVED" if exists else "ARTIFACT_MISSING",
        }
        rows.append(row)
    return rows


def generate_run_id(captured_at: str, artifact_rows: Iterable[dict[str, str]]) -> str:
    metadata = [
        {
            "artifact_path": row["artifact_path"],
            "artifact_exists": row["artifact_exists"],
            "row_count": row["row_count"],
            "file_size_bytes": row["file_size_bytes"],
            "modified_at": row["modified_at"],
            "content_hash": row["content_hash"],
        }
        for row in artifact_rows
    ]
    digest_source = {
        "captured_at": captured_at,
        "artifacts": metadata,
    }
    digest = hashlib.sha256(
        json.dumps(digest_source, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    timestamp = captured_at.replace("-", "").replace(":", "").replace("+00:00", "Z")
    timestamp = timestamp.replace("Z", "").replace("T", "T")
    return f"{timestamp}Z-{digest}"


def identity_key(row: dict[str, str], hash_column: str) -> str:
    row_hash_value = normalize_cell(row.get(hash_column))
    if row_hash_value:
        return f"hash:{row_hash_value}"
    ticker = normalize_cell(row.get("ticker"))
    date = normalize_cell(row.get("date"))
    if ticker and date:
        return f"ticker_date:{ticker}:{date}"
    return ""


def reporting_identity_key(row: dict[str, str]) -> str:
    return identity_key(row, "source_input_row_hash")


def build_reporting_index(
    reporting_rows: list[dict[str, str]]
) -> tuple[dict[str, list[int]], set[str]]:
    index: dict[str, list[int]] = {}
    duplicate_keys: set[str] = set()
    for row_index, row in enumerate(reporting_rows):
        key = reporting_identity_key(row)
        if not key:
            continue
        index.setdefault(key, []).append(row_index)
    for key, rows in index.items():
        if len(rows) > 1:
            duplicate_keys.add(key)
    return index, duplicate_keys


def duplicate_decision_keys(decision_rows: list[dict[str, str]]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in decision_rows:
        key = identity_key(row, "input_row_hash")
        if not key:
            continue
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return duplicates


def observation_from_decision(
    *,
    run_id: str,
    captured_at: str,
    decision_row: dict[str, str],
    decision_row_index: int,
    reporting_row: dict[str, str] | None,
    reporting_row_index: int | None,
    diagnostic_note: str,
) -> dict[str, str]:
    return {
        "run_id": run_id,
        "captured_at": captured_at,
        "ticker": normalize_cell(decision_row.get("ticker")),
        "date": normalize_cell(decision_row.get("date")),
        "decision_row_index": str(decision_row_index),
        "reporting_row_index": "" if reporting_row_index is None else str(reporting_row_index),
        "decision_artifact_path": DECISION_ARTIFACT_PATH,
        "reporting_artifact_path": REPORTING_ARTIFACT_PATH if reporting_row is not None else "",
        "decision_input_row_hash": normalize_cell(decision_row.get("input_row_hash")),
        "decision_row_hash": row_hash(decision_row),
        "reporting_source_row_identity": normalize_cell(
            reporting_row.get("source_row_identity") if reporting_row else ""
        ),
        "reporting_source_row_index": normalize_cell(
            reporting_row.get("source_row_index") if reporting_row else ""
        ),
        "reporting_source_input_row_hash": normalize_cell(
            reporting_row.get("source_input_row_hash") if reporting_row else ""
        ),
        "reporting_represented_flag": "True" if reporting_row is not None else "False",
        "source_final_action": normalize_cell(decision_row.get("final_action")),
        "source_allocation_decision": normalize_cell(decision_row.get("allocation_decision")),
        "source_execution_decision": normalize_cell(decision_row.get("execution_decision")),
        "diagnostic_note": diagnostic_note,
    }


def unmatched_reporting_observation(
    *,
    run_id: str,
    captured_at: str,
    reporting_row: dict[str, str],
    reporting_row_index: int,
    diagnostic_note: str,
) -> dict[str, str]:
    return {
        "run_id": run_id,
        "captured_at": captured_at,
        "ticker": normalize_cell(reporting_row.get("ticker")),
        "date": normalize_cell(reporting_row.get("date")),
        "decision_row_index": "",
        "reporting_row_index": str(reporting_row_index),
        "decision_artifact_path": DECISION_ARTIFACT_PATH,
        "reporting_artifact_path": REPORTING_ARTIFACT_PATH,
        "decision_input_row_hash": "",
        "decision_row_hash": "",
        "reporting_source_row_identity": normalize_cell(reporting_row.get("source_row_identity")),
        "reporting_source_row_index": normalize_cell(reporting_row.get("source_row_index")),
        "reporting_source_input_row_hash": normalize_cell(reporting_row.get("source_input_row_hash")),
        "reporting_represented_flag": "True",
        "source_final_action": normalize_cell(reporting_row.get("source_final_action")),
        "source_allocation_decision": normalize_cell(reporting_row.get("source_allocation_decision")),
        "source_execution_decision": normalize_cell(reporting_row.get("source_execution_decision")),
        "diagnostic_note": diagnostic_note,
    }


def build_decision_reporting_observations(
    *,
    run_id: str,
    captured_at: str,
    decision_rows: list[dict[str, str]],
    reporting_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], str, str]:
    observations: list[dict[str, str]] = []
    reporting_index, duplicate_reporting_keys = build_reporting_index(reporting_rows)
    duplicate_decision_identity = duplicate_decision_keys(decision_rows)
    matched_reporting_rows: set[int] = set()
    diagnostic_notes: list[str] = []

    for decision_row_index, decision_row in enumerate(decision_rows):
        key = identity_key(decision_row, "input_row_hash")
        reporting_row = None
        reporting_row_index = None
        diagnostic_note = "LINKED"

        if not key:
            diagnostic_note = "MISSING_DECISION_IDENTITY"
        elif key in duplicate_decision_identity:
            diagnostic_note = "DUPLICATE_DECISION_IDENTITY"
        elif key in duplicate_reporting_keys:
            diagnostic_note = "DUPLICATE_REPORTING_IDENTITY"
        else:
            matches = reporting_index.get(key, [])
            if len(matches) == 1:
                reporting_row_index = matches[0]
                reporting_row = reporting_rows[reporting_row_index]
                matched_reporting_rows.add(reporting_row_index)
            else:
                diagnostic_note = "REPORTING_ROW_NOT_MATCHED"

        if diagnostic_note != "LINKED":
            diagnostic_notes.append(diagnostic_note)

        observations.append(
            observation_from_decision(
                run_id=run_id,
                captured_at=captured_at,
                decision_row=decision_row,
                decision_row_index=decision_row_index,
                reporting_row=reporting_row,
                reporting_row_index=reporting_row_index,
                diagnostic_note=diagnostic_note,
            )
        )

    for reporting_row_index, reporting_row in enumerate(reporting_rows):
        if reporting_row_index in matched_reporting_rows:
            continue
        diagnostic_note = "REPORTING_ROW_WITHOUT_DECISION_MATCH"
        key = reporting_identity_key(reporting_row)
        if not key:
            diagnostic_note = "MISSING_REPORTING_IDENTITY"
        elif key in duplicate_reporting_keys:
            diagnostic_note = "DUPLICATE_REPORTING_IDENTITY"
        diagnostic_notes.append(diagnostic_note)
        observations.append(
            unmatched_reporting_observation(
                run_id=run_id,
                captured_at=captured_at,
                reporting_row=reporting_row,
                reporting_row_index=reporting_row_index,
                diagnostic_note=diagnostic_note,
            )
        )

    if not decision_rows and not reporting_rows:
        return observations, "NO_DECISION_OR_REPORTING_ROWS", "NO_ROWS_AVAILABLE"
    if decision_rows and not reporting_rows:
        return observations, "REPORTING_UNAVAILABLE", "REPORTING_ROWS_MISSING"
    if not decision_rows and reporting_rows:
        return observations, "DECISION_UNAVAILABLE", "DECISION_ROWS_MISSING"
    if diagnostic_notes:
        return observations, "PARTIAL", ";".join(sorted(set(diagnostic_notes)))
    return observations, "LINKED", "DECISION_REPORTING_LINKED"


def append_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        if rows:
            for row in rows:
                writer.writerow(row)


def capture_historical_evidence(
    project_root: Path = PROJECT_ROOT,
    *,
    captured_at: str | None = None,
    history_dir: Path | None = None,
) -> HistoricalCaptureResult:
    project_root = Path(project_root)
    captured_at = captured_at or utc_now_iso()
    history_dir = history_dir or project_root / "data" / "history"

    artifact_rows = artifact_metadata(project_root, captured_at)
    run_id = generate_run_id(captured_at, artifact_rows)
    artifact_rows = [{**row, "run_id": run_id} for row in artifact_rows]

    decision_rows, _ = read_csv_rows(project_root / DECISION_ARTIFACT_PATH)
    reporting_rows, _ = read_csv_rows(project_root / REPORTING_ARTIFACT_PATH)
    observations, linkage_status, observation_notes = build_decision_reporting_observations(
        run_id=run_id,
        captured_at=captured_at,
        decision_rows=decision_rows,
        reporting_rows=reporting_rows,
    )

    missing_artifact_count = sum(row["artifact_exists"] == "False" for row in artifact_rows)
    existing_artifact_count = len(artifact_rows) - missing_artifact_count
    capture_status = "CAPTURED" if existing_artifact_count else "NO_ARTIFACTS_OBSERVED"
    run_row = {
        "run_id": run_id,
        "captured_at": captured_at,
        "capture_status": capture_status,
        "artifact_count": str(existing_artifact_count),
        "missing_artifact_count": str(missing_artifact_count),
        "decision_row_count": str(len(decision_rows)) if decision_rows else "",
        "reporting_row_count": str(len(reporting_rows)) if reporting_rows else "",
        "decision_reporting_linkage_status": linkage_status,
        "decision_reporting_observation_count": str(len(observations)),
        "diagnostic_notes": observation_notes,
    }

    pipeline_runs_file = history_dir / "pipeline_runs.csv"
    pipeline_artifacts_file = history_dir / "pipeline_artifacts.csv"
    observations_file = history_dir / "decision_reporting_observations.csv"

    append_csv(pipeline_runs_file, PIPELINE_RUN_COLUMNS, [run_row])
    append_csv(pipeline_artifacts_file, PIPELINE_ARTIFACT_COLUMNS, artifact_rows)
    append_csv(observations_file, DECISION_REPORTING_OBSERVATION_COLUMNS, observations)

    return HistoricalCaptureResult(
        run_id=run_id,
        captured_at=captured_at,
        pipeline_runs_file=pipeline_runs_file,
        pipeline_artifacts_file=pipeline_artifacts_file,
        decision_reporting_observations_file=observations_file,
        artifact_rows=len(artifact_rows),
        observation_rows=len(observations),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture observational historical evidence from latest pipeline artifacts."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--history-dir", type=Path, default=None)
    parser.add_argument("--captured-at", default=None)
    args = parser.parse_args()

    result = capture_historical_evidence(
        project_root=args.project_root,
        history_dir=args.history_dir,
        captured_at=args.captured_at,
    )
    print(f"historical evidence capture completed: run_id={result.run_id}")
    print(f"artifact rows written: {result.artifact_rows}")
    print(f"decision/reporting observation rows written: {result.observation_rows}")
    print(f"pipeline runs artifact: {result.pipeline_runs_file}")
    print(f"pipeline artifacts artifact: {result.pipeline_artifacts_file}")
    print(f"decision/reporting observations artifact: {result.decision_reporting_observations_file}")


if __name__ == "__main__":
    main()
