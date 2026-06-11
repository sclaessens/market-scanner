from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

FORBIDDEN_TERMS = {
    "allocation",
    "rank",
    "ranking",
    "score",
    "scoring",
    "tradeable",
    "tradeability",
    "urgency",
    "conviction",
    "hidden_filter",
    "decision_bypass",
    "final_action",
    "buy",
    "sell",
    "hold",
    "trim",
}


@dataclass(frozen=True)
class PrefillAudit:
    run_timestamp: str
    provider_source_label: str
    requested_ticker_count: int
    matched_ticker_count: int
    missing_ticker_count: int
    written_row_count: int
    stale_row_count: int
    invalid_row_count: int
    partial_row_count: int
    duplicate_detection_result: str
    artifact_write_path: str
    validation_status: str
    failure_reason: str
    refresh_mode: str
    source_artifact_target: str
    dry_run: bool
    credential_safe_status: str = "NO_CREDENTIALS_USED"

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_timestamp": self.run_timestamp,
            "provider_source_label": self.provider_source_label,
            "requested_ticker_count": self.requested_ticker_count,
            "matched_ticker_count": self.matched_ticker_count,
            "missing_ticker_count": self.missing_ticker_count,
            "written_row_count": self.written_row_count,
            "stale_row_count": self.stale_row_count,
            "invalid_row_count": self.invalid_row_count,
            "partial_row_count": self.partial_row_count,
            "duplicate_detection_result": self.duplicate_detection_result,
            "artifact_write_path": self.artifact_write_path,
            "validation_status": self.validation_status,
            "failure_reason": self.failure_reason,
            "refresh_mode": self.refresh_mode,
            "source_artifact_target": self.source_artifact_target,
            "dry_run": self.dry_run,
            "credential_safe_status": self.credential_safe_status,
        }


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalize_ticker(value: Any) -> str:
    return clean_text(value).upper()


def parse_iso_date(value: Any, field_name: str) -> date:
    text = clean_text(value)
    if len(text) != 10:
        raise ValueError(f"{field_name} must be an ISO date in YYYY-MM-DD form: {text}")
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO date in YYYY-MM-DD form: {text}") from exc


def read_provider_export(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"provider export input file not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("provider export JSON must contain a list of row objects")
        return pd.DataFrame(rows, dtype=str).fillna("")
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"provider export input file is empty: {path}") from exc


def require_columns(df: pd.DataFrame, required_columns: list[str], label: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def reject_blank_tickers(df: pd.DataFrame, label: str) -> pd.DataFrame:
    normalized = df.copy()
    normalized["ticker"] = normalized["ticker"].map(normalize_ticker)
    blank_mask = normalized["ticker"] == ""
    if blank_mask.any():
        rows = df.loc[blank_mask, ["ticker"]].to_dict(orient="records")
        raise ValueError(f"{label} contains blank ticker values: {rows}")
    return normalized


def reject_duplicate_identity(df: pd.DataFrame, identity_columns: list[str], label: str) -> None:
    duplicate_mask = df.duplicated(subset=identity_columns, keep=False)
    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, identity_columns].to_dict(orient="records")
        raise ValueError(f"{label} contains duplicate row identity values: {duplicates}")


def validate_no_forbidden_columns(columns: list[str], label: str) -> None:
    lowered = [column.lower() for column in columns]
    forbidden = sorted(
        column
        for column in lowered
        if any(term in column for term in FORBIDDEN_TERMS)
    )
    if forbidden:
        raise ValueError(f"{label} contains forbidden semantic columns: {forbidden}")


def ensure_output_path(path: Path, allow_overwrite: bool) -> None:
    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"output artifact already exists; pass --allow-overwrite to replace it: {path}")


def require_governed_output_path(path: Path, expected_parts: tuple[str, ...]) -> None:
    parts = path.parts
    expected = list(expected_parts)
    if len(parts) < len(expected) or list(parts[-len(expected) :]) != expected:
        expected_path = "/".join(expected)
        raise ValueError(f"output path must target governed source artifact: {expected_path}")


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            df.to_csv(handle, index=False)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def print_audit(audit: PrefillAudit) -> None:
    print(json.dumps(audit.as_dict(), sort_keys=True, separators=(",", ":")))


def common_parser(description: str, default_output: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", required=True, type=Path, help="Local provider export or operator-provided source file.")
    parser.add_argument("--output", default=default_output, type=Path, help="Governed source artifact target.")
    parser.add_argument("--source-label", default="operator_provided", help="Credential-safe provider or source label.")
    parser.add_argument("--as-of-date", default=date.today().isoformat(), help="Validation date in YYYY-MM-DD form.")
    parser.add_argument("--write", action="store_true", help="Write the governed source artifact after validation.")
    parser.add_argument("--allow-overwrite", action="store_true", help="Allow replacing an existing governed source artifact.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print audit output without writing.")
    return parser
