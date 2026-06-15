from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SEC_COMPANYFACTS_SOURCE_NAME = "sec_companyfacts"
SEC_COMPANYFACTS_SNAPSHOT_FORMAT_VERSION = "sec-companyfacts-raw-v1"
SEC_COMPANYFACTS_SNAPSHOT_ROOT = Path("data/market_engine/source_snapshots/sec_companyfacts")


class SecCompanyFactsSnapshotError(Exception):
    """Base error for controlled SEC CompanyFacts snapshot failures."""


class SecCompanyFactsSnapshotMissingError(SecCompanyFactsSnapshotError):
    """Raised when a requested SEC CompanyFacts snapshot cannot be found."""


class SecCompanyFactsSnapshotJsonError(SecCompanyFactsSnapshotError):
    """Raised when a SEC CompanyFacts snapshot is not valid JSON."""


class SecCompanyFactsSnapshotMetadataError(SecCompanyFactsSnapshotError):
    """Raised when required SEC CompanyFacts snapshot metadata is missing."""


class SecCompanyFactsSnapshotMismatchError(SecCompanyFactsSnapshotError):
    """Raised when cached SEC CompanyFacts metadata does not match the request."""


class SecCompanyFactsSnapshotUnsupportedFormatError(SecCompanyFactsSnapshotError):
    """Raised when a SEC CompanyFacts snapshot format is unsupported."""


@dataclass(frozen=True)
class SecCompanyFactsRawSnapshot:
    ticker: str
    cik: str
    source_name: str
    fetched_at: str
    snapshot_id: str
    payload_format_version: str
    raw_payload: dict[str, Any]
    path: Path | None = None


def default_sec_companyfacts_source_snapshot_root() -> Path:
    return SEC_COMPANYFACTS_SNAPSHOT_ROOT


def persist_sec_companyfacts_raw_snapshot(
    *,
    raw_payload: dict[str, Any],
    ticker: str,
    cik: str,
    run_id: str,
    snapshot_id: str | None = None,
    fetched_at: datetime | str | None = None,
    root_dir: Path | None = None,
) -> Path:
    root = root_dir or default_sec_companyfacts_source_snapshot_root()
    normalized_ticker = _normalize_ticker(ticker)
    normalized_cik = _normalize_cik(cik)
    resolved_snapshot_id = snapshot_id or f"{normalized_ticker}_companyfacts"
    raw_dir = root / run_id / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = raw_dir / f"{resolved_snapshot_id}.json"
    if snapshot_path.exists():
        raise FileExistsError(f"refusing to overwrite existing SEC CompanyFacts snapshot: {snapshot_path}")

    envelope = {
        "metadata": {
            "ticker": normalized_ticker,
            "cik": normalized_cik,
            "source_name": SEC_COMPANYFACTS_SOURCE_NAME,
            "fetched_at": _timestamp_text(fetched_at),
            "snapshot_id": resolved_snapshot_id,
            "payload_format_version": SEC_COMPANYFACTS_SNAPSHOT_FORMAT_VERSION,
        },
        "raw_payload": raw_payload,
    }
    snapshot_path.write_text(json.dumps(envelope, indent=2, sort_keys=True), encoding="utf-8")
    _append_ticker_manifest(
        root=root,
        run_id=run_id,
        ticker=normalized_ticker,
        cik=normalized_cik,
        snapshot_id=resolved_snapshot_id,
        snapshot_path=snapshot_path,
    )
    _write_snapshot_metadata(root=root, run_id=run_id, source_name=SEC_COMPANYFACTS_SOURCE_NAME)
    return snapshot_path


def persist_sec_companyfacts_provider_error(
    *,
    ticker: str,
    cik: str | None,
    run_id: str,
    error_type: str,
    error_message: str,
    root_dir: Path | None = None,
) -> Path:
    root = root_dir or default_sec_companyfacts_source_snapshot_root()
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    error_path = run_dir / "provider_errors.csv"
    write_header = not error_path.exists()
    with error_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(("ticker", "cik", "error_type", "error_message", "source_name"))
        writer.writerow((
            _normalize_ticker(ticker),
            _normalize_cik(cik) if cik else "",
            error_type,
            error_message,
            SEC_COMPANYFACTS_SOURCE_NAME,
        ))
    _write_snapshot_metadata(root=root, run_id=run_id, source_name=SEC_COMPANYFACTS_SOURCE_NAME)
    return error_path


def load_sec_companyfacts_raw_snapshot(
    snapshot_path: Path,
    *,
    expected_ticker: str | None = None,
    expected_cik: str | None = None,
) -> SecCompanyFactsRawSnapshot:
    if not snapshot_path.exists():
        raise SecCompanyFactsSnapshotMissingError(f"SEC CompanyFacts snapshot not found: {snapshot_path}")
    try:
        envelope = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SecCompanyFactsSnapshotJsonError(
            f"SEC CompanyFacts snapshot contains invalid JSON: {snapshot_path}"
        ) from error
    if not isinstance(envelope, dict):
        raise SecCompanyFactsSnapshotMetadataError("SEC CompanyFacts snapshot envelope must be an object")

    metadata = envelope.get("metadata")
    raw_payload = envelope.get("raw_payload")
    if not isinstance(metadata, dict):
        raise SecCompanyFactsSnapshotMetadataError("SEC CompanyFacts snapshot metadata is missing")
    if not isinstance(raw_payload, dict):
        raise SecCompanyFactsSnapshotMetadataError("SEC CompanyFacts raw payload is missing")

    snapshot = _snapshot_from_parts(metadata=metadata, raw_payload=raw_payload, path=snapshot_path)
    _validate_expected_entity(snapshot, expected_ticker=expected_ticker, expected_cik=expected_cik)
    return snapshot


def load_latest_sec_companyfacts_raw_snapshot(
    *,
    root_dir: Path | None = None,
    ticker: str | None = None,
    cik: str | None = None,
) -> SecCompanyFactsRawSnapshot:
    root = root_dir or default_sec_companyfacts_source_snapshot_root()
    candidates = sorted((root.glob("*/raw/*.json")), key=lambda path: str(path), reverse=True)
    if not candidates:
        raise SecCompanyFactsSnapshotMissingError(f"no SEC CompanyFacts snapshots found under {root}")

    expected_ticker = _normalize_ticker(ticker) if ticker else None
    expected_cik = _normalize_cik(cik) if cik else None
    mismatches: list[SecCompanyFactsSnapshotMismatchError] = []
    for candidate in candidates:
        try:
            return load_sec_companyfacts_raw_snapshot(
                candidate,
                expected_ticker=expected_ticker,
                expected_cik=expected_cik,
            )
        except SecCompanyFactsSnapshotMismatchError as error:
            mismatches.append(error)
            continue
    if mismatches:
        raise SecCompanyFactsSnapshotMissingError(
            f"no matching SEC CompanyFacts snapshot found under {root}"
        ) from mismatches[-1]
    raise SecCompanyFactsSnapshotMissingError(f"no SEC CompanyFacts snapshots found under {root}")


def _snapshot_from_parts(
    *,
    metadata: dict[str, Any],
    raw_payload: dict[str, Any],
    path: Path,
) -> SecCompanyFactsRawSnapshot:
    required_metadata = (
        "ticker",
        "cik",
        "source_name",
        "fetched_at",
        "snapshot_id",
        "payload_format_version",
    )
    missing = [field_name for field_name in required_metadata if not metadata.get(field_name)]
    if missing:
        raise SecCompanyFactsSnapshotMetadataError(
            f"SEC CompanyFacts snapshot metadata missing required fields: {', '.join(missing)}"
        )
    if metadata["source_name"] != SEC_COMPANYFACTS_SOURCE_NAME:
        raise SecCompanyFactsSnapshotMetadataError(
            f"unexpected SEC CompanyFacts source name: {metadata['source_name']}"
        )
    if metadata["payload_format_version"] != SEC_COMPANYFACTS_SNAPSHOT_FORMAT_VERSION:
        raise SecCompanyFactsSnapshotUnsupportedFormatError(
            f"unsupported SEC CompanyFacts snapshot format: {metadata['payload_format_version']}"
        )
    return SecCompanyFactsRawSnapshot(
        ticker=_normalize_ticker(str(metadata["ticker"])),
        cik=_normalize_cik(str(metadata["cik"])),
        source_name=str(metadata["source_name"]),
        fetched_at=str(metadata["fetched_at"]),
        snapshot_id=str(metadata["snapshot_id"]),
        payload_format_version=str(metadata["payload_format_version"]),
        raw_payload=raw_payload,
        path=path,
    )


def _validate_expected_entity(
    snapshot: SecCompanyFactsRawSnapshot,
    *,
    expected_ticker: str | None,
    expected_cik: str | None,
) -> None:
    if expected_ticker is not None and snapshot.ticker != _normalize_ticker(expected_ticker):
        raise SecCompanyFactsSnapshotMismatchError(
            f"snapshot ticker {snapshot.ticker} does not match requested ticker {_normalize_ticker(expected_ticker)}"
        )
    if expected_cik is not None and snapshot.cik != _normalize_cik(expected_cik):
        raise SecCompanyFactsSnapshotMismatchError(
            f"snapshot CIK {snapshot.cik} does not match requested CIK {_normalize_cik(expected_cik)}"
        )


def _append_ticker_manifest(
    *,
    root: Path,
    run_id: str,
    ticker: str,
    cik: str,
    snapshot_id: str,
    snapshot_path: Path,
) -> None:
    manifest_path = root / run_id / "ticker_manifest.csv"
    write_header = not manifest_path.exists()
    with manifest_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(("ticker", "cik", "snapshot_id", "snapshot_path", "source_name"))
        writer.writerow((ticker, cik, snapshot_id, snapshot_path.as_posix(), SEC_COMPANYFACTS_SOURCE_NAME))


def _write_snapshot_metadata(
    *,
    root: Path,
    run_id: str,
    source_name: str,
) -> None:
    metadata_path = root / run_id / "snapshot_metadata.json"
    if metadata_path.exists():
        return
    metadata = {
        "run_id": run_id,
        "source_name": source_name,
        "payload_format_version": SEC_COMPANYFACTS_SNAPSHOT_FORMAT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "raw source snapshot evidence for Market Engine Source Refresh",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _timestamp_text(value: datetime | str | None) -> str:
    if value is None:
        return datetime.now(UTC).isoformat()
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return value


def _normalize_ticker(ticker: str) -> str:
    normalized = str(ticker).strip().upper()
    if not normalized or any(character.isspace() for character in normalized):
        raise ValueError(f"{ticker!r} is not a valid SEC CompanyFacts snapshot ticker")
    return normalized


def _normalize_cik(cik: str | int) -> str:
    digits = "".join(character for character in str(cik) if character.isdigit())
    if not digits:
        raise ValueError("CIK must contain digits")
    return digits.zfill(10)
