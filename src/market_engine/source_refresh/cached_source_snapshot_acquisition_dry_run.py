from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from market_engine.source_refresh.cached_source_snapshot_inventory import (
    CACHED_SOURCE_SNAPSHOT_ACQUISITION_MANIFEST_FORMAT_VERSION,
)


CACHED_SOURCE_SNAPSHOT_ACQUISITION_DRY_RUN_FORMAT_VERSION = (
    "market-engine-cached-source-snapshot-acquisition-dry-run-v1"
)

SUPPORTED_SOURCE_FAMILIES: tuple[str, ...] = ("sec_companyfacts",)
REQUIRED_ACQUISITION_MANIFEST_FIELDS: tuple[str, ...] = (
    "manifest_format_version",
    "snapshot_id",
    "batch_id",
    "created_at_utc",
    "acquired_at_utc",
    "acquisition_mode",
    "source_family",
    "source_name",
    "source_url",
    "source_license_note",
    "redistribution_allowed",
    "local_use_allowed",
    "commit_allowed",
    "source_material_type",
    "ticker",
    "entity_name",
    "entity_country",
    "entity_exchange",
    "source_entity_identifier",
    "cik",
    "requested_document_type",
    "resolved_document_type",
    "requested_period",
    "resolved_period",
    "source_publication_date",
    "source_retrieved_at_utc",
    "local_snapshot_path",
    "local_manifest_path",
    "local_payload_sha256",
    "local_payload_size_bytes",
    "payload_mime_type",
    "payload_encoding",
    "normalization_status",
    "validation_status",
    "validation_errors",
    "validation_warnings",
    "staleness_status",
    "staleness_reason",
    "usable_for_cached_source_dry_run",
    "blocked_reason",
    "notes",
)
REQUIRED_PAYLOAD_METADATA_FIELDS: tuple[str, ...] = (
    "local_snapshot_path",
    "local_payload_path",
    "local_payload_sha256",
    "local_payload_size_bytes",
    "payload_mime_type",
    "payload_encoding",
    "validation_status",
    "staleness_status",
    "usable_for_cached_source_dry_run",
)
_TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")


def build_cached_source_snapshot_acquisition_dry_run(
    *,
    tickers: Sequence[str] | None = None,
    source_families: Sequence[str] | None = None,
    output_root: str | Path | None = None,
    dry_run_at: str | None = None,
    batch_id: str | None = None,
) -> dict[str, Any]:
    requested_tickers = _normalize_tickers(tickers or ())
    requested_source_families = _normalize_source_families(source_families or ())
    root = Path(output_root) if output_root is not None else None
    effective_batch_id = batch_id or _batch_id_from_timestamp(dry_run_at)

    entries = _entries(
        requested_tickers=requested_tickers,
        requested_source_families=requested_source_families,
        output_root=root,
        batch_id=effective_batch_id,
    )
    counts = _counts(
        entries=entries,
        requested_tickers=requested_tickers,
        requested_source_families=requested_source_families,
    )
    return {
        "report_format_version": CACHED_SOURCE_SNAPSHOT_ACQUISITION_DRY_RUN_FORMAT_VERSION,
        "dry_run_at": dry_run_at or _utc_now_text(),
        "output_root": root.as_posix() if root is not None else None,
        "batch_id": effective_batch_id,
        "requested_tickers": requested_tickers,
        "requested_source_families": requested_source_families,
        "supported_source_families": SUPPORTED_SOURCE_FAMILIES,
        "counts": counts,
        "entries": entries,
        "staging_validator_handoff": (
            "A future real acquisition or operator import must write payload files and "
            "market-engine-cached-source-snapshot-acquisition-manifest-v1 manifests, "
            "then run the ME-SR10 cached-source snapshot staging validator before any "
            "cached-source dry-run may consume the snapshots."
        ),
        "forbidden_side_effect_confirmation": (
            "Acquisition dry-run planned local intent only. No provider, network, "
            "SEC/EDGAR, yfinance, broker, Telegram, portfolio, watchlist, production "
            "write, payload write, acquisition manifest write, Decision Engine, "
            "Recommendation Review, ranking, scoring, allocation, order, execution, "
            "or tradeability behavior was invoked."
        ),
    }


def _entries(
    *,
    requested_tickers: Sequence[str],
    requested_source_families: Sequence[str],
    output_root: Path | None,
    batch_id: str,
) -> list[dict[str, Any]]:
    if not requested_tickers or not requested_source_families:
        return []

    entries = [
        _entry(
            ticker=ticker,
            source_family=source_family,
            output_root=output_root,
            batch_id=batch_id,
        )
        for ticker in requested_tickers
        for source_family in requested_source_families
    ]
    return sorted(
        entries,
        key=lambda entry: (
            str(entry["ticker"]),
            str(entry["source_family"]),
        ),
    )


def _entry(
    *,
    ticker: str,
    source_family: str,
    output_root: Path | None,
    batch_id: str,
) -> dict[str, Any]:
    issues = _entry_issues(
        ticker=ticker,
        source_family=source_family,
        output_root=output_root,
    )
    status = "blocked" if issues else "planned"
    snapshot_id = _snapshot_id(ticker=ticker, source_family=source_family)
    staging_path = (
        (output_root / batch_id / ticker / snapshot_id).as_posix()
        if output_root is not None
        else None
    )
    manifest_path = f"{staging_path}/manifest.json" if staging_path else None
    return {
        "ticker": ticker,
        "source_family": source_family,
        "acquisition_mode": "dry_run_only",
        "acquisition_dry_run_status": status,
        "would_acquire": status == "planned",
        "would_acquire_external_data": False,
        "would_write_payload": False,
        "would_write_manifest": False,
        "expected_manifest_format_version": (
            CACHED_SOURCE_SNAPSHOT_ACQUISITION_MANIFEST_FORMAT_VERSION
        ),
        "proposed_staging_path": staging_path,
        "proposed_manifest_path": manifest_path,
        "required_manifest_fields": REQUIRED_ACQUISITION_MANIFEST_FIELDS,
        "required_payload_metadata_fields": REQUIRED_PAYLOAD_METADATA_FIELDS,
        "validation_handoff": (
            "Run market_engine.source_refresh."
            "cached_source_snapshot_staging_validator_command after a real local "
            "payload and manifest are staged."
        ),
        "issues": tuple(issues),
    }


def _entry_issues(
    *,
    ticker: str,
    source_family: str,
    output_root: Path | None,
) -> list[str]:
    issues: list[str] = []
    if not _valid_ticker(ticker):
        issues.append("ticker_invalid")
    if source_family not in SUPPORTED_SOURCE_FAMILIES:
        issues.append("source_family_unsupported")
    if output_root is None:
        issues.append("output_root_missing")
    return sorted(issues)


def _counts(
    *,
    entries: Sequence[Mapping[str, Any]],
    requested_tickers: Sequence[str],
    requested_source_families: Sequence[str],
) -> dict[str, int]:
    return {
        "total_requested_entries": len(entries),
        "planned_entries": sum(
            1 for entry in entries if entry["acquisition_dry_run_status"] == "planned"
        ),
        "blocked_entries": sum(
            1 for entry in entries if entry["acquisition_dry_run_status"] == "blocked"
        ),
        "invalid_ticker_count": sum(
            1 for ticker in requested_tickers if not _valid_ticker(ticker)
        ),
        "unsupported_source_family_count": sum(
            1
            for source_family in requested_source_families
            if source_family not in SUPPORTED_SOURCE_FAMILIES
        ),
        "missing_ticker_count": 0 if requested_tickers else 1,
        "missing_source_family_count": 0 if requested_source_families else 1,
    }


def _normalize_tickers(tickers: Sequence[str]) -> tuple[str, ...]:
    normalized: set[str] = set()
    for value in tickers:
        for part in str(value).split(","):
            ticker = part.strip().upper()
            if ticker:
                normalized.add(ticker)
    return tuple(sorted(normalized))


def _normalize_source_families(source_families: Sequence[str]) -> tuple[str, ...]:
    normalized: set[str] = set()
    for value in source_families:
        for part in str(value).split(","):
            source_family = part.strip().lower().replace("-", "_")
            if source_family:
                normalized.add(source_family)
    return tuple(sorted(normalized))


def _valid_ticker(ticker: str) -> bool:
    return bool(_TICKER_PATTERN.fullmatch(ticker))


def _snapshot_id(*, ticker: str, source_family: str) -> str:
    return f"{ticker.lower()}-{source_family}-snapshot"


def _batch_id_from_timestamp(dry_run_at: str | None) -> str:
    if not dry_run_at:
        return "cached-source-acquisition-dry-run"
    return (
        "cached-source-acquisition-dry-run-"
        + dry_run_at.replace(":", "").replace("-", "").replace(".", "").replace("Z", "z")
    )


def _utc_now_text() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
