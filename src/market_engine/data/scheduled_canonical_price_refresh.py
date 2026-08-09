from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, time as wall_time, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TextIO
from zoneinfo import ZoneInfo

import pandas as pd

from market_engine.data.complete_local_market_dataset import _to_yfinance_symbol
from market_engine.data.incremental_market_data_refresh import (
    _download_yfinance_history,
    download_yfinance_batch,
    refresh_one_instrument,
)
from market_engine.data.instrument_lifecycle import (
    DEFAULT_LIFECYCLE_REGISTRY,
    LIFECYCLE_SCHEMA_VERSION,
    InstrumentLifecycleError,
    apply_lifecycle_registry,
    load_lifecycle_registry,
)
from market_engine.data.local_market_data_universe import (
    DEFAULT_MIN_HISTORY_ROWS,
    UNIVERSE_SNAPSHOT_SCHEMA_VERSION,
    validate_price_history_csv,
)
from market_engine.data.observation_receipts import (
    DEFAULT_SOURCE_POLICY,
    POLICY_SCHEMA_VERSION,
    ObservationReceiptError,
    load_source_policy,
    observation_receipt_root,
    replay_observation_receipts,
)


SCHEMA_VERSION = "market-engine-me-sr23-canonical-price-freshness-manifest-v7"
VALIDATION_SCHEMA_VERSION = "market-engine-me-sr23-published-price-dataset-validation-v7"
DEFAULT_UNIVERSE_SNAPSHOT = Path(
    "artifacts/market_engine/data_runs/"
    "me-data04-complete-dataset-20260713T133000Z-coverage-after/universe_snapshot.json"
)
DATA_BRANCH = "market-data"
DATA_RELATIVE_ROOT = Path("data/processed")
LATEST_MANIFEST = Path("manifests/canonical_price_freshness_latest.json")
PROVIDER_IDENTITY = "Yahoo Finance via yfinance"
DEFAULT_BATCH_SIZE = 25
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_PROVIDER_TIMEOUT_SECONDS = 15
DEFAULT_SCHEDULED_OVERLAP_CALENDAR_DAYS = 0
STATUS_ORDER = (
    "updated",
    "already_current",
    "not_expected",
    "stale",
    "failed",
    "unsupported",
)
HISTORY_COVERAGE_ORDER = (
    "sufficient",
    "limited_history",
    "insufficient_unexplained",
    "retained_inactive",
    "not_applicable",
)
DEGRADED_STATUSES = frozenset({"stale", "failed", "unsupported"})
DEGRADED_HISTORY_COVERAGE = frozenset({"insufficient_unexplained"})
MAX_LISTING_START_SESSION_LAG = 1
REQUIRED_LISTING_SESSION_COVERAGE_RATIO = 1.0
SHA1 = re.compile(r"^[0-9a-f]{40}$")

Provider = Callable[[str, str, str], pd.DataFrame]
Sleeper = Callable[[float], None]


class ScheduledPriceRefreshError(ValueError):
    pass


class ProviderBoundaryError(RuntimeError):
    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        diagnostic: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.diagnostic = dict(diagnostic or {})


@dataclass(frozen=True)
class MarketProfile:
    market: str
    timezone: str
    close_time: wall_time
    holiday_calendar: str


MARKET_PROFILES: Mapping[str, MarketProfile] = {
    "US": MarketProfile("US", "America/New_York", wall_time(16, 0), "us_equities"),
    "XAMS": MarketProfile("XAMS", "Europe/Amsterdam", wall_time(17, 30), "europe_continental"),
    "XBRU": MarketProfile("XBRU", "Europe/Brussels", wall_time(17, 30), "europe_continental"),
    "XPAR": MarketProfile("XPAR", "Europe/Paris", wall_time(17, 30), "europe_continental"),
    "XETR": MarketProfile("XETR", "Europe/Berlin", wall_time(17, 30), "europe_continental"),
    "XLON": MarketProfile("XLON", "Europe/London", wall_time(16, 30), "uk_equities"),
}
EXCHANGE_ALIASES = {
    "NYSE": "US",
    "NASDAQ": "US",
    "XNAS": "US",
    "XNYS": "US",
    "AMEX": "US",
    "ARCA": "US",
    "EURONEXT_AMSTERDAM": "XAMS",
    "EURONEXT_BRUSSELS": "XBRU",
    "EURONEXT_PARIS": "XPAR",
    "LSE": "XLON",
}


def run_scheduled_refresh(
    *,
    run_id: str,
    source_main_sha: str,
    universe_snapshot_path: str | Path = DEFAULT_UNIVERSE_SNAPSHOT,
    lifecycle_registry_path: str | Path | None = None,
    published_root: str | Path,
    staging_root: str | Path,
    report_output: str | Path,
    run_at: datetime | None = None,
    workflow_run_id: str | None = None,
    provider: Provider | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    overlap_calendar_days: int = DEFAULT_SCHEDULED_OVERLAP_CALENDAR_DAYS,
    sleeper: Sleeper = time.sleep,
    source_policy_path: str | Path = DEFAULT_SOURCE_POLICY,
    fallback_receipts: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    if not run_id or not SHA1.fullmatch(source_main_sha):
        raise ScheduledPriceRefreshError("run ID and full source main SHA are required")
    if batch_size < 1 or max_attempts < 1:
        raise ScheduledPriceRefreshError("batch size and maximum attempts must be positive")
    generated_at = _as_utc(run_at or datetime.now(UTC))
    universe = load_authoritative_universe(universe_snapshot_path)
    lifecycle_registry = _lifecycle_registry_for_universe(
        universe_snapshot_path,
        lifecycle_registry_path,
    )
    governed = apply_lifecycle_registry(
        universe["instruments"],
        lifecycle_registry,
        as_of=generated_at.date(),
    )
    source_policy = load_source_policy(source_policy_path)
    source_root = Path(published_root)
    stage_root = Path(staging_root)
    _prepare_staging_root(source_root, stage_root)
    price_root = stage_root / DATA_RELATIVE_ROOT
    price_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    refreshable_inactive = [
        instrument
        for instrument in governed["inactive_instruments"]
        if _inactive_history_needs_backfill(instrument, price_root=price_root)
    ]
    instruments = governed["active_instruments"] + refreshable_inactive
    for offset in range(0, len(instruments), batch_size):
        batch = instruments[offset : offset + batch_size]
        selected_provider = provider or _prefetch_batch_provider(
            batch,
            price_root=price_root,
            run_at=generated_at,
            max_attempts=max_attempts,
            sleeper=sleeper,
        )
        for instrument in batch:
            refreshed = _refresh_instrument(
                instrument,
                price_root=price_root,
                run_at=generated_at,
                provider=selected_provider,
                max_attempts=max_attempts,
                overlap_calendar_days=overlap_calendar_days,
                sleeper=sleeper,
                fallback_receipts=(fallback_receipts or {}).get(
                    instrument["instrument_id"], []
                ),
                source_policy=source_policy,
                artifact_root=stage_root,
            )
            if instrument["lifecycle_status"] == "inactive":
                retained = _non_refreshable_lifecycle_row(
                    instrument, price_root=price_root
                )
                refreshed = {
                    **retained,
                    "rows_added": refreshed["rows_added"],
                    "previous_file_checksum": refreshed[
                        "previous_file_checksum"
                    ],
                    "persisted_file_checksum": refreshed[
                        "persisted_file_checksum"
                    ],
                    "previous_last_observation": refreshed[
                        "previous_last_observation"
                    ],
                    "resulting_last_observation": refreshed[
                        "resulting_last_observation"
                    ],
                    "previous_row_count": refreshed["previous_row_count"],
                    "resulting_row_count": refreshed["resulting_row_count"],
                    "expected_backfill_sessions": refreshed.get(
                        "expected_backfill_sessions", []
                    ),
                    "observed_backfill_sessions": refreshed.get(
                        "observed_backfill_sessions", []
                    ),
                    "explained_missing_sessions": refreshed.get(
                        "explained_missing_sessions", []
                    ),
                    "provider_identity": refreshed["provider_identity"],
                    "provider_retrieval": refreshed.get("provider_retrieval", []),
                    "rejected_bar_diagnostics": refreshed.get(
                        "rejected_bar_diagnostics", []
                    ),
                    "primary_observed_sessions": refreshed.get(
                        "primary_observed_sessions", []
                    ),
                    "observation_receipts": refreshed.get(
                        "observation_receipts", []
                    ),
                    "observation_receipt_root": refreshed.get(
                        "observation_receipt_root"
                    ),
                    "fallback_required_sessions": refreshed.get(
                        "fallback_required_sessions", []
                    ),
                }
            rows.append(refreshed)
    refreshed_inactive_ids = {
        row["instrument_id"] for row in refreshable_inactive
    }
    for instrument in governed["inactive_instruments"] + governed["pending_instruments"]:
        if instrument["instrument_id"] in refreshed_inactive_ids:
            continue
        rows.append(_non_refreshable_lifecycle_row(instrument, price_root=price_root))

    rows.sort(key=lambda row: (row["instrument_id"], row["ticker"]))
    counts = Counter(row["freshness_status"] for row in rows)
    history_counts = Counter(row["history_coverage_status"] for row in rows)
    changed_price_files = sorted(
        str(row["persisted_file_path"])
        for row in rows
        if isinstance(row.get("persisted_file_checksum"), str)
        and row.get("previous_file_checksum")
        != row.get("persisted_file_checksum")
    )
    changed_price_file_count = len(changed_price_files)
    bound_rows = [
        row for row in rows if row["lifecycle_status"] != "pending"
    ]
    publication_set_valid = all(
        row["validation_status"] == "valid"
        and isinstance(row.get("persisted_file_checksum"), str)
        for row in bound_rows
    ) and all(
        row.get("persisted_file_checksum") is None
        for row in rows
        if row["lifecycle_status"] == "pending"
    )
    previous_manifest = _load_optional_manifest(source_root / LATEST_MANIFEST)
    manifest_change_required = (
        previous_manifest.get("schema_version") != SCHEMA_VERSION
        or previous_manifest.get("canonical_universe_checksum")
        != universe["universe_checksum"]
        or previous_manifest.get("lifecycle_registry_checksum")
        != governed["registry_checksum"]
        or previous_manifest.get("active_universe_checksum")
        != governed["active_universe_checksum"]
        or previous_manifest.get("governed_universe_checksum")
        != governed["governed_universe_checksum"]
        or previous_manifest.get("market_price_source_policy_checksum")
        != source_policy["policy_checksum"]
    )
    degraded = (
        any(counts.get(status, 0) for status in DEGRADED_STATUSES)
        or any(
            history_counts.get(status, 0)
            for status in DEGRADED_HISTORY_COVERAGE
        )
    )
    run_status = "failed" if not publication_set_valid else "degraded" if degraded else "completed"
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": _utc_text(generated_at),
        "source_main_sha": source_main_sha,
        "workflow_run_id": workflow_run_id,
        "data_branch": DATA_BRANCH,
        "universe_version": universe["universe_version"],
        "canonical_universe_checksum": universe["universe_checksum"],
        "canonical_universe_size": len(universe["instruments"]),
        "lifecycle_schema_version": LIFECYCLE_SCHEMA_VERSION,
        "lifecycle_registry_checksum": governed["registry_checksum"],
        "market_price_source_policy_schema_version": POLICY_SCHEMA_VERSION,
        "market_price_source_policy_checksum": source_policy["policy_checksum"],
        "active_universe_checksum": governed["active_universe_checksum"],
        "governed_universe_checksum": governed["governed_universe_checksum"],
        "active_universe_size": governed["active_universe_size"],
        "inactive_retained_instrument_count": governed[
            "inactive_retained_instrument_count"
        ],
        "pending_instrument_count": governed["pending_instrument_count"],
        "provider_configuration": {
            "identity": PROVIDER_IDENTITY,
            "batch_size": batch_size,
            "max_attempts": max_attempts,
            "timeout_seconds": DEFAULT_PROVIDER_TIMEOUT_SECONDS,
            "overlap_calendar_days": overlap_calendar_days,
            "missing_range_only": True,
            "parallel_requests": 1,
            "request_mode": "bounded_multi_symbol_batches" if provider is None else "injected_offline_provider",
        },
        "expected_completed_sessions": _expected_session_summary(rows),
        "status_counts": {status: counts.get(status, 0) for status in STATUS_ORDER},
        "history_coverage_counts": {
            status: history_counts.get(status, 0)
            for status in HISTORY_COVERAGE_ORDER
        },
        "run_status": run_status,
        "publication": {
            "publication_set_valid": publication_set_valid,
            "publication_required": run_status == "completed"
            and publication_set_valid
            and (changed_price_file_count > 0 or manifest_change_required),
            "changed_price_file_count": changed_price_file_count,
            "changed_price_files": changed_price_files,
            "manifest_change_required": manifest_change_required,
            "empty_commit_required": False,
        },
        "fundamental_evidence": {
            "status": "not_evaluated",
            "reason_code": "NO_RELIABLE_AUTOMATED_FUNDAMENTAL_FRESHNESS_CONTRACT",
            "approval_required": False,
            "approval_generated": False,
        },
        "tickers": rows,
        "manifest_checksum": None,
    }
    report["manifest_checksum"] = _manifest_checksum(report)
    _atomic_write_json(Path(report_output), report)
    if report["publication"]["publication_required"]:
        _atomic_write_json(stage_root / LATEST_MANIFEST, report)
    return report


def load_authoritative_universe(path: str | Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("schema_version") != UNIVERSE_SNAPSHOT_SCHEMA_VERSION:
        raise ScheduledPriceRefreshError("authoritative universe snapshot schema is unsupported")
    instruments = payload.get("instruments")
    if not isinstance(instruments, list) or not instruments:
        raise ScheduledPriceRefreshError("authoritative universe snapshot has no instruments")
    identities: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(instruments):
        if not isinstance(raw, Mapping):
            raise ScheduledPriceRefreshError(f"universe instrument {index} is not an object")
        instrument_id = _required_text(raw, "instrument_id")
        symbol = _required_text(raw, "symbol")
        source_symbol = _required_text(raw, "source_symbol")
        if instrument_id in identities:
            raise ScheduledPriceRefreshError(f"duplicate universe instrument ID: {instrument_id}")
        identities.add(instrument_id)
        normalized.append({**dict(raw), "instrument_id": instrument_id, "symbol": symbol, "source_symbol": source_symbol})
    normalized.sort(key=lambda row: (row["instrument_id"], row["symbol"]))
    checksum_payload = {
        "schema_version": payload["schema_version"],
        "universe_version": payload.get("universe_version"),
        "instruments": normalized,
    }
    return {
        **dict(payload),
        "instruments": normalized,
        "universe_checksum": _canonical_checksum(checksum_payload),
    }


def expected_completed_session(instrument: Mapping[str, Any], run_at: datetime) -> tuple[MarketProfile | None, date | None]:
    profile = _resolve_market_profile(instrument)
    if profile is None:
        return None, None
    local_now = _as_utc(run_at).astimezone(ZoneInfo(profile.timezone))
    candidate = local_now.date()
    if local_now.timetz().replace(tzinfo=None) < profile.close_time:
        candidate -= timedelta(days=1)
    while not _is_trading_session(candidate, profile.holiday_calendar):
        candidate -= timedelta(days=1)
    last_trading_session = instrument.get("last_trading_session")
    if isinstance(last_trading_session, str):
        lifecycle_cutoff = date.fromisoformat(last_trading_session)
        if lifecycle_cutoff <= candidate:
            candidate = lifecycle_cutoff
    return profile, candidate


def validate_published_dataset(
    publication_root: str | Path,
    *,
    universe_snapshot_path: str | Path = DEFAULT_UNIVERSE_SNAPSHOT,
    lifecycle_registry_path: str | Path | None = None,
    run_at: datetime | None = None,
    allow_degraded: bool = False,
    expected_source_main_sha: str | None = None,
    baseline_publication_root: str | Path | None = None,
    source_policy_path: str | Path = DEFAULT_SOURCE_POLICY,
) -> dict[str, Any]:
    root = Path(publication_root)
    issues: list[dict[str, str]] = []
    manifest_path = root / LATEST_MANIFEST
    try:
        manifest = _load_json(manifest_path)
    except ScheduledPriceRefreshError:
        return _validation_result(issues=[_validation_issue("PUBLISHED_MANIFEST_MISSING_OR_MALFORMED", "manifest")])
    try:
        universe = load_authoritative_universe(universe_snapshot_path)
        lifecycle_registry = _lifecycle_registry_for_universe(
            universe_snapshot_path,
            lifecycle_registry_path,
        )
        governed = apply_lifecycle_registry(
            universe["instruments"],
            lifecycle_registry,
            as_of=_as_utc(run_at or datetime.now(UTC)).date(),
        )
        source_policy = load_source_policy(source_policy_path)
    except (
        ScheduledPriceRefreshError,
        InstrumentLifecycleError,
        ObservationReceiptError,
    ):
        return _validation_result(issues=[_validation_issue("AUTHORITATIVE_UNIVERSE_INVALID", "universe")])

    if manifest.get("schema_version") != SCHEMA_VERSION:
        issues.append(_validation_issue("PUBLISHED_MANIFEST_SCHEMA_MISMATCH", "schema_version"))
    if manifest.get("data_branch") != DATA_BRANCH:
        issues.append(_validation_issue("PUBLISHED_DATA_BRANCH_MISMATCH", "data_branch"))
    source_main_sha = manifest.get("source_main_sha")
    if not isinstance(source_main_sha, str) or not SHA1.fullmatch(source_main_sha):
        issues.append(_validation_issue("PUBLISHED_SOURCE_MAIN_SHA_INVALID", "source_main_sha"))
    elif expected_source_main_sha is not None and source_main_sha != expected_source_main_sha:
        issues.append(_validation_issue("PUBLISHED_SOURCE_MAIN_SHA_MISMATCH", "source_main_sha"))
    if manifest.get("manifest_checksum") != _manifest_checksum(manifest):
        issues.append(_validation_issue("PUBLISHED_MANIFEST_CHECKSUM_MISMATCH", "manifest_checksum"))
    if (
        manifest.get("universe_version") != universe.get("universe_version")
        or manifest.get("canonical_universe_checksum")
        != universe.get("universe_checksum")
        or manifest.get("canonical_universe_size")
        != len(universe["instruments"])
        or manifest.get("lifecycle_schema_version")
        != LIFECYCLE_SCHEMA_VERSION
        or manifest.get("lifecycle_registry_checksum")
        != governed["registry_checksum"]
        or manifest.get("active_universe_checksum")
        != governed["active_universe_checksum"]
        or manifest.get("governed_universe_checksum")
        != governed["governed_universe_checksum"]
        or manifest.get("active_universe_size")
        != governed["active_universe_size"]
        or manifest.get("inactive_retained_instrument_count")
        != governed["inactive_retained_instrument_count"]
        or manifest.get("pending_instrument_count")
        != governed["pending_instrument_count"]
        or manifest.get("market_price_source_policy_schema_version")
        != POLICY_SCHEMA_VERSION
        or manifest.get("market_price_source_policy_checksum")
        != source_policy["policy_checksum"]
    ):
        issues.append(_validation_issue("PUBLISHED_UNIVERSE_BINDING_MISMATCH", "universe"))
    if _contains_executable_content(root):
        issues.append(_validation_issue("PUBLISHED_DATA_BRANCH_CONTENT_INVALID", "publication_root"))

    publication = manifest.get("publication")
    manifest_status_counts = manifest.get("status_counts")
    manifest_change_required = (
        publication.get("manifest_change_required")
        if isinstance(publication, Mapping)
        else None
    )
    if not isinstance(publication, Mapping) or not (
        publication.get("publication_set_valid") is True
        and publication.get("publication_required") is True
        and isinstance(publication.get("changed_price_file_count"), int)
        and isinstance(publication.get("changed_price_files"), list)
        and isinstance(manifest_change_required, bool)
        and (
            publication["changed_price_file_count"] > 0
            or manifest_change_required
        )
        and publication.get("empty_commit_required") is False
    ):
        issues.append(_validation_issue("PUBLISHED_PUBLICATION_DECISION_INVALID", "publication"))
    if manifest.get("fundamental_evidence") != {
        "status": "not_evaluated",
        "reason_code": "NO_RELIABLE_AUTOMATED_FUNDAMENTAL_FRESHNESS_CONTRACT",
        "approval_required": False,
        "approval_generated": False,
    }:
        issues.append(_validation_issue("PUBLISHED_FUNDAMENTAL_BOUNDARY_INVALID", "fundamental_evidence"))

    entries = manifest.get("tickers")
    if not isinstance(entries, list):
        entries = []
        issues.append(_validation_issue("PUBLISHED_TICKER_ENTRIES_INVALID", "tickers"))
    expected_ids = [row["instrument_id"] for row in governed["instruments"]]
    actual_ids = [row.get("instrument_id") for row in entries if isinstance(row, Mapping)]
    if actual_ids != expected_ids:
        issues.append(_validation_issue("PUBLISHED_TICKER_SET_MISMATCH", "tickers"))
    expected_by_id = {
        row["instrument_id"]: row for row in governed["instruments"]
    }
    actual_status_counts = Counter(
        str(row.get("freshness_status")) for row in entries if isinstance(row, Mapping)
    )
    expected_status_counts = {status: actual_status_counts.get(status, 0) for status in STATUS_ORDER}
    if (
        any(status not in STATUS_ORDER for status in actual_status_counts)
        or manifest_status_counts != expected_status_counts
    ):
        issues.append(_validation_issue("PUBLISHED_STATUS_COUNTS_MISMATCH", "status_counts"))
    manifest_history_counts = manifest.get("history_coverage_counts")
    actual_history_counts = Counter(
        str(row.get("history_coverage_status"))
        for row in entries
        if isinstance(row, Mapping)
    )
    expected_history_counts = {
        status: actual_history_counts.get(status, 0)
        for status in HISTORY_COVERAGE_ORDER
    }
    if (
        any(
            status not in HISTORY_COVERAGE_ORDER
            for status in actual_history_counts
        )
        or manifest_history_counts != expected_history_counts
    ):
        issues.append(
            _validation_issue(
                "PUBLISHED_HISTORY_COVERAGE_COUNTS_MISMATCH",
                "history_coverage_counts",
            )
        )
    bound_files = {
        str(row.get("persisted_file_path"))
        for row in entries
        if isinstance(row, Mapping)
        and isinstance(row.get("persisted_file_checksum"), str)
        and isinstance(row.get("persisted_file_path"), str)
    }
    actual_files = {
        path.relative_to(root).as_posix()
        for path in (root / DATA_RELATIVE_ROOT).glob("*.csv")
        if path.is_file()
    }
    if actual_files != bound_files:
        issues.append(_validation_issue("PUBLISHED_UNBOUND_PRICE_FILE_SET", "data/processed"))
    declared_raw_artifacts = {
        str(receipt.get("raw_artifact_locator"))
        for entry in entries
        if isinstance(entry, Mapping)
        for receipt in entry.get("observation_receipts", [])
        if isinstance(receipt, Mapping)
        and isinstance(receipt.get("raw_artifact_locator"), str)
    }
    evidence_root = root / "evidence" / "market_price"
    actual_raw_artifacts = {
        path.relative_to(root).as_posix()
        for path in evidence_root.rglob("*.json")
        if path.is_file()
    } if evidence_root.is_dir() else set()
    if actual_raw_artifacts != declared_raw_artifacts:
        issues.append(
            _validation_issue(
                "PUBLISHED_RAW_OBSERVATION_ARTIFACT_SET_MISMATCH",
                "evidence/market_price",
            )
        )
    declared_changed_files = (
        publication.get("changed_price_files", [])
        if isinstance(publication, Mapping)
        else []
    )
    reconciled_changed_files = sorted(
        str(row.get("persisted_file_path"))
        for row in entries
        if isinstance(row, Mapping)
        and isinstance(row.get("persisted_file_checksum"), str)
        and row.get("previous_file_checksum")
        != row.get("persisted_file_checksum")
    )
    previous_checksums_valid = all(
        row.get("previous_file_checksum") is None
        or (
            isinstance(row.get("previous_file_checksum"), str)
            and re.fullmatch(r"[0-9a-f]{64}", row["previous_file_checksum"])
        )
        for row in entries
        if isinstance(row, Mapping)
    )
    if (
        not previous_checksums_valid
        or declared_changed_files != sorted(set(declared_changed_files))
        or declared_changed_files != reconciled_changed_files
        or publication.get("changed_price_file_count")
        != len(reconciled_changed_files)
        or not set(reconciled_changed_files).issubset(actual_files)
    ):
        issues.append(
            _validation_issue(
                "PUBLISHED_CHANGED_PRICE_FILE_SET_MISMATCH",
                "publication.changed_price_files",
            )
        )
    if baseline_publication_root is not None:
        baseline_root = Path(baseline_publication_root)
        baseline_changed_files: list[str] = []
        baseline_previous_mismatch = False
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            relative = entry.get("persisted_file_path")
            persisted_checksum = entry.get("persisted_file_checksum")
            if not isinstance(relative, str) or not isinstance(
                persisted_checksum, str
            ):
                continue
            baseline_path = baseline_root / relative
            baseline_checksum = (
                _sha256_file(baseline_path) if baseline_path.is_file() else None
            )
            if entry.get("previous_file_checksum") != baseline_checksum:
                baseline_previous_mismatch = True
            if baseline_checksum != persisted_checksum:
                baseline_changed_files.append(relative)
        if (
            baseline_previous_mismatch
            or sorted(baseline_changed_files) != reconciled_changed_files
        ):
            issues.append(
                _validation_issue(
                    "PUBLISHED_CHANGED_PRICE_BASELINE_MISMATCH",
                    "publication.changed_price_files",
                )
            )

    validation_at = _as_utc(run_at or datetime.now(UTC))
    stale: list[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            continue
        expected_instrument = expected_by_id.get(entry.get("instrument_id"))
        relative = entry.get("persisted_file_path")
        checksum = entry.get("persisted_file_checksum")
        expected_relative = (
            (DATA_RELATIVE_ROOT / f"{expected_instrument['source_symbol']}.csv").as_posix()
            if expected_instrument is not None
            else None
        )
        if (
            expected_instrument is None
            or entry.get("ticker") != expected_instrument["symbol"]
            or relative != expected_relative
            or not _entry_lifecycle_matches(entry, expected_instrument)
        ):
            issues.append(_validation_issue("PUBLISHED_TICKER_IDENTITY_MISMATCH", f"tickers[{index}]"))
            continue
        if expected_instrument["lifecycle_status"] == "pending":
            if checksum is not None:
                issues.append(
                    _validation_issue(
                        "PUBLISHED_PRE_LISTING_FILE_UNEXPECTED",
                        f"tickers[{index}]",
                    )
                )
            continue
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or relative_path.parent != DATA_RELATIVE_ROOT or relative_path.suffix != ".csv":
            issues.append(_validation_issue("PUBLISHED_FILE_PATH_INVALID", f"tickers[{index}]"))
            continue
        path = root / relative
        if not path.is_file() or not isinstance(checksum, str) or _sha256_file(path) != checksum:
            issues.append(_validation_issue("PUBLISHED_FILE_CHECKSUM_MISMATCH", f"tickers[{index}]"))
            continue
        validation = _validate_price_history(path)
        if validation.get("status") != "valid":
            issues.append(_validation_issue("PUBLISHED_PRICE_FILE_INVALID", f"tickers[{index}]"))
            continue
        if entry.get("validation_status") != "valid":
            issues.append(_validation_issue("PUBLISHED_TICKER_VALIDATION_STATUS_INVALID", f"tickers[{index}]"))
        resulting = entry.get("resulting_last_observation")
        if resulting != validation.get("end_date"):
            issues.append(_validation_issue("PUBLISHED_LAST_OBSERVATION_MISMATCH", f"tickers[{index}]"))
        previous = entry.get("previous_last_observation")
        previous_row_count = entry.get("previous_row_count")
        resulting_row_count = entry.get("resulting_row_count")
        rows_added = entry.get("rows_added")
        observation_metadata_valid = (
            (previous is None or isinstance(previous, str))
            and isinstance(resulting, str)
            and isinstance(previous_row_count, int)
            and isinstance(resulting_row_count, int)
            and isinstance(rows_added, int)
            and resulting_row_count == int(validation.get("row_count") or 0)
            and resulting_row_count - previous_row_count == rows_added
            and (previous is None or previous <= resulting)
            and (
                rows_added == 0
                or previous is None
                or previous < resulting
            )
        )
        if not observation_metadata_valid:
            issues.append(
                _validation_issue(
                    "PUBLISHED_OBSERVATION_MUTATION_METADATA_INVALID",
                    f"tickers[{index}]",
                )
            )
        if baseline_publication_root is not None:
            baseline_path = Path(baseline_publication_root) / str(relative)
            baseline_validation = (
                _validate_price_history(baseline_path)
                if baseline_path.is_file()
                else {"status": "missing"}
            )
            if (
                entry.get("previous_file_checksum")
                != baseline_validation.get("checksum")
                or previous != baseline_validation.get("end_date")
                or previous_row_count
                != int(baseline_validation.get("row_count") or 0)
            ):
                issues.append(
                    _validation_issue(
                        "PUBLISHED_OBSERVATION_BASELINE_MISMATCH",
                        f"tickers[{index}]",
                    )
                )
        runtime_observation = _runtime_observation_fields(
            expected_instrument,
            validation=validation,
        )
        if not _row_fields_match(entry, runtime_observation):
            issues.append(
                _validation_issue(
                    "PUBLISHED_DAILY_OHLCV_OBSERVATION_STATUS_INVALID",
                    f"tickers[{index}]",
                )
            )
        profile, required_session = expected_completed_session(
            expected_instrument,
            validation_at,
        )
        expected_backfill = (
            _expected_sessions_between(
                profile,
                date.fromisoformat(previous) + timedelta(days=1),
                required_session,
            )
            if profile is not None
            and required_session is not None
            and isinstance(previous, str)
            else []
        )
        observed_dates = set(validation.get("observation_dates") or ())
        expected_observed = [
            session for session in expected_backfill if session in observed_dates
        ]
        expected_missing = [
            session for session in expected_backfill if session not in observed_dates
        ]
        expected_explained = (
            _explained_missing_daily_ohlcv_sessions(
                expected_instrument,
                missing_sessions=expected_missing,
                expected_session=required_session,
            )
            if required_session is not None
            else []
        )
        if (
            entry.get("expected_backfill_sessions", [])
            != [session.isoformat() for session in expected_backfill]
            or entry.get("observed_backfill_sessions", [])
            != [session.isoformat() for session in expected_observed]
            or entry.get("explained_missing_sessions", [])
            != expected_explained
            or len(expected_missing) != len(expected_explained)
        ):
            issues.append(
                _validation_issue(
                    "PUBLISHED_EXPECTED_SESSION_COMPLETENESS_INVALID",
                    f"tickers[{index}]",
                )
            )
        receipt_issue = _validate_published_observation_receipts(
            entry,
            staged_path=path,
            baseline_path=(
                Path(baseline_publication_root) / str(relative)
                if baseline_publication_root is not None
                else None
            ),
            publication_root=root,
            source_policy=source_policy,
            lifecycle_cutoff=expected_instrument.get("last_trading_session"),
            instrument_exchange=str(expected_instrument.get("exchange")),
        )
        if receipt_issue is not None:
            issues.append(
                _validation_issue(receipt_issue, f"tickers[{index}]")
            )
        if expected_instrument["lifecycle_status"] == "inactive":
            retained_boundary = _retained_history_boundary(
                expected_instrument,
                validation=validation,
            )
            if (
                retained_boundary["retained_history_boundary_status"]
                != "aligned"
                or not _row_fields_match(entry, retained_boundary)
            ):
                issues.append(
                    _validation_issue(
                        "PUBLISHED_RETAINED_HISTORY_BOUNDARY_INVALID",
                        f"tickers[{index}]",
                    )
                )
            if (
                entry.get("freshness_status") != "not_expected"
                or entry.get("history_coverage_status")
                != "retained_inactive"
            ):
                issues.append(
                    _validation_issue(
                        "PUBLISHED_INACTIVE_CLASSIFICATION_INVALID",
                        f"tickers[{index}]",
                    )
                )
            continue
        required = required_session
        expected_history = _history_coverage(
            expected_instrument,
            validation=validation,
            expected_session=required,
        )
        if not _row_fields_match(entry, expected_history):
            issues.append(
                _validation_issue(
                    "PUBLISHED_HISTORY_COVERAGE_CLASSIFICATION_INVALID",
                    f"tickers[{index}]",
                )
            )
        expected_retained_boundary = _retained_history_boundary(
            expected_instrument,
            validation=validation,
        )
        if not _row_fields_match(entry, expected_retained_boundary):
            issues.append(
                _validation_issue(
                    "PUBLISHED_RETAINED_HISTORY_BOUNDARY_INVALID",
                    f"tickers[{index}]",
                )
            )
        actual_end = validation.get("end_date")
        declared_freshness = entry.get("freshness_status")
        actual_is_stale = (
            profile is None
            or required is None
            or not isinstance(actual_end, str)
            or actual_end < required.isoformat()
        )
        if (
            actual_is_stale
            and declared_freshness not in {"stale", "failed", "unsupported"}
        ) or (
            not actual_is_stale and declared_freshness == "stale"
        ):
            issues.append(
                _validation_issue(
                    "PUBLISHED_FRESHNESS_CLASSIFICATION_INVALID",
                    f"tickers[{index}]",
                )
            )
        if actual_is_stale:
            stale.append(str(entry.get("ticker") or entry.get("instrument_id") or index))
    if stale and not allow_degraded:
        issues.append(_validation_issue("PUBLISHED_DATASET_STALE", "tickers", ",".join(sorted(stale))))
    run_status = manifest.get("run_status")
    expected_run_status = (
        "degraded"
        if (
            any(
                actual_status_counts.get(status, 0)
                for status in DEGRADED_STATUSES
            )
            or any(
                actual_history_counts.get(status, 0)
                for status in DEGRADED_HISTORY_COVERAGE
            )
        )
        else "completed"
    )
    if run_status != expected_run_status:
        issues.append(_validation_issue("PUBLISHED_RUN_STATUS_INVALID", "run_status"))
    elif run_status != "completed" and not allow_degraded:
        issues.append(_validation_issue("PUBLISHED_DATASET_DEGRADED", "run_status"))
    return _validation_result(issues=issues, manifest=manifest, price_history_root=root / DATA_RELATIVE_ROOT, stale=stale)


def run_validated_analysis(
    publication_root: str | Path,
    *,
    universe_snapshot_path: str | Path = DEFAULT_UNIVERSE_SNAPSHOT,
    lifecycle_registry_path: str | Path | None = None,
    run_at: datetime | None = None,
    analysis_runner: Callable[..., Any],
    analysis_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_published_dataset(
        publication_root,
        universe_snapshot_path=universe_snapshot_path,
        lifecycle_registry_path=lifecycle_registry_path,
        run_at=run_at,
    )
    if not validation["validated"]:
        return {"status": "blocked", "analysis_executed": False, "validation": validation}
    universe = load_authoritative_universe(universe_snapshot_path)
    governed = apply_lifecycle_registry(
        universe["instruments"],
        _lifecycle_registry_for_universe(
            universe_snapshot_path,
            lifecycle_registry_path,
        ),
        as_of=_as_utc(run_at or datetime.now(UTC)).date(),
    )
    manifest = _load_json(Path(publication_root) / LATEST_MANIFEST)
    entries_by_id = {
        row["instrument_id"]: row
        for row in manifest["tickers"]
        if isinstance(row, Mapping)
    }
    active_instruments = [
        {
            **instrument,
            "freshness_status": entries_by_id[instrument["instrument_id"]][
                "freshness_status"
            ],
            "history_coverage_status": entries_by_id[
                instrument["instrument_id"]
            ]["history_coverage_status"],
            "history_coverage_reason_code": entries_by_id[
                instrument["instrument_id"]
            ]["history_coverage_reason_code"],
        }
        for instrument in governed["active_instruments"]
    ]
    analysis_universe = {
        **universe,
        "instruments": active_instruments,
        "summary": {
            **dict(universe.get("summary") or {}),
            "total_instruments": len(active_instruments),
        },
        "lifecycle_schema_version": LIFECYCLE_SCHEMA_VERSION,
        "lifecycle_registry_checksum": governed["registry_checksum"],
        "active_universe_checksum": governed["active_universe_checksum"],
    }
    result = analysis_runner(
        price_history_root=Path(publication_root) / DATA_RELATIVE_ROOT,
        universe_snapshot=analysis_universe,
        **dict(analysis_kwargs),
    )
    return {"status": "completed", "analysis_executed": True, "validation": validation, "analysis_result": result}


def _refresh_instrument(
    instrument: Mapping[str, Any],
    *,
    price_root: Path,
    run_at: datetime,
    provider: Provider,
    max_attempts: int,
    overlap_calendar_days: int,
    sleeper: Sleeper,
    fallback_receipts: Sequence[Mapping[str, Any]] = (),
    source_policy: Mapping[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    ticker = str(instrument["symbol"])
    source_symbol = str(instrument["source_symbol"])
    path = price_root / f"{source_symbol}.csv"
    profile, expected = expected_completed_session(instrument, run_at)
    initial_validation = (
        _validate_price_history(path)
        if path.is_file()
        else {"status": "missing"}
    )
    initial_history = _history_coverage(
        instrument,
        validation=initial_validation,
        expected_session=expected,
    )
    base = {
        "ticker": ticker,
        "instrument_id": str(instrument["instrument_id"]),
        "exchange": profile.market if profile else str(instrument.get("exchange") or "UNKNOWN"),
        "market_timezone": profile.timezone if profile else None,
        "provider_identity": PROVIDER_IDENTITY if instrument.get("source_mapping_status") == "mapped" else None,
        "previous_last_observation": initial_validation.get("end_date"),
        "resulting_last_observation": initial_validation.get("end_date"),
        "expected_completed_session": expected.isoformat() if expected else None,
        "rows_added": 0,
        "previous_row_count": int(initial_validation.get("row_count") or 0),
        "resulting_row_count": int(initial_validation.get("row_count") or 0),
        "validation_status": (
            "valid" if initial_validation.get("status") == "valid" else "blocked"
        ),
        "freshness_status": "unsupported",
        "reason_code": "UNSUPPORTED_EXCHANGE" if profile is None else "PROVIDER_MAPPING_MISSING",
        "primary_observed_sessions": [],
        "fallback_required_sessions": [],
        "observation_receipts": [],
        "observation_receipt_root": None,
        "persisted_file_path": (DATA_RELATIVE_ROOT / f"{source_symbol}.csv").as_posix(),
        "previous_file_checksum": (
            initial_validation.get("checksum")
            if initial_validation.get("status") == "valid"
            else None
        ),
        "persisted_file_checksum": _sha256_file(path) if path.is_file() else None,
        **_lifecycle_fields(instrument),
        **initial_history,
        **_retained_history_boundary(
            instrument,
            validation=initial_validation,
        ),
    }
    if profile is None or expected is None:
        return base
    if instrument.get("source_mapping_status") != "mapped" or not source_symbol:
        return base

    before_bytes = path.read_bytes() if path.is_file() else None
    before_validation = initial_validation
    before_frame = pd.read_csv(path) if before_validation.get("status") == "valid" else None
    previous_end = before_validation.get("end_date")
    error: dict[str, str] = {}
    terminal_rejected_bar_diagnostics: list[dict[str, Any]] = []
    terminal_observation_receipts: list[dict[str, Any]] = []
    terminal_primary_observed_sessions: set[str] = set()
    terminal_fallback_required_sessions: set[str] = set()

    def guarded_provider(symbol: str, start: str, end: str) -> pd.DataFrame:
        try:
            accumulated = pd.DataFrame()
            required_sessions = (
                _expected_sessions_between(
                    profile,
                    date.fromisoformat(previous_end) + timedelta(days=1),
                    expected,
                )
                if isinstance(previous_end, str)
                else []
            )
            for completeness_attempt in range(1, max_attempts + 1):
                validated = _provider_with_retries(
                    provider,
                    symbol,
                    start,
                    end,
                    expected_session=expected,
                    max_attempts=(max_attempts if completeness_attempt == 1 else 1),
                    sleeper=sleeper,
                    lifecycle_context=instrument,
                )
                terminal_rejected_bar_diagnostics.extend(
                    validated.attrs.get("rejected_bar_diagnostics", [])
                )
                accumulated = _merge_provider_attempts(accumulated, validated)
                terminal_primary_observed_sessions.update(
                    value.isoformat()
                    for value in pd.to_datetime(
                        accumulated.get("Date", pd.Series(dtype=str))
                    ).dt.date
                )
                primary_missing = _missing_provider_sessions(
                    accumulated,
                    required_sessions=required_sessions,
                )
                primary_explained = set(
                    _explained_missing_daily_ohlcv_sessions(
                        instrument,
                        missing_sessions=primary_missing,
                        expected_session=expected,
                    )
                )
                terminal_fallback_required_sessions.clear()
                terminal_fallback_required_sessions.update(
                    session.isoformat()
                    for session in primary_missing
                    if session.isoformat() not in primary_explained
                )
                supplemented = _supplement_observation_receipts(
                    accumulated,
                    receipts=fallback_receipts,
                    instrument=instrument,
                    requested_start=start,
                    requested_end=end,
                    expected_session=expected,
                    artifact_root=artifact_root,
                    source_policy=source_policy,
                )
                terminal_observation_receipts.extend(
                    supplemented.attrs.get("observation_receipts", [])
                )
                missing = _missing_provider_sessions(
                    supplemented,
                    required_sessions=required_sessions,
                )
                explained = set(
                    _explained_missing_daily_ohlcv_sessions(
                        instrument,
                        missing_sessions=missing,
                        expected_session=expected,
                    )
                )
                unexplained = [
                    session
                    for session in missing
                    if session.isoformat() not in explained
                ]
                if not unexplained:
                    return supplemented
                if completeness_attempt < max_attempts:
                    sleeper(float(2 ** (completeness_attempt - 1)))
            raise ProviderBoundaryError(
                "EXPECTED_SESSION_COVERAGE_INCOMPLETE",
                "provider did not return every expected exchange session",
                diagnostic={
                    "ticker": ticker,
                    "request_start": start,
                    "request_end_exclusive": end,
                    "expected_sessions": [day.isoformat() for day in required_sessions],
                    "observed_sessions": sorted(
                        value.isoformat()
                        for value in pd.to_datetime(
                            supplemented.get("Date", pd.Series(dtype=str))
                        ).dt.date
                    ),
                    "missing_sessions": [
                        day.isoformat() for day in unexplained
                    ],
                    "attempts": max_attempts,
                    "disposition": "blocked_not_persisted",
                },
            )
        except ObservationReceiptError as exc:
            error["reason_code"] = "FALLBACK_OBSERVATION_RECEIPT_INVALID"
            terminal_rejected_bar_diagnostics.append(
                {
                    "reason_code": "FALLBACK_OBSERVATION_RECEIPT_INVALID",
                    "message": str(exc),
                    "disposition": "blocked_not_persisted",
                }
            )
            raise ProviderBoundaryError(
                "FALLBACK_OBSERVATION_RECEIPT_INVALID", str(exc)
            ) from exc
        except ProviderBoundaryError as exc:
            error["reason_code"] = exc.reason_code
            if exc.diagnostic:
                terminal_rejected_bar_diagnostics.append(exc.diagnostic)
            raise

    result = refresh_one_instrument(
        instrument,
        price_history_root=price_root,
        cutoff_date=expected.isoformat(),
        overlap_calendar_days=overlap_calendar_days,
        provider=guarded_provider,
        missing_range_only=True,
    )
    if before_frame is not None and path.is_file() and result.get("file_changed"):
        after_frame = pd.read_csv(path)
        conflict = _historical_conflict(before_frame, after_frame)
        if conflict is not None:
            if before_bytes is not None:
                _atomic_write_bytes(path, before_bytes)
            result = {**result, "status": "historical_conflict", "file_changed": False, "rows_added": 0}
            error["reason_code"] = conflict

    final_validation = (
        _validate_price_history(path)
        if path.is_file()
        else {"status": "missing"}
    )
    expected_backfill_sessions = _expected_sessions_between(
        profile,
        (
            date.fromisoformat(previous_end) + timedelta(days=1)
            if isinstance(previous_end, str)
            else None
        ),
        expected,
    )
    observed_dates = set(final_validation.get("observation_dates") or ())
    observed_backfill_sessions = [
        session for session in expected_backfill_sessions if session in observed_dates
    ]
    missing_sessions = [
        session for session in expected_backfill_sessions if session not in observed_dates
    ]
    explained_missing_sessions = _explained_missing_daily_ohlcv_sessions(
        instrument,
        missing_sessions=missing_sessions,
        expected_session=expected,
    )
    unexplained_missing_sessions = [
        session
        for session in missing_sessions
        if session.isoformat() not in explained_missing_sessions
    ]
    if unexplained_missing_sessions and not error:
        if before_bytes is not None:
            _atomic_write_bytes(path, before_bytes)
        result = {
            **result,
            "status": "incomplete_expected_sessions",
            "file_changed": False,
            "rows_added": 0,
        }
        error["reason_code"] = "EXPECTED_SESSION_COVERAGE_INCOMPLETE"
        final_validation = before_validation
        observed_dates = set(final_validation.get("observation_dates") or ())
        observed_backfill_sessions = [
            session
            for session in expected_backfill_sessions
            if session in observed_dates
        ]
    resulting_end = final_validation.get("end_date")
    status, reason = _normalize_status(
        result,
        expected_session=expected,
        resulting_end=resulting_end,
        error_code=error.get("reason_code"),
        final_row_count=int(final_validation.get("row_count") or 0),
        had_existing_valid=before_validation.get("status") == "valid",
        no_session_expected=_no_new_session_expected(profile, run_at, expected),
    )
    history_coverage = _history_coverage(
        instrument,
        validation=final_validation,
        expected_session=expected,
    )
    if (
        history_coverage["history_coverage_reason_code"]
        == "LISTING_START_AFTER_FIRST_OBSERVATION"
    ):
        status = "failed"
        reason = history_coverage["history_coverage_reason_code"]
    provider_retrieval = (
        provider.diagnostics_for(source_symbol)
        if hasattr(provider, "diagnostics_for")
        else []
    )
    rejected_bar_diagnostics = list(terminal_rejected_bar_diagnostics)
    if hasattr(provider, "rejected_diagnostics_for"):
        rejected_bar_diagnostics.extend(
            provider.rejected_diagnostics_for(source_symbol)
        )
    unique_receipts = sorted(
        {
            str(row["receipt_sha256"]): dict(row)
            for row in terminal_observation_receipts
        }.values(),
        key=lambda row: (row["instrument_id"], row["session_date"]),
    )
    return {
        **base,
        "previous_last_observation": previous_end,
        "resulting_last_observation": resulting_end,
        "rows_added": int(result.get("rows_added") or 0),
        "previous_row_count": int(before_validation.get("row_count") or 0),
        "resulting_row_count": int(final_validation.get("row_count") or 0),
        "expected_backfill_sessions": [
            session.isoformat() for session in expected_backfill_sessions
        ],
        "observed_backfill_sessions": [
            session.isoformat() for session in observed_backfill_sessions
        ],
        "explained_missing_sessions": explained_missing_sessions,
        "validation_status": "valid" if final_validation.get("status") == "valid" else "blocked",
        "freshness_status": status,
        "reason_code": reason,
        "provider_retrieval": provider_retrieval,
        "rejected_bar_diagnostics": rejected_bar_diagnostics,
        "primary_observed_sessions": sorted(
            terminal_primary_observed_sessions
        ),
        "observation_receipts": unique_receipts,
        "observation_receipt_root": (
            observation_receipt_root(unique_receipts)
            if unique_receipts
            else None
        ),
        "fallback_required_sessions": sorted(
            terminal_fallback_required_sessions
        ),
        **history_coverage,
        **_retained_history_boundary(
            instrument,
            validation=final_validation,
        ),
        **_runtime_observation_fields(instrument, validation=final_validation),
        "persisted_file_checksum": final_validation.get("checksum"),
    }


def _non_refreshable_lifecycle_row(
    instrument: Mapping[str, Any],
    *,
    price_root: Path,
) -> dict[str, Any]:
    source_symbol = str(instrument["source_symbol"])
    path = price_root / f"{source_symbol}.csv"
    validation = (
        _validate_price_history(path)
        if path.is_file()
        else {"status": "missing"}
    )
    lifecycle_status = str(instrument["lifecycle_status"])
    is_retained = lifecycle_status == "inactive"
    valid = validation.get("status") == "valid"
    status = "not_expected"
    reason = (
        "INACTIVE_AFTER_COMPLETED_CORPORATE_ACTION"
        if is_retained
        else "PRE_LISTING_NOT_EXPECTED"
    )
    history_status = "retained_inactive" if is_retained else "not_applicable"
    history_reason = (
        "RETAINED_INACTIVE_HISTORY"
        if is_retained
        else "PRE_LISTING_NOT_APPLICABLE"
    )
    retained_boundary = _retained_history_boundary(
        instrument,
        validation=validation,
    )
    if is_retained and not valid:
        status = "failed"
        reason = "RETAINED_INACTIVE_HISTORY_INVALID"
        history_status = "not_applicable"
        history_reason = "RETAINED_INACTIVE_HISTORY_INVALID"
    elif is_retained and (
        retained_boundary["retained_history_boundary_status"] != "aligned"
    ):
        status = "failed"
        reason = retained_boundary["retained_history_boundary_reason_code"]
        history_status = "not_applicable"
        history_reason = reason
    elif not is_retained and valid:
        status = "failed"
        reason = "PRE_LISTING_HISTORY_UNEXPECTED"
        history_reason = "PRE_LISTING_HISTORY_UNEXPECTED"
    return {
        "ticker": str(instrument["symbol"]),
        "instrument_id": str(instrument["instrument_id"]),
        "exchange": str(instrument.get("exchange") or "UNKNOWN"),
        "market_timezone": None,
        "provider_identity": None,
        "previous_last_observation": validation.get("end_date"),
        "resulting_last_observation": validation.get("end_date"),
        "expected_completed_session": instrument.get("delisting_end_date"),
        "rows_added": 0,
        "previous_row_count": int(validation.get("row_count") or 0),
        "resulting_row_count": int(validation.get("row_count") or 0),
        "validation_status": (
            "valid"
            if valid
            and (
                not is_retained
                or retained_boundary["retained_history_boundary_status"]
                == "aligned"
            )
            else "blocked"
            if is_retained
            else "not_applicable"
        ),
        "freshness_status": status,
        "reason_code": reason,
        "primary_observed_sessions": [],
        "observation_receipts": [],
        "observation_receipt_root": None,
        "fallback_required_sessions": [],
        "history_coverage_status": history_status,
        "history_coverage_reason_code": history_reason,
        "persisted_file_path": (
            DATA_RELATIVE_ROOT / f"{source_symbol}.csv"
        ).as_posix(),
        "previous_file_checksum": validation.get("checksum"),
        "persisted_file_checksum": validation.get("checksum"),
        **_lifecycle_fields(instrument),
        **retained_boundary,
        **_runtime_observation_fields(instrument, validation=validation),
    }


def _inactive_history_needs_backfill(
    instrument: Mapping[str, Any],
    *,
    price_root: Path,
) -> bool:
    path = price_root / f"{instrument['source_symbol']}.csv"
    validation = _validate_price_history(path) if path.is_file() else {"status": "missing"}
    cutoff = instrument.get("last_trading_session")
    return (
        isinstance(cutoff, str)
        and (
            validation.get("status") != "valid"
            or not isinstance(validation.get("end_date"), str)
            or validation["end_date"] < cutoff
        )
    )


def _expected_sessions_between(
    profile: MarketProfile,
    start: date | None,
    end: date,
) -> list[date]:
    if start is None or start > end:
        return []
    sessions: list[date] = []
    cursor = start
    while cursor <= end:
        if _is_trading_session(cursor, profile.holiday_calendar):
            sessions.append(cursor)
        cursor += timedelta(days=1)
    return sessions


def _validate_published_observation_receipts(
    entry: Mapping[str, Any],
    *,
    staged_path: Path,
    baseline_path: Path | None,
    publication_root: Path,
    source_policy: Mapping[str, Any],
    lifecycle_cutoff: Any,
    instrument_exchange: str,
) -> str | None:
    receipts = entry.get("observation_receipts")
    required = entry.get("fallback_required_sessions")
    primary = entry.get("primary_observed_sessions")
    root = entry.get("observation_receipt_root")
    if (
        not isinstance(receipts, list)
        or not isinstance(required, list)
        or not isinstance(primary, list)
    ):
        return "PUBLISHED_OBSERVATION_RECEIPT_CONTRACT_INVALID"
    if required != sorted(set(required)) or primary != sorted(set(primary)):
        return "PUBLISHED_OBSERVATION_ACQUISITION_JOURNAL_INVALID"
    try:
        staged = pd.read_csv(staged_path, dtype={"Date": str})
        baseline = None
        if baseline_path is not None:
            baseline = (
                pd.read_csv(baseline_path, dtype={"Date": str})
                if baseline_path.is_file()
                else pd.DataFrame(columns=staged.columns)
            )
    except (OSError, ValueError):
        return "PUBLISHED_OBSERVATION_BASELINE_INVALID"
    staged_by_date = {str(row["Date"]): row for _, row in staged.iterrows()}
    if baseline is not None:
        baseline_by_date = {
            str(row["Date"]): row for _, row in baseline.iterrows()
        }
        added = sorted(set(staged_by_date) - set(baseline_by_date))
        changed_existing = sorted(
            session
            for session in set(staged_by_date).intersection(baseline_by_date)
            if not _canonical_csv_rows_equal(
                staged_by_date[session], baseline_by_date[session]
            )
        )
        if changed_existing:
            return "PUBLISHED_FALLBACK_HISTORICAL_REWRITE_INVALID"
        required_set = set(required)
        primary_added = set(primary).intersection(added)
        if required_set.union(primary_added) != set(added):
            return "PUBLISHED_OBSERVATION_ACQUISITION_JOURNAL_INVALID"
    if not receipts:
        if required:
            return "PUBLISHED_OBSERVATION_RECEIPT_MISSING"
        if root is not None:
            return "PUBLISHED_OBSERVATION_RECEIPT_ROOT_INVALID"
        return None
    try:
        replayed = replay_observation_receipts(
            receipts,
            artifact_root=publication_root,
            policy=source_policy,
        )
        if root != observation_receipt_root(replayed):
            return "PUBLISHED_OBSERVATION_RECEIPT_ROOT_INVALID"
    except ObservationReceiptError:
        return "PUBLISHED_OBSERVATION_RECEIPT_REPLAY_INVALID"
    receipt_sessions = [str(row["session_date"]) for row in replayed]
    if receipt_sessions != sorted(required) or len(receipt_sessions) != len(set(receipt_sessions)):
        return "PUBLISHED_OBSERVATION_RECEIPT_SET_MISMATCH"
    for receipt in replayed:
        session = str(receipt["session_date"])
        if (
            receipt.get("instrument_id") != entry.get("instrument_id")
            or receipt.get("ticker") != entry.get("ticker")
            or receipt.get("exchange") != instrument_exchange
            or session not in staged_by_date
            or (
                isinstance(lifecycle_cutoff, str)
                and session > lifecycle_cutoff
            )
            or not _receipt_matches_csv_row(receipt, staged_by_date[session])
        ):
            return "PUBLISHED_OBSERVATION_RECEIPT_CANONICAL_MISMATCH"
    return None


def _receipt_matches_csv_row(
    receipt: Mapping[str, Any], row: Mapping[str, Any]
) -> bool:
    try:
        return all(
            Decimal(str(receipt[receipt_key]))
            == Decimal(str(row[csv_key]))
            for receipt_key, csv_key in (
                ("open", "Open"),
                ("high", "High"),
                ("low", "Low"),
                ("close", "Close"),
                ("adj_close", "Adj Close"),
                ("volume", "Volume"),
            )
        )
    except (KeyError, InvalidOperation, ValueError):
        return False


def _canonical_csv_rows_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return _receipt_matches_csv_row(
        {
            "open": left.get("Open"),
            "high": left.get("High"),
            "low": left.get("Low"),
            "close": left.get("Close"),
            "adj_close": left.get("Adj Close"),
            "volume": left.get("Volume"),
        },
        right,
    )


def _supplement_observation_receipts(
    frame: pd.DataFrame,
    *,
    receipts: Sequence[Mapping[str, Any]],
    instrument: Mapping[str, Any],
    requested_start: str,
    requested_end: str,
    expected_session: date,
    artifact_root: Path,
    source_policy: Mapping[str, Any],
) -> pd.DataFrame:
    if not receipts:
        return frame
    replayed = replay_observation_receipts(
        receipts,
        artifact_root=artifact_root,
        policy=source_policy,
    )
    observed = {
        value.isoformat()
        for value in pd.to_datetime(frame.get("Date", pd.Series(dtype=str))).dt.date
    }
    evidence_rows = [
        row
        for row in replayed
        if row.get("instrument_id") == instrument.get("instrument_id")
        and row.get("ticker") == instrument.get("symbol")
        and row.get("exchange") == instrument.get("exchange")
        and requested_start <= str(row.get("session_date")) < requested_end
        and str(row.get("session_date")) not in observed
    ]
    if len(evidence_rows) != len(receipts):
        raise ProviderBoundaryError(
            "FALLBACK_OBSERVATION_RECEIPT_IDENTITY_MISMATCH",
            "fallback receipts do not exactly match the requested instrument window",
            diagnostic={
                "reason_code": (
                    "FALLBACK_OBSERVATION_RECEIPT_IDENTITY_MISMATCH"
                ),
                "ticker": instrument.get("symbol"),
                "request_start": requested_start,
                "request_end_exclusive": requested_end,
                "declared_receipt_sessions": sorted(
                    str(row.get("session_date")) for row in receipts
                ),
                "eligible_receipt_sessions": sorted(
                    str(row.get("session_date")) for row in evidence_rows
                ),
                "disposition": "blocked_not_persisted",
            },
        )
    if not evidence_rows:
        return frame
    additions = pd.DataFrame(
        [
            {
                "Date": row["session_date"],
                "Open": row["open"],
                "High": row["high"],
                "Low": row["low"],
                "Close": row["close"],
                "Adj Close": row["adj_close"],
                "Volume": row["volume"],
            }
            for row in evidence_rows
        ]
    )
    combined = pd.concat([frame, additions], ignore_index=True).sort_values("Date")
    validated = _validate_provider_frame(
        combined,
        expected_session,
        provider_symbol=str(instrument["source_symbol"]),
        lifecycle_context=instrument,
    )
    validated.attrs["observation_receipts"] = [dict(row) for row in receipts]
    return validated


def _merge_provider_attempts(
    current: pd.DataFrame,
    received: pd.DataFrame,
) -> pd.DataFrame:
    if current.empty:
        return received.copy()
    if received.empty:
        return current.copy()
    return (
        pd.concat([current, received], ignore_index=True)
        .drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )


def _missing_provider_sessions(
    frame: pd.DataFrame,
    *,
    required_sessions: Sequence[date],
) -> list[date]:
    observed = set(
        pd.to_datetime(frame.get("Date", pd.Series(dtype=str))).dt.date
    )
    return [session for session in required_sessions if session not in observed]


def _explained_missing_daily_ohlcv_sessions(
    instrument: Mapping[str, Any],
    *,
    missing_sessions: Sequence[date],
    expected_session: date,
) -> list[str]:
    if missing_sessions != [expected_session]:
        return []
    last_observed_text = instrument.get(
        "canonical_ohlcv_last_observed_session"
    )
    try:
        last_observed = (
            date.fromisoformat(last_observed_text)
            if isinstance(last_observed_text, str)
            else None
        )
    except ValueError:
        last_observed = None
    if (
        instrument.get("terminal_session_daily_ohlcv_status")
        != "no_valid_daily_ohlcv_bar_from_provider_as_of"
        or last_observed is None
        or last_observed >= expected_session
    ):
        return []
    evidence = instrument.get("observation_evidence")
    if not isinstance(evidence, Mapping) or evidence.get(
        "relevant_session"
    ) != expected_session.isoformat():
        return []
    return [expected_session.isoformat()]


def _history_coverage(
    instrument: Mapping[str, Any],
    *,
    validation: Mapping[str, Any],
    expected_session: date | None,
) -> dict[str, Any]:
    base = {
        "history_coverage_status": "not_applicable",
        "history_coverage_reason_code": "PRICE_HISTORY_INVALID",
        "history_coverage_boundary_date": (
            expected_session.isoformat() if expected_session else None
        ),
        "history_listing_boundary_type": None,
        "history_expected_session_count": None,
        "history_observed_session_count": None,
        "history_missing_session_count": None,
        "history_initial_session_lag": None,
        "history_session_coverage_ratio": None,
        "history_bounded_session_coverage_ratio": None,
        "history_required_session_coverage_ratio": None,
    }
    if validation.get("status") != "valid":
        return base
    row_count = int(validation.get("row_count") or 0)
    listing_start = instrument.get("listing_start_date")
    first_observation = validation.get("start_date")
    if not isinstance(listing_start, str):
        return {
            **base,
            "history_coverage_status": (
                "sufficient"
                if row_count >= DEFAULT_MIN_HISTORY_ROWS
                else "insufficient_unexplained"
            ),
            "history_coverage_reason_code": (
                "MINIMUM_ANALYTICAL_HISTORY_AVAILABLE"
                if row_count >= DEFAULT_MIN_HISTORY_ROWS
                else "INSUFFICIENT_HISTORY_WITHOUT_LISTING_EVIDENCE"
            ),
        }
    if not isinstance(first_observation, str) or first_observation < listing_start:
        return {
            **base,
            "history_coverage_reason_code": (
                "LISTING_START_AFTER_FIRST_OBSERVATION"
            ),
        }
    profile = _resolve_market_profile(instrument)
    if profile is None or expected_session is None:
        return {
            **base,
            "history_coverage_reason_code": (
                "HISTORY_COVERAGE_BOUNDARY_UNAVAILABLE"
            ),
        }
    listing_date = date.fromisoformat(listing_start)
    regular_way = instrument.get("regular_way_listing_date")
    boundary_type = (
        "when_issued_start"
        if isinstance(regular_way, str) and listing_start < regular_way
        else "regular_way_start"
    )
    first_expected = listing_date
    while not _is_trading_session(
        first_expected,
        profile.holiday_calendar,
    ):
        first_expected += timedelta(days=1)
    if expected_session < first_expected:
        return {
            **base,
            "history_coverage_reason_code": "FUTURE_LISTING_NOT_APPLICABLE",
            "history_listing_boundary_type": boundary_type,
        }
    expected_sessions = _trading_sessions(
        first_expected,
        expected_session,
        profile.holiday_calendar,
    )
    observed_dates = {
        day
        for day in validation.get("observation_dates", ())
        if isinstance(day, date)
    }
    observed_sessions = [
        day for day in expected_sessions if day in observed_dates
    ]
    observed_count = len(observed_sessions)
    expected_count = len(expected_sessions)
    missing_count = expected_count - observed_count
    first_observed_session = (
        observed_sessions[0] if observed_sessions else None
    )
    initial_lag = (
        expected_sessions.index(first_observed_session)
        if first_observed_session is not None
        else expected_count
    )
    coverage_ratio = (
        round(observed_count / expected_count, 8)
        if expected_count
        else 0.0
    )
    required_observed_count = expected_count - initial_lag
    bounded_ratio = (
        observed_count / required_observed_count
        if required_observed_count
        else 0.0
    )
    detail = {
        **base,
        "history_coverage_boundary_date": expected_session.isoformat(),
        "history_listing_boundary_type": boundary_type,
        "history_expected_session_count": expected_count,
        "history_observed_session_count": observed_count,
        "history_missing_session_count": missing_count,
        "history_initial_session_lag": initial_lag,
        "history_session_coverage_ratio": coverage_ratio,
        "history_bounded_session_coverage_ratio": round(
            bounded_ratio,
            8,
        ),
        "history_required_session_coverage_ratio": (
            REQUIRED_LISTING_SESSION_COVERAGE_RATIO
        ),
    }
    if str(validation.get("end_date")) < expected_session.isoformat():
        return {
            **detail,
            "history_coverage_status": "insufficient_unexplained",
            "history_coverage_reason_code": (
                "HISTORY_END_BEFORE_EXPECTED_SESSION"
            ),
        }
    if initial_lag > MAX_LISTING_START_SESSION_LAG:
        return {
            **detail,
            "history_coverage_status": "insufficient_unexplained",
            "history_coverage_reason_code": (
                "HISTORY_START_TOO_LATE_AFTER_LISTING"
            ),
        }
    missing_after_start = [
        day
        for day in expected_sessions[initial_lag:]
        if day not in observed_dates
    ]
    if missing_after_start:
        return {
            **detail,
            "history_coverage_status": "insufficient_unexplained",
            "history_coverage_reason_code": (
                "HISTORY_SESSION_GAPS_AFTER_LISTING"
            ),
        }
    if bounded_ratio < REQUIRED_LISTING_SESSION_COVERAGE_RATIO:
        return {
            **detail,
            "history_coverage_status": "insufficient_unexplained",
            "history_coverage_reason_code": (
                "HISTORY_SESSION_COVERAGE_BELOW_REQUIRED_RATIO"
            ),
        }
    return {
        **detail,
        "history_coverage_status": (
            "sufficient"
            if observed_count >= DEFAULT_MIN_HISTORY_ROWS
            else "limited_history"
        ),
        "history_coverage_reason_code": (
            "MINIMUM_ANALYTICAL_HISTORY_AVAILABLE"
            if observed_count >= DEFAULT_MIN_HISTORY_ROWS
            else "LIMITED_HISTORY_SINCE_LISTING"
        ),
    }


def _retained_history_boundary(
    instrument: Mapping[str, Any],
    *,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    lifecycle_status = str(instrument.get("lifecycle_status") or "")
    expected_end = instrument.get("last_trading_session")
    actual_end = validation.get("end_date")
    base = {
        "retained_history_boundary_status": "not_applicable",
        "retained_history_boundary_reason_code": (
            "RETAINED_HISTORY_BOUNDARY_NOT_APPLICABLE"
        ),
        "retained_history_expected_end_date": (
            expected_end if isinstance(expected_end, str) else None
        ),
        "retained_history_actual_end_date": (
            actual_end if isinstance(actual_end, str) else None
        ),
    }
    if lifecycle_status != "inactive":
        return base
    if (
        validation.get("status") != "valid"
        or not isinstance(expected_end, str)
        or not isinstance(actual_end, str)
    ):
        return {
            **base,
            "retained_history_boundary_status": "invalid",
            "retained_history_boundary_reason_code": (
                "RETAINED_HISTORY_DATE_BOUNDARY_INVALID"
            ),
        }
    if actual_end < expected_end:
        if (
            instrument.get("terminal_session_daily_ohlcv_status")
            == "no_valid_daily_ohlcv_bar_from_provider_as_of"
            and instrument.get("canonical_ohlcv_last_observed_session")
            == actual_end
            and isinstance(instrument.get("observation_evidence"), Mapping)
            and instrument["observation_evidence"].get("relevant_session")
            == expected_end
        ):
            return {
                **base,
                "retained_history_boundary_status": "aligned",
                "retained_history_boundary_reason_code": (
                    "RETAINED_HISTORY_TERMINAL_DAILY_OHLCV_NOT_RETURNED_AS_OF"
                ),
            }
        return {
            **base,
            "retained_history_boundary_status": "ends_before",
            "retained_history_boundary_reason_code": (
                "RETAINED_HISTORY_ENDS_BEFORE_EXPECTED_SESSION"
            ),
        }
    if actual_end > expected_end:
        return {
            **base,
            "retained_history_boundary_status": "extends_after",
            "retained_history_boundary_reason_code": (
                "RETAINED_HISTORY_EXTENDS_AFTER_DELISTING"
            ),
        }
    return {
        **base,
        "retained_history_boundary_status": "aligned",
        "retained_history_boundary_reason_code": (
            "RETAINED_HISTORY_ENDS_ON_EXPECTED_SESSION"
        ),
    }


def _validate_price_history(path: Path) -> dict[str, Any]:
    validation = validate_price_history_csv(path, min_history_rows=1)
    if validation.get("status") != "valid":
        return validation
    frame = pd.read_csv(path, usecols=lambda column: column.strip().lower() == "date")
    date_column = next(
        (
            column
            for column in frame.columns
            if column.strip().lower() == "date"
        ),
        None,
    )
    if date_column is None:
        return {
            **validation,
            "status": "validation_failed",
            "note": "Price-history date column is unavailable.",
        }
    return {
        **validation,
        "observation_dates": tuple(
            pd.to_datetime(frame[date_column], errors="raise").dt.date
        ),
    }


def _row_fields_match(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    return all(row.get(key) == value for key, value in expected.items())


def _lifecycle_fields(instrument: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "lifecycle_status": instrument["lifecycle_status"],
        "lifecycle_record_status": instrument["lifecycle_record_status"],
        "lifecycle_status_effective_date": instrument[
            "lifecycle_status_effective_date"
        ],
        "listing_start_date": instrument["listing_start_date"],
        "regular_way_listing_date": instrument["regular_way_listing_date"],
        "delisting_end_date": instrument["delisting_end_date"],
        "last_trading_session": instrument["last_trading_session"],
        "transaction_closing_date": instrument["transaction_closing_date"],
        "trading_suspension_effective_date": instrument[
            "trading_suspension_effective_date"
        ],
        "inactive_effective_date": instrument["inactive_effective_date"],
        "canonical_ohlcv_last_observed_session": instrument[
            "canonical_ohlcv_last_observed_session"
        ],
        "terminal_session_daily_ohlcv_status": instrument[
            "terminal_session_daily_ohlcv_status"
        ],
        "observation_status_as_of": instrument["observation_status_as_of"],
        "observation_evidence": instrument["observation_evidence"],
        "trading_suspension_effective_timing": instrument[
            "trading_suspension_effective_timing"
        ],
        "lifecycle_reason": instrument["lifecycle_reason"],
        "corporate_action_type": instrument["corporate_action_type"],
        "lifecycle_provenance_checksum": instrument[
            "lifecycle_provenance_checksum"
        ],
    }


def _runtime_observation_fields(
    instrument: Mapping[str, Any],
    *,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    actual_end = validation.get("end_date")
    formal_end = instrument.get("last_trading_session")
    if isinstance(actual_end, str) and actual_end == formal_end:
        return {
            "canonical_ohlcv_last_observed_session": actual_end,
            "terminal_session_daily_ohlcv_status": "observed_daily_ohlcv",
            "observation_status_as_of": None,
            "observation_evidence": None,
        }
    return {
        "canonical_ohlcv_last_observed_session": instrument.get(
            "canonical_ohlcv_last_observed_session"
        ),
        "terminal_session_daily_ohlcv_status": instrument.get(
            "terminal_session_daily_ohlcv_status"
        ),
        "observation_status_as_of": instrument.get("observation_status_as_of"),
        "observation_evidence": instrument.get("observation_evidence"),
    }


def _provider_with_retries(
    provider: Provider,
    symbol: str,
    start: str,
    end: str,
    *,
    expected_session: date,
    max_attempts: int,
    sleeper: Sleeper,
    lifecycle_context: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    last_error: ProviderBoundaryError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            frame = provider(symbol, start, end)
            try:
                return _validate_provider_frame(
                    frame,
                    expected_session,
                    provider_symbol=symbol,
                    retry_number=attempt,
                    lifecycle_context=lifecycle_context,
                )
            except ProviderBoundaryError as exc:
                if (
                    exc.reason_code == "PROVIDER_OHLC_INVALID"
                    and hasattr(provider, "revalidate_single_ticker")
                ):
                    provider.record_original_invalid_bar(symbol, exc.diagnostic)
                    return provider.revalidate_single_ticker(
                        symbol,
                        start,
                        end,
                        expected_session=expected_session,
                        max_attempts=max_attempts,
                        sleeper=sleeper,
                        lifecycle_context=lifecycle_context,
                    )
                raise
        except ProviderBoundaryError:
            raise
        except (TimeoutError, ConnectionError) as exc:
            last_error = ProviderBoundaryError("PROVIDER_TIMEOUT", type(exc).__name__)
        except Exception as exc:
            text = str(exc).lower()
            reason = "PROVIDER_RATE_LIMITED" if "429" in text or "rate limit" in text else "PROVIDER_ERROR"
            last_error = ProviderBoundaryError(reason, type(exc).__name__)
        if attempt < max_attempts:
            sleeper(float(2 ** (attempt - 1)))
    assert last_error is not None
    raise last_error


def _prefetch_batch_provider(
    instruments: Sequence[Mapping[str, Any]],
    *,
    price_root: Path,
    run_at: datetime,
    max_attempts: int,
    sleeper: Sleeper,
) -> Provider:
    requests: list[tuple[str, date, date]] = []
    for instrument in instruments:
        if instrument.get("source_mapping_status") != "mapped":
            continue
        _profile, expected = expected_completed_session(instrument, run_at)
        if expected is None:
            continue
        path = price_root / f"{instrument['source_symbol']}.csv"
        validation = (
            _validate_price_history(path)
            if path.is_file()
            else {"status": "missing"}
        )
        if validation.get("status") == "valid" and validation.get("end_date") >= expected.isoformat():
            continue
        start = (
            date.fromisoformat(str(validation["end_date"])) + timedelta(days=1)
            if validation.get("status") == "valid"
            else date(2025, 1, 1)
        )
        requests.append((_to_yfinance_symbol(str(instrument["source_symbol"])), start, expected))
    if not requests:
        return _download_yfinance_history
    symbols = [row[0] for row in requests]
    start = min(row[1] for row in requests).isoformat()
    end = (max(row[2] for row in requests) + timedelta(days=1)).isoformat()
    cache: dict[str, pd.DataFrame] = {}
    terminal_failures: dict[str, ProviderBoundaryError] = {}
    diagnostics: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in symbols}
    rejected_diagnostics: dict[str, list[dict[str, Any]]] = {
        symbol: [] for symbol in symbols
    }

    def fetch_batch(batch_symbols: Sequence[str], *, split_depth: int) -> None:
        unresolved = list(batch_symbols)
        for attempt in range(1, max_attempts + 1):
            attempt_symbols = list(unresolved)
            attempt_received: dict[str, pd.DataFrame] = {}
            attempt_failure: ProviderBoundaryError | None = None
            try:
                attempt_received = download_yfinance_batch(
                    attempt_symbols,
                    start,
                    end,
                )
            except (TimeoutError, ConnectionError) as exc:
                attempt_failure = ProviderBoundaryError(
                    "PROVIDER_TIMEOUT",
                    type(exc).__name__,
                )
            except Exception as exc:
                detail = str(exc).lower()
                reason = (
                    "PROVIDER_RATE_LIMITED"
                    if "429" in detail or "rate limit" in detail
                    else "PROVIDER_ERROR"
                )
                attempt_failure = ProviderBoundaryError(reason, type(exc).__name__)
            missing = [
                symbol
                for symbol in attempt_symbols
                if attempt_received.get(symbol, pd.DataFrame()).empty
            ]
            classification = (
                attempt_failure.reason_code
                if attempt_failure is not None
                else "PROVIDER_BATCH_EMPTY"
                if len(missing) == len(attempt_symbols)
                else "PROVIDER_BATCH_PARTIAL"
                if missing
                else "PROVIDER_BATCH_COMPLETE"
            )
            for symbol in attempt_symbols:
                diagnostics[symbol].append(
                    {
                        "request_mode": "batch",
                        "batch_symbols": attempt_symbols,
                        "attempt": attempt,
                        "split_depth": split_depth,
                        "classification": classification,
                        "received_bar_count": int(
                            len(attempt_received.get(symbol, pd.DataFrame()))
                        ),
                    }
                )
            if attempt_failure is None:
                for symbol in attempt_symbols:
                    frame = attempt_received.get(symbol, pd.DataFrame())
                    if frame.empty:
                        continue
                    cache[symbol] = frame.copy()
                    terminal_failures.pop(symbol, None)
                unresolved = missing
            if not unresolved:
                return
            if attempt < max_attempts:
                sleeper(float(2 ** (attempt - 1)))

        if not unresolved:
            return
        if len(unresolved) > 1:
            midpoint = len(unresolved) // 2
            fetch_batch(unresolved[:midpoint], split_depth=split_depth + 1)
            fetch_batch(unresolved[midpoint:], split_depth=split_depth + 1)
            return
        symbol = unresolved[0]
        last_failure: ProviderBoundaryError | None = None
        for attempt in range(1, max_attempts + 1):
            frame = pd.DataFrame()
            attempt_failure: ProviderBoundaryError | None = None
            try:
                frame = _download_yfinance_history(symbol, start, end)
            except (TimeoutError, ConnectionError) as exc:
                attempt_failure = ProviderBoundaryError(
                    "PROVIDER_TIMEOUT",
                    type(exc).__name__,
                )
            except Exception as exc:
                detail = str(exc).lower()
                reason = (
                    "PROVIDER_RATE_LIMITED"
                    if "429" in detail or "rate limit" in detail
                    else "PROVIDER_ERROR"
                )
                attempt_failure = ProviderBoundaryError(reason, type(exc).__name__)
            classification = (
                attempt_failure.reason_code
                if attempt_failure is not None
                else "PROVIDER_SINGLE_TICKER_EMPTY"
                if frame.empty
                else "PROVIDER_SINGLE_TICKER_COMPLETE"
            )
            diagnostics[symbol].append(
                {
                    "request_mode": "single_ticker",
                    "batch_symbols": [symbol],
                    "attempt": attempt,
                    "split_depth": split_depth + 1,
                    "classification": classification,
                    "received_bar_count": int(len(frame)),
                }
            )
            if attempt_failure is not None:
                last_failure = attempt_failure
            elif not frame.empty:
                cache[symbol] = frame.copy()
                terminal_failures.pop(symbol, None)
                return
            else:
                last_failure = None
            if attempt < max_attempts:
                sleeper(float(2 ** (attempt - 1)))
        if last_failure is not None:
            terminal_failures[symbol] = last_failure

    fetch_batch(symbols, split_depth=0)

    def cached_provider(symbol: str, requested_start: str, requested_end: str) -> pd.DataFrame:
        provider_symbol = _to_yfinance_symbol(str(symbol))
        failure = terminal_failures.get(provider_symbol)
        if failure is not None:
            raise failure
        frame = cache.get(provider_symbol, pd.DataFrame()).copy()
        if frame.empty:
            return frame
        dates = pd.to_datetime(frame["Date"])
        return frame[(dates >= pd.Timestamp(requested_start)) & (dates < pd.Timestamp(requested_end))].copy()

    def record_original_invalid_bar(
        symbol: str,
        diagnostic: Mapping[str, Any],
    ) -> None:
        provider_symbol = _to_yfinance_symbol(str(symbol))
        rejected_diagnostics[provider_symbol].append(dict(diagnostic))
        diagnostics[provider_symbol].append(
            {
                "request_mode": "validation",
                "phase": "original_invalid_bar",
                "batch_symbols": [provider_symbol],
                "attempt": diagnostic.get("retry_number"),
                "split_depth": None,
                "classification": "original_invalid_bar",
                "received_bar_count": 1,
            }
        )

    def revalidate_single_ticker(
        symbol: str,
        requested_start: str,
        requested_end: str,
        *,
        expected_session: date,
        max_attempts: int,
        sleeper: Sleeper,
        lifecycle_context: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        provider_symbol = _to_yfinance_symbol(str(symbol))
        terminal_error: ProviderBoundaryError | None = None
        terminal_classification = "single_ticker_refetch_missing"
        for attempt in range(1, max_attempts + 1):
            frame = pd.DataFrame()
            attempt_failure: ProviderBoundaryError | None = None
            try:
                frame = _download_yfinance_history(
                    provider_symbol,
                    requested_start,
                    requested_end,
                )
            except (TimeoutError, ConnectionError) as exc:
                attempt_failure = ProviderBoundaryError(
                    "PROVIDER_TIMEOUT",
                    type(exc).__name__,
                )
            except Exception as exc:
                detail = str(exc).lower()
                reason = (
                    "PROVIDER_RATE_LIMITED"
                    if "429" in detail or "rate limit" in detail
                    else "PROVIDER_ERROR"
                )
                attempt_failure = ProviderBoundaryError(reason, type(exc).__name__)

            if attempt_failure is not None:
                terminal_error = attempt_failure
                terminal_classification = "single_ticker_refetch_provider_failure"
                received_bar_count = 0
            elif frame.empty:
                terminal_error = None
                terminal_classification = "single_ticker_refetch_missing"
                received_bar_count = 0
            else:
                received_bar_count = int(len(frame))
                try:
                    validated = _validate_provider_frame(
                        frame,
                        expected_session,
                        provider_symbol=provider_symbol,
                        retry_number=attempt,
                        lifecycle_context=lifecycle_context,
                    )
                except ProviderBoundaryError as exc:
                    terminal_error = exc
                    terminal_classification = "single_ticker_refetch_invalid"
                    if exc.diagnostic:
                        rejected_diagnostics[provider_symbol].append(
                            dict(exc.diagnostic)
                        )
                else:
                    diagnostics[provider_symbol].append(
                        {
                            "request_mode": "single_ticker_revalidation",
                            "phase": "single_ticker_refetch_attempt",
                            "batch_symbols": [provider_symbol],
                            "attempt": attempt,
                            "split_depth": None,
                            "classification": "single_ticker_refetch_valid",
                            "received_bar_count": received_bar_count,
                        }
                    )
                    return validated

            diagnostics[provider_symbol].append(
                {
                    "request_mode": "single_ticker_revalidation",
                    "phase": "single_ticker_refetch_attempt",
                    "batch_symbols": [provider_symbol],
                    "attempt": attempt,
                    "split_depth": None,
                    "classification": terminal_classification,
                    "received_bar_count": received_bar_count,
                }
            )
            if attempt < max_attempts:
                sleeper(float(2 ** (attempt - 1)))

        if terminal_error is not None:
            raise terminal_error
        raise ProviderBoundaryError(
            "EXPECTED_SESSION_NOT_AVAILABLE",
            "single-ticker OHLC revalidation returned no session",
        )

    cached_provider.diagnostics_for = lambda symbol: list(
        diagnostics.get(_to_yfinance_symbol(str(symbol)), [])
    )
    cached_provider.rejected_diagnostics_for = lambda symbol: list(
        rejected_diagnostics.get(_to_yfinance_symbol(str(symbol)), [])
    )
    cached_provider.record_original_invalid_bar = record_original_invalid_bar
    cached_provider.revalidate_single_ticker = revalidate_single_ticker
    return cached_provider


def _validate_provider_frame(
    frame: pd.DataFrame | None,
    expected_session: date,
    *,
    provider_symbol: str | None = None,
    retry_number: int | None = None,
    lifecycle_context: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    required = {"Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"}
    if not required.issubset(frame.columns):
        raise ProviderBoundaryError("PROVIDER_PAYLOAD_SCHEMA_INVALID", "required OHLCV fields are missing")
    try:
        dates = pd.to_datetime(frame["Date"], errors="raise").dt.date
    except (TypeError, ValueError) as exc:
        raise ProviderBoundaryError("PROVIDER_PAYLOAD_MALFORMED", "provider dates are invalid") from exc
    if dates.duplicated().any():
        raise ProviderBoundaryError("PROVIDER_DUPLICATE_TIMESTAMP", "provider dates are duplicated")
    if list(dates) != sorted(dates):
        raise ProviderBoundaryError("PROVIDER_PAYLOAD_NOT_CHRONOLOGICAL", "provider dates are not ordered")
    post_cutoff = [day for day in dates if day > expected_session]
    if post_cutoff:
        if not lifecycle_context or not lifecycle_context.get("last_trading_session"):
            raise ProviderBoundaryError(
                "PROVIDER_FUTURE_DATED_BAR",
                "provider returned a future or incomplete bar",
            )
        diagnostics = [
            {
                "ticker": str(lifecycle_context["symbol"]),
                "session_date": day.isoformat(),
                "cutoff_date": expected_session.isoformat(),
                "last_trading_session": lifecycle_context[
                    "last_trading_session"
                ],
                "canonical_ohlcv_last_observed_session": lifecycle_context[
                    "canonical_ohlcv_last_observed_session"
                ],
                "lifecycle_event": str(lifecycle_context["lifecycle_reason"]),
                "lifecycle_provenance_checksum": lifecycle_context[
                    "lifecycle_provenance_checksum"
                ],
                "provider": PROVIDER_IDENTITY,
                "retry_number": retry_number,
                "final_reason_code": "PROVIDER_BAR_AFTER_LIFECYCLE_CUTOFF",
                "disposition": "quarantined_not_persisted",
            }
            for day in post_cutoff
        ]
        frame = frame.loc[
            [day <= expected_session for day in dates]
        ].copy()
        frame.attrs["rejected_bar_diagnostics"] = diagnostics
        dates = pd.to_datetime(frame["Date"], errors="raise").dt.date
    for column in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or any(not math.isfinite(float(value)) for value in values):
            raise ProviderBoundaryError("PROVIDER_PAYLOAD_MALFORMED", f"provider {column} values are invalid")
    fetch_timestamp = _utc_text(datetime.now(UTC))
    for row in frame.itertuples(index=False):
        open_price = float(getattr(row, "Open"))
        high = float(getattr(row, "High"))
        low = float(getattr(row, "Low"))
        close = float(getattr(row, "Close"))
        violations = _ohlc_violations(
            open_price=open_price,
            high=high,
            low=low,
            close=close,
        )
        if violations:
            raw = {
                "open": getattr(row, "Open"),
                "high": getattr(row, "High"),
                "low": getattr(row, "Low"),
                "close": getattr(row, "Close"),
                "volume": getattr(row, "Volume"),
            }
            canonicalized = {
                key: float(value) if key != "volume" else int(value)
                for key, value in raw.items()
            }
            raise ProviderBoundaryError(
                "PROVIDER_OHLC_INVALID",
                "provider OHLC relationship is invalid",
                diagnostic={
                    "ticker": provider_symbol,
                    "session_date": pd.Timestamp(getattr(row, "Date")).date().isoformat(),
                    "provider": PROVIDER_IDENTITY,
                    "fetch_timestamp": fetch_timestamp,
                    "raw_ohlcv": raw,
                    "canonicalized_ohlcv": canonicalized,
                    "numeric_representation": {
                        key: type(value).__name__ for key, value in raw.items()
                    },
                    "violations": violations,
                    "retry_number": retry_number,
                    "final_reason_code": "PROVIDER_OHLC_INVALID",
                },
            )
    return frame


def _ohlc_violations(
    *,
    open_price: float,
    high: float,
    low: float,
    close: float,
) -> list[dict[str, Any]]:
    checks = (
        ("LOW_ABOVE_OPEN", low, open_price, low - open_price),
        ("LOW_ABOVE_CLOSE", low, close, low - close),
        ("HIGH_BELOW_OPEN", high, open_price, open_price - high),
        ("HIGH_BELOW_CLOSE", high, close, close - high),
        ("HIGH_BELOW_LOW", high, low, low - high),
    )
    violations = []
    for relation, left, right, absolute_deviation in checks:
        if absolute_deviation <= 0:
            continue
        denominator = max(abs(right), abs(left))
        violations.append(
            {
                "relation": relation,
                "left_value": left,
                "right_value": right,
                "absolute_deviation": absolute_deviation,
                "relative_deviation": (
                    absolute_deviation / denominator if denominator else None
                ),
            }
        )
    return violations


def _historical_conflict(before: pd.DataFrame, after: pd.DataFrame) -> str | None:
    if len(after) < len(before):
        return "HISTORY_TRUNCATION_BLOCKED"
    before_copy = before.copy()
    after_copy = after.copy()
    before_copy["Date"] = pd.to_datetime(before_copy["Date"]).dt.strftime("%Y-%m-%d")
    after_copy["Date"] = pd.to_datetime(after_copy["Date"]).dt.strftime("%Y-%m-%d")
    indexed = after_copy.set_index("Date")
    columns = [column for column in before_copy.columns if column != "Date"]
    for _, row in before_copy.iterrows():
        day = row["Date"]
        if day not in indexed.index:
            return "HISTORY_TRUNCATION_BLOCKED"
        current = indexed.loc[day]
        if isinstance(current, pd.DataFrame):
            return "HISTORY_DUPLICATE_DATE_BLOCKED"
        for column in columns:
            if not _equivalent_value(row[column], current[column]):
                return "HISTORICAL_VALUE_REWRITE_BLOCKED"
    return None


def _normalize_status(
    result: Mapping[str, Any],
    *,
    expected_session: date,
    resulting_end: str | None,
    error_code: str | None,
    final_row_count: int,
    had_existing_valid: bool,
    no_session_expected: bool,
) -> tuple[str, str]:
    status = result.get("status")
    if error_code:
        return "failed", error_code
    if status == "already_current":
        return "already_current", "NO_NEW_SESSION_EXPECTED" if no_session_expected else "ALREADY_CURRENT"
    if status in {"incrementally_updated", "new_snapshot_created", "full_rebuild_completed"}:
        if resulting_end is None or resulting_end < expected_session.isoformat():
            return "stale", "EXPECTED_SESSION_NOT_AVAILABLE"
        return "updated", "VALIDATED_UPDATE_PERSISTED"
    if status in {"stale_after_update", "empty_provider_response"}:
        return (
            "stale",
            "EXPECTED_SESSION_NOT_AVAILABLE" if had_existing_valid else "LOCAL_HISTORY_MISSING_AND_PROVIDER_EMPTY",
        )
    if status == "insufficient_history":
        if resulting_end is not None and resulting_end >= expected_session.isoformat():
            return "already_current", "ALREADY_CURRENT"
        return "stale", "EXPECTED_SESSION_NOT_AVAILABLE"
    if status == "unsupported_mapping":
        return "unsupported", "PROVIDER_MAPPING_MISSING"
    if status == "historical_conflict":
        return "failed", error_code or "HISTORICAL_VALUE_REWRITE_BLOCKED"
    if status == "validation_failed":
        return "failed", "PRICE_VALIDATION_FAILED"
    if status == "merge_failed":
        return "failed", "PRICE_MERGE_FAILED"
    return "failed", "PROVIDER_ERROR"


def _no_new_session_expected(profile: MarketProfile, run_at: datetime, expected: date) -> bool:
    local_now = _as_utc(run_at).astimezone(ZoneInfo(profile.timezone))
    candidate = local_now.date()
    if local_now.timetz().replace(tzinfo=None) < profile.close_time:
        candidate -= timedelta(days=1)
    return candidate > expected


def _resolve_market_profile(instrument: Mapping[str, Any]) -> MarketProfile | None:
    raw = str(instrument.get("exchange") or instrument.get("market") or "").upper()
    key = EXCHANGE_ALIASES.get(raw, raw)
    if key in MARKET_PROFILES:
        return MARKET_PROFILES[key]
    country = str(instrument.get("country") or "").upper()
    if raw in {"", "UNKNOWN"} and country == "US":
        return MARKET_PROFILES["US"]
    country_exchange = {"NL": "XAMS", "BE": "XBRU", "FR": "XPAR", "DE": "XETR", "GB": "XLON"}.get(country)
    return MARKET_PROFILES.get(country_exchange) if country_exchange else None


def _is_trading_session(day: date, calendar_name: str) -> bool:
    if day.weekday() >= 5:
        return False
    holidays = (
        _us_equity_holidays(day.year)
        if calendar_name == "us_equities"
        else _uk_equity_holidays(day.year)
        if calendar_name == "uk_equities"
        else _continental_equity_holidays(day.year)
    )
    return day not in holidays


def _trading_sessions(
    start: date,
    end: date,
    calendar_name: str,
) -> list[date]:
    sessions: list[date] = []
    cursor = start
    while cursor <= end:
        if _is_trading_session(cursor, calendar_name):
            sessions.append(cursor)
        cursor += timedelta(days=1)
    return sessions


def _us_equity_holidays(year: int) -> set[date]:
    easter = _easter_sunday(year)
    return {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        easter - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed(date(year, 6, 19)),
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed(date(year, 12, 25)),
    }


def _continental_equity_holidays(year: int) -> set[date]:
    easter = _easter_sunday(year)
    return {
        date(year, 1, 1),
        easter - timedelta(days=2),
        easter + timedelta(days=1),
        date(year, 5, 1),
        date(year, 12, 25),
        date(year, 12, 26),
    }


def _uk_equity_holidays(year: int) -> set[date]:
    easter = _easter_sunday(year)
    holidays = {
        _observed(date(year, 1, 1)),
        easter - timedelta(days=2),
        easter + timedelta(days=1),
        _nth_weekday(year, 5, 0, 1),
        _last_weekday(year, 5, 0),
        _last_weekday(year, 8, 0),
        _observed(date(year, 12, 25)),
        _observed(date(year, 12, 26)),
    }
    if _observed(date(year, 12, 25)) == _observed(date(year, 12, 26)):
        holidays.add(date(year, 12, 28))
    return holidays


def _easter_sunday(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    cursor = date(year, month, 1)
    cursor += timedelta(days=(weekday - cursor.weekday()) % 7)
    return cursor + timedelta(weeks=occurrence - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    cursor = date(year + (month == 12), 1 if month == 12 else month + 1, 1) - timedelta(days=1)
    return cursor - timedelta(days=(cursor.weekday() - weekday) % 7)


def _observed(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _prepare_staging_root(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        raise ScheduledPriceRefreshError("published and staging roots must differ")
    if destination.exists() and any(destination.iterdir()):
        raise ScheduledPriceRefreshError("staging root must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    source_data = source / DATA_RELATIVE_ROOT
    if source_data.is_dir():
        shutil.copytree(source_data, destination / DATA_RELATIVE_ROOT, dirs_exist_ok=True)
    source_evidence = source / "evidence" / "market_price"
    destination_evidence = destination / "evidence" / "market_price"
    destination_evidence.mkdir(parents=True, exist_ok=True)
    if source_evidence.is_dir():
        shutil.copytree(
            source_evidence,
            destination_evidence,
            dirs_exist_ok=True,
        )


def _expected_session_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    summary: dict[str, str] = {}
    for row in rows:
        exchange = row.get("exchange")
        session = row.get("expected_completed_session")
        if isinstance(exchange, str) and isinstance(session, str):
            summary[exchange] = max(session, summary.get(exchange, session))
    return dict(sorted(summary.items()))


def _entry_lifecycle_matches(
    entry: Mapping[str, Any],
    instrument: Mapping[str, Any],
) -> bool:
    expected = {
        key: value
        for key, value in _lifecycle_fields(instrument).items()
        if key
        not in {
            "canonical_ohlcv_last_observed_session",
            "terminal_session_daily_ohlcv_status",
            "observation_status_as_of",
            "observation_evidence",
        }
    }
    return all(entry.get(key) == value for key, value in expected.items())


def _lifecycle_registry_for_universe(
    universe_snapshot_path: str | Path,
    lifecycle_registry_path: str | Path | None,
) -> dict[str, Any]:
    if lifecycle_registry_path is not None:
        return load_lifecycle_registry(lifecycle_registry_path)
    if Path(universe_snapshot_path) == DEFAULT_UNIVERSE_SNAPSHOT:
        return load_lifecycle_registry(DEFAULT_LIFECYCLE_REGISTRY)
    payload = {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "records": [],
    }
    return {
        **payload,
        "registry_checksum": _canonical_checksum(payload),
        "records_by_instrument_id": {},
    }


def _load_optional_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return _load_json(path)
    except ScheduledPriceRefreshError:
        return {}


def _contains_executable_content(root: Path) -> bool:
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.is_symlink():
            return True
        try:
            relative = path.relative_to(root)
        except ValueError:
            return True
        is_price_file = relative.parent == DATA_RELATIVE_ROOT and relative.suffix.lower() == ".csv"
        is_raw_market_price_evidence = (
            len(relative.parts) == 4
            and relative.parts[:2] == ("evidence", "market_price")
            and relative.suffix.lower() == ".json"
        )
        if (
            relative != LATEST_MANIFEST
            and not is_price_file
            and not is_raw_market_price_evidence
        ):
            return True
    return False


def _validation_result(
    *,
    issues: Sequence[Mapping[str, str]],
    manifest: Mapping[str, Any] | None = None,
    price_history_root: Path | None = None,
    stale: Sequence[str] = (),
) -> dict[str, Any]:
    reason_codes = sorted({str(issue["reason_code"]) for issue in issues})
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "validated": not issues,
        "status": "validated" if not issues else "blocked",
        "reason_codes": reason_codes,
        "issues": list(issues),
        "manifest_run_id": manifest.get("run_id") if manifest else None,
        "price_history_root": price_history_root.as_posix() if price_history_root else None,
        "stale_tickers": sorted(stale),
    }


def _validation_issue(reason_code: str, path: str, detail: str = "") -> dict[str, str]:
    return {"reason_code": reason_code, "path": path, "detail": detail}


def _manifest_checksum(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_checksum", None)
    return _canonical_checksum(payload)


def _canonical_checksum(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_write_bytes(path, data.encode("utf-8"))


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ScheduledPriceRefreshError(f"strict JSON artifact is invalid: {source}") from exc
    if not isinstance(value, dict):
        raise ScheduledPriceRefreshError(f"JSON artifact must be an object: {source}")
    return value


def _required_text(value: Mapping[str, Any], key: str) -> str:
    text = value.get(key)
    if not isinstance(text, str) or not text.strip():
        raise ScheduledPriceRefreshError(f"required universe field is missing: {key}")
    return text.strip()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _equivalent_value(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return str(left) == str(right)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ScheduledPriceRefreshError("run timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _utc_text(value: datetime) -> str:
    return _as_utc(value).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh and validate the published canonical price dataset.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    refresh = subparsers.add_parser("refresh")
    refresh.add_argument("--run-id", required=True)
    refresh.add_argument("--source-main-sha", required=True)
    refresh.add_argument("--universe-snapshot", default=DEFAULT_UNIVERSE_SNAPSHOT.as_posix())
    refresh.add_argument(
        "--lifecycle-registry",
        default=DEFAULT_LIFECYCLE_REGISTRY.as_posix(),
    )
    refresh.add_argument("--published-root", required=True)
    refresh.add_argument("--staging-root", required=True)
    refresh.add_argument("--report-output", required=True)
    refresh.add_argument("--workflow-run-id")
    refresh.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    refresh.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    validate = subparsers.add_parser("validate-publication")
    validate.add_argument("--publication-root", required=True)
    validate.add_argument("--universe-snapshot", default=DEFAULT_UNIVERSE_SNAPSHOT.as_posix())
    validate.add_argument(
        "--lifecycle-registry",
        default=DEFAULT_LIFECYCLE_REGISTRY.as_posix(),
    )
    validate.add_argument("--allow-degraded", action="store_true")
    validate.add_argument("--expected-source-main-sha")
    validate.add_argument("--baseline-publication-root")
    validate.add_argument(
        "--source-policy",
        default=DEFAULT_SOURCE_POLICY.as_posix(),
    )
    consume = subparsers.add_parser("consume-analysis")
    consume.add_argument("--publication-root", required=True)
    consume.add_argument("--universe-snapshot", default=DEFAULT_UNIVERSE_SNAPSHOT.as_posix())
    consume.add_argument(
        "--lifecycle-registry",
        default=DEFAULT_LIFECYCLE_REGISTRY.as_posix(),
    )
    consume.add_argument("--run-id", required=True)
    consume.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(argv=argv, stdout=sys.stdout, stderr=sys.stderr)


def run_command(argv: Sequence[str] | None, *, stdout: TextIO, stderr: TextIO) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    if args.command == "refresh":
        try:
            report = run_scheduled_refresh(
                run_id=args.run_id,
                source_main_sha=args.source_main_sha,
                universe_snapshot_path=args.universe_snapshot,
                lifecycle_registry_path=args.lifecycle_registry,
                published_root=args.published_root,
                staging_root=args.staging_root,
                report_output=args.report_output,
                workflow_run_id=args.workflow_run_id,
                batch_size=args.batch_size,
                max_attempts=args.max_attempts,
            )
        except Exception as exc:
            diagnostic = {
                "schema_version": SCHEMA_VERSION,
                "run_id": args.run_id,
                "run_status": "failed",
                "reason_code": "GLOBAL_REFRESH_FAILURE",
                "error_type": type(exc).__name__,
                "approval_generated": False,
            }
            _atomic_write_json(Path(args.report_output), diagnostic)
            print(json.dumps(diagnostic, sort_keys=True), file=stderr)
            return 2
        print(
            json.dumps(
                {
                    "run_id": report["run_id"],
                    "run_status": report["run_status"],
                    "status_counts": report["status_counts"],
                    "publication": report["publication"],
                },
                sort_keys=True,
            ),
            file=stdout,
        )
        return 0 if report["run_status"] == "completed" else 1
    if args.command == "validate-publication":
        validation = validate_published_dataset(
            args.publication_root,
            universe_snapshot_path=args.universe_snapshot,
            lifecycle_registry_path=args.lifecycle_registry,
            allow_degraded=args.allow_degraded,
            expected_source_main_sha=args.expected_source_main_sha,
            baseline_publication_root=args.baseline_publication_root,
            source_policy_path=args.source_policy,
        )
        print(json.dumps(validation, sort_keys=True), file=stdout if validation["validated"] else stderr)
        return 0 if validation["validated"] else 2

    from market_engine.run.full_canonical_universe_analysis import run_full_canonical_universe_analysis

    result = run_validated_analysis(
        args.publication_root,
        universe_snapshot_path=args.universe_snapshot,
        lifecycle_registry_path=args.lifecycle_registry,
        analysis_runner=run_full_canonical_universe_analysis,
        analysis_kwargs={
            "run_id": args.run_id,
            "universe_path": "config/market_engine/universes/canonical_universe.json",
            "output_root": args.output_root,
        },
    )
    print(json.dumps(result, sort_keys=True, default=str), file=stdout if result["analysis_executed"] else stderr)
    return 0 if result["analysis_executed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
