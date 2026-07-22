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
from market_engine.data.local_market_data_universe import (
    DEFAULT_MIN_HISTORY_ROWS,
    UNIVERSE_SNAPSHOT_SCHEMA_VERSION,
    validate_price_history_csv,
)


SCHEMA_VERSION = "market-engine-me-sr17-canonical-price-freshness-manifest-v1"
VALIDATION_SCHEMA_VERSION = "market-engine-me-sr17-published-price-dataset-validation-v1"
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
STATUS_ORDER = ("updated", "already_current", "stale", "failed", "insufficient", "unsupported")
DEGRADED_STATUSES = frozenset({"stale", "failed", "insufficient", "unsupported"})
SHA1 = re.compile(r"^[0-9a-f]{40}$")

Provider = Callable[[str, str, str], pd.DataFrame]
Sleeper = Callable[[float], None]


class ScheduledPriceRefreshError(ValueError):
    pass


class ProviderBoundaryError(RuntimeError):
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


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
) -> dict[str, Any]:
    if not run_id or not SHA1.fullmatch(source_main_sha):
        raise ScheduledPriceRefreshError("run ID and full source main SHA are required")
    if batch_size < 1 or max_attempts < 1:
        raise ScheduledPriceRefreshError("batch size and maximum attempts must be positive")
    generated_at = _as_utc(run_at or datetime.now(UTC))
    universe = load_authoritative_universe(universe_snapshot_path)
    instruments = [row for row in universe["instruments"] if row.get("active", True)]
    source_root = Path(published_root)
    stage_root = Path(staging_root)
    _prepare_staging_root(source_root, stage_root)
    price_root = stage_root / DATA_RELATIVE_ROOT
    price_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
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
            rows.append(
                _refresh_instrument(
                    instrument,
                    price_root=price_root,
                    run_at=generated_at,
                    provider=selected_provider,
                    max_attempts=max_attempts,
                    overlap_calendar_days=overlap_calendar_days,
                    sleeper=sleeper,
                )
            )

    rows.sort(key=lambda row: (row["instrument_id"], row["ticker"]))
    counts = Counter(row["freshness_status"] for row in rows)
    updated_count = counts.get("updated", 0)
    mapped_rows = [row for row in rows if row["provider_identity"] is not None]
    publication_set_valid = all(
        row["validation_status"] == "valid" and isinstance(row.get("persisted_file_checksum"), str)
        for row in mapped_rows
    )
    degraded = any(counts.get(status, 0) for status in DEGRADED_STATUSES)
    run_status = "failed" if not publication_set_valid else "degraded" if degraded else "completed"
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": _utc_text(generated_at),
        "source_main_sha": source_main_sha,
        "workflow_run_id": workflow_run_id,
        "data_branch": DATA_BRANCH,
        "universe_version": universe["universe_version"],
        "universe_checksum": universe["universe_checksum"],
        "universe_size": len(instruments),
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
        "run_status": run_status,
        "publication": {
            "publication_set_valid": publication_set_valid,
            "publication_required": publication_set_valid and updated_count > 0,
            "changed_price_file_count": updated_count,
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
    return profile, candidate


def validate_published_dataset(
    publication_root: str | Path,
    *,
    universe_snapshot_path: str | Path = DEFAULT_UNIVERSE_SNAPSHOT,
    run_at: datetime | None = None,
    allow_degraded: bool = False,
    expected_source_main_sha: str | None = None,
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
    except ScheduledPriceRefreshError:
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
    active_instruments = [row for row in universe["instruments"] if row.get("active", True)]
    if (
        manifest.get("universe_version") != universe.get("universe_version")
        or manifest.get("universe_checksum") != universe.get("universe_checksum")
        or manifest.get("universe_size") != len(active_instruments)
    ):
        issues.append(_validation_issue("PUBLISHED_UNIVERSE_BINDING_MISMATCH", "universe"))
    if _contains_executable_content(root):
        issues.append(_validation_issue("PUBLISHED_DATA_BRANCH_CONTENT_INVALID", "publication_root"))

    publication = manifest.get("publication")
    manifest_status_counts = manifest.get("status_counts")
    manifest_updated_count = (
        manifest_status_counts.get("updated", 0) if isinstance(manifest_status_counts, Mapping) else 0
    )
    if not isinstance(publication, Mapping) or not (
        publication.get("publication_set_valid") is True
        and publication.get("publication_required") is True
        and isinstance(publication.get("changed_price_file_count"), int)
        and publication["changed_price_file_count"] == manifest_updated_count
        and manifest_updated_count > 0
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
    expected_ids = [row["instrument_id"] for row in active_instruments]
    actual_ids = [row.get("instrument_id") for row in entries if isinstance(row, Mapping)]
    if actual_ids != expected_ids:
        issues.append(_validation_issue("PUBLISHED_TICKER_SET_MISMATCH", "tickers"))
    expected_by_id = {row["instrument_id"]: row for row in active_instruments}
    actual_status_counts = Counter(
        str(row.get("freshness_status")) for row in entries if isinstance(row, Mapping)
    )
    expected_status_counts = {status: actual_status_counts.get(status, 0) for status in STATUS_ORDER}
    if (
        any(status not in STATUS_ORDER for status in actual_status_counts)
        or manifest_status_counts != expected_status_counts
    ):
        issues.append(_validation_issue("PUBLISHED_STATUS_COUNTS_MISMATCH", "status_counts"))
    bound_files = {
        str(row.get("persisted_file_path"))
        for row in entries
        if isinstance(row, Mapping) and isinstance(row.get("persisted_file_path"), str)
    }
    actual_files = {
        path.relative_to(root).as_posix()
        for path in (root / DATA_RELATIVE_ROOT).glob("*.csv")
        if path.is_file()
    }
    if actual_files != bound_files:
        issues.append(_validation_issue("PUBLISHED_UNBOUND_PRICE_FILE_SET", "data/processed"))

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
        ):
            issues.append(_validation_issue("PUBLISHED_TICKER_IDENTITY_MISMATCH", f"tickers[{index}]"))
            continue
        relative_path = Path(relative)
        if relative_path.is_absolute() or relative_path.parent != DATA_RELATIVE_ROOT or relative_path.suffix != ".csv":
            issues.append(_validation_issue("PUBLISHED_FILE_PATH_INVALID", f"tickers[{index}]"))
            continue
        path = root / relative
        if not path.is_file() or not isinstance(checksum, str) or _sha256_file(path) != checksum:
            issues.append(_validation_issue("PUBLISHED_FILE_CHECKSUM_MISMATCH", f"tickers[{index}]"))
            continue
        validation = validate_price_history_csv(path, min_history_rows=1)
        if validation.get("status") != "valid":
            issues.append(_validation_issue("PUBLISHED_PRICE_FILE_INVALID", f"tickers[{index}]"))
            continue
        if entry.get("validation_status") != "valid":
            issues.append(_validation_issue("PUBLISHED_TICKER_VALIDATION_STATUS_INVALID", f"tickers[{index}]"))
        resulting = entry.get("resulting_last_observation")
        if resulting != validation.get("end_date"):
            issues.append(_validation_issue("PUBLISHED_LAST_OBSERVATION_MISMATCH", f"tickers[{index}]"))
        profile, required = expected_completed_session(expected_instrument, validation_at)
        actual_end = validation.get("end_date")
        if profile is None or required is None or not isinstance(actual_end, str) or actual_end < required.isoformat():
            stale.append(str(entry.get("ticker") or entry.get("instrument_id") or index))
    if stale and not allow_degraded:
        issues.append(_validation_issue("PUBLISHED_DATASET_STALE", "tickers", ",".join(sorted(stale))))
    run_status = manifest.get("run_status")
    expected_run_status = (
        "degraded"
        if any(actual_status_counts.get(status, 0) for status in DEGRADED_STATUSES)
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
    run_at: datetime | None = None,
    analysis_runner: Callable[..., Any],
    analysis_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_published_dataset(
        publication_root,
        universe_snapshot_path=universe_snapshot_path,
        run_at=run_at,
    )
    if not validation["validated"]:
        return {"status": "blocked", "analysis_executed": False, "validation": validation}
    result = analysis_runner(
        price_history_root=Path(publication_root) / DATA_RELATIVE_ROOT,
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
) -> dict[str, Any]:
    ticker = str(instrument["symbol"])
    source_symbol = str(instrument["source_symbol"])
    path = price_root / f"{source_symbol}.csv"
    profile, expected = expected_completed_session(instrument, run_at)
    base = {
        "ticker": ticker,
        "instrument_id": str(instrument["instrument_id"]),
        "exchange": profile.market if profile else str(instrument.get("exchange") or "UNKNOWN"),
        "market_timezone": profile.timezone if profile else None,
        "provider_identity": PROVIDER_IDENTITY if instrument.get("source_mapping_status") == "mapped" else None,
        "previous_last_observation": None,
        "resulting_last_observation": None,
        "expected_completed_session": expected.isoformat() if expected else None,
        "rows_added": 0,
        "validation_status": "blocked",
        "freshness_status": "unsupported",
        "reason_code": "UNSUPPORTED_EXCHANGE" if profile is None else "PROVIDER_MAPPING_MISSING",
        "persisted_file_path": (DATA_RELATIVE_ROOT / f"{source_symbol}.csv").as_posix(),
        "persisted_file_checksum": _sha256_file(path) if path.is_file() else None,
    }
    if profile is None or expected is None:
        return base
    if instrument.get("source_mapping_status") != "mapped" or not source_symbol:
        return base

    before_bytes = path.read_bytes() if path.is_file() else None
    before_validation = validate_price_history_csv(path, min_history_rows=1) if path.is_file() else {"status": "missing"}
    before_frame = pd.read_csv(path) if before_validation.get("status") == "valid" else None
    previous_end = before_validation.get("end_date")
    error: dict[str, str] = {}

    def guarded_provider(symbol: str, start: str, end: str) -> pd.DataFrame:
        try:
            return _provider_with_retries(
                provider,
                symbol,
                start,
                end,
                expected_session=expected,
                max_attempts=max_attempts,
                sleeper=sleeper,
            )
        except ProviderBoundaryError as exc:
            error["reason_code"] = exc.reason_code
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

    final_validation = validate_price_history_csv(path, min_history_rows=1) if path.is_file() else {"status": "missing"}
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
    return {
        **base,
        "previous_last_observation": previous_end,
        "resulting_last_observation": resulting_end,
        "rows_added": int(result.get("rows_added") or 0),
        "validation_status": "valid" if final_validation.get("status") == "valid" else "blocked",
        "freshness_status": status,
        "reason_code": reason,
        "persisted_file_checksum": final_validation.get("checksum"),
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
) -> pd.DataFrame:
    last_error: ProviderBoundaryError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            frame = provider(symbol, start, end)
            return _validate_provider_frame(frame, expected_session)
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
        validation = validate_price_history_csv(path, min_history_rows=1) if path.is_file() else {"status": "missing"}
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
    cache: dict[str, pd.DataFrame] | None = None
    failure: ProviderBoundaryError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            cache = download_yfinance_batch(symbols, start, end)
            break
        except (TimeoutError, ConnectionError) as exc:
            failure = ProviderBoundaryError("PROVIDER_TIMEOUT", type(exc).__name__)
        except Exception as exc:
            detail = str(exc).lower()
            reason = "PROVIDER_RATE_LIMITED" if "429" in detail or "rate limit" in detail else "PROVIDER_ERROR"
            failure = ProviderBoundaryError(reason, type(exc).__name__)
        if attempt < max_attempts:
            sleeper(float(2 ** (attempt - 1)))

    def cached_provider(symbol: str, requested_start: str, requested_end: str) -> pd.DataFrame:
        if cache is None:
            assert failure is not None
            raise failure
        frame = cache.get(symbol, pd.DataFrame()).copy()
        if frame.empty:
            return frame
        dates = pd.to_datetime(frame["Date"])
        return frame[(dates >= pd.Timestamp(requested_start)) & (dates < pd.Timestamp(requested_end))].copy()

    return cached_provider


def _validate_provider_frame(frame: pd.DataFrame | None, expected_session: date) -> pd.DataFrame:
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
    if any(day > expected_session for day in dates):
        raise ProviderBoundaryError("PROVIDER_FUTURE_DATED_BAR", "provider returned a future or incomplete bar")
    for column in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or any(not math.isfinite(float(value)) for value in values):
            raise ProviderBoundaryError("PROVIDER_PAYLOAD_MALFORMED", f"provider {column} values are invalid")
    for row in frame.itertuples(index=False):
        open_price = float(getattr(row, "Open"))
        high = float(getattr(row, "High"))
        low = float(getattr(row, "Low"))
        close = float(getattr(row, "Close"))
        if low > min(open_price, close) or high < max(open_price, close) or high < low:
            raise ProviderBoundaryError("PROVIDER_OHLC_INVALID", "provider OHLC relationship is invalid")
    return frame


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
        if final_row_count < DEFAULT_MIN_HISTORY_ROWS:
            return "insufficient", "VALID_HISTORY_INSUFFICIENT_ROWS"
        return "already_current", "NO_NEW_SESSION_EXPECTED" if no_session_expected else "ALREADY_CURRENT"
    if status in {"incrementally_updated", "new_snapshot_created", "full_rebuild_completed"}:
        if resulting_end is None or resulting_end < expected_session.isoformat():
            return "stale", "EXPECTED_SESSION_NOT_AVAILABLE"
        if final_row_count < DEFAULT_MIN_HISTORY_ROWS:
            return "insufficient", "VALID_HISTORY_INSUFFICIENT_ROWS"
        return "updated", "VALIDATED_UPDATE_PERSISTED"
    if status in {"stale_after_update", "empty_provider_response"}:
        return (
            "stale",
            "EXPECTED_SESSION_NOT_AVAILABLE" if had_existing_valid else "LOCAL_HISTORY_MISSING_AND_PROVIDER_EMPTY",
        )
    if status == "insufficient_history":
        return "insufficient", "VALID_HISTORY_INSUFFICIENT_ROWS"
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


def _expected_session_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    summary: dict[str, str] = {}
    for row in rows:
        exchange = row.get("exchange")
        session = row.get("expected_completed_session")
        if isinstance(exchange, str) and isinstance(session, str):
            summary[exchange] = max(session, summary.get(exchange, session))
    return dict(sorted(summary.items()))


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
        if relative != LATEST_MANIFEST and not is_price_file:
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
    refresh.add_argument("--published-root", required=True)
    refresh.add_argument("--staging-root", required=True)
    refresh.add_argument("--report-output", required=True)
    refresh.add_argument("--workflow-run-id")
    refresh.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    refresh.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    validate = subparsers.add_parser("validate-publication")
    validate.add_argument("--publication-root", required=True)
    validate.add_argument("--universe-snapshot", default=DEFAULT_UNIVERSE_SNAPSHOT.as_posix())
    validate.add_argument("--allow-degraded", action="store_true")
    validate.add_argument("--expected-source-main-sha")
    consume = subparsers.add_parser("consume-analysis")
    consume.add_argument("--publication-root", required=True)
    consume.add_argument("--universe-snapshot", default=DEFAULT_UNIVERSE_SNAPSHOT.as_posix())
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
            allow_degraded=args.allow_degraded,
            expected_source_main_sha=args.expected_source_main_sha,
        )
        print(json.dumps(validation, sort_keys=True), file=stdout if validation["validated"] else stderr)
        return 0 if validation["validated"] else 2

    from market_engine.run.full_canonical_universe_analysis import run_full_canonical_universe_analysis

    result = run_validated_analysis(
        args.publication_root,
        universe_snapshot_path=args.universe_snapshot,
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
