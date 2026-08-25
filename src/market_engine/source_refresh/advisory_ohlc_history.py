from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TextIO

import pandas as pd

from market_engine.data.complete_local_market_dataset import _to_yfinance_symbol
from market_engine.data.incremental_market_data_refresh import (
    _download_yfinance_history,
    download_yfinance_batch,
)
from market_engine.data.scheduled_canonical_price_refresh import (
    DEFAULT_UNIVERSE_SNAPSHOT,
    expected_completed_session,
    load_authoritative_universe,
)


ARTIFACT_VERSION = "me-sr26-advisory-ohlc-history-v1"
MANIFEST_VERSION = "market-engine-advisory-ohlc-history-manifest-v1"
INDEX_VERSION = "market-engine-advisory-ohlc-history-index-v1"
SERIES_VERSION = "market-engine-advisory-ohlc-series-v1"
ELIGIBILITY_VERSION = "market-engine-advisory-ohlc-screening-eligibility-v1"
CHECKSUM_VERSION = "market-engine-checksum-index-v1"
POLICY_VERSION = "market-engine-advisory-ohlc-history-policy-v1"
SOURCE_ID = "yahoo-finance-yfinance"
ADAPTER_ID = "existing_yfinance_daily_history_adapter"
DEFAULT_POLICY_PATH = Path("config/market_engine/advisory_ohlc_history_policy.json")
DEFAULT_OUTPUT_ROOT = Path("artifacts/market_engine/advisory_ohlc_history_runs")
STATUSES = (
    "fresh", "stale", "insufficient_history", "missing", "invalid",
    "blocked_identity", "blocked_adjustment_policy",
)
DECIMAL_TEXT = re.compile(r"^(?:0\.[0-9]*[1-9][0-9]*|[1-9][0-9]*(?:\.[0-9]+)?)$")
INTEGER_TEXT = re.compile(r"^(?:0|[1-9][0-9]*)$")
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")


class AdvisoryHistoryIssue(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


Provider = Callable[[Sequence[Mapping[str, Any]], datetime, Mapping[str, Any]], Mapping[str, Mapping[str, Any]]]


@dataclass(frozen=True)
class _ValidatedHistoryContext:
    manifest: Mapping[str, Any]
    index: tuple[Mapping[str, Any], ...]
    series: Mapping[str, Mapping[str, Any]]
    effective_status: Mapping[str, str]
    universe: Mapping[str, Any]
    policy: Mapping[str, Any]
    root: Path


def build_advisory_ohlc_history(
    *, run_id: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[dict[str, Any], Path]:
    """Build history with repository-owned authority and internal provenance."""
    _load_canonical_universe()
    _load_canonical_policy()
    return _build_advisory_ohlc_history_impl(
        run_id=run_id,
        source_main_sha=_current_repository_head_sha(),
        output_root=output_root,
        universe_path=DEFAULT_UNIVERSE_SNAPSHOT,
        policy_path=DEFAULT_POLICY_PATH,
        provider=_acquire_with_existing_adapter,
        clock=_system_utc_now,
    )


def _build_advisory_ohlc_history_impl(
    *, run_id: str, source_main_sha: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    universe_path: str | Path = DEFAULT_UNIVERSE_SNAPSHOT,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    provider: Provider,
    clock: Callable[[], datetime],
) -> tuple[dict[str, Any], Path]:
    """Private deterministic construction seam; never a production authority API."""
    destination = _validate_output_destination(output_root, run_id)
    if not re.fullmatch(r"[0-9a-f]{40}", source_main_sha):
        raise AdvisoryHistoryIssue("SOURCE_MAIN_SHA_INVALID", "source_main_sha must be a full Git SHA")
    acquisition_started = _clock_now(clock)
    universe_source, policy_source = Path(universe_path), Path(policy_path)
    universe = load_authoritative_universe(universe_source)
    policy = _load_policy(policy_source)
    instruments = sorted(universe["instruments"], key=lambda row: str(row["instrument_id"]))
    expected = {str(row["instrument_id"]) for row in instruments}
    try:
        raw = dict(provider(instruments, acquisition_started, policy))
    except Exception as exc:
        raw = {key: {"error_code": "PROVIDER_ACQUISITION_FAILED", "error_detail": type(exc).__name__} for key in expected}
    extras = sorted(set(raw) - expected)
    if extras:
        raise AdvisoryHistoryIssue("UNEXPECTED_INSTRUMENT", "provider returned non-universe identities")

    series_by_id: dict[str, dict[str, Any]] = {}
    index_rows: list[dict[str, Any]] = []
    for instrument in instruments:
        try:
            row, series = _classify_acquisition(instrument, raw.get(str(instrument["instrument_id"])), acquisition_started, policy)
        except Exception as exc:
            row = {"instrument_id": str(instrument["instrument_id"]), "canonical_ticker": str(instrument["symbol"]), "source_symbol": str(instrument["source_symbol"]), "currency": str(instrument["currency"]), "expected_session": None, "previous_expected_session": None, "history_status": "invalid", "reason_codes": ["COMPLETED_SESSION_RESOLUTION_FAILED"], "row_count": 0, "first_session": None, "last_session": None, "series_file": None}
            series = None
        index_rows.append(row)
        if series is not None:
            series_by_id[str(instrument["instrument_id"])] = series
    acquisition_completed = _clock_now(clock)
    if acquisition_completed < acquisition_started:
        raise AdvisoryHistoryIssue("CLOCK_INVALID", "acquisition completion precedes acquisition start")
    replay = _semantic_replay(
        instruments, index_rows, series_by_id, acquisition_time=acquisition_started, policy=policy
    )
    index_rows = replay["rows"]
    eligibility_rows = replay["eligibility"]
    global_semantics = replay["global"]
    index_payload = {"schema_version": INDEX_VERSION, "run_id": run_id, "records": index_rows}
    eligibility = {"schema_version": ELIGIBILITY_VERSION, "run_id": run_id, "records": eligibility_rows}
    observation_digest = _sha256(_canonical_json({"index": index_payload, "series": series_by_id}))
    manifest_base = {
        "schema_version": MANIFEST_VERSION, "artifact_version": ARTIFACT_VERSION,
        "artifact_type": "advisory_ohlc_history", "run_id": run_id,
        "source_main_sha": source_main_sha, "acquisition_started_at": _utc_text(acquisition_started),
        "acquisition_completed_at": _utc_text(acquisition_completed), "source_id": SOURCE_ID,
        "adapter_id": ADAPTER_ID, "universe_schema_version": universe["schema_version"],
        "universe_version": universe.get("universe_version"), "universe_source_path": universe_source.as_posix(),
        "universe_sha256": _sha256_file(universe_source), "universe_identity_digest": universe["universe_checksum"],
        "history_policy_version": policy["schema_version"], "history_policy_path": policy_source.as_posix(),
        "history_policy_sha256": _sha256_file(policy_source), "expected_last_completed_sessions": sorted({row["expected_session"] for row in index_rows if row["expected_session"]}),
        "price_basis": policy["price_basis"], "corporate_action_adjustment_policy": policy["corporate_action_adjustment_policy"],
        "minimum_history_sessions": policy["minimum_history_sessions"], "maximum_history_sessions": policy["maximum_history_sessions"],
        **global_semantics,
        "observations_sha256": observation_digest, "retention_days": policy["artifact_retention_days"],
        "authority_boundary": "advisory_only_no_canonical_publication",
    }
    manifest = {**manifest_base, "artifact_sha256": _sha256(_canonical_json(manifest_base))}
    payloads: dict[str, Any] = {
        "manifest.json": manifest, "history_index.json": index_payload,
        "screening_eligibility.json": eligibility,
    }
    for instrument_id, series in series_by_id.items():
        payloads[f"series/{_safe_filename(instrument_id)}.json"] = series
    checksums = {name: _sha256(_canonical_json(value) + b"\n") for name, value in sorted(payloads.items())}
    checksum_payload = {"schema_version": CHECKSUM_VERSION, "files": checksums}
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in payloads.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_canonical_json(value) + b"\n")
    (destination / "checksum_index.json").write_bytes(_canonical_json(checksum_payload) + b"\n")
    return manifest, destination


def load_advisory_ohlc_history(
    artifact_root: str | Path,
) -> _ValidatedHistoryContext:
    """Load history against repository-owned authority and current UTC time."""
    _load_canonical_universe()
    _load_canonical_policy()
    return _load_advisory_ohlc_history_impl(
        artifact_root,
        universe_path=DEFAULT_UNIVERSE_SNAPSHOT,
        policy_path=DEFAULT_POLICY_PATH,
        now=_system_utc_now(),
    )


def _load_advisory_ohlc_history_impl(
    artifact_root: str | Path, *, universe_path: str | Path = DEFAULT_UNIVERSE_SNAPSHOT,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    now: datetime,
) -> _ValidatedHistoryContext:
    """Private deterministic load seam; callers of public authority cannot set time."""
    now = _clock_now(lambda: now)
    if isinstance(artifact_root, Mapping):
        raise AdvisoryHistoryIssue("CALLER_CONTENT_FORBIDDEN", "history authority requires an artifact path")
    root = Path(artifact_root)
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise AdvisoryHistoryIssue("ARTIFACT_PATH_INVALID", "symlinks are forbidden in history artifacts")
    manifest = _json(root / "manifest.json")
    index_payload = _json(root / "history_index.json")
    stored_eligibility = _json(root / "screening_eligibility.json")
    checksums = _json(root / "checksum_index.json")
    _validate_manifest(manifest)
    if checksums.get("schema_version") != CHECKSUM_VERSION or not isinstance(checksums.get("files"), Mapping):
        raise AdvisoryHistoryIssue("CHECKSUM_INDEX_INVALID", "checksum index contract is invalid")
    actual_files = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name != "checksum_index.json")
    if actual_files != sorted(checksums["files"]):
        raise AdvisoryHistoryIssue("CHECKSUM_INDEX_INVALID", "checksum index does not enumerate the artifact exactly")
    for name, digest in checksums["files"].items():
        if _sha256_file(root / name) != digest:
            raise AdvisoryHistoryIssue("ARTIFACT_INTEGRITY_INVALID", f"checksum mismatch for {name}")
    integrity = dict(manifest); artifact_sha = integrity.pop("artifact_sha256")
    if artifact_sha != _sha256(_canonical_json(integrity)):
        raise AdvisoryHistoryIssue("ARTIFACT_INTEGRITY_INVALID", "manifest integrity is invalid")
    universe_source, policy_source = Path(universe_path), Path(policy_path)
    universe, policy = load_authoritative_universe(universe_source), _load_policy(policy_source)
    if manifest["universe_sha256"] != _sha256_file(universe_source) or manifest["universe_identity_digest"] != universe["universe_checksum"]:
        raise AdvisoryHistoryIssue("UNIVERSE_BINDING_INVALID", "canonical universe changed")
    if manifest["history_policy_sha256"] != _sha256_file(policy_source):
        raise AdvisoryHistoryIssue("POLICY_BINDING_INVALID", "history policy changed")
    expected_static_manifest = {
        "artifact_type": "advisory_ohlc_history",
        "adapter_id": ADAPTER_ID,
        "universe_schema_version": universe["schema_version"],
        "universe_version": universe.get("universe_version"),
        "universe_source_path": universe_source.as_posix(),
        "history_policy_version": policy["schema_version"],
        "history_policy_path": policy_source.as_posix(),
        "price_basis": policy["price_basis"],
        "corporate_action_adjustment_policy": policy["corporate_action_adjustment_policy"],
        "minimum_history_sessions": policy["minimum_history_sessions"],
        "maximum_history_sessions": policy["maximum_history_sessions"],
        "retention_days": policy["artifact_retention_days"],
    }
    if any(manifest.get(key) != value for key, value in expected_static_manifest.items()):
        raise AdvisoryHistoryIssue("MANIFEST_SEMANTIC_REPLAY_INVALID", "manifest static authority differs from canonical inputs")
    records = index_payload.get("records")
    if index_payload.get("schema_version") != INDEX_VERSION or not isinstance(records, list):
        raise AdvisoryHistoryIssue("HISTORY_INDEX_INVALID", "history index contract is invalid")
    expected = {(str(row["instrument_id"]), str(row["symbol"])) for row in universe["instruments"]}
    actual = [(str(row.get("instrument_id")), str(row.get("canonical_ticker"))) for row in records]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise AdvisoryHistoryIssue("IDENTITY_RECONCILIATION_INVALID", "history index does not exactly reconcile the universe")
    acquisition_started = _timestamp(manifest["acquisition_started_at"], "acquisition_started_at")
    acquisition_completed = _timestamp(manifest["acquisition_completed_at"], "acquisition_completed_at")
    if acquisition_completed < acquisition_started:
        raise AdvisoryHistoryIssue("MANIFEST_SEMANTIC_REPLAY_INVALID", "acquisition timestamps are inconsistent")
    series: dict[str, Mapping[str, Any]] = {}
    for row in records:
        if row.get("series_file") is not None:
            expected_series_file = f"series/{_safe_filename(str(row['instrument_id']))}.json"
            if row.get("series_file") != expected_series_file:
                raise AdvisoryHistoryIssue("ARTIFACT_PATH_INVALID", "series path is not canonical for its instrument")
            path = root / expected_series_file
            payload = _json(path)
            series[str(row["instrument_id"])] = payload
    expected_series_files = sorted(str(row["series_file"]) for row in records if row.get("series_file") is not None)
    actual_series_files = sorted(path.relative_to(root).as_posix() for path in (root / "series").rglob("*") if path.is_file()) if (root / "series").is_dir() else []
    if actual_series_files != expected_series_files:
        raise AdvisoryHistoryIssue("HISTORY_SEMANTIC_REPLAY_INVALID", "series files do not exactly match index authority")
    if actual_files != sorted(["manifest.json", "history_index.json", "screening_eligibility.json", *expected_series_files]):
        raise AdvisoryHistoryIssue("CHECKSUM_INDEX_INVALID", "history artifact contains unexpected files")
    replay = _semantic_replay(
        sorted(universe["instruments"], key=lambda row: str(row["instrument_id"])),
        records,
        series,
        acquisition_time=acquisition_started,
        policy=policy,
    )
    replayed_index = {"schema_version": INDEX_VERSION, "run_id": manifest["run_id"], "records": replay["rows"]}
    replayed_eligibility = {"schema_version": ELIGIBILITY_VERSION, "run_id": manifest["run_id"], "records": replay["eligibility"]}
    if index_payload != replayed_index:
        raise AdvisoryHistoryIssue("HISTORY_SEMANTIC_REPLAY_INVALID", "stored history index differs from bar-derived semantics")
    if stored_eligibility != replayed_eligibility:
        raise AdvisoryHistoryIssue("ELIGIBILITY_SEMANTIC_REPLAY_INVALID", "stored eligibility differs from replayed semantics")
    for key, value in replay["global"].items():
        if manifest.get(key) != value:
            raise AdvisoryHistoryIssue("MANIFEST_SEMANTIC_REPLAY_INVALID", f"manifest {key} differs from replayed semantics")
    expected_sessions = sorted({row["expected_session"] for row in replay["rows"] if row["expected_session"]})
    if manifest.get("expected_last_completed_sessions") != expected_sessions:
        raise AdvisoryHistoryIssue("MANIFEST_SEMANTIC_REPLAY_INVALID", "manifest expected sessions differ from replayed semantics")
    if manifest["observations_sha256"] != _sha256(_canonical_json({"index": replayed_index, "series": series})):
        raise AdvisoryHistoryIssue("OBSERVATIONS_BINDING_INVALID", "history observations binding is invalid")
    if acquisition_completed > now:
        raise AdvisoryHistoryIssue("CLOCK_INVALID", "load clock precedes artifact acquisition completion")
    effective: dict[str, str] = {}
    instruments = {str(row["instrument_id"]): row for row in universe["instruments"]}
    for row in records:
        status = str(row["history_status"])
        if status == "fresh":
            _profile, expected_session = expected_completed_session(instruments[str(row["instrument_id"])], now)
            status = "fresh" if row["last_session"] == expected_session.isoformat() else "stale"
        effective[str(row["instrument_id"])] = status
    return _ValidatedHistoryContext(manifest, tuple(replay["rows"]), series, effective, universe, policy, root)


def validate_series_payload(payload: Mapping[str, Any]) -> None:
    _validate_series(payload, expected_instrument=None, policy=None, acquired_at=_clock_now(None))


def _effective_analytic_authority_status(context: _ValidatedHistoryContext) -> str:
    fresh = sum(status == "fresh" for status in context.effective_status.values())
    total = len(context.effective_status)
    coverage = Decimal(fresh) / Decimal(total) if total else Decimal(0)
    return "usable" if (
        context.manifest.get("analytic_authority_status") == "usable"
        and coverage >= Decimal(context.policy["minimum_fresh_screening_coverage_ratio"])
    ) else "unusable"


def _classify_acquisition(instrument: Mapping[str, Any], acquired: Mapping[str, Any] | None, at: datetime, policy: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    instrument_id, ticker = str(instrument["instrument_id"]), str(instrument["symbol"])
    _profile, expected_session = expected_completed_session(instrument, at)
    previous_session = _previous_completed_session(instrument, at, expected_session)
    base = {"instrument_id": instrument_id, "canonical_ticker": ticker, "source_symbol": str(instrument["source_symbol"]), "currency": str(instrument["currency"]), "expected_session": expected_session.isoformat(), "previous_expected_session": previous_session.isoformat()}
    status, reasons, series = "missing", ["HISTORY_NOT_RETURNED"], None
    if instrument.get("source_mapping_status") != "mapped":
        status, reasons = "blocked_identity", ["SOURCE_MAPPING_UNAUTHORIZED"]
    elif acquired is not None:
        if acquired.get("error_code"):
            reasons = [str(acquired["error_code"])]
            if reasons[0] in {"AMBIGUOUS_PROVIDER_IDENTITY", "SOURCE_MAPPING_UNAUTHORIZED"}:
                status = "blocked_identity"
            elif reasons[0] == "PROVIDER_SERIES_INVALID":
                status = "invalid"
        elif acquired.get("instrument_id", instrument_id) != instrument_id or acquired.get("canonical_ticker", ticker) != ticker or acquired.get("source_symbol", instrument["source_symbol"]) != instrument["source_symbol"] or acquired.get("currency", instrument["currency"]) != instrument["currency"]:
            status, reasons = "blocked_identity", ["PROVIDER_IDENTITY_MISMATCH"]
        elif acquired.get("price_basis", policy["price_basis"]) != policy["price_basis"] or acquired.get("corporate_action_adjustment_policy", policy["corporate_action_adjustment_policy"]) != policy["corporate_action_adjustment_policy"]:
            status, reasons = "blocked_adjustment_policy", ["ADJUSTMENT_POLICY_MISMATCH"]
        else:
            bars = list(acquired.get("bars") or [])[-int(policy["maximum_history_sessions"]):]
            series_identity = {key: base[key] for key in ("instrument_id", "canonical_ticker", "source_symbol", "currency", "expected_session")}
            series = {"schema_version": SERIES_VERSION, **series_identity, "source_id": SOURCE_ID, "price_basis": policy["price_basis"], "corporate_action_adjustment_policy": policy["corporate_action_adjustment_policy"], "bars": bars}
            try:
                _validate_series(series, expected_instrument=base, policy=policy, acquired_at=at)
                last = bars[-1]["session"]
                status = "insufficient_history" if len(bars) < policy["minimum_history_sessions"] else ("fresh" if last == base["expected_session"] else "stale")
                reasons = [] if status == "fresh" else (["INSUFFICIENT_HISTORY"] if status == "insufficient_history" else ["LATEST_SESSION_BEHIND_EXPECTED"])
            except AdvisoryHistoryIssue as exc:
                status, reasons, series = "invalid", [exc.code], None
    row = {**base, "history_status": status, "reason_codes": reasons, "row_count": len(series["bars"]) if series else 0, "first_session": series["bars"][0]["session"] if series else None, "last_session": series["bars"][-1]["session"] if series else None, "series_file": f"series/{_safe_filename(instrument_id)}.json" if series else None}
    return row, series


_NO_SERIES_REASON_CODES = {
    "missing": frozenset({"HISTORY_NOT_RETURNED", "PROVIDER_ACQUISITION_FAILED", "PROVIDER_HISTORY_MISSING"}),
    "invalid": frozenset({
        "PROVIDER_SERIES_INVALID", "COMPLETED_SESSION_RESOLUTION_FAILED", "SERIES_SCHEMA_INVALID",
        "SERIES_IDENTITY_INVALID", "ADJUSTMENT_POLICY_INVALID", "SERIES_EMPTY", "BAR_INVALID",
        "SESSION_INVALID", "FUTURE_SESSION", "SESSION_AFTER_EXPECTED", "PRICE_DOMAIN_INVALID",
        "OHLC_RELATION_INVALID", "VOLUME_INVALID", "SESSION_ORDER_INVALID", "SERIES_TOO_LONG",
        "EXPECTED_SESSION_INVALID", "SOURCE_ID_INVALID",
    }),
    "blocked_identity": frozenset({"SOURCE_MAPPING_UNAUTHORIZED", "AMBIGUOUS_PROVIDER_IDENTITY", "PROVIDER_IDENTITY_MISMATCH"}),
    "blocked_adjustment_policy": frozenset({"ADJUSTMENT_POLICY_MISMATCH"}),
}


def _semantic_replay(
    instruments: Sequence[Mapping[str, Any]],
    stored_rows: Sequence[Mapping[str, Any]],
    series_by_id: Mapping[str, Mapping[str, Any]],
    *,
    acquisition_time: datetime,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    stored_by_id = {str(row.get("instrument_id")): row for row in stored_rows}
    expected_ids = {str(row["instrument_id"]) for row in instruments}
    if len(stored_by_id) != len(stored_rows) or set(stored_by_id) != expected_ids:
        raise AdvisoryHistoryIssue("IDENTITY_RECONCILIATION_INVALID", "stored history identities are not canonical and unique")
    replayed: list[dict[str, Any]] = []
    for instrument in instruments:
        instrument_id = str(instrument["instrument_id"])
        stored = stored_by_id[instrument_id]
        _profile, expected = expected_completed_session(instrument, acquisition_time)
        previous = _previous_completed_session(instrument, acquisition_time, expected)
        base = {
            "instrument_id": instrument_id,
            "canonical_ticker": str(instrument["symbol"]),
            "source_symbol": str(instrument["source_symbol"]),
            "currency": str(instrument["currency"]),
            "expected_session": expected.isoformat(),
            "previous_expected_session": previous.isoformat(),
        }
        series = series_by_id.get(instrument_id)
        if series is not None:
            if instrument.get("source_mapping_status") != "mapped":
                raise AdvisoryHistoryIssue("HISTORY_SEMANTIC_REPLAY_INVALID", "series exists without an authorized source mapping")
            _validate_series(series, expected_instrument=base, policy=policy, acquired_at=acquisition_time)
            bars = series["bars"]
            if len(bars) > int(policy["maximum_history_sessions"]):
                raise AdvisoryHistoryIssue("SERIES_TOO_LONG", "series exceeds the maximum retained history")
            last = bars[-1]["session"]
            if len(bars) < int(policy["minimum_history_sessions"]):
                status, reasons = "insufficient_history", ["INSUFFICIENT_HISTORY"]
            elif last == base["expected_session"]:
                status, reasons = "fresh", []
            else:
                status, reasons = "stale", ["LATEST_SESSION_BEHIND_EXPECTED"]
            row = {
                **base,
                "history_status": status,
                "reason_codes": reasons,
                "row_count": len(bars),
                "first_session": bars[0]["session"],
                "last_session": last,
                "series_file": f"series/{_safe_filename(instrument_id)}.json",
            }
        else:
            status = str(stored.get("history_status"))
            reasons = stored.get("reason_codes")
            allowed = _NO_SERIES_REASON_CODES.get(status)
            if allowed is None or not isinstance(reasons, list) or len(reasons) != 1 or reasons[0] not in allowed:
                raise AdvisoryHistoryIssue("HISTORY_SEMANTIC_REPLAY_INVALID", "non-series status has unauthorized reason semantics")
            if any(stored.get(key) is not None for key in ("first_session", "last_session", "series_file")) or stored.get("row_count") != 0:
                raise AdvisoryHistoryIssue("HISTORY_SEMANTIC_REPLAY_INVALID", "non-series status contains conflicting series claims")
            row = {
                **base,
                "history_status": status,
                "reason_codes": list(reasons),
                "row_count": 0,
                "first_session": None,
                "last_session": None,
                "series_file": None,
            }
        replayed.append(row)
    global_semantics = _global_semantics(replayed, policy)
    usable = global_semantics["analytic_authority_status"] == "usable"
    eligibility = [{
        "instrument_id": row["instrument_id"],
        "canonical_ticker": row["canonical_ticker"],
        "history_status": row["history_status"],
        "eligible_for_current_screening": row["history_status"] == "fresh" and usable,
        "reason_codes": list(row["reason_codes"]) + (
            ["GLOBAL_ANALYTIC_AUTHORITY_UNUSABLE"] if row["history_status"] == "fresh" and not usable else []
        ),
    } for row in replayed]
    return {"rows": replayed, "eligibility": eligibility, "global": global_semantics}


def _global_semantics(rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]) -> dict[str, Any]:
    counts = Counter(str(row["history_status"]) for row in rows)
    status_counts = {status: counts[status] for status in STATUSES}
    status_counts["attempted"] = len(rows)
    lag = _provider_lag(rows, policy)
    provider_failure_count = sum(any("PROVIDER" in code for code in row["reason_codes"]) for row in rows)
    provider_failure = {
        "detected": provider_failure_count > len(rows) / 2,
        "affected_count": provider_failure_count,
        "threshold": "strict_majority",
    }
    coverage = Decimal(status_counts["fresh"]) / Decimal(len(rows)) if rows else Decimal(0)
    coverage_threshold = Decimal(policy["minimum_fresh_screening_coverage_ratio"])
    run_status = "blocked_provider_session_lag" if lag["detected"] else (
        "blocked_provider_failure" if provider_failure["detected"] else
        ("completed_with_blockers" if status_counts["fresh"] != len(rows) else "completed")
    )
    usable = not lag["detected"] and not provider_failure["detected"] and coverage >= coverage_threshold
    return {
        "status_counts": status_counts,
        "provider_session_lag": lag,
        "provider_failure": provider_failure,
        "fresh_screening_coverage": {
            "ratio": format(coverage, "f"),
            "threshold": format(coverage_threshold, "f"),
            "meets_threshold": coverage >= coverage_threshold,
        },
        "run_status": run_status,
        "analytic_authority_status": "usable" if usable else "unusable",
    }


def _validate_series(payload: Mapping[str, Any], *, expected_instrument: Mapping[str, Any] | None, policy: Mapping[str, Any] | None, acquired_at: datetime) -> None:
    allowed_series = {"schema_version", "instrument_id", "canonical_ticker", "source_symbol", "currency", "expected_session", "source_id", "price_basis", "corporate_action_adjustment_policy", "bars"}
    if set(payload) != allowed_series:
        raise AdvisoryHistoryIssue("SERIES_SCHEMA_INVALID", "series fields are not exact")
    if payload.get("schema_version") != SERIES_VERSION:
        raise AdvisoryHistoryIssue("SERIES_SCHEMA_INVALID", "series schema version is invalid")
    if payload.get("source_id") != SOURCE_ID:
        raise AdvisoryHistoryIssue("SOURCE_ID_INVALID", "series source is not the approved advisory source")
    for key in ("instrument_id", "canonical_ticker", "source_symbol", "currency", "expected_session", "price_basis", "corporate_action_adjustment_policy"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise AdvisoryHistoryIssue("SERIES_IDENTITY_INVALID", f"{key} is required")
    if expected_instrument:
        for source, target in (("instrument_id", "instrument_id"), ("canonical_ticker", "canonical_ticker"), ("source_symbol", "source_symbol"), ("currency", "currency")):
            if payload[source] != expected_instrument[target]:
                raise AdvisoryHistoryIssue("SERIES_IDENTITY_INVALID", f"{source} differs from canonical identity")
        if payload.get("expected_session") != expected_instrument.get("expected_session"):
            raise AdvisoryHistoryIssue("EXPECTED_SESSION_INVALID", "series expected session differs from producer-time policy")
    if policy and (payload["price_basis"] != policy["price_basis"] or payload["corporate_action_adjustment_policy"] != policy["corporate_action_adjustment_policy"]):
        raise AdvisoryHistoryIssue("ADJUSTMENT_POLICY_INVALID", "series adjustment policy differs")
    bars = payload.get("bars")
    if not isinstance(bars, list) or not bars:
        raise AdvisoryHistoryIssue("SERIES_EMPTY", "series must contain bars")
    if policy and len(bars) > int(policy["maximum_history_sessions"]):
        raise AdvisoryHistoryIssue("SERIES_TOO_LONG", "series exceeds the maximum retained history")
    try:
        expected_date = date.fromisoformat(str(payload.get("expected_session")))
    except ValueError as exc:
        raise AdvisoryHistoryIssue("EXPECTED_SESSION_INVALID", "series expected session is invalid") from exc
    sessions: list[str] = []
    for bar in bars:
        if not isinstance(bar, Mapping):
            raise AdvisoryHistoryIssue("BAR_INVALID", "bar must be an object")
        if set(bar) != {"session", "open", "high", "low", "close", "volume", "volume_status"}:
            raise AdvisoryHistoryIssue("BAR_INVALID", "bar fields are not exact")
        try: session = date.fromisoformat(str(bar.get("session")))
        except ValueError as exc: raise AdvisoryHistoryIssue("SESSION_INVALID", "session must be an ISO date") from exc
        if session > acquired_at.date():
            raise AdvisoryHistoryIssue("FUTURE_SESSION", "future sessions are forbidden")
        if session > expected_date:
            raise AdvisoryHistoryIssue("SESSION_AFTER_EXPECTED", "series contains a bar after the producer-time expected session")
        sessions.append(session.isoformat())
        values = {key: _positive_decimal(bar.get(key), key) for key in ("open", "high", "low", "close")}
        if values["high"] < max(values["open"], values["close"], values["low"]) or values["low"] > min(values["open"], values["close"], values["high"]):
            raise AdvisoryHistoryIssue("OHLC_RELATION_INVALID", "OHLC relationships are invalid")
        volume = bar.get("volume")
        if volume is not None and (isinstance(volume, bool) or not isinstance(volume, str) or not INTEGER_TEXT.fullmatch(volume)):
            raise AdvisoryHistoryIssue("VOLUME_INVALID", "volume must be an exact nonnegative integer string or null")
        if bar.get("volume_status") not in ({"provider_reported"} if volume is not None else {"not_reported"}):
            raise AdvisoryHistoryIssue("VOLUME_INVALID", "volume status must state whether volume was reported")
    if sessions != sorted(sessions) or len(sessions) != len(set(sessions)):
        raise AdvisoryHistoryIssue("SESSION_ORDER_INVALID", "sessions must be strictly increasing and unique")


def _provider_lag(rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]) -> dict[str, Any]:
    valid = [row for row in rows if row["history_status"] in {"fresh", "stale", "insufficient_history"} and row["last_session"]]
    exact_one = 0
    for row in valid:
        if row["history_status"] == "stale" and row["last_session"] == row.get("previous_expected_session"):
            exact_one += 1
    ratio = Decimal(exact_one) / Decimal(len(valid)) if valid else Decimal(0)
    threshold = Decimal(policy["widespread_one_session_lag_ratio"])
    return {"detected": bool(valid) and ratio >= threshold, "otherwise_valid_count": len(valid), "exactly_one_session_late_count": exact_one, "ratio": format(ratio, "f"), "threshold": format(threshold, "f")}


def _acquire_with_existing_adapter(instruments: Sequence[Mapping[str, Any]], acquired_at: datetime, policy: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    start = (acquired_at.date() - timedelta(days=int(policy["request_calendar_days"]))).isoformat()
    end = (acquired_at.date() + timedelta(days=1)).isoformat()
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    result: dict[str, Mapping[str, Any]] = {}
    for row in instruments:
        if row.get("source_mapping_status") != "mapped":
            result[str(row["instrument_id"])] = {"error_code": "SOURCE_MAPPING_UNAUTHORIZED"}
            continue
        grouped.setdefault(_to_yfinance_symbol(str(row["source_symbol"])), []).append(row)
    by_symbol = {symbol: rows[0] for symbol, rows in grouped.items() if len(rows) == 1}
    batch = download_yfinance_batch(tuple(by_symbol), start, end)
    for rows in grouped.values():
        if len(rows) > 1:
            for row in rows:
                result[str(row["instrument_id"])] = {"error_code": "AMBIGUOUS_PROVIDER_IDENTITY"}
    fallback_count = 0
    for provider_symbol, instrument in by_symbol.items():
        frame = batch.get(provider_symbol)
        if (frame is None or frame.empty) and fallback_count < int(policy["max_individual_fallbacks"]):
            fallback_count += 1
            try: frame = _download_yfinance_history(provider_symbol, start, end)
            except Exception: frame = None
        instrument_id = str(instrument["instrument_id"])
        if frame is None or frame.empty:
            result[instrument_id] = {"error_code": "PROVIDER_HISTORY_MISSING"}
        else:
            try:
                bars = _frame_bars(frame)
                result[instrument_id] = {"instrument_id": instrument_id, "canonical_ticker": instrument["symbol"], "source_symbol": instrument["source_symbol"], "currency": instrument["currency"], "price_basis": policy["price_basis"], "corporate_action_adjustment_policy": policy["corporate_action_adjustment_policy"], "bars": bars}
            except Exception as exc:
                result[instrument_id] = {"error_code": "PROVIDER_SERIES_INVALID", "error_detail": type(exc).__name__}
    return result


def _frame_bars(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _, row in frame.sort_values("Date").iterrows():
        volume = row.get("Volume")
        reported = volume is not None and not pd.isna(volume)
        rows.append({"session": pd.Timestamp(row["Date"]).date().isoformat(), "open": _provider_decimal(row["Open"]), "high": _provider_decimal(row["High"]), "low": _provider_decimal(row["Low"]), "close": _provider_decimal(row["Close"]), "volume": str(int(volume)) if reported else None, "volume_status": "provider_reported" if reported else "not_reported"})
    return rows


def _load_policy(path: Path) -> dict[str, Any]:
    value = _json(path)
    if value.get("schema_version") != POLICY_VERSION or value.get("minimum_history_sessions") != value.get("indicator_max_warmup_sessions") + value.get("warmup_safety_margin_sessions"):
        raise AdvisoryHistoryIssue("POLICY_INVALID", "history policy is inconsistent")
    if value.get("maximum_history_sessions", 0) < value["minimum_history_sessions"] or value.get("artifact_retention_days") != 14 or value.get("max_individual_fallbacks", 999) > 25:
        raise AdvisoryHistoryIssue("POLICY_INVALID", "history policy bounds are invalid")
    Decimal(value["widespread_one_session_lag_ratio"])
    coverage = Decimal(value["minimum_fresh_screening_coverage_ratio"])
    if coverage <= 0 or coverage > 1:
        raise AdvisoryHistoryIssue("POLICY_INVALID", "fresh screening coverage threshold is invalid")
    return value


def _load_canonical_policy() -> dict[str, Any]:
    value = _load_policy(DEFAULT_POLICY_PATH)
    governed = {
        "schema_version": POLICY_VERSION,
        "indicator_max_warmup_sessions": 200,
        "warmup_safety_margin_sessions": 52,
        "minimum_history_sessions": 252,
        "maximum_history_sessions": 420,
        "widespread_one_session_lag_ratio": "0.80",
        "minimum_fresh_screening_coverage_ratio": "0.99",
        "max_individual_fallbacks": 25,
        "artifact_retention_days": 14,
        "price_basis": "unadjusted_ohlc",
        "corporate_action_adjustment_policy": "provider_reported_unadjusted_with_adj_close_excluded",
    }
    if any(value.get(field) != expected for field, expected in governed.items()):
        raise AdvisoryHistoryIssue("POLICY_INVALID", "governed history policy semantics changed")
    return value


def _load_canonical_universe() -> dict[str, Any]:
    value = load_authoritative_universe(DEFAULT_UNIVERSE_SNAPSHOT)
    if len(value.get("instruments", ())) != 952:
        raise AdvisoryHistoryIssue("UNIVERSE_INVALID", "canonical universe must contain exactly 952 identities")
    return value


def _validate_manifest(value: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "artifact_version", "artifact_type", "run_id", "source_main_sha",
        "acquisition_started_at", "acquisition_completed_at", "source_id", "adapter_id",
        "universe_schema_version", "universe_version", "universe_source_path", "universe_sha256",
        "universe_identity_digest", "history_policy_version", "history_policy_path",
        "history_policy_sha256", "expected_last_completed_sessions", "price_basis",
        "corporate_action_adjustment_policy", "minimum_history_sessions", "maximum_history_sessions",
        "status_counts", "provider_session_lag", "provider_failure", "fresh_screening_coverage",
        "run_status", "analytic_authority_status", "observations_sha256", "retention_days",
        "authority_boundary", "artifact_sha256",
    }
    if set(value) != required or value.get("schema_version") != MANIFEST_VERSION or value.get("artifact_version") != ARTIFACT_VERSION or value.get("authority_boundary") != "advisory_only_no_canonical_publication" or value.get("source_id") != SOURCE_ID or not re.fullmatch(r"[0-9a-f]{40}", str(value.get("source_main_sha", ""))) or not IDENTIFIER.fullmatch(str(value.get("run_id", ""))):
        raise AdvisoryHistoryIssue("MANIFEST_INVALID", "history manifest contract is invalid")


def _validate_output_destination(output_root: str | Path, run_id: str) -> Path:
    if not IDENTIFIER.fullmatch(run_id): raise AdvisoryHistoryIssue("RUN_ID_INVALID", "run_id is invalid")
    root = Path(output_root)
    if root.is_absolute() or ".." in root.parts or root != DEFAULT_OUTPUT_ROOT:
        raise AdvisoryHistoryIssue("OUTPUT_PATH_INVALID", "output root must be the approved repository-relative history root")
    repository = _repository_root(); destination = (repository / root / run_id).resolve()
    approved = (repository / DEFAULT_OUTPUT_ROOT).resolve()
    if approved not in destination.parents or destination.exists():
        raise AdvisoryHistoryIssue("OUTPUT_PATH_INVALID", "output destination is outside authority or already exists")
    cursor = repository
    for part in root.parts:
        cursor = cursor / part
        if cursor.is_symlink(): raise AdvisoryHistoryIssue("OUTPUT_PATH_INVALID", "symlink output paths are forbidden")
    return destination


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _current_repository_head_sha() -> str:
    """Resolve immutable production provenance from this source repository."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(_repository_root()), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AdvisoryHistoryIssue("SOURCE_MAIN_SHA_UNRESOLVED", "repository HEAD cannot be resolved") from exc
    source_main_sha = completed.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source_main_sha):
        raise AdvisoryHistoryIssue("SOURCE_MAIN_SHA_UNRESOLVED", "repository HEAD is not a full lowercase Git SHA")
    return source_main_sha


def _positive_decimal(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, str) or not DECIMAL_TEXT.fullmatch(value):
        raise AdvisoryHistoryIssue("PRICE_DOMAIN_INVALID", f"{field} must be an exact positive decimal string")
    try: parsed = Decimal(value)
    except InvalidOperation as exc: raise AdvisoryHistoryIssue("PRICE_DOMAIN_INVALID", f"{field} is invalid") from exc
    if not parsed.is_finite() or parsed <= 0: raise AdvisoryHistoryIssue("PRICE_DOMAIN_INVALID", f"{field} must be finite and positive")
    return parsed


def _provider_decimal(value: Any) -> str:
    parsed = Decimal(str(value))
    if not parsed.is_finite() or parsed <= 0: raise AdvisoryHistoryIssue("PROVIDER_PRICE_INVALID", "provider returned an invalid price")
    return format(parsed, "f")


def _previous_completed_session(instrument: Mapping[str, Any], reference: datetime, current: date) -> date:
    for days in range(1, 15):
        _profile, candidate = expected_completed_session(instrument, reference - timedelta(days=days))
        if candidate < current:
            return candidate
    raise AdvisoryHistoryIssue("EXPECTED_SESSION_UNAVAILABLE", "previous completed session cannot be resolved")


def _timestamp(value: str, field: str) -> datetime:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
        raise AdvisoryHistoryIssue("TIMESTAMP_INVALID", f"{field} must be canonical UTC with Z")
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _clock_now(clock: Callable[[], datetime] | None) -> datetime:
    value = datetime.now(UTC).replace(microsecond=0) if clock is None else clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise AdvisoryHistoryIssue("CLOCK_INVALID", "clock must produce timezone-aware UTC")
    if value.microsecond != 0:
        raise AdvisoryHistoryIssue("CLOCK_INVALID", "clock must produce canonical whole-second UTC")
    return value.astimezone(UTC)


def _system_utc_now() -> datetime:
    return _clock_now(None)


def _utc_text(value: datetime) -> str: return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
def _canonical_json(value: Any) -> bytes: return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
def _sha256(value: bytes) -> str: return hashlib.sha256(value).hexdigest()
def _sha256_file(path: Path) -> str: return _sha256(path.read_bytes())
def _json(path: Path) -> dict[str, Any]:
    try: value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc: raise AdvisoryHistoryIssue("ARTIFACT_READ_INVALID", f"cannot read {path.name}") from exc
    if not isinstance(value, dict): raise AdvisoryHistoryIssue("ARTIFACT_READ_INVALID", f"{path.name} must contain an object")
    return value
def _safe_filename(value: str) -> str: return hashlib.sha256(value.encode()).hexdigest()


def run_command(argv: Sequence[str] | None = None, *, stdout: TextIO = sys.stdout, stderr: TextIO = sys.stderr) -> int:
    parser = argparse.ArgumentParser(description="Build and validate advisory-only current OHLC history evidence")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--run-id", required=True)
    build.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT.as_posix())
    gate = commands.add_parser("quality-gate")
    gate.add_argument("--artifact-root", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "quality-gate":
            context = load_advisory_ohlc_history(args.artifact_root)
            status = _effective_analytic_authority_status(context)
            print(json.dumps({"status": status, "run_status": context.manifest["run_status"]}, sort_keys=True), file=stdout)
            return 0 if status == "usable" else 3
        manifest, path = build_advisory_ohlc_history(run_id=args.run_id, output_root=args.output_root)
    except AdvisoryHistoryIssue as exc:
        print(json.dumps({"status": "blocked", "code": exc.code}), file=stderr); return 2
    print(json.dumps({"status": manifest["run_status"], "artifact_path": path.as_posix(), "manifest": manifest}, sort_keys=True), file=stdout)
    return 0


if __name__ == "__main__": raise SystemExit(run_command())
