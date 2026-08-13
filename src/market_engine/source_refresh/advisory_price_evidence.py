from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TextIO
from zoneinfo import ZoneInfo

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


ARTIFACT_VERSION = "me-sr25-advisory-price-evidence-v1"
OBSERVATION_SCHEMA_VERSION = "market-engine-advisory-price-observation-v1"
OBSERVATIONS_SCHEMA_VERSION = "market-engine-advisory-price-observations-v1"
MANIFEST_SCHEMA_VERSION = "market-engine-advisory-price-manifest-v1"
CONSUMER_SCHEMA_VERSION = "market-engine-advisory-price-context-v1"
POLICY_SCHEMA_VERSION = "market-engine-advisory-price-freshness-policy-v1"
ARTIFACT_TYPE = "advisory_price_evidence"
OBSERVATION_TYPE = "provider_reported_daily_close"
SOURCE_ID = "yahoo-finance-yfinance"
SOURCE_ADAPTER = "existing_yfinance_daily_history_adapter"
OBSERVATIONS_FILE = "advisory_price_observations.json"
MANIFEST_FILE = "advisory_price_manifest.json"
DEFAULT_POLICY_PATH = Path("config/market_engine/advisory_price_freshness_policy.json")
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")
TICKER = re.compile(r"^[A-Z0-9][A-Z0-9.^_-]{0,49}$")
ERROR_CODE = re.compile(r"^[A-Z][A-Z0-9_]{2,99}$")
DECIMAL_TEXT = re.compile(r"^(?:0\.[0-9]*[1-9][0-9]*|[1-9][0-9]*(?:\.[0-9]+)?)$")


class AdvisoryPriceIssue(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


Provider = Callable[[Sequence[Mapping[str, Any]], datetime], Mapping[str, Mapping[str, Any]]]


def build_advisory_price_artifact(
    *,
    run_id: str,
    source_main_sha: str,
    output_root: str | Path,
    universe_path: str | Path = DEFAULT_UNIVERSE_SNAPSHOT,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    retrieval_timestamp: str | None = None,
    provider: Provider | None = None,
) -> tuple[dict[str, Any], Path]:
    _safe_identifier(run_id, "run_id")
    if not re.fullmatch(r"[0-9a-f]{40}", source_main_sha):
        raise AdvisoryPriceIssue("SOURCE_MAIN_SHA_INVALID", "source_main_sha must be a full Git SHA")
    retrieval = _timestamp(retrieval_timestamp or datetime.now(UTC).isoformat(), "retrieval_timestamp")
    if retrieval > datetime.now(UTC):
        raise AdvisoryPriceIssue("FUTURE_RETRIEVAL_TIMESTAMP", "retrieval timestamp cannot be in the future")
    universe_source = Path(universe_path)
    universe = load_authoritative_universe(universe_source)
    policy_source = Path(policy_path)
    policy = _load_policy(policy_source)
    instruments = sorted(universe["instruments"], key=lambda row: (row["instrument_id"], row["symbol"]))
    try:
        acquisitions = dict((provider or _acquire_with_existing_adapter)(instruments, retrieval))
    except Exception as exc:
        acquisitions = {
            str(row["instrument_id"]): {
                "acquisition_error_code": "ACQUISITION_FAILED",
                "acquisition_error_detail": type(exc).__name__,
            }
            for row in instruments
        }
    expected_ids = {str(row["instrument_id"]) for row in instruments}
    extra_ids = sorted(set(acquisitions) - expected_ids)
    if extra_ids:
        raise AdvisoryPriceIssue("UNEXPECTED_INSTRUMENT", f"provider returned unknown instrument IDs: {', '.join(extra_ids)}")

    records = []
    for instrument in instruments:
        instrument_id = str(instrument["instrument_id"])
        acquired = acquisitions.get(instrument_id)
        records.append(
            _record_from_acquisition(
                instrument,
                acquired,
                run_id=run_id,
                retrieval=retrieval,
                policy=policy,
            )
        )
    records.sort(key=lambda row: (row["instrument_id"], row["canonical_ticker"]))
    observations = {
        "schema_version": OBSERVATIONS_SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "run_id": run_id,
        "records": records,
    }
    validate_observations_payload(observations, trusted_now=datetime.now(UTC))
    observations_bytes = _canonical_json(observations) + b"\n"
    counts = Counter(row["freshness_status"] for row in records)
    status_counts = {
        "attempted": len(records),
        "fresh": counts["fresh"],
        "stale": counts["stale"],
        "missing": counts["missing"],
        "invalid": counts["invalid"],
    }
    manifest_without_integrity = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "run_id": run_id,
        "generated_at": _utc_text(retrieval),
        "source_main_sha": source_main_sha,
        "universe_schema_version": universe["schema_version"],
        "universe_source_path": universe_source.as_posix(),
        "universe_sha256": _sha256_file(universe_source),
        "universe_snapshot_sha256": universe["universe_checksum"],
        "freshness_policy_path": policy_source.as_posix(),
        "freshness_policy_sha256": _sha256_file(policy_source),
        "source_adapter": SOURCE_ADAPTER,
        "source_id": SOURCE_ID,
        "observation_type": OBSERVATION_TYPE,
        "expected_instrument_count": len(instruments),
        "status_counts": status_counts,
        "observations_file": OBSERVATIONS_FILE,
        "observations_sha256": _sha256(observations_bytes),
        "integrity_algorithm": "sha256-canonical-json-v1",
        "retention_days": policy["artifact_retention_days"],
        "canonical_publication_status": "not_authorized_advisory_only",
    }
    manifest = {
        **manifest_without_integrity,
        "artifact_sha256": _sha256(_canonical_json(manifest_without_integrity)),
    }
    validate_manifest_payload(manifest)
    destination = Path(output_root) / run_id
    destination.mkdir(parents=True, exist_ok=False)
    _write_bytes(destination / OBSERVATIONS_FILE, observations_bytes)
    _write_bytes(destination / MANIFEST_FILE, _canonical_json(manifest) + b"\n")
    return manifest, destination


def load_advisory_price_artifact(
    artifact_root: str | Path,
    *,
    universe_path: str | Path = DEFAULT_UNIVERSE_SNAPSHOT,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    trusted_now: str | None = None,
) -> dict[str, Any]:
    root = Path(artifact_root)
    manifest = _load_json(root / MANIFEST_FILE)
    observations = _load_json(root / OBSERVATIONS_FILE)
    validate_manifest_payload(manifest)
    now = _timestamp(trusted_now or datetime.now(UTC).isoformat(), "trusted_now")
    validate_observations_payload(observations, trusted_now=now)
    if manifest["observations_sha256"] != _sha256_file(root / OBSERVATIONS_FILE):
        raise AdvisoryPriceIssue("ARTIFACT_INTEGRITY_INVALID", "observations file checksum differs from manifest")
    integrity = dict(manifest)
    artifact_sha = integrity.pop("artifact_sha256")
    if artifact_sha != _sha256(_canonical_json(integrity)):
        raise AdvisoryPriceIssue("ARTIFACT_INTEGRITY_INVALID", "manifest integrity checksum is invalid")
    universe_source = Path(universe_path)
    policy_source = Path(policy_path)
    universe = load_authoritative_universe(universe_source)
    policy = _load_policy(policy_source)
    generated_at = _timestamp(manifest["generated_at"], "generated_at")
    if generated_at > now:
        raise AdvisoryPriceIssue("FUTURE_RETRIEVAL_TIMESTAMP", "manifest generation timestamp cannot be in the future")
    if manifest["run_id"] != observations["run_id"]:
        raise AdvisoryPriceIssue("RUN_ID_MISMATCH", "manifest and observations run IDs differ")
    if manifest["universe_schema_version"] != universe["schema_version"]:
        raise AdvisoryPriceIssue("UNIVERSE_BINDING_INVALID", "universe schema version changed")
    if manifest["universe_sha256"] != _sha256_file(universe_source):
        raise AdvisoryPriceIssue("UNIVERSE_BINDING_INVALID", "universe source checksum changed")
    if manifest["universe_snapshot_sha256"] != universe["universe_checksum"]:
        raise AdvisoryPriceIssue("UNIVERSE_BINDING_INVALID", "universe identity digest changed")
    if manifest["freshness_policy_sha256"] != _sha256_file(policy_source):
        raise AdvisoryPriceIssue("POLICY_BINDING_INVALID", "freshness policy checksum changed")
    if manifest["retention_days"] != policy["artifact_retention_days"]:
        raise AdvisoryPriceIssue("POLICY_BINDING_INVALID", "artifact retention differs from policy")
    _validate_complete_universe(observations["records"], universe["instruments"])
    _validate_loaded_freshness(
        observations["records"],
        universe["instruments"],
        policy=policy,
        generated_at=generated_at,
    )
    expected_counts = Counter(row["freshness_status"] for row in observations["records"])
    reconciled = {
        "attempted": len(observations["records"]),
        "fresh": expected_counts["fresh"],
        "stale": expected_counts["stale"],
        "missing": expected_counts["missing"],
        "invalid": expected_counts["invalid"],
    }
    if manifest["status_counts"] != reconciled:
        raise AdvisoryPriceIssue("ARTIFACT_RECONCILIATION_INVALID", "manifest totals do not reconcile")
    if manifest["expected_instrument_count"] != len(universe["instruments"]):
        raise AdvisoryPriceIssue("ARTIFACT_RECONCILIATION_INVALID", "expected instrument total is invalid")
    return {"manifest": manifest, "observations": observations, "policy": policy}


def consume_advisory_price_context(
    artifact_root: str | Path,
    *,
    instrument_id: str,
    canonical_ticker: str,
    universe_path: str | Path = DEFAULT_UNIVERSE_SNAPSHOT,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    trusted_now: str | None = None,
) -> dict[str, Any]:
    if isinstance(artifact_root, Mapping):
        raise AdvisoryPriceIssue("FORGED_CALLER_CONTEXT", "consumer requires a validated artifact path")
    loaded = load_advisory_price_artifact(
        artifact_root,
        universe_path=universe_path,
        policy_path=policy_path,
        trusted_now=trusted_now,
    )
    matches = [
        row for row in loaded["observations"]["records"]
        if row["instrument_id"] == instrument_id and row["canonical_ticker"] == canonical_ticker
    ]
    if len(matches) != 1:
        raise AdvisoryPriceIssue("INSTRUMENT_IDENTITY_MISMATCH", "requested instrument identity is absent or mismatched")
    row = matches[0]
    usable = row["freshness_status"] == "fresh" and row["acquisition_status"] == "succeeded"
    return {
        "schema_version": CONSUMER_SCHEMA_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "instrument_id": instrument_id,
        "canonical_ticker": canonical_ticker,
        "price_context_status": row["freshness_status"],
        "current_price": row["price"] if usable else None,
        "currency": row["currency"],
        "observation_type": row["observation_type"],
        "observation_timestamp": row["observation_timestamp"],
        "retrieval_timestamp": row["retrieval_timestamp"],
        "source_id": row["source_id"],
        "acquisition_status": row["acquisition_status"],
        "error_code": row["error_code"],
        "error_detail": row["error_detail"],
        "artifact_sha256": loaded["manifest"]["artifact_sha256"],
        "advisory_only": True,
    }


def validate_observations_payload(payload: Mapping[str, Any], *, trusted_now: datetime) -> None:
    if set(payload) != {"schema_version", "artifact_type", "run_id", "records"}:
        raise AdvisoryPriceIssue("ARTIFACT_SCHEMA_INVALID", "observation container fields are invalid")
    if payload.get("schema_version") != OBSERVATIONS_SCHEMA_VERSION or payload.get("artifact_type") != ARTIFACT_TYPE:
        raise AdvisoryPriceIssue("ARTIFACT_VERSION_INVALID", "observation container version is unsupported")
    _safe_identifier(payload.get("run_id"), "run_id")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise AdvisoryPriceIssue("ARTIFACT_SCHEMA_INVALID", "observations must be a non-empty list")
    seen: set[str] = set()
    order: list[tuple[str, str]] = []
    for row in records:
        _validate_record(row, run_id=payload["run_id"], trusted_now=trusted_now)
        if row["instrument_id"] in seen:
            raise AdvisoryPriceIssue("DUPLICATE_INSTRUMENT", "artifact contains duplicate instrument identity")
        seen.add(row["instrument_id"])
        order.append((row["instrument_id"], row["canonical_ticker"]))
    if order != sorted(order):
        raise AdvisoryPriceIssue("ARTIFACT_ORDER_INVALID", "records are not deterministically sorted")


def validate_manifest_payload(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "artifact_version", "artifact_type", "run_id", "generated_at",
        "source_main_sha", "universe_schema_version", "universe_source_path", "universe_sha256",
        "universe_snapshot_sha256", "freshness_policy_path", "freshness_policy_sha256",
        "source_adapter", "source_id", "observation_type", "expected_instrument_count",
        "status_counts", "observations_file", "observations_sha256", "integrity_algorithm",
        "artifact_sha256", "retention_days", "canonical_publication_status",
    }
    if set(manifest) != required:
        raise AdvisoryPriceIssue("ARTIFACT_SCHEMA_INVALID", "manifest fields are incomplete or unexpected")
    constants = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "source_adapter": SOURCE_ADAPTER,
        "source_id": SOURCE_ID,
        "observation_type": OBSERVATION_TYPE,
        "observations_file": OBSERVATIONS_FILE,
        "integrity_algorithm": "sha256-canonical-json-v1",
        "canonical_publication_status": "not_authorized_advisory_only",
    }
    if any(manifest.get(key) != value for key, value in constants.items()):
        raise AdvisoryPriceIssue("ARTIFACT_VERSION_INVALID", "manifest authority constants are invalid")
    _safe_identifier(manifest.get("run_id"), "run_id")
    _timestamp(manifest.get("generated_at"), "generated_at")
    if not isinstance(manifest.get("source_main_sha"), str) or not re.fullmatch(r"[0-9a-f]{40}", manifest["source_main_sha"]):
        raise AdvisoryPriceIssue("ARTIFACT_SCHEMA_INVALID", "source_main_sha is not a full Git SHA")
    for field in ("universe_schema_version", "universe_source_path", "freshness_policy_path"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise AdvisoryPriceIssue("ARTIFACT_SCHEMA_INVALID", f"{field} must be a non-empty string")
    for field in ("universe_sha256", "universe_snapshot_sha256", "freshness_policy_sha256", "observations_sha256", "artifact_sha256"):
        if not isinstance(manifest.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", manifest[field]):
            raise AdvisoryPriceIssue("ARTIFACT_SCHEMA_INVALID", f"{field} is not a SHA-256 digest")
    if not isinstance(manifest.get("expected_instrument_count"), int) or manifest["expected_instrument_count"] < 1:
        raise AdvisoryPriceIssue("ARTIFACT_SCHEMA_INVALID", "expected instrument count is invalid")
    if isinstance(manifest.get("retention_days"), bool) or not isinstance(manifest.get("retention_days"), int) or manifest["retention_days"] < 1:
        raise AdvisoryPriceIssue("ARTIFACT_SCHEMA_INVALID", "retention_days must be a positive integer")
    counts = manifest.get("status_counts")
    if not isinstance(counts, Mapping) or set(counts) != {"attempted", "fresh", "stale", "missing", "invalid"}:
        raise AdvisoryPriceIssue("ARTIFACT_SCHEMA_INVALID", "status counts are invalid")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values()):
        raise AdvisoryPriceIssue("ARTIFACT_SCHEMA_INVALID", "status counts must be non-negative integers")
    if counts["attempted"] != sum(counts[key] for key in ("fresh", "stale", "missing", "invalid")):
        raise AdvisoryPriceIssue("ARTIFACT_RECONCILIATION_INVALID", "status counts do not reconcile")


def _record_from_acquisition(
    instrument: Mapping[str, Any],
    acquired: Mapping[str, Any] | None,
    *,
    run_id: str,
    retrieval: datetime,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    base = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "run_id": run_id,
        "instrument_id": str(instrument["instrument_id"]),
        "canonical_ticker": str(instrument["symbol"]),
        "retrieval_timestamp": _utc_text(retrieval),
    }
    if acquired is None:
        return {**base, "price": None, "currency": instrument["currency"],
                "observation_type": OBSERVATION_TYPE, "observation_timestamp": None, "source_id": SOURCE_ID,
                "freshness_status": "missing", "observation_age_completed_sessions": None,
                "acquisition_status": "missing", "error_code": "PRICE_OBSERVATION_MISSING",
                "error_detail": "No observation was returned for the authoritative instrument."}
    if not isinstance(acquired, Mapping):
        acquired = {
            "acquisition_error_code": "ACQUISITION_RESPONSE_INVALID",
            "acquisition_error_detail": "Provider response is not an object.",
        }
    if acquired.get("acquisition_error_code"):
        code = str(acquired["acquisition_error_code"])
        detail = str(acquired.get("acquisition_error_detail") or "Acquisition failed.")[:500]
        if not ERROR_CODE.fullmatch(code):
            code = "ACQUISITION_RESPONSE_INVALID"
        row = {**base, "price": None, "currency": instrument["currency"],
               "observation_type": OBSERVATION_TYPE, "observation_timestamp": None, "source_id": SOURCE_ID,
               "freshness_status": "invalid", "observation_age_completed_sessions": None,
               "acquisition_status": "failed", "error_code": code,
               "error_detail": detail}
        _validate_record(row, run_id=run_id, trusted_now=datetime.now(UTC))
        return row
    try:
        if acquired.get("instrument_id") != instrument["instrument_id"] or acquired.get("canonical_ticker") != instrument["symbol"]:
            raise AdvisoryPriceIssue("INSTRUMENT_IDENTITY_MISMATCH", "acquisition identity differs from canonical universe")
        if acquired.get("source_id") != SOURCE_ID:
            raise AdvisoryPriceIssue("SOURCE_ID_INVALID", "acquisition source identity is unsupported")
        if acquired.get("observation_type") != OBSERVATION_TYPE:
            raise AdvisoryPriceIssue("OBSERVATION_TYPE_INVALID", "observation type is unsupported")
        price = _price_text(acquired.get("price"))
        currency = str(acquired.get("currency") or "")
        if currency != instrument.get("currency") or currency not in policy["supported_currencies"]:
            raise AdvisoryPriceIssue("CURRENCY_INVALID", "observation currency differs from canonical instrument")
        observed = _timestamp(acquired.get("observation_timestamp"), "observation_timestamp")
        if observed > retrieval:
            raise AdvisoryPriceIssue("OBSERVATION_AFTER_RETRIEVAL", "observation timestamp follows retrieval")
        _profile, expected = expected_completed_session(instrument, retrieval)
        if expected is None:
            raise AdvisoryPriceIssue("EXPECTED_SESSION_UNAVAILABLE", "completed session cannot be resolved")
        age = _weekday_session_lag(observed.date(), expected)
        if age < 0:
            raise AdvisoryPriceIssue("FUTURE_OBSERVATION", "observation follows expected completed session")
        freshness = "fresh" if age <= policy["max_completed_session_lag"] else "stale"
        row = {**base, "price": price, "currency": currency,
               "observation_type": OBSERVATION_TYPE, "observation_timestamp": _utc_text(observed),
               "source_id": SOURCE_ID, "freshness_status": freshness,
               "observation_age_completed_sessions": age, "acquisition_status": "succeeded",
               "error_code": None, "error_detail": None}
    except AdvisoryPriceIssue as exc:
        row = {**base, "price": None, "currency": instrument["currency"],
               "observation_type": OBSERVATION_TYPE, "observation_timestamp": None, "source_id": SOURCE_ID,
               "freshness_status": "invalid", "observation_age_completed_sessions": None,
               "acquisition_status": "failed", "error_code": exc.code,
               "error_detail": str(exc).split(": ", 1)[-1][:500]}
    _validate_record(row, run_id=run_id, trusted_now=datetime.now(UTC))
    return row


def _validate_record(row: Any, *, run_id: str, trusted_now: datetime) -> None:
    fields = {
        "schema_version", "artifact_version", "run_id", "instrument_id", "canonical_ticker",
        "price", "currency", "observation_type", "observation_timestamp", "retrieval_timestamp",
        "source_id", "freshness_status", "observation_age_completed_sessions",
        "acquisition_status", "error_code", "error_detail",
    }
    if not isinstance(row, Mapping) or set(row) != fields:
        raise AdvisoryPriceIssue("ARTIFACT_SCHEMA_INVALID", "observation fields are incomplete or unexpected")
    if row.get("schema_version") != OBSERVATION_SCHEMA_VERSION or row.get("artifact_version") != ARTIFACT_VERSION:
        raise AdvisoryPriceIssue("ARTIFACT_VERSION_INVALID", "observation version is unsupported")
    if row.get("run_id") != run_id:
        raise AdvisoryPriceIssue("RUN_ID_MISMATCH", "observation run ID differs from container")
    _safe_identifier(row.get("instrument_id"), "instrument_id")
    if not isinstance(row.get("canonical_ticker"), str) or not TICKER.fullmatch(row["canonical_ticker"]):
        raise AdvisoryPriceIssue("INSTRUMENT_IDENTITY_INVALID", "canonical ticker is malformed")
    retrieval = _timestamp(row.get("retrieval_timestamp"), "retrieval_timestamp")
    if retrieval > trusted_now:
        raise AdvisoryPriceIssue("FUTURE_RETRIEVAL_TIMESTAMP", "retrieval timestamp cannot be in the future")
    acquisition = row.get("acquisition_status")
    freshness = row.get("freshness_status")
    if row.get("currency") is None or row.get("observation_type") != OBSERVATION_TYPE or row.get("source_id") != SOURCE_ID:
        raise AdvisoryPriceIssue("OBSERVATION_SEMANTICS_INVALID", "observation provenance metadata is invalid")
    if acquisition == "succeeded":
        _price_text(row.get("price"))
        observed = _timestamp(row.get("observation_timestamp"), "observation_timestamp")
        if observed > retrieval:
            raise AdvisoryPriceIssue("OBSERVATION_AFTER_RETRIEVAL", "observation timestamp follows retrieval")
        age = row.get("observation_age_completed_sessions")
        if isinstance(age, bool) or not isinstance(age, int) or age < 0 or freshness not in {"fresh", "stale"}:
            raise AdvisoryPriceIssue("FRESHNESS_INVALID", "successful freshness evidence is invalid")
        if row.get("error_code") is not None or row.get("error_detail") is not None:
            raise AdvisoryPriceIssue("OBSERVATION_SEMANTICS_INVALID", "successful observation cannot carry an error")
    elif acquisition in {"missing", "failed"}:
        if any(row.get(field) is not None for field in ("price", "observation_timestamp", "observation_age_completed_sessions")):
            raise AdvisoryPriceIssue("OBSERVATION_SEMANTICS_INVALID", "failed observation cannot carry price evidence")
        if freshness not in {"missing", "invalid"} or not isinstance(row.get("error_code"), str) or not ERROR_CODE.fullmatch(row["error_code"]):
            raise AdvisoryPriceIssue("OBSERVATION_SEMANTICS_INVALID", "failed observation status is invalid")
        if not isinstance(row.get("error_detail"), str) or not row["error_detail"].strip() or len(row["error_detail"]) > 500:
            raise AdvisoryPriceIssue("OBSERVATION_SEMANTICS_INVALID", "failed observation detail is invalid")
    else:
        raise AdvisoryPriceIssue("ACQUISITION_STATUS_INVALID", "acquisition status is unsupported")


def _validate_complete_universe(records: Sequence[Mapping[str, Any]], instruments: Sequence[Mapping[str, Any]]) -> None:
    actual = {(row["instrument_id"], row["canonical_ticker"]) for row in records}
    expected = {(str(row["instrument_id"]), str(row["symbol"])) for row in instruments}
    if len(actual) != len(records):
        raise AdvisoryPriceIssue("DUPLICATE_INSTRUMENT", "duplicate universe records detected")
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise AdvisoryPriceIssue("MISSING_INSTRUMENT", f"artifact omits {len(missing)} canonical instruments")
    if extra:
        raise AdvisoryPriceIssue("UNEXPECTED_INSTRUMENT", f"artifact adds {len(extra)} non-canonical instruments")


def _validate_loaded_freshness(
    records: Sequence[Mapping[str, Any]],
    instruments: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
    generated_at: datetime,
) -> None:
    instruments_by_id = {str(row["instrument_id"]): row for row in instruments}
    expected_retrieval = _utc_text(generated_at)
    for row in records:
        if row["retrieval_timestamp"] != expected_retrieval:
            raise AdvisoryPriceIssue("RETRIEVAL_BINDING_INVALID", "record retrieval differs from manifest generation")
        instrument = instruments_by_id[row["instrument_id"]]
        if row["currency"] != instrument.get("currency") or row["currency"] not in policy["supported_currencies"]:
            raise AdvisoryPriceIssue("CURRENCY_INVALID", "artifact currency differs from canonical instrument")
        if row["acquisition_status"] != "succeeded":
            continue
        observed = _timestamp(row["observation_timestamp"], "observation_timestamp")
        _profile, expected_session = expected_completed_session(instrument, generated_at)
        if expected_session is None:
            raise AdvisoryPriceIssue("EXPECTED_SESSION_UNAVAILABLE", "completed session cannot be resolved")
        expected_age = _weekday_session_lag(observed.date(), expected_session)
        expected_status = "fresh" if expected_age <= policy["max_completed_session_lag"] else "stale"
        if expected_age < 0 or row["observation_age_completed_sessions"] != expected_age or row["freshness_status"] != expected_status:
            raise AdvisoryPriceIssue("FRESHNESS_INVALID", "artifact freshness evidence does not match trusted policy")


def _acquire_with_existing_adapter(instruments: Sequence[Mapping[str, Any]], retrieval: datetime) -> Mapping[str, Mapping[str, Any]]:
    grouped_by_provider_symbol: dict[str, list[Mapping[str, Any]]] = {}
    result: dict[str, Mapping[str, Any]] = {}
    for instrument in instruments:
        if instrument.get("source_mapping_status") != "mapped":
            result[str(instrument["instrument_id"])] = {
                "acquisition_error_code": "SOURCE_MAPPING_UNAUTHORIZED",
                "acquisition_error_detail": "Canonical universe has no authorized provider mapping.",
            }
            continue
        provider_symbol = _to_yfinance_symbol(str(instrument["source_symbol"]))
        grouped_by_provider_symbol.setdefault(provider_symbol, []).append(instrument)
    by_provider_symbol: dict[str, Mapping[str, Any]] = {}
    for provider_symbol, mapped_instruments in grouped_by_provider_symbol.items():
        if len(mapped_instruments) != 1:
            for instrument in mapped_instruments:
                result[str(instrument["instrument_id"])] = {
                    "acquisition_error_code": "SOURCE_MAPPING_AMBIGUOUS",
                    "acquisition_error_detail": "Provider symbol maps to multiple canonical instruments.",
                }
            continue
        by_provider_symbol[provider_symbol] = mapped_instruments[0]
    requests = sorted(by_provider_symbol)
    start = (retrieval.date() - timedelta(days=14)).isoformat()
    end = (retrieval.date() + timedelta(days=1)).isoformat()
    if requests:
        try:
            frames = download_yfinance_batch(requests, start, end)
        except Exception:
            frames = {}
    else:
        frames = {}
    errors: dict[str, str] = {}
    for provider_symbol in requests:
        if provider_symbol in frames and frames[provider_symbol] is not None and not frames[provider_symbol].empty:
            continue
        for _attempt in range(2):
            try:
                frame = _download_yfinance_history(provider_symbol, start, end)
            except Exception as exc:
                errors[provider_symbol] = type(exc).__name__
                continue
            if frame is not None and not frame.empty:
                frames[provider_symbol] = frame
                break
    for provider_symbol, instrument in by_provider_symbol.items():
        frame = frames.get(provider_symbol)
        if frame is None or frame.empty:
            result[str(instrument["instrument_id"])] = {
                "acquisition_error_code": "ACQUISITION_FAILED",
                "acquisition_error_detail": errors.get(provider_symbol, "No provider rows returned."),
            }
    for provider_symbol, frame in frames.items():
        instrument = by_provider_symbol.get(provider_symbol)
        if instrument is None or frame is None or frame.empty:
            continue
        try:
            last = frame.sort_values("Date").iloc[-1]
            observed_date = date.fromisoformat(str(last["Date"])[:10])
            result[str(instrument["instrument_id"])] = {
                "instrument_id": instrument["instrument_id"],
                "canonical_ticker": instrument["symbol"],
                "price": _provider_decimal_text(last["Close"]),
                "currency": instrument["currency"],
                "observation_type": OBSERVATION_TYPE,
                "observation_timestamp": _completed_session_close_timestamp(instrument, observed_date),
                "source_id": SOURCE_ID,
            }
        except (AdvisoryPriceIssue, KeyError, TypeError, ValueError) as exc:
            result[str(instrument["instrument_id"])] = {
                "acquisition_error_code": exc.code if isinstance(exc, AdvisoryPriceIssue) else "ACQUISITION_RESPONSE_INVALID",
                "acquisition_error_detail": str(exc)[:500] or type(exc).__name__,
            }
    return result


def _completed_session_close_timestamp(instrument: Mapping[str, Any], observed_date: date) -> str:
    profile, _expected = expected_completed_session(instrument, datetime.combine(observed_date, time(23, 59), UTC))
    if profile is None:
        raise AdvisoryPriceIssue("EXPECTED_SESSION_UNAVAILABLE", "market profile is unavailable")
    local_close = datetime.combine(observed_date, profile.close_time, ZoneInfo(profile.timezone))
    return _utc_text(local_close.astimezone(UTC))


def _weekday_session_lag(observed: date, expected: date) -> int:
    if observed > expected:
        return -1
    lag = 0
    cursor = observed
    while cursor < expected:
        cursor = date.fromordinal(cursor.toordinal() + 1)
        if cursor.weekday() < 5:
            lag += 1
    return lag


def _load_policy(path: Path) -> dict[str, Any]:
    policy = _load_json(path)
    required = {"schema_version", "policy_id", "observation_type", "max_completed_session_lag", "age_method", "artifact_retention_days", "supported_currencies", "limitations"}
    if set(policy) != required or policy.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise AdvisoryPriceIssue("FRESHNESS_POLICY_INVALID", "freshness policy fields or version are invalid")
    if policy.get("observation_type") != OBSERVATION_TYPE or policy.get("age_method") != "weekday_completed_session_lag_v1":
        raise AdvisoryPriceIssue("FRESHNESS_POLICY_INVALID", "freshness policy semantics are unsupported")
    if isinstance(policy.get("max_completed_session_lag"), bool) or not isinstance(policy.get("max_completed_session_lag"), int) or policy["max_completed_session_lag"] < 0:
        raise AdvisoryPriceIssue("FRESHNESS_POLICY_INVALID", "freshness lag must be non-negative")
    if not isinstance(policy.get("artifact_retention_days"), int) or policy["artifact_retention_days"] < 1:
        raise AdvisoryPriceIssue("FRESHNESS_POLICY_INVALID", "retention must be positive")
    currencies = policy.get("supported_currencies")
    if not isinstance(currencies, list) or not currencies or currencies != sorted(set(currencies)):
        raise AdvisoryPriceIssue("FRESHNESS_POLICY_INVALID", "supported currencies must be unique and sorted")
    return dict(policy)


def _price_text(value: Any) -> str:
    if isinstance(value, (bool, float)) or not isinstance(value, str) or not DECIMAL_TEXT.fullmatch(value):
        raise AdvisoryPriceIssue("PRICE_INVALID", "price must be a canonical positive decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise AdvisoryPriceIssue("PRICE_INVALID", "price is malformed") from exc
    if not parsed.is_finite() or parsed <= 0 or _decimal_text(parsed) != value:
        raise AdvisoryPriceIssue("PRICE_INVALID", "price must be finite, positive and losslessly normalized")
    return value


def _provider_decimal_text(value: Any) -> str:
    if isinstance(value, bool):
        raise AdvisoryPriceIssue("PRICE_INVALID", "provider price is boolean")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise AdvisoryPriceIssue("PRICE_INVALID", "provider price is malformed") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise AdvisoryPriceIssue("PRICE_INVALID", "provider price must be finite and positive")
    return _decimal_text(parsed)


def _decimal_text(value: Decimal) -> str:
    with localcontext() as context:
        context.prec = 50
        return format(+value, "f")


def _safe_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise AdvisoryPriceIssue("IDENTIFIER_INVALID", f"{field} must be a safe identifier")
    return value


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise AdvisoryPriceIssue("TIMESTAMP_INVALID", f"{field} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AdvisoryPriceIssue("TIMESTAMP_INVALID", f"{field} is malformed") from exc
    if parsed.tzinfo is None:
        raise AdvisoryPriceIssue("TIMESTAMP_INVALID", f"{field} requires a timezone")
    normalized = parsed.astimezone(UTC)
    if _utc_text(normalized) != value:
        raise AdvisoryPriceIssue("TIMESTAMP_INVALID", f"{field} must be canonical UTC")
    return normalized


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256(path.read_bytes())
    except OSError as exc:
        raise AdvisoryPriceIssue("ARTIFACT_READ_FAILED", f"cannot read {path.name}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise AdvisoryPriceIssue("ARTIFACT_READ_FAILED", f"cannot load {path.name}") from exc
    if not isinstance(value, dict):
        raise AdvisoryPriceIssue("ARTIFACT_SCHEMA_INVALID", f"{path.name} must contain an object")
    return value


def _write_bytes(path: Path, value: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()


def run_command(argv: Sequence[str] | None = None, *, stdout: TextIO = sys.stdout, stderr: TextIO = sys.stderr) -> int:
    parser = argparse.ArgumentParser(description="Build or consume non-canonical ME-SR25 advisory price evidence.")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--run-id", required=True)
    build.add_argument("--source-main-sha", required=True)
    build.add_argument("--output-root", required=True)
    build.add_argument("--universe", default=DEFAULT_UNIVERSE_SNAPSHOT.as_posix())
    build.add_argument("--policy", default=DEFAULT_POLICY_PATH.as_posix())
    consume = commands.add_parser("consume")
    consume.add_argument("--artifact-root", required=True)
    consume.add_argument("--instrument-id", required=True)
    consume.add_argument("--ticker", required=True)
    consume.add_argument("--universe", default=DEFAULT_UNIVERSE_SNAPSHOT.as_posix())
    consume.add_argument("--policy", default=DEFAULT_POLICY_PATH.as_posix())
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            manifest, destination = build_advisory_price_artifact(
                run_id=args.run_id, source_main_sha=args.source_main_sha,
                output_root=args.output_root, universe_path=args.universe, policy_path=args.policy,
            )
            result = {"manifest": manifest, "artifact_path": destination.as_posix()}
        else:
            result = consume_advisory_price_context(
                args.artifact_root, instrument_id=args.instrument_id,
                canonical_ticker=args.ticker, universe_path=args.universe, policy_path=args.policy,
            )
    except AdvisoryPriceIssue as exc:
        print(f"ERROR: {exc}", file=stderr)
        return 2
    json.dump(result, stdout, indent=2, sort_keys=True)
    stdout.write("\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_command(argv)


if __name__ == "__main__":
    raise SystemExit(main())
