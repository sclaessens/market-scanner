from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from market_engine.source_refresh.cached_source_snapshot_inventory import (
    CACHED_SOURCE_SNAPSHOT_ACQUISITION_MANIFEST_FORMAT_VERSION,
)


REQUEST_FORMAT = "market-engine-automated-cached-source-acquisition-request-v1"
RESULT_FORMAT = "market-engine-automated-cached-source-acquisition-result-v1"
SUPPORTED_RUN_MODES = frozenset({"dry_run", "local_non_production"})
SUPPORTED_SOURCE_FAMILIES = frozenset({"company_profile"})
SUPPORTED_TICKERS = frozenset({"NVDA", "AMD", "ASML"})
REQUIRED_FALSE_SAFETY_FLAGS = (
    "allow_provider_calls",
    "allow_network",
    "allow_production_writes",
    "allow_telegram_send",
    "allow_portfolio_writes",
    "allow_watchlist_writes",
    "allow_broker_actions",
)
TICKER_PATTERN = re.compile(r"^[A-Z0-9.-]{1,12}$")


class AutomatedCachedSourceAcquisitionError(ValueError):
    """Raised when an acquisition request is invalid before execution."""

    def __init__(self, reason: str, *, issues: Sequence[str]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.issues = tuple(issues)


@dataclass(frozen=True)
class CompanyProfilePayload:
    ticker: str
    entity_name: str
    entity_country: str
    entity_exchange: str
    source_timestamp: str | None
    retrieved_at: str
    payload: Mapping[str, Any]
    provenance: Mapping[str, Any]


class CompanyProfileAdapter(Protocol):
    adapter_id: str
    adapter_version: str
    provider_name: str

    def acquire(
        self,
        *,
        ticker: str,
        retrieved_at: str,
    ) -> CompanyProfilePayload:
        ...


class DeterministicFakeCompanyProfileAdapter:
    adapter_id = "fake_company_profile_adapter"
    adapter_version = "test-v1"
    provider_name = "deterministic_fake_provider"

    def __init__(self, *, failing_tickers: Sequence[str] = ()) -> None:
        self._failing_tickers = frozenset(ticker.upper() for ticker in failing_tickers)

    def acquire(
        self,
        *,
        ticker: str,
        retrieved_at: str,
    ) -> CompanyProfilePayload:
        normalized_ticker = ticker.upper()
        if normalized_ticker in self._failing_tickers:
            raise RuntimeError(f"deterministic adapter failure for {normalized_ticker}")
        profile = _profile_for_ticker(normalized_ticker)
        canonical_source_identity = f"fake://company_profile/{normalized_ticker}"
        payload = {
            "payload_format": "market-engine-company-profile-snapshot-v1",
            "ticker": normalized_ticker,
            "entity_name": profile["entity_name"],
            "entity_country": profile["entity_country"],
            "entity_exchange": profile["entity_exchange"],
            "source_family": "company_profile",
            "profile": {
                "business_summary": (
                    f"Deterministic non-production company profile for "
                    f"{normalized_ticker}."
                ),
                "missing_data": [],
            },
            "provenance": {
                "adapter_id": self.adapter_id,
                "adapter_version": self.adapter_version,
                "provider_name": self.provider_name,
                "canonical_source_identity": canonical_source_identity,
                "retrieved_at": retrieved_at,
                "source_timestamp": None,
                "request_metadata": {
                    "network_used": False,
                    "provider_calls_performed": False,
                    "deterministic_fake_adapter": True,
                },
            },
        }
        return CompanyProfilePayload(
            ticker=normalized_ticker,
            entity_name=str(profile["entity_name"]),
            entity_country=str(profile["entity_country"]),
            entity_exchange=str(profile["entity_exchange"]),
            source_timestamp=None,
            retrieved_at=retrieved_at,
            payload=payload,
            provenance=payload["provenance"],
        )


def run_automated_cached_source_acquisition(
    request: Mapping[str, Any],
    *,
    company_profile_adapter: CompanyProfileAdapter | None = None,
) -> dict[str, Any]:
    validated = _validate_request(request)
    adapter = company_profile_adapter or DeterministicFakeCompanyProfileAdapter()
    destination_root = Path(validated["destination_root"])
    entries = []
    for ticker in validated["tickers"]:
        for source_family in validated["source_families"]:
            if ticker not in SUPPORTED_TICKERS:
                entries.append(
                    _entry_without_artifacts(
                        ticker=ticker,
                        source_family=source_family,
                        status="unsupported",
                        issues=("unsupported_ticker",),
                    )
                )
                continue
            if source_family != "company_profile":
                entries.append(
                    _entry_without_artifacts(
                        ticker=ticker,
                        source_family=source_family,
                        status="unsupported",
                        issues=("unsupported_source_family",),
                    )
                )
                continue
            entries.append(
                _acquire_company_profile_entry(
                    ticker=ticker,
                    request=validated,
                    destination_root=destination_root,
                    adapter=adapter,
                )
            )

    result = {
        "result_format": RESULT_FORMAT,
        "request_id": validated["request_id"],
        "run_id": validated["request_id"],
        "generated_at": validated["generated_at"],
        "destination_root": destination_root.as_posix(),
        "entries": entries,
        "summary": _summary(
            entries=entries,
            requested_ticker_count=len(validated["tickers"]),
            requested_source_family_count=len(validated["source_families"]),
        ),
        "safety": {
            "provider_calls_performed": False,
            "network_used": False,
            "telegram_sent": False,
            "portfolio_written": False,
            "watchlist_written": False,
            "broker_action_performed": False,
            "production_write_performed": False,
        },
        "next_step": {
            "recommended_action": "run_existing_import_staging_validation",
            "import_candidate_root": destination_root.as_posix(),
            "dry_run_candidate": any(
                entry["status"] in {"completed", "completed_with_limitations"}
                for entry in entries
            ),
            "blocked_reason": None
            if any(
                entry["status"] in {"completed", "completed_with_limitations"}
                for entry in entries
            )
            else "no_usable_acquisition_entries",
        },
    }
    _write_result_payload(destination_root=destination_root, result=result)
    return result


def _validate_request(request: Mapping[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    if request.get("request_format") != REQUEST_FORMAT:
        issues.append("request_format_invalid")
    request_id = _required_text(request, "request_id", issues)
    requested_at = _required_text(request, "requested_at", issues)
    generated_at = str(request.get("generated_at") or requested_at or _utc_now_text())
    run_mode = _required_text(request, "run_mode", issues)
    if run_mode and run_mode not in SUPPORTED_RUN_MODES:
        issues.append("run_mode_unsupported")

    ticker_source = request.get("ticker_source")
    if not isinstance(ticker_source, Mapping):
        issues.append("ticker_source_invalid")
        ticker_source_mode = None
    else:
        ticker_source_mode = ticker_source.get("mode")
        if ticker_source_mode != "explicit_list":
            issues.append("ticker_source_mode_unsupported")

    tickers = _validated_tickers(request.get("tickers"), issues)
    source_families = _validated_source_families(request.get("source_families"), issues)
    destination_root = _validated_destination_root(
        request.get("destination_root"),
        issues,
    )
    _validate_freshness_policy(request.get("freshness_policy"), source_families, issues)
    _validate_provider_policy(
        request.get("provider_policy"),
        source_families,
        run_mode,
        issues,
    )
    _validate_safety_flags(request.get("safety_flags"), issues)

    if issues:
        raise AutomatedCachedSourceAcquisitionError(
            "automated cached-source acquisition request is invalid",
            issues=tuple(sorted(set(issues))),
        )
    return {
        "request_id": request_id,
        "requested_at": requested_at,
        "generated_at": generated_at,
        "run_mode": run_mode,
        "ticker_source_mode": ticker_source_mode,
        "tickers": tickers,
        "source_families": source_families,
        "destination_root": destination_root,
        "freshness_policy": request["freshness_policy"],
        "provider_policy": request["provider_policy"],
        "operator_context": request.get("operator_context") or {},
    }


def _validated_tickers(value: object, issues: list[str]) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        issues.append("tickers_invalid")
        return ()
    if not value:
        issues.append("tickers_empty")
        return ()
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            issues.append("ticker_invalid")
            continue
        ticker = item.strip().upper()
        if ticker != item:
            issues.append("ticker_must_be_uppercase_without_padding")
        if not TICKER_PATTERN.fullmatch(ticker):
            issues.append("ticker_invalid")
            continue
        if "/" in ticker or "\\" in ticker or "://" in ticker:
            issues.append("ticker_invalid")
            continue
        if ticker in seen:
            issues.append("ticker_duplicate")
            continue
        seen.add(ticker)
        normalized.append(ticker)
    return tuple(normalized)


def _validated_source_families(value: object, issues: list[str]) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        issues.append("source_families_invalid")
        return ()
    if not value:
        issues.append("source_families_empty")
        return ()
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            issues.append("source_family_invalid")
            continue
        source_family = item.strip()
        if source_family not in SUPPORTED_SOURCE_FAMILIES:
            issues.append("source_family_unsupported")
            continue
        if source_family in seen:
            issues.append("source_family_duplicate")
            continue
        seen.add(source_family)
        normalized.append(source_family)
    return tuple(normalized)


def _validated_destination_root(value: object, issues: list[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        issues.append("destination_root_required")
        return ""
    destination = Path(value)
    if destination.is_absolute():
        return destination.as_posix()
    parts = destination.parts
    if not parts:
        issues.append("destination_root_required")
    if parts and parts[0] in {"data", "operator_input"}:
        issues.append("destination_root_must_be_non_production")
    if ".." in parts:
        issues.append("destination_root_must_not_escape")
    return destination.as_posix()


def _validate_freshness_policy(
    value: object,
    source_families: Sequence[str],
    issues: list[str],
) -> None:
    if not isinstance(value, Mapping):
        issues.append("freshness_policy_invalid")
        return
    per_source_family = value.get("per_source_family")
    default_max_age_days = value.get("default_max_age_days")
    if default_max_age_days is not None and (
        not isinstance(default_max_age_days, int) or default_max_age_days <= 0
    ):
        issues.append("freshness_default_max_age_days_invalid")
    if per_source_family is not None and not isinstance(per_source_family, Mapping):
        issues.append("freshness_per_source_family_invalid")
        return
    for source_family in source_families:
        family_policy = (
            per_source_family.get(source_family)
            if isinstance(per_source_family, Mapping)
            else None
        )
        if family_policy is None and default_max_age_days is None:
            issues.append("freshness_policy_missing_for_source_family")
            continue
        if family_policy is not None:
            if not isinstance(family_policy, Mapping):
                issues.append("freshness_family_policy_invalid")
                continue
            max_age_days = family_policy.get("max_age_days", default_max_age_days)
            if not isinstance(max_age_days, int) or max_age_days <= 0:
                issues.append("freshness_max_age_days_invalid")
            if not isinstance(family_policy.get("source_timestamp_required"), bool):
                issues.append("freshness_source_timestamp_required_invalid")


def _validate_provider_policy(
    value: object,
    source_families: Sequence[str],
    run_mode: str | None,
    issues: list[str],
) -> None:
    if not isinstance(value, Mapping):
        issues.append("provider_policy_invalid")
        return
    if value.get("allow_hidden_fallback") is not False:
        issues.append("provider_policy_hidden_fallback_must_be_false")
    if value.get("allow_silent_substitution") is not False:
        issues.append("provider_policy_silent_substitution_must_be_false")
    if value.get("allow_fabricated_data") is not False:
        issues.append("provider_policy_fabricated_data_must_be_false")
    adapters = value.get("approved_adapters")
    if not isinstance(adapters, Sequence) or isinstance(adapters, (str, bytes)):
        issues.append("provider_policy_approved_adapters_invalid")
        return
    for source_family in source_families:
        if not any(
            _adapter_supports_source_family(
                adapter,
                source_family=source_family,
                run_mode=run_mode,
            )
            for adapter in adapters
        ):
            issues.append("approved_adapter_missing_for_source_family")


def _adapter_supports_source_family(
    adapter: object,
    *,
    source_family: str,
    run_mode: str | None,
) -> bool:
    if not isinstance(adapter, Mapping):
        return False
    source_families = adapter.get("source_families")
    allowed_run_modes = adapter.get("allowed_run_modes")
    return (
        isinstance(adapter.get("adapter_id"), str)
        and isinstance(adapter.get("adapter_version"), str)
        and isinstance(source_families, Sequence)
        and not isinstance(source_families, (str, bytes))
        and source_family in source_families
        and isinstance(allowed_run_modes, Sequence)
        and not isinstance(allowed_run_modes, (str, bytes))
        and run_mode in allowed_run_modes
        and adapter.get("network_required") is False
    )


def _validate_safety_flags(value: object, issues: list[str]) -> None:
    if not isinstance(value, Mapping):
        issues.append("safety_flags_invalid")
        return
    for flag in REQUIRED_FALSE_SAFETY_FLAGS:
        if value.get(flag) is not False:
            issues.append(f"{flag}_must_be_false")


def _required_text(
    request: Mapping[str, Any],
    field_name: str,
    issues: list[str],
) -> str | None:
    value = request.get(field_name)
    if not isinstance(value, str) or not value.strip():
        issues.append(f"{field_name}_required")
        return None
    return value.strip()


def _acquire_company_profile_entry(
    *,
    ticker: str,
    request: Mapping[str, Any],
    destination_root: Path,
    adapter: CompanyProfileAdapter,
) -> dict[str, Any]:
    retrieved_at = str(request["generated_at"])
    batch_id = str(request["request_id"])
    snapshot_id = f"{ticker}-company_profile-{batch_id}"
    snapshot_dir = destination_root / ticker / "company_profile"
    if snapshot_dir.exists():
        return _entry_without_artifacts(
            ticker=ticker,
            source_family="company_profile",
            status="blocked",
            issues=("snapshot_path_already_exists",),
        )
    try:
        payload = adapter.acquire(ticker=ticker, retrieved_at=retrieved_at)
    except RuntimeError:
        return _entry_without_artifacts(
            ticker=ticker,
            source_family="company_profile",
            status="provider_error",
            issues=("adapter_error",),
        )
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    payload_path = snapshot_dir / "company_profile.json"
    _write_json(payload_path, payload.payload)
    payload_hash = _sha256(payload_path)
    payload_size = payload_path.stat().st_size
    manifest_path = snapshot_dir / "manifest.json"
    manifest = _manifest_payload(
        request=request,
        payload=payload,
        batch_id=batch_id,
        snapshot_id=snapshot_id,
        payload_hash=payload_hash,
        payload_size=payload_size,
    )
    _write_json(manifest_path, manifest)
    provenance = {
        **payload.provenance,
        "payload_sha256": payload_hash,
        "payload_size_bytes": payload_size,
        "snapshot_id": snapshot_id,
        "batch_id": batch_id,
    }
    freshness = _freshness_for_source_family(
        request=request,
        source_family="company_profile",
        source_timestamp=payload.source_timestamp,
    )
    return {
        "ticker": ticker,
        "source_family": "company_profile",
        "status": "completed",
        "snapshot_path": snapshot_dir.as_posix(),
        "manifest_path": manifest_path.as_posix(),
        "payload_paths": (payload_path.as_posix(),),
        "provenance": provenance,
        "freshness": freshness,
        "issues": (),
    }


def _manifest_payload(
    *,
    request: Mapping[str, Any],
    payload: CompanyProfilePayload,
    batch_id: str,
    snapshot_id: str,
    payload_hash: str,
    payload_size: int,
) -> dict[str, Any]:
    return {
        "manifest_format_version": CACHED_SOURCE_SNAPSHOT_ACQUISITION_MANIFEST_FORMAT_VERSION,
        "snapshot_id": snapshot_id,
        "batch_id": batch_id,
        "created_at_utc": str(request["generated_at"]),
        "acquired_at_utc": payload.retrieved_at,
        "acquisition_mode": "automated_dry_run",
        "source_family": "company_profile",
        "source_name": payload.provenance["provider_name"],
        "source_url": payload.provenance["canonical_source_identity"],
        "source_license_note": (
            "Deterministic local non-production company_profile adapter output. "
            "No network or live provider call was performed."
        ),
        "redistribution_allowed": False,
        "local_use_allowed": True,
        "commit_allowed": False,
        "source_material_type": "company_profile_snapshot",
        "ticker": payload.ticker,
        "entity_name": payload.entity_name,
        "entity_country": payload.entity_country,
        "entity_exchange": payload.entity_exchange,
        "source_entity_identifier": payload.ticker,
        "cik": None,
        "requested_document_type": "company_profile",
        "resolved_document_type": "company_profile",
        "requested_period": "latest",
        "resolved_period": "latest",
        "source_publication_date": payload.source_timestamp,
        "source_retrieved_at_utc": payload.retrieved_at,
        "local_snapshot_path": "company_profile.json",
        "local_manifest_path": "manifest.json",
        "local_payload_sha256": payload_hash,
        "local_payload_size_bytes": payload_size,
        "payload_mime_type": "application/json",
        "payload_encoding": "utf-8",
        "normalization_status": "raw_only",
        "validation_status": "passed",
        "validation_errors": [],
        "validation_warnings": [],
        "staleness_status": "fresh",
        "staleness_reason": "Source timestamp is optional for ME-SA02 company_profile.",
        "usable_for_cached_source_dry_run": True,
        "blocked_reason": None,
        "notes": (
            "ME-SA02 bounded automated cached-source acquisition package. "
            "Existing cached-source dry-run may still block if source-family "
            "consumption is not implemented downstream."
        ),
    }


def _freshness_for_source_family(
    *,
    request: Mapping[str, Any],
    source_family: str,
    source_timestamp: str | None,
) -> dict[str, Any]:
    policy = request["freshness_policy"]
    family_policy = policy.get("per_source_family", {}).get(source_family, {})
    max_age_days = family_policy.get("max_age_days", policy.get("default_max_age_days"))
    source_timestamp_required = bool(family_policy.get("source_timestamp_required"))
    return {
        "retrieval_timestamp_required": True,
        "source_timestamp_required": source_timestamp_required,
        "max_age_days": max_age_days,
        "state": "current"
        if source_timestamp is not None or not source_timestamp_required
        else "unknown_source_timestamp",
    }


def _entry_without_artifacts(
    *,
    ticker: str,
    source_family: str,
    status: str,
    issues: Sequence[str],
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "source_family": source_family,
        "status": status,
        "snapshot_path": None,
        "manifest_path": None,
        "payload_paths": (),
        "provenance": None,
        "freshness": None,
        "issues": tuple(issues),
    }


def _summary(
    *,
    entries: Sequence[Mapping[str, Any]],
    requested_ticker_count: int,
    requested_source_family_count: int,
) -> dict[str, int]:
    return {
        "requested_ticker_count": requested_ticker_count,
        "requested_source_family_count": requested_source_family_count,
        "entry_count": len(entries),
        "completed_count": _status_count(entries, "completed"),
        "completed_with_limitations_count": _status_count(
            entries,
            "completed_with_limitations",
        ),
        "blocked_count": _status_count(entries, "blocked"),
        "rejected_count": _status_count(entries, "rejected"),
        "provider_error_count": _status_count(entries, "provider_error"),
        "unsupported_count": _status_count(entries, "unsupported"),
        "stale_count": _status_count(entries, "stale"),
        "invalid_manifest_count": _status_count(entries, "invalid_manifest"),
    }


def _status_count(entries: Sequence[Mapping[str, Any]], status: str) -> int:
    return sum(1 for entry in entries if entry["status"] == status)


def _write_result_payload(
    *,
    destination_root: Path,
    result: Mapping[str, Any],
) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    _write_json(destination_root / "acquisition_result.json", result)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now_text() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _profile_for_ticker(ticker: str) -> Mapping[str, str]:
    profiles = {
        "NVDA": {
            "entity_name": "NVIDIA Corporation",
            "entity_country": "US",
            "entity_exchange": "NASDAQ",
        },
        "AMD": {
            "entity_name": "Advanced Micro Devices, Inc.",
            "entity_country": "US",
            "entity_exchange": "NASDAQ",
        },
        "ASML": {
            "entity_name": "ASML Holding N.V.",
            "entity_country": "NL",
            "entity_exchange": "NASDAQ",
        },
    }
    return profiles[ticker]
