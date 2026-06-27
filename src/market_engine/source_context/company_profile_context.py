from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


COMPANY_PROFILE_SOURCE_CONTEXT_FORMAT_VERSION = (
    "market-engine-company-profile-source-context-v1"
)
COMPANY_PROFILE_SOURCE_CONTEXT_AUTHORITY_BOUNDARY = (
    "Company Profile Source Context exposes descriptive identity, profile, "
    "provenance, freshness, and compatibility evidence only. It does not emit "
    "fundamental observations, recommendations, rankings, targets, allocation, "
    "or Decision Engine authority."
)


@dataclass(frozen=True)
class CompanyProfileCompatibilityGateOutcome:
    compatibility_gate_version: str
    allowed: bool
    result: str
    source_family: str
    reason_codes: tuple[str, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CompanyProfileSourceContext:
    context_format_version: str
    input_family: str
    consumption_state: str
    ticker: str
    symbol: str
    entity_name: str | None
    entity_country: str | None
    entity_exchange: str | None
    provider_name: str
    source_name: str
    source_refresh_snapshot_id: str
    source_refresh_fetched_at: str | None
    source_refresh_payload_format_version: str
    source_refresh_snapshot_path: str
    manifest_path: str
    as_of: str | None
    profile: dict[str, Any]
    provenance: dict[str, Any]
    compatibility_gate: CompanyProfileCompatibilityGateOutcome
    consumption_reason_codes: tuple[str, ...]
    missing_data_markers: tuple[str, ...] = field(default_factory=tuple)
    stale_data_markers: tuple[str, ...] = field(default_factory=tuple)
    authority_boundary: str = COMPANY_PROFILE_SOURCE_CONTEXT_AUTHORITY_BOUNDARY


def build_company_profile_source_context(
    *,
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    snapshot_path: str,
    manifest_path: str,
    gate_outcome: CompanyProfileCompatibilityGateOutcome,
) -> CompanyProfileSourceContext:
    if not gate_outcome.allowed:
        raise ValueError(
            "Company Profile Source Context requires an allowed gate outcome."
        )

    profile = payload["profile"]
    provenance = payload["provenance"]
    ticker = str(payload["ticker"]).upper()
    source_timestamp = provenance.get("source_timestamp")
    retrieved_at = provenance.get("retrieved_at")
    missing_data = profile.get("missing_data") or ()
    stale_markers = (
        ("source_context.company_profile.source_timestamp_unknown",)
        if source_timestamp in (None, "")
        else ()
    )
    consumption_reason_codes = ["company_profile_consumed_into_source_context"]
    consumption_reason_codes.extend(gate_outcome.reason_codes)

    return CompanyProfileSourceContext(
        context_format_version=COMPANY_PROFILE_SOURCE_CONTEXT_FORMAT_VERSION,
        input_family="company_profile",
        consumption_state="consumed",
        ticker=ticker,
        symbol=ticker,
        entity_name=_optional_text(payload.get("entity_name")),
        entity_country=_optional_text(payload.get("entity_country")),
        entity_exchange=_optional_text(payload.get("entity_exchange")),
        provider_name=str(provenance["provider_name"]),
        source_name=str(manifest["source_name"]),
        source_refresh_snapshot_id=str(manifest["snapshot_id"]),
        source_refresh_fetched_at=_optional_text(retrieved_at),
        source_refresh_payload_format_version=str(payload["payload_format"]),
        source_refresh_snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        as_of=_optional_text(source_timestamp) or _optional_text(retrieved_at),
        profile=dict(profile),
        provenance=dict(provenance),
        compatibility_gate=gate_outcome,
        consumption_reason_codes=tuple(dict.fromkeys(consumption_reason_codes)),
        missing_data_markers=tuple(
            f"source_context.company_profile.{item}" for item in missing_data
        ),
        stale_data_markers=stale_markers,
    )


def _optional_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None
