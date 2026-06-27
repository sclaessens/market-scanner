from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from market_engine.source_context.company_profile_context import (
    CompanyProfileSourceContext,
)


COMPANY_PROFILE_FUNDAMENTAL_OBSERVATIONS_FORMAT_VERSION = (
    "market-engine-company-profile-fundamental-observations-v1"
)
COMPANY_PROFILE_OBSERVATION_SEVERITY = "informational"
COMPANY_PROFILE_NON_ADVISORY_NOTICE = (
    "Company profile observations are descriptive source-derived evidence only "
    "and confer no analytical, action, or allocation authority."
)


@dataclass(frozen=True)
class CompanyProfileObservation:
    observation_code: str
    severity: str
    value: Any
    source_field: str


@dataclass(frozen=True)
class CompanyProfileFundamentalObservationSet:
    ticker: str
    symbol: str
    provider_name: str
    observation_format_version: str
    input_family: str
    source_context_format_version: str
    source_context_state: str
    source_context_reference: str
    source_refresh_snapshot_id: str
    source_refresh_fetched_at: str | None
    source_refresh_payload_format_version: str
    as_of: str | None
    provenance: dict[str, Any]
    observations: tuple[CompanyProfileObservation, ...]
    non_advisory_notice: str = COMPANY_PROFILE_NON_ADVISORY_NOTICE


def build_company_profile_fundamental_observations(
    source_context: CompanyProfileSourceContext,
) -> CompanyProfileFundamentalObservationSet:
    if source_context.consumption_state != "consumed":
        raise ValueError(
            "Company profile observations require consumed Source Context."
        )

    profile = source_context.profile
    observations: list[CompanyProfileObservation] = []
    _append_text_observation(
        observations,
        code="company_profile_identity_observed",
        value=source_context.entity_name or profile.get("company_name"),
        source_field=(
            "entity_name" if source_context.entity_name else "profile.company_name"
        ),
    )
    _append_text_observation(
        observations,
        code="company_profile_symbol_observed",
        value=source_context.symbol,
        source_field="symbol",
    )
    _append_text_observation(
        observations,
        code="company_profile_exchange_observed",
        value=source_context.entity_exchange or profile.get("exchange"),
        source_field=(
            "entity_exchange"
            if source_context.entity_exchange
            else "profile.exchange"
        ),
    )
    for field_name in ("sector", "industry"):
        _append_text_observation(
            observations,
            code=f"company_profile_{field_name}_observed",
            value=profile.get(field_name),
            source_field=f"profile.{field_name}",
        )
    _append_text_observation(
        observations,
        code="company_profile_country_observed",
        value=source_context.entity_country or profile.get("country"),
        source_field=(
            "entity_country"
            if source_context.entity_country
            else "profile.country"
        ),
    )
    _append_text_observation(
        observations,
        code="company_profile_currency_observed",
        value=profile.get("currency"),
        source_field="profile.currency",
    )
    if _non_empty_text(profile.get("business_summary")):
        observations.append(
            CompanyProfileObservation(
                observation_code="company_profile_description_available",
                severity=COMPANY_PROFILE_OBSERVATION_SEVERITY,
                value=True,
                source_field="profile.business_summary",
            )
        )
    _append_text_observation(
        observations,
        code="company_profile_website_observed",
        value=profile.get("website"),
        source_field="profile.website",
    )
    observations.append(
        CompanyProfileObservation(
            observation_code="company_profile_provenance_retained",
            severity=COMPANY_PROFILE_OBSERVATION_SEVERITY,
            value=source_context.provider_name,
            source_field="provenance.provider_name",
        )
    )
    if source_context.as_of is not None:
        observations.append(
            CompanyProfileObservation(
                observation_code="company_profile_as_of_retained",
                severity=COMPANY_PROFILE_OBSERVATION_SEVERITY,
                value=source_context.as_of,
                source_field="as_of",
            )
        )

    return CompanyProfileFundamentalObservationSet(
        ticker=source_context.ticker,
        symbol=source_context.symbol,
        provider_name=source_context.provider_name,
        observation_format_version=(
            COMPANY_PROFILE_FUNDAMENTAL_OBSERVATIONS_FORMAT_VERSION
        ),
        input_family=source_context.input_family,
        source_context_format_version=source_context.context_format_version,
        source_context_state=source_context.consumption_state,
        source_context_reference=source_context.source_refresh_snapshot_path,
        source_refresh_snapshot_id=source_context.source_refresh_snapshot_id,
        source_refresh_fetched_at=source_context.source_refresh_fetched_at,
        source_refresh_payload_format_version=(
            source_context.source_refresh_payload_format_version
        ),
        as_of=source_context.as_of,
        provenance=dict(source_context.provenance),
        observations=tuple(observations),
    )


def _append_text_observation(
    observations: list[CompanyProfileObservation],
    *,
    code: str,
    value: Any,
    source_field: str,
) -> None:
    if not _non_empty_text(value):
        return
    observations.append(
        CompanyProfileObservation(
            observation_code=code,
            severity=COMPANY_PROFILE_OBSERVATION_SEVERITY,
            value=value,
            source_field=source_field,
        )
    )


def _non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
