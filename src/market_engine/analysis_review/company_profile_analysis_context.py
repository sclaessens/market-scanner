from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from market_engine.derived_observations.company_profile_context_bridge import (
    CompanyProfileDerivedContextBridge,
)
from market_engine.fundamental_observations.company_profile_observations import (
    CompanyProfileFundamentalObservationSet,
)
from market_engine.setup_detection.company_profile_not_applicable import (
    CompanyProfileSetupNotApplicable,
)


COMPANY_PROFILE_ANALYSIS_CONTEXT_FORMAT_VERSION = (
    "market-engine-company-profile-analysis-context-v1"
)
COMPANY_PROFILE_ANALYSIS_CONTEXT_BOUNDARY = (
    "Company profile context is descriptive source-derived context only and "
    "confers no evaluative or action authority."
)

_CONTEXT_TYPE_BY_OBSERVATION_CODE = {
    "company_profile_identity_observed": "company_identity_context",
    "company_profile_symbol_observed": "symbol_context",
    "company_profile_exchange_observed": "exchange_context",
    "company_profile_sector_observed": "sector_context",
    "company_profile_industry_observed": "industry_context",
    "company_profile_country_observed": "country_context",
    "company_profile_currency_observed": "currency_context",
    "company_profile_description_available": "description_availability_context",
    "company_profile_website_observed": "website_context",
    "company_profile_provenance_retained": "provenance_context",
    "company_profile_as_of_retained": "as_of_context",
}


@dataclass(frozen=True)
class CompanyProfileDescriptiveContextItem:
    context_type: str
    value: Any
    source_observation_code: str
    source_field: str


@dataclass(frozen=True)
class CompanyProfileAnalysisContext:
    ticker: str
    symbol: str
    provider_name: str
    analysis_review_format_version: str
    analysis_review_run_id: str
    input_family: str
    context_state: str
    source_observation_format_version: str
    source_bridge_format_version: str
    setup_boundary_format_version: str
    source_refresh_snapshot_id: str
    source_refresh_fetched_at: str | None
    source_refresh_payload_format_version: str
    as_of: str | None
    provenance: dict[str, Any]
    descriptive_context: tuple[CompanyProfileDescriptiveContextItem, ...]
    non_advisory_boundary: str = COMPANY_PROFILE_ANALYSIS_CONTEXT_BOUNDARY


def build_company_profile_analysis_context(
    observation_set: CompanyProfileFundamentalObservationSet,
    bridge: CompanyProfileDerivedContextBridge,
    setup_boundary: CompanyProfileSetupNotApplicable,
    *,
    analysis_review_run_id: str,
) -> CompanyProfileAnalysisContext:
    _validate_alignment(
        observation_set=observation_set,
        bridge=bridge,
        setup_boundary=setup_boundary,
    )
    return CompanyProfileAnalysisContext(
        ticker=observation_set.ticker,
        symbol=observation_set.symbol,
        provider_name=observation_set.provider_name,
        analysis_review_format_version=(
            COMPANY_PROFILE_ANALYSIS_CONTEXT_FORMAT_VERSION
        ),
        analysis_review_run_id=analysis_review_run_id,
        input_family=observation_set.input_family,
        context_state="descriptive_context_available",
        source_observation_format_version=observation_set.observation_format_version,
        source_bridge_format_version=bridge.derived_observation_format_version,
        setup_boundary_format_version=setup_boundary.setup_detection_format_version,
        source_refresh_snapshot_id=observation_set.source_refresh_snapshot_id,
        source_refresh_fetched_at=observation_set.source_refresh_fetched_at,
        source_refresh_payload_format_version=(
            observation_set.source_refresh_payload_format_version
        ),
        as_of=observation_set.as_of,
        provenance=dict(observation_set.provenance),
        descriptive_context=tuple(
            CompanyProfileDescriptiveContextItem(
                context_type=_CONTEXT_TYPE_BY_OBSERVATION_CODE[
                    observation.observation_code
                ],
                value=observation.value,
                source_observation_code=observation.observation_code,
                source_field=observation.source_field,
            )
            for observation in observation_set.observations
            if observation.observation_code in _CONTEXT_TYPE_BY_OBSERVATION_CODE
        ),
    )


def attach_company_profile_context_to_analysis_review(
    analysis_review_payload: Mapping[str, Any],
    company_profile_context: CompanyProfileAnalysisContext,
) -> dict[str, Any]:
    if "company_profile_context" in analysis_review_payload:
        raise ValueError("Analysis Review already contains company profile context.")
    existing_ticker = analysis_review_payload.get("ticker")
    if (
        isinstance(existing_ticker, str)
        and existing_ticker
        and existing_ticker.upper() != company_profile_context.ticker.upper()
    ):
        raise ValueError("Combined Analysis Review ticker alignment failed.")
    return {
        **dict(analysis_review_payload),
        "company_profile_context": asdict(company_profile_context),
    }


def _validate_alignment(
    *,
    observation_set: CompanyProfileFundamentalObservationSet,
    bridge: CompanyProfileDerivedContextBridge,
    setup_boundary: CompanyProfileSetupNotApplicable,
) -> None:
    identities = {
        observation_set.ticker,
        observation_set.symbol,
        bridge.ticker,
        bridge.symbol,
        setup_boundary.ticker,
    }
    if len(identities) != 1:
        raise ValueError("Company profile analysis context ticker alignment failed.")
    if observation_set.source_context_state != "consumed":
        raise ValueError(
            "Company profile analysis context requires consumed Source Context."
        )
    if bridge.source_refresh_snapshot_id != observation_set.source_refresh_snapshot_id:
        raise ValueError(
            "Company profile analysis context snapshot alignment failed."
        )
    if setup_boundary.source_refresh_snapshot_id != (
        observation_set.source_refresh_snapshot_id
    ):
        raise ValueError(
            "Company profile analysis setup boundary snapshot alignment failed."
        )
