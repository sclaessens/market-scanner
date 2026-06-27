from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from market_engine.fundamental_observations.company_profile_observations import (
    CompanyProfileFundamentalObservationSet,
)


COMPANY_PROFILE_DERIVED_CONTEXT_BRIDGE_FORMAT_VERSION = (
    "market-engine-company-profile-derived-context-bridge-v1"
)
COMPANY_PROFILE_DERIVED_CONTEXT_BOUNDARY = (
    "This bridge preserves descriptive source observations without computing "
    "financial metrics, evaluations, signals, or action authority."
)


@dataclass(frozen=True)
class CompanyProfileObservationReference:
    observation_code: str
    value: Any
    source_field: str


@dataclass(frozen=True)
class CompanyProfileDerivedContextBridge:
    ticker: str
    symbol: str
    provider_name: str
    derived_observation_format_version: str
    fundamental_observation_format_version: str
    input_family: str
    bridge_state: str
    source_refresh_snapshot_id: str
    source_refresh_fetched_at: str | None
    source_refresh_payload_format_version: str
    as_of: str | None
    provenance: dict[str, Any]
    source_observation_references: tuple[CompanyProfileObservationReference, ...]
    financial_derivations_performed: bool = False
    non_derivation_boundary: str = COMPANY_PROFILE_DERIVED_CONTEXT_BOUNDARY


def build_company_profile_derived_context_bridge(
    observation_set: CompanyProfileFundamentalObservationSet,
) -> CompanyProfileDerivedContextBridge:
    if observation_set.source_context_state != "consumed":
        raise ValueError(
            "Company profile context bridge requires consumed Source Context."
        )

    return CompanyProfileDerivedContextBridge(
        ticker=observation_set.ticker,
        symbol=observation_set.symbol,
        provider_name=observation_set.provider_name,
        derived_observation_format_version=(
            COMPANY_PROFILE_DERIVED_CONTEXT_BRIDGE_FORMAT_VERSION
        ),
        fundamental_observation_format_version=(
            observation_set.observation_format_version
        ),
        input_family=observation_set.input_family,
        bridge_state="descriptive_context_preserved",
        source_refresh_snapshot_id=observation_set.source_refresh_snapshot_id,
        source_refresh_fetched_at=observation_set.source_refresh_fetched_at,
        source_refresh_payload_format_version=(
            observation_set.source_refresh_payload_format_version
        ),
        as_of=observation_set.as_of,
        provenance=dict(observation_set.provenance),
        source_observation_references=tuple(
            CompanyProfileObservationReference(
                observation_code=observation.observation_code,
                value=observation.value,
                source_field=observation.source_field,
            )
            for observation in observation_set.observations
        ),
    )
