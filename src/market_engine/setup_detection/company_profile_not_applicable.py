from __future__ import annotations

from dataclasses import dataclass

from market_engine.derived_observations.company_profile_context_bridge import (
    CompanyProfileDerivedContextBridge,
)


COMPANY_PROFILE_SETUP_NOT_APPLICABLE_FORMAT_VERSION = (
    "market-engine-company-profile-setup-not-applicable-v1"
)
COMPANY_PROFILE_SETUP_BOUNDARY = (
    "Descriptive company profile context does not produce setup evidence, "
    "signals, evaluations, or action authority."
)


@dataclass(frozen=True)
class CompanyProfileSetupNotApplicable:
    ticker: str
    provider_name: str
    setup_detection_format_version: str
    setup_detection_run_id: str
    setup_detection_state: str
    input_family: str
    source_bridge_format_version: str
    source_refresh_snapshot_id: str
    setup_items: tuple[object, ...] = ()
    non_actionable_boundary: str = COMPANY_PROFILE_SETUP_BOUNDARY


def build_company_profile_setup_not_applicable(
    bridge: CompanyProfileDerivedContextBridge,
    *,
    setup_detection_run_id: str,
) -> CompanyProfileSetupNotApplicable:
    if bridge.bridge_state != "descriptive_context_preserved":
        raise ValueError(
            "Company profile setup boundary requires preserved descriptive context."
        )
    return CompanyProfileSetupNotApplicable(
        ticker=bridge.ticker,
        provider_name=bridge.provider_name,
        setup_detection_format_version=(
            COMPANY_PROFILE_SETUP_NOT_APPLICABLE_FORMAT_VERSION
        ),
        setup_detection_run_id=setup_detection_run_id,
        setup_detection_state="not_applicable_for_descriptive_context",
        input_family=bridge.input_family,
        source_bridge_format_version=bridge.derived_observation_format_version,
        source_refresh_snapshot_id=bridge.source_refresh_snapshot_id,
    )
