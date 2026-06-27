from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from market_engine.fundamental_observations.company_profile_observations import (
    COMPANY_PROFILE_FUNDAMENTAL_OBSERVATIONS_FORMAT_VERSION,
    build_company_profile_fundamental_observations,
)
from market_engine.source_context.company_profile_context import (
    COMPANY_PROFILE_SOURCE_CONTEXT_FORMAT_VERSION,
    CompanyProfileCompatibilityGateOutcome,
    CompanyProfileSourceContext,
)


def test_builds_descriptive_observations_from_consumed_profile_context() -> None:
    observation_set = build_company_profile_fundamental_observations(
        _source_context()
    )
    observations = {
        observation.observation_code: observation
        for observation in observation_set.observations
    }
    assert tuple(
        observation.observation_code
        for observation in observation_set.observations
    ) == (
        "company_profile_identity_observed",
        "company_profile_symbol_observed",
        "company_profile_exchange_observed",
        "company_profile_sector_observed",
        "company_profile_industry_observed",
        "company_profile_country_observed",
        "company_profile_currency_observed",
        "company_profile_description_available",
        "company_profile_website_observed",
        "company_profile_provenance_retained",
        "company_profile_as_of_retained",
    )

    assert observation_set.observation_format_version == (
        COMPANY_PROFILE_FUNDAMENTAL_OBSERVATIONS_FORMAT_VERSION
    )
    assert observation_set.input_family == "company_profile"
    assert observation_set.source_context_format_version == (
        COMPANY_PROFILE_SOURCE_CONTEXT_FORMAT_VERSION
    )
    assert observation_set.source_context_state == "consumed"
    assert observation_set.ticker == "NVDA"
    assert observation_set.symbol == "NVDA"
    assert observation_set.as_of == "2026-06-26T11:59:00Z"
    assert observation_set.provenance["provider_name"] == (
        "deterministic_fake_provider"
    )
    assert observations["company_profile_identity_observed"].value == (
        "NVIDIA Corporation"
    )
    assert observations["company_profile_exchange_observed"].value == "NASDAQ"
    assert observations["company_profile_sector_observed"].value == "Technology"
    assert observations["company_profile_industry_observed"].value == (
        "Semiconductors"
    )
    assert observations["company_profile_country_observed"].value == "US"
    assert observations["company_profile_currency_observed"].value == "USD"
    assert observations["company_profile_description_available"].value is True
    assert observations["company_profile_website_observed"].value == (
        "https://example.invalid/nvda"
    )
    assert all(
        observation.severity == "informational"
        for observation in observation_set.observations
    )


def test_omits_optional_observations_when_profile_fields_are_missing() -> None:
    source_context = replace(
        _source_context(),
        entity_name=None,
        entity_country=None,
        entity_exchange=None,
        profile={"missing_data": ["sector", "industry"]},
    )

    observation_set = build_company_profile_fundamental_observations(source_context)
    observation_codes = {
        observation.observation_code
        for observation in observation_set.observations
    }

    assert "company_profile_identity_observed" not in observation_codes
    assert "company_profile_exchange_observed" not in observation_codes
    assert "company_profile_sector_observed" not in observation_codes
    assert "company_profile_industry_observed" not in observation_codes
    assert "company_profile_description_available" not in observation_codes
    assert "company_profile_symbol_observed" in observation_codes
    assert "company_profile_provenance_retained" in observation_codes
    assert "company_profile_as_of_retained" in observation_codes


def test_rejects_non_consumed_source_context() -> None:
    with pytest.raises(ValueError, match="require consumed Source Context"):
        build_company_profile_fundamental_observations(
            replace(_source_context(), consumption_state="blocked")
        )


def test_observation_payload_contains_no_advisory_semantics() -> None:
    payload_text = json.dumps(
        asdict(build_company_profile_fundamental_observations(_source_context()))
    ).lower()

    forbidden_terms = (
        "recommendation",
        "target_price",
        "ranking",
        "conviction",
        "urgency",
        "buy",
        "sell",
        "hold",
        "trade",
        "broker",
        "price",
        "performance",
        "valuation",
        "trend",
        "momentum",
        "setup",
    )
    assert not any(term in payload_text for term in forbidden_terms)


def _source_context() -> CompanyProfileSourceContext:
    provenance = {
        "adapter_id": "fake_company_profile_adapter",
        "adapter_version": "test-v1",
        "provider_name": "deterministic_fake_provider",
        "canonical_source_identity": "fake://company_profile/NVDA",
        "retrieved_at": "2026-06-26T12:00:00Z",
        "source_timestamp": "2026-06-26T11:59:00Z",
        "request_metadata": {
            "network_used": False,
            "provider_calls_performed": False,
        },
    }
    return CompanyProfileSourceContext(
        context_format_version=COMPANY_PROFILE_SOURCE_CONTEXT_FORMAT_VERSION,
        input_family="company_profile",
        consumption_state="consumed",
        ticker="NVDA",
        symbol="NVDA",
        entity_name="NVIDIA Corporation",
        entity_country="US",
        entity_exchange="NASDAQ",
        provider_name="deterministic_fake_provider",
        source_name="deterministic_fake_provider",
        source_refresh_snapshot_id="NVDA-company_profile-test-run",
        source_refresh_fetched_at="2026-06-26T12:00:00Z",
        source_refresh_payload_format_version=(
            "market-engine-company-profile-snapshot-v1"
        ),
        source_refresh_snapshot_path="/tmp/NVDA/company_profile/company_profile.json",
        manifest_path="/tmp/NVDA/company_profile/manifest.json",
        as_of="2026-06-26T11:59:00Z",
        profile={
            "business_summary": "Deterministic profile description.",
            "sector": "Technology",
            "industry": "Semiconductors",
            "currency": "USD",
            "website": "https://example.invalid/nvda",
            "missing_data": [],
        },
        provenance=provenance,
        compatibility_gate=CompanyProfileCompatibilityGateOutcome(
            compatibility_gate_version=(
                "market-engine-company-profile-cached-source-compatibility-gate-v1"
            ),
            allowed=True,
            result="company_profile_consumption_allowed",
            source_family="company_profile",
            reason_codes=("company_profile_consumption_allowed",),
        ),
        consumption_reason_codes=(
            "company_profile_consumed_into_source_context",
        ),
    )
