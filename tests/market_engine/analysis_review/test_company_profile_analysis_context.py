from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from market_engine.analysis_review.company_profile_analysis_context import (
    COMPANY_PROFILE_ANALYSIS_CONTEXT_FORMAT_VERSION,
    attach_company_profile_context_to_analysis_review,
    build_company_profile_analysis_context,
)
from market_engine.derived_observations.company_profile_context_bridge import (
    COMPANY_PROFILE_DERIVED_CONTEXT_BRIDGE_FORMAT_VERSION,
    build_company_profile_derived_context_bridge,
)
from market_engine.fundamental_observations.company_profile_observations import (
    CompanyProfileFundamentalObservationSet,
    CompanyProfileObservation,
)
from market_engine.setup_detection.company_profile_not_applicable import (
    COMPANY_PROFILE_SETUP_NOT_APPLICABLE_FORMAT_VERSION,
    build_company_profile_setup_not_applicable,
)


def test_builds_descriptive_analysis_context_from_profile_observations() -> None:
    observations = _observation_set()
    bridge = build_company_profile_derived_context_bridge(observations)
    setup_boundary = build_company_profile_setup_not_applicable(
        bridge,
        setup_detection_run_id="profile-setup-boundary",
    )

    context = build_company_profile_analysis_context(
        observations,
        bridge,
        setup_boundary,
        analysis_review_run_id="profile-analysis-context",
    )

    assert context.analysis_review_format_version == (
        COMPANY_PROFILE_ANALYSIS_CONTEXT_FORMAT_VERSION
    )
    assert context.source_bridge_format_version == (
        COMPANY_PROFILE_DERIVED_CONTEXT_BRIDGE_FORMAT_VERSION
    )
    assert context.setup_boundary_format_version == (
        COMPANY_PROFILE_SETUP_NOT_APPLICABLE_FORMAT_VERSION
    )
    assert context.context_state == "descriptive_context_available"
    assert context.ticker == "SYNTH"
    assert context.symbol == "SYNTH"
    assert context.as_of == "2026-06-27T10:00:00Z"
    assert context.provenance["provider_name"] == "deterministic_fake_provider"
    assert tuple(item.context_type for item in context.descriptive_context) == (
        "company_identity_context",
        "symbol_context",
        "exchange_context",
        "sector_context",
        "industry_context",
        "country_context",
        "currency_context",
        "description_availability_context",
        "website_context",
        "provenance_context",
        "as_of_context",
    )
    assert context.descriptive_context[0].source_observation_code == (
        "company_profile_identity_observed"
    )


def test_bridge_and_setup_boundary_do_not_create_financial_or_setup_output() -> None:
    observations = _observation_set()
    bridge = build_company_profile_derived_context_bridge(observations)
    setup_boundary = build_company_profile_setup_not_applicable(
        bridge,
        setup_detection_run_id="profile-setup-boundary",
    )

    assert bridge.financial_derivations_performed is False
    assert bridge.bridge_state == "descriptive_context_preserved"
    assert setup_boundary.setup_detection_state == (
        "not_applicable_for_descriptive_context"
    )
    assert setup_boundary.setup_items == ()


def test_analysis_context_is_ticker_agnostic_for_non_us_profile() -> None:
    observations = replace(
        _observation_set(),
        ticker="ASML",
        symbol="ASML",
        observations=(
            _observation(
                "company_profile_identity_observed",
                "ASML Holding N.V.",
                "entity_name",
            ),
            _observation(
                "company_profile_symbol_observed",
                "ASML",
                "symbol",
            ),
            _observation(
                "company_profile_country_observed",
                "NL",
                "entity_country",
            ),
        ),
    )
    bridge = build_company_profile_derived_context_bridge(observations)
    setup_boundary = build_company_profile_setup_not_applicable(
        bridge,
        setup_detection_run_id="asml-setup-boundary",
    )

    context = build_company_profile_analysis_context(
        observations,
        bridge,
        setup_boundary,
        analysis_review_run_id="asml-analysis-context",
    )

    assert context.ticker == "ASML"
    assert [item.value for item in context.descriptive_context] == [
        "ASML Holding N.V.",
        "ASML",
        "NL",
    ]


def test_analysis_context_rejects_misaligned_ticker() -> None:
    observations = _observation_set()
    bridge = replace(
        build_company_profile_derived_context_bridge(observations),
        ticker="OTHER",
        symbol="OTHER",
    )
    setup_boundary = build_company_profile_setup_not_applicable(
        bridge,
        setup_detection_run_id="profile-setup-boundary",
    )

    with pytest.raises(ValueError, match="ticker alignment failed"):
        build_company_profile_analysis_context(
            observations,
            bridge,
            setup_boundary,
            analysis_review_run_id="profile-analysis-context",
        )


def test_company_profile_context_is_additive_to_existing_analysis_review() -> None:
    observations = _observation_set()
    bridge = build_company_profile_derived_context_bridge(observations)
    setup_boundary = build_company_profile_setup_not_applicable(
        bridge,
        setup_detection_run_id="profile-setup-boundary",
    )
    context = build_company_profile_analysis_context(
        observations,
        bridge,
        setup_boundary,
        analysis_review_run_id="profile-analysis-context",
    )
    existing_review = {
        "analysis_review_format_version": "sec-companyfacts-analysis-review-v1",
        "ticker": "SYNTH",
        "review_items": [{"category": "SOURCE_AVAILABILITY_REVIEW"}],
    }

    combined = attach_company_profile_context_to_analysis_review(
        existing_review,
        context,
    )

    assert combined["analysis_review_format_version"] == (
        "sec-companyfacts-analysis-review-v1"
    )
    assert combined["review_items"] == existing_review["review_items"]
    assert combined["company_profile_context"]["context_state"] == (
        "descriptive_context_available"
    )
    assert "company_profile_context" not in existing_review


def test_combined_analysis_review_rejects_ticker_mismatch() -> None:
    observations = _observation_set()
    bridge = build_company_profile_derived_context_bridge(observations)
    setup_boundary = build_company_profile_setup_not_applicable(
        bridge,
        setup_detection_run_id="profile-setup-boundary",
    )
    context = build_company_profile_analysis_context(
        observations,
        bridge,
        setup_boundary,
        analysis_review_run_id="profile-analysis-context",
    )

    with pytest.raises(ValueError, match="ticker alignment failed"):
        attach_company_profile_context_to_analysis_review(
            {
                "analysis_review_format_version": (
                    "sec-companyfacts-analysis-review-v1"
                ),
                "ticker": "OTHER",
            },
            context,
        )


def test_analysis_context_contains_no_advisory_semantics() -> None:
    observations = _observation_set()
    bridge = build_company_profile_derived_context_bridge(observations)
    setup_boundary = build_company_profile_setup_not_applicable(
        bridge,
        setup_detection_run_id="profile-setup-boundary",
    )
    payload_text = json.dumps(
        asdict(
            build_company_profile_analysis_context(
                observations,
                bridge,
                setup_boundary,
                analysis_review_run_id="profile-analysis-context",
            )
        )
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
        "score",
    )
    assert not any(term in payload_text for term in forbidden_terms)


def _observation_set() -> CompanyProfileFundamentalObservationSet:
    values = (
        (
            "company_profile_identity_observed",
            "Synthetic Equipment N.V.",
            "entity_name",
        ),
        ("company_profile_symbol_observed", "SYNTH", "symbol"),
        ("company_profile_exchange_observed", "XAMS", "entity_exchange"),
        ("company_profile_sector_observed", "Technology", "profile.sector"),
        ("company_profile_industry_observed", "Equipment", "profile.industry"),
        ("company_profile_country_observed", "NL", "entity_country"),
        ("company_profile_currency_observed", "EUR", "profile.currency"),
        (
            "company_profile_description_available",
            True,
            "profile.business_summary",
        ),
        (
            "company_profile_website_observed",
            "https://example.invalid/synth",
            "profile.website",
        ),
        (
            "company_profile_provenance_retained",
            "deterministic_fake_provider",
            "provenance.provider_name",
        ),
        (
            "company_profile_as_of_retained",
            "2026-06-27T10:00:00Z",
            "as_of",
        ),
    )
    return CompanyProfileFundamentalObservationSet(
        ticker="SYNTH",
        symbol="SYNTH",
        provider_name="deterministic_fake_provider",
        observation_format_version=(
            "market-engine-company-profile-fundamental-observations-v1"
        ),
        input_family="company_profile",
        source_context_format_version=(
            "market-engine-company-profile-source-context-v1"
        ),
        source_context_state="consumed",
        source_context_reference="/tmp/SYNTH/company_profile.json",
        source_refresh_snapshot_id="SYNTH-company-profile",
        source_refresh_fetched_at="2026-06-27T10:01:00Z",
        source_refresh_payload_format_version=(
            "market-engine-company-profile-snapshot-v1"
        ),
        as_of="2026-06-27T10:00:00Z",
        provenance={"provider_name": "deterministic_fake_provider"},
        observations=tuple(_observation(*value) for value in values),
    )


def _observation(
    code: str,
    value: object,
    source_field: str,
) -> CompanyProfileObservation:
    return CompanyProfileObservation(
        observation_code=code,
        severity="informational",
        value=value,
        source_field=source_field,
    )
