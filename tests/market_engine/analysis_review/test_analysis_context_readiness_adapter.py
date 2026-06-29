from __future__ import annotations

from copy import deepcopy

from market_engine.analysis_review.analysis_context_readiness import (
    COMPANY_PROFILE_ONLY_CONTEXT_NON_ACTIONABLE,
    MISSING_FUNDAMENTAL_EVIDENCE,
    MISSING_SETUP_OR_PRICE_CONTEXT,
    STALE_OR_UNPROVENANCED_ANALYSIS_CONTEXT,
    AnalysisContextEvidenceFamily,
    AnalysisContextReadinessLevel,
    AnalysisContextReadinessResult,
)
from market_engine.analysis_review.analysis_context_readiness_adapter import (
    classify_analysis_context_readiness_from_stage_payloads,
)


PROFILE = AnalysisContextEvidenceFamily.COMPANY_PROFILE
FUNDAMENTALS = AnalysisContextEvidenceFamily.FUNDAMENTALS
SETUP = AnalysisContextEvidenceFamily.SETUP_PRICE_MARKET
PROVENANCE = AnalysisContextEvidenceFamily.PROVENANCE_MANIFEST_STALENESS


def test_company_profile_only_maps_to_descriptive_context() -> None:
    result = classify_analysis_context_readiness_from_stage_payloads(
        _company_profile_stage_payloads()
    )

    assert result.readiness_level == AnalysisContextReadinessLevel.DESCRIPTIVE_ONLY
    assert result.evidence_families_present == (PROFILE, PROVENANCE)
    assert COMPANY_PROFILE_ONLY_CONTEXT_NON_ACTIONABLE in result.blocked_reasons
    _assert_non_authoritative(result)


def test_fundamentals_only_maps_to_partial_analysis() -> None:
    payloads = _sec_stage_payloads()
    payloads.pop("setup_detection")

    result = classify_analysis_context_readiness_from_stage_payloads(payloads)

    assert result.readiness_level == AnalysisContextReadinessLevel.PARTIAL_ANALYSIS
    assert FUNDAMENTALS in result.evidence_families_present
    assert SETUP not in result.evidence_families_present
    assert MISSING_SETUP_OR_PRICE_CONTEXT in result.blocked_reasons
    _assert_non_authoritative(result)


def test_setup_only_maps_to_partial_analysis() -> None:
    payloads = _sec_stage_payloads()
    payloads.pop("fundamental_observations")

    result = classify_analysis_context_readiness_from_stage_payloads(payloads)

    assert result.readiness_level == AnalysisContextReadinessLevel.PARTIAL_ANALYSIS
    assert SETUP in result.evidence_families_present
    assert FUNDAMENTALS not in result.evidence_families_present
    assert MISSING_FUNDAMENTAL_EVIDENCE in result.blocked_reasons
    _assert_non_authoritative(result)


def test_fundamentals_setup_and_provenance_are_recommendation_eligible() -> None:
    result = classify_analysis_context_readiness_from_stage_payloads(
        _sec_stage_payloads()
    )

    assert (
        result.readiness_level
        == AnalysisContextReadinessLevel.RECOMMENDATION_ELIGIBLE
    )
    assert result.evidence_families_present == (
        FUNDAMENTALS,
        SETUP,
        PROVENANCE,
    )
    assert result.recommendation_review_eligible is True
    assert result.actionable_review_allowed is False
    assert result.decision_engine_ready is False


def test_stale_context_fails_closed() -> None:
    payloads = _sec_stage_payloads()
    payloads["source_context"]["stale_data_markers"] = (
        "source_context.snapshot_stale",
    )

    result = classify_analysis_context_readiness_from_stage_payloads(payloads)

    assert result.readiness_level == AnalysisContextReadinessLevel.PARTIAL_ANALYSIS
    assert PROVENANCE not in result.evidence_families_present
    assert STALE_OR_UNPROVENANCED_ANALYSIS_CONTEXT in result.blocked_reasons
    _assert_non_authoritative(result)


def test_unprovenanced_context_fails_closed() -> None:
    payloads = _sec_stage_payloads()
    payloads["source_context"].pop("fixture_backed")

    result = classify_analysis_context_readiness_from_stage_payloads(payloads)

    assert result.readiness_level == AnalysisContextReadinessLevel.PARTIAL_ANALYSIS
    assert PROVENANCE not in result.evidence_families_present
    assert STALE_OR_UNPROVENANCED_ANALYSIS_CONTEXT in result.blocked_reasons
    _assert_non_authoritative(result)


def test_limited_fundamentals_do_not_count_as_fundamental_evidence() -> None:
    payloads = _sec_stage_payloads()
    payloads["fundamental_observations"]["observations"] = (
        {"state": "MISSING_DATA"},
    )

    result = classify_analysis_context_readiness_from_stage_payloads(payloads)

    assert result.readiness_level == AnalysisContextReadinessLevel.PARTIAL_ANALYSIS
    assert FUNDAMENTALS not in result.evidence_families_present
    assert MISSING_FUNDAMENTAL_EVIDENCE in result.blocked_reasons
    _assert_non_authoritative(result)


def test_incomplete_setup_does_not_count_as_setup_evidence() -> None:
    payloads = _sec_stage_payloads()
    payloads["setup_detection"]["setup_items"] = (
        {"state": "setup_blocked_by_missing_data"},
    )

    result = classify_analysis_context_readiness_from_stage_payloads(payloads)

    assert result.readiness_level == AnalysisContextReadinessLevel.PARTIAL_ANALYSIS
    assert SETUP not in result.evidence_families_present
    assert MISSING_SETUP_OR_PRICE_CONTEXT in result.blocked_reasons
    _assert_non_authoritative(result)


def test_company_profile_does_not_upgrade_partial_evidence() -> None:
    payloads = _sec_stage_payloads()
    payloads.pop("setup_detection")
    payloads["analysis_review"]["company_profile_context"] = {
        "analysis_review_format_version": (
            "market-engine-company-profile-analysis-context-v1"
        ),
        "ticker": "SYNTH",
        "input_family": "company_profile",
        "context_state": "descriptive_context_available",
        "source_refresh_snapshot_id": "profile-snapshot-001",
        "provenance": {"provider_name": "deterministic_fixture"},
    }

    without_profile = classify_analysis_context_readiness_from_stage_payloads(
        {
            key: value
            for key, value in payloads.items()
            if key != "analysis_review"
        }
    )
    with_profile = classify_analysis_context_readiness_from_stage_payloads(payloads)

    assert PROFILE in with_profile.evidence_families_present
    assert with_profile.readiness_level == without_profile.readiness_level
    assert MISSING_SETUP_OR_PRICE_CONTEXT in with_profile.blocked_reasons
    _assert_non_authoritative(with_profile)


def test_unknown_or_missing_stage_payloads_fail_closed() -> None:
    for payloads in (None, {}, {"analysis_review": object()}):
        result = classify_analysis_context_readiness_from_stage_payloads(payloads)

        assert (
            result.readiness_level
            == AnalysisContextReadinessLevel.DESCRIPTIVE_ONLY
        )
        _assert_non_authoritative(result)


def test_adapter_does_not_invent_valuation_context() -> None:
    payloads = _sec_stage_payloads()
    payloads["analysis_review"]["valuation_summary"] = {"status": "available"}

    result = classify_analysis_context_readiness_from_stage_payloads(payloads)

    assert AnalysisContextEvidenceFamily.VALUATION not in (
        result.evidence_families_present
    )


def _sec_stage_payloads() -> dict[str, dict[str, object]]:
    return {
        "source_context": {
            "source_context_format_version": "sec-companyfacts-source-context-v1",
            "ticker": "SYNTH",
            "source_refresh_snapshot_id": "source-snapshot-001",
            "fixture_backed": True,
        },
        "fundamental_observations": {
            "fundamental_observations_format_version": (
                "sec-companyfacts-fundamental-observations-v1"
            ),
            "ticker": "SYNTH",
            "source_context_reference": {
                "source_refresh_snapshot_id": "source-snapshot-001"
            },
        },
        "setup_detection": {
            "setup_detection_format_version": "sec-companyfacts-setup-detection-v1",
            "ticker": "SYNTH",
            "source_refresh_snapshot_id": "source-snapshot-001",
        },
        "analysis_review": {
            "analysis_review_format_version": "sec-companyfacts-analysis-review-v1",
            "ticker": "SYNTH",
            "source_refresh_snapshot_id": "source-snapshot-001",
        },
    }


def _company_profile_stage_payloads() -> dict[str, dict[str, object]]:
    cached_source_reference = {
        "input_mode": "cached_source_snapshot",
        "source_snapshot_path": "/fixtures/SYNTH/company_profile.json",
    }
    return {
        "source_context": {
            "source_context_format_version": (
                "market-engine-company-profile-source-context-v1"
            ),
            "ticker": "SYNTH",
            "consumption_state": "consumed",
            "source_refresh_snapshot_id": "profile-snapshot-001",
            "manifest_path": "/fixtures/SYNTH/manifest.json",
            "provenance": {"provider_name": "deterministic_fixture"},
            "compatibility_gate": {"allowed": True},
            "cached_source_reference": deepcopy(cached_source_reference),
        },
        "analysis_review": {
            "analysis_review_format_version": (
                "market-engine-company-profile-analysis-context-v1"
            ),
            "ticker": "SYNTH",
            "input_family": "company_profile",
            "context_state": "descriptive_context_available",
            "source_refresh_snapshot_id": "profile-snapshot-001",
            "cached_source_reference": deepcopy(cached_source_reference),
        },
    }


def _assert_non_authoritative(result: AnalysisContextReadinessResult) -> None:
    assert result.recommendation_review_eligible is False
    assert result.actionable_review_allowed is False
    assert result.decision_engine_ready is False
