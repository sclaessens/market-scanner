from __future__ import annotations

import json

import pytest

from market_engine.analysis_review.analysis_context_readiness import (
    ANALYSIS_CONTEXT_READINESS_BOUNDARY,
    ANALYSIS_CONTEXT_READINESS_FORMAT_VERSION,
    COMPANY_PROFILE_ONLY_CONTEXT_NON_ACTIONABLE,
    INSUFFICIENT_ANALYSIS_CONTEXT,
    MISSING_FUNDAMENTAL_EVIDENCE,
    MISSING_SETUP_OR_PRICE_CONTEXT,
    STALE_OR_UNPROVENANCED_ANALYSIS_CONTEXT,
    AnalysisContextEvidenceFamily,
    AnalysisContextReadinessLevel,
    AnalysisContextReadinessResult,
    classify_analysis_context_readiness,
)


PROVENANCE = AnalysisContextEvidenceFamily.PROVENANCE_MANIFEST_STALENESS
FUNDAMENTALS = AnalysisContextEvidenceFamily.FUNDAMENTALS
SETUP = AnalysisContextEvidenceFamily.SETUP_PRICE_MARKET
PROFILE = AnalysisContextEvidenceFamily.COMPANY_PROFILE


def test_company_profile_only_is_descriptive_and_non_actionable() -> None:
    result = classify_analysis_context_readiness(
        (PROFILE, PROVENANCE),
        provenance_valid=True,
    )

    assert result.readiness_level == AnalysisContextReadinessLevel.DESCRIPTIVE_ONLY
    assert result.blocked_reasons == (
        COMPANY_PROFILE_ONLY_CONTEXT_NON_ACTIONABLE,
    )
    assert result.evidence_families_missing == (
        FUNDAMENTALS,
        SETUP,
    )
    _assert_not_higher_readiness(result)


def test_empty_context_fails_closed() -> None:
    result = classify_analysis_context_readiness(())

    assert result.readiness_level == AnalysisContextReadinessLevel.DESCRIPTIVE_ONLY
    assert result.blocked_reasons == (INSUFFICIENT_ANALYSIS_CONTEXT,)
    assert result.evidence_families_present == ()
    assert result.evidence_families_missing == (
        FUNDAMENTALS,
        SETUP,
        PROVENANCE,
    )
    _assert_not_higher_readiness(result)


def test_fundamentals_only_is_partial_without_setup_context() -> None:
    result = classify_analysis_context_readiness(
        (FUNDAMENTALS, PROVENANCE),
        provenance_valid=True,
    )

    assert result.readiness_level == AnalysisContextReadinessLevel.PARTIAL_ANALYSIS
    assert result.blocked_reasons == (MISSING_SETUP_OR_PRICE_CONTEXT,)
    assert result.evidence_families_missing == (SETUP,)
    _assert_not_higher_readiness(result)


def test_setup_only_is_partial_without_fundamentals() -> None:
    result = classify_analysis_context_readiness(
        (SETUP, PROVENANCE),
        provenance_valid=True,
    )

    assert result.readiness_level == AnalysisContextReadinessLevel.PARTIAL_ANALYSIS
    assert result.blocked_reasons == (MISSING_FUNDAMENTAL_EVIDENCE,)
    assert result.evidence_families_missing == (FUNDAMENTALS,)
    _assert_not_higher_readiness(result)


def test_complete_analytical_context_is_recommendation_eligible_only() -> None:
    result = classify_analysis_context_readiness(
        (FUNDAMENTALS, SETUP, PROVENANCE),
        provenance_valid=True,
    )

    assert (
        result.readiness_level
        == AnalysisContextReadinessLevel.RECOMMENDATION_ELIGIBLE
    )
    assert result.blocked_reasons == ()
    assert result.evidence_families_missing == ()
    assert result.recommendation_review_eligible is True
    assert result.actionable_review_allowed is False
    assert result.decision_engine_ready is False


@pytest.mark.parametrize(
    ("provenance_valid", "context_stale"),
    (
        (False, False),
        (True, True),
    ),
)
def test_stale_or_unprovenanced_context_is_at_most_partial(
    provenance_valid: bool,
    context_stale: bool,
) -> None:
    result = classify_analysis_context_readiness(
        (FUNDAMENTALS, SETUP, PROVENANCE),
        provenance_valid=provenance_valid,
        context_stale=context_stale,
    )

    assert result.readiness_level == AnalysisContextReadinessLevel.PARTIAL_ANALYSIS
    assert STALE_OR_UNPROVENANCED_ANALYSIS_CONTEXT in result.blocked_reasons
    assert PROVENANCE in result.evidence_families_present
    assert PROVENANCE not in result.evidence_families_missing
    _assert_not_higher_readiness(result)


def test_profile_does_not_upgrade_fundamentals_without_setup() -> None:
    without_profile = classify_analysis_context_readiness(
        (FUNDAMENTALS, PROVENANCE),
        provenance_valid=True,
    )
    with_profile = classify_analysis_context_readiness(
        (PROFILE, FUNDAMENTALS, PROVENANCE),
        provenance_valid=True,
    )

    assert with_profile.readiness_level == without_profile.readiness_level
    assert with_profile.blocked_reasons == without_profile.blocked_reasons
    assert MISSING_SETUP_OR_PRICE_CONTEXT in with_profile.blocked_reasons
    _assert_not_higher_readiness(with_profile)


def test_profile_does_not_upgrade_setup_without_fundamentals() -> None:
    without_profile = classify_analysis_context_readiness(
        (SETUP, PROVENANCE),
        provenance_valid=True,
    )
    with_profile = classify_analysis_context_readiness(
        (PROFILE, SETUP, PROVENANCE),
        provenance_valid=True,
    )

    assert with_profile.readiness_level == without_profile.readiness_level
    assert with_profile.blocked_reasons == without_profile.blocked_reasons
    assert MISSING_FUNDAMENTAL_EVIDENCE in with_profile.blocked_reasons
    _assert_not_higher_readiness(with_profile)


def test_unknown_evidence_family_fails_closed() -> None:
    result = classify_analysis_context_readiness(
        (FUNDAMENTALS, SETUP, PROVENANCE, "future_evidence_family"),  # type: ignore[arg-type]
        provenance_valid=True,
    )

    assert result.readiness_level == AnalysisContextReadinessLevel.DESCRIPTIVE_ONLY
    assert result.blocked_reasons == (INSUFFICIENT_ANALYSIS_CONTEXT,)
    assert result.unknown_evidence_families == ("future_evidence_family",)
    assert "Malformed readiness input failed closed." in result.input_notes
    _assert_not_higher_readiness(result)


def test_non_iterable_evidence_input_fails_closed() -> None:
    result = classify_analysis_context_readiness(123)  # type: ignore[arg-type]

    assert result.readiness_level == AnalysisContextReadinessLevel.DESCRIPTIVE_ONLY
    assert result.blocked_reasons == (INSUFFICIENT_ANALYSIS_CONTEXT,)
    assert result.unknown_evidence_families == ("123",)
    assert "Malformed readiness input failed closed." in result.input_notes
    _assert_not_higher_readiness(result)


def test_non_boolean_gate_fails_closed() -> None:
    result = classify_analysis_context_readiness(
        (FUNDAMENTALS, SETUP, PROVENANCE),
        provenance_valid=1,  # type: ignore[arg-type]
    )

    assert result.readiness_level == AnalysisContextReadinessLevel.DESCRIPTIVE_ONLY
    assert result.blocked_reasons == (INSUFFICIENT_ANALYSIS_CONTEXT,)
    assert "Readiness gate flags must be booleans." in result.input_notes
    assert "Malformed readiness input failed closed." in result.input_notes
    _assert_not_higher_readiness(result)


def test_missing_required_valuation_remains_partial() -> None:
    result = classify_analysis_context_readiness(
        (FUNDAMENTALS, SETUP, PROVENANCE),
        provenance_valid=True,
        valuation_required=True,
    )

    assert result.readiness_level == AnalysisContextReadinessLevel.PARTIAL_ANALYSIS
    assert result.blocked_reasons == (INSUFFICIENT_ANALYSIS_CONTEXT,)
    assert result.evidence_families_missing == (
        AnalysisContextEvidenceFamily.VALUATION,
    )
    _assert_not_higher_readiness(result)


def test_serialized_result_is_stable_and_preserves_safety_boundary() -> None:
    result = classify_analysis_context_readiness(
        (SETUP, FUNDAMENTALS, PROVENANCE),
        provenance_valid=True,
    )

    payload = result.to_payload()

    assert payload["readiness_format_version"] == (
        ANALYSIS_CONTEXT_READINESS_FORMAT_VERSION
    )
    assert payload["readiness_level"] == "recommendation_eligible"
    assert payload["evidence_families_present"] == [
        "fundamentals",
        "setup_price_market",
        "provenance_manifest_staleness",
    ]
    assert payload["actionable_review_allowed"] is False
    assert payload["decision_engine_ready"] is False
    assert payload["non_authority_boundary"] == ANALYSIS_CONTEXT_READINESS_BOUNDARY
    json.dumps(payload)


def test_actionable_and_decision_levels_are_declared_but_unreachable() -> None:
    assert AnalysisContextReadinessLevel.ACTIONABLE_REVIEW.value == (
        "actionable_review"
    )
    assert AnalysisContextReadinessLevel.DECISION_READY.value == "decision_ready"

    evidence_sets = (
        (),
        (PROFILE, PROVENANCE),
        (FUNDAMENTALS, PROVENANCE),
        (SETUP, PROVENANCE),
        (FUNDAMENTALS, SETUP, PROVENANCE),
        tuple(AnalysisContextEvidenceFamily),
    )
    observed_levels = {
        classify_analysis_context_readiness(
            evidence,
            provenance_valid=True,
        ).readiness_level
        for evidence in evidence_sets
    }

    assert AnalysisContextReadinessLevel.ACTIONABLE_REVIEW not in observed_levels
    assert AnalysisContextReadinessLevel.DECISION_READY not in observed_levels


def _assert_not_higher_readiness(result: AnalysisContextReadinessResult) -> None:
    assert result.recommendation_review_eligible is False
    assert result.actionable_review_allowed is False
    assert result.decision_engine_ready is False
