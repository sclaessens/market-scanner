from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from market_engine.governor.evaluation import (
    GOVERNOR_FACTOR_TAXONOMY_VERSION,
    GOVERNOR_INVESTMENT_EVALUATION_CONTRACT_VERSION,
    EvaluationState,
    FactorFamily,
    FactorState,
    GovernorEvaluationError,
    evaluate_governor_evidence,
    to_plain_dict,
)


FIXTURE_PATH = Path(
    "tests/fixtures/market_engine/governor/"
    "me_gv03_governor_evidence_cases.json"
)
EVALUATED_AT = "2026-07-05T12:00:00Z"
INPUT_REFERENCE = "fixture://me-gv03"


def _cases() -> dict[str, dict[str, object]]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return {item["ticker"]: item for item in payload["cases"]}


def _evaluate(ticker: str):
    return evaluate_governor_evidence(
        _cases()[ticker],
        evaluation_timestamp=EVALUATED_AT,
        input_reference=INPUT_REFERENCE,
    )


def _factor(evaluation, family: FactorFamily):
    return next(
        item
        for item in evaluation.factor_evaluations
        if item.factor is family
    )


def test_contract_versions_factor_families_and_states_are_exact() -> None:
    assert GOVERNOR_INVESTMENT_EVALUATION_CONTRACT_VERSION == (
        "market-engine-governor-investment-evaluation-v1"
    )
    assert GOVERNOR_FACTOR_TAXONOMY_VERSION == (
        "market-engine-governor-factor-taxonomy-v1"
    )
    assert tuple(item.value for item in FactorFamily) == (
        "fundamentals",
        "growth",
        "valuation",
        "trend",
        "momentum",
        "risk",
        "technical_setup",
        "portfolio_fit",
        "data_confidence",
    )
    assert tuple(item.value for item in FactorState) == (
        "not_started",
        "blocked",
        "unavailable",
        "insufficient_evidence",
        "partial",
        "qualitative_only",
        "evaluable",
    )


def test_profile_only_case_is_descriptive_and_non_actionable() -> None:
    evaluation = _evaluate("GX001")

    assert evaluation.evaluation_state is EvaluationState.DESCRIPTIVE_ONLY
    assert (
        _factor(evaluation, FactorFamily.FUNDAMENTALS).state
        is FactorState.QUALITATIVE_ONLY
    )
    assert (
        _factor(evaluation, FactorFamily.VALUATION).state
        is FactorState.UNAVAILABLE
    )
    assert evaluation.recommendation_state["state"] == "blocked_not_authorized"
    assert evaluation.recommendation_state["actionable"] is False
    assert evaluation.recommendation_state["decision_engine_ready"] is False


def test_partial_fundamentals_and_momentum_remain_partial() -> None:
    evaluation = _evaluate("GX002")

    assert evaluation.evaluation_state is EvaluationState.PARTIAL_EVALUATION
    assert (
        _factor(evaluation, FactorFamily.FUNDAMENTALS).state
        is FactorState.PARTIAL
    )
    assert (
        _factor(evaluation, FactorFamily.MOMENTUM).state
        is FactorState.PARTIAL
    )
    assert (
        _factor(evaluation, FactorFamily.PORTFOLIO_FIT).state
        is FactorState.BLOCKED
    )
    assert (
        "blocked_missing_approved_portfolio_context"
        in evaluation.blocked_reasons
    )


def test_stale_market_and_setup_evidence_blocks_affected_factors() -> None:
    evaluation = _evaluate("GX003")

    assert evaluation.evaluation_state is EvaluationState.PARTIAL_EVALUATION
    for family in (
        FactorFamily.TREND,
        FactorFamily.MOMENTUM,
        FactorFamily.TECHNICAL_SETUP,
    ):
        factor = _factor(evaluation, family)
        assert factor.state is FactorState.BLOCKED
        assert "stale_evidence" in factor.blocked_reasons
    confidence = _factor(evaluation, FactorFamily.DATA_CONFIDENCE)
    assert confidence.state is FactorState.PARTIAL
    assert "stale_evidence_remains_inspectable" in confidence.limitations


def test_unprovenanced_evidence_fails_closed_globally() -> None:
    evaluation = _evaluate("GX004")

    assert evaluation.evaluation_state is EvaluationState.BLOCKED
    fundamentals = _factor(evaluation, FactorFamily.FUNDAMENTALS)
    assert fundamentals.state is FactorState.BLOCKED
    assert "missing_provenance" in fundamentals.blocked_reasons


def test_broad_evidence_can_be_evaluable_without_scores() -> None:
    evaluation = _evaluate("GX006")

    assert (
        evaluation.evaluation_state
        is EvaluationState.EVALUATION_COMPLETED_NON_ACTIONABLE
    )
    for family in (
        FactorFamily.FUNDAMENTALS,
        FactorFamily.GROWTH,
        FactorFamily.VALUATION,
        FactorFamily.TREND,
        FactorFamily.MOMENTUM,
        FactorFamily.RISK,
        FactorFamily.TECHNICAL_SETUP,
        FactorFamily.DATA_CONFIDENCE,
    ):
        factor = _factor(evaluation, family)
        assert factor.state is FactorState.EVALUABLE
        assert factor.score is None
        assert factor.score_scale is None
        assert factor.weight is None
        assert factor.weighted_score is None
    assert evaluation.overall_evaluation["score"] is None
    assert evaluation.overall_evaluation["weighted_score"] is None
    assert evaluation.overall_evaluation["rank"] is None


def test_conflicting_evidence_is_preserved_without_averaging_or_score() -> None:
    evaluation = _evaluate("GX005")
    growth = _factor(evaluation, FactorFamily.GROWTH)

    assert evaluation.evaluation_state is EvaluationState.PARTIAL_EVALUATION
    assert growth.state is FactorState.PARTIAL
    assert growth.conflicting_evidence_references == (
        "fixture://conflict/growth-series-a",
        "fixture://conflict/growth-series-b",
    )
    assert (
        "conflicting_evidence_preserved_without_averaging"
        in growth.limitations
    )
    assert growth.score is None


def test_invalid_manifest_and_malformed_evidence_fail_closed() -> None:
    invalid_manifest = deepcopy(_cases()["GX006"])
    invalid_manifest["evidence_readiness"]["manifest_valid"] = False
    malformed = deepcopy(_cases()["GX006"])
    malformed["factor_evidence"]["growth"]["structurally_valid"] = False

    first = evaluate_governor_evidence(
        invalid_manifest,
        evaluation_timestamp=EVALUATED_AT,
        input_reference=INPUT_REFERENCE,
    )
    second = evaluate_governor_evidence(
        malformed,
        evaluation_timestamp=EVALUATED_AT,
        input_reference=INPUT_REFERENCE,
    )

    assert first.evaluation_state is EvaluationState.BLOCKED
    assert "invalid_manifest" in first.blocked_reasons
    assert second.evaluation_state is EvaluationState.BLOCKED
    assert "malformed_evidence" in second.blocked_reasons


@pytest.mark.parametrize(
    "gate, reason",
    (
        ("manifest_valid", "invalid_manifest"),
        ("provenance_valid", "missing_provenance"),
        ("consumable", "evidence_not_consumable"),
        ("structurally_valid", "malformed_evidence"),
    ),
)
def test_global_evidence_gates_fail_closed(
    gate: str,
    reason: str,
) -> None:
    case = deepcopy(_cases()["GX006"])
    case["evidence_readiness"][gate] = False

    evaluation = evaluate_governor_evidence(
        case,
        evaluation_timestamp=EVALUATED_AT,
        input_reference=INPUT_REFERENCE,
    )

    assert evaluation.evaluation_state is EvaluationState.BLOCKED
    assert reason in evaluation.blocked_reasons


def test_incomplete_valuation_inputs_are_insufficient() -> None:
    case = deepcopy(_cases()["GX001"])
    case["factor_evidence"]["valuation"] = {
        "level": "limited",
        "evidence_references": ["fixture://valuation/incomplete"],
        "missing_evidence": ["valuation_denominator"],
    }
    evaluation = evaluate_governor_evidence(
        case,
        evaluation_timestamp=EVALUATED_AT,
        input_reference=INPUT_REFERENCE,
    )

    assert (
        _factor(evaluation, FactorFamily.VALUATION).state
        is FactorState.INSUFFICIENT_EVIDENCE
    )


def test_boundary_sections_and_reserved_states_are_fixed_false() -> None:
    payload = to_plain_dict(_evaluate("GX006"))

    assert payload["buy_zone_explanation"]["state"] == "blocked_not_authorized"
    assert (
        payload["position_management_explanation"]["state"]
        == "blocked_not_authorized"
    )
    assert payload["recommendation_state"]["recommendation_state_ready"] is False
    for field in (
        "actionable",
        "actionable_review",
        "recommendation_state_ready",
        "decision_ready",
        "de_ready",
        "decision_engine_ready",
    ):
        assert payload["authority_boundary"][field] is False
    assert payload["authority_boundary"]["scoring_authorized"] is True


def test_unknown_factor_or_evidence_level_is_rejected() -> None:
    unknown_factor = deepcopy(_cases()["GX001"])
    unknown_factor["factor_evidence"]["invented_factor"] = {"level": "complete"}
    unknown_level = deepcopy(_cases()["GX001"])
    unknown_level["factor_evidence"]["fundamentals"]["level"] = "positive"

    with pytest.raises(GovernorEvaluationError, match="factor family"):
        evaluate_governor_evidence(
            unknown_factor,
            evaluation_timestamp=EVALUATED_AT,
            input_reference=INPUT_REFERENCE,
        )
    with pytest.raises(GovernorEvaluationError, match="evidence level"):
        evaluate_governor_evidence(
            unknown_level,
            evaluation_timestamp=EVALUATED_AT,
            input_reference=INPUT_REFERENCE,
        )
