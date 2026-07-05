from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from market_engine.governor.evaluation import (
    EvaluationState,
    FactorFamily,
    FactorState,
    evaluate_governor_evidence,
    to_plain_dict,
)
from market_engine.governor.recommendation import (
    APPROVED_RECOMMENDATION_REVIEW_CONTRACT_VERSION,
    DATA_CONFIDENCE_ELIGIBILITY_THRESHOLD,
    GOVERNOR_RECOMMENDATION_STATE_CONTRACT_VERSION,
    RecommendationEligibilityState,
    RecommendationState,
    map_recommendation_state,
)


FIXTURE_PATH = Path(
    "tests/fixtures/market_engine/governor/"
    "me_gv05_governor_recommendation_case.json"
)
EVALUATED_AT = "2026-07-05T16:00:00Z"
INPUT_REFERENCE = "fixture://me-gv05"


def _case() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _evaluate(case: dict[str, object] | None = None):
    return evaluate_governor_evidence(
        case or _case(),
        evaluation_timestamp=EVALUATED_AT,
        input_reference=INPUT_REFERENCE,
    )


def _factor(evaluation, family: FactorFamily):
    return next(
        item for item in evaluation.factor_evaluations if item.factor is family
    )


def _set_values(
    case: dict[str, object],
    factor: str,
    values: tuple[float, float, float],
) -> None:
    components = case["factor_evidence"][factor]["score_inputs"]["components"]
    for component, value in zip(components, values, strict=True):
        component["input_value"] = value


def _moderate_case() -> dict[str, object]:
    case = _case()
    _set_values(case, "fundamentals", (0.16, 0.16, 0.1125))
    _set_values(case, "growth", (0.14, 0.16, 0.16))
    _set_values(case, "risk", (0.41, 1.75, 1.3))
    _set_values(case, "data_confidence", (0.8, 0.8, 0.8))
    return case


def test_contract_versions_and_state_sets_are_exact() -> None:
    assert GOVERNOR_RECOMMENDATION_STATE_CONTRACT_VERSION == (
        "market-engine-governor-recommendation-state-v1"
    )
    assert APPROVED_RECOMMENDATION_REVIEW_CONTRACT_VERSION == (
        "sec-companyfacts-recommendation-review-v1"
    )
    assert DATA_CONFIDENCE_ELIGIBILITY_THRESHOLD == 75.0
    assert tuple(item.value for item in RecommendationEligibilityState) == (
        "eligible",
        "ineligible",
    )
    assert tuple(item.value for item in RecommendationState) == (
        "blocked",
        "insufficient_evidence",
        "avoid",
        "watch",
        "consider",
        "preferred",
    )
    with pytest.raises(ValueError):
        RecommendationState("buy")


def test_complete_approved_case_is_eligible_and_preferred() -> None:
    evaluation = _evaluate()
    recommendation = evaluation.recommendation_state

    assert (
        evaluation.evaluation_state
        is EvaluationState.EVALUATION_COMPLETED_NON_ACTIONABLE
    )
    assert recommendation["eligibility_state"] == "eligible"
    assert recommendation["state"] == "preferred"
    assert recommendation["reason_codes"] == (
        "recommendation_eligible",
        "mapped_favorable_critical_factor_pattern",
    )
    assert recommendation["limitations"] == (
        "portfolio_fit_not_used_without_approved_context",
    )
    assert recommendation["actionable"] is False
    assert recommendation["recommendation_state_ready"] is False
    assert recommendation["decision_engine_ready"] is False


def test_global_blocked_evaluation_precedes_other_mapping() -> None:
    case = _case()
    case["evidence_readiness"]["manifest_valid"] = False
    recommendation = _evaluate(case).recommendation_state

    assert recommendation["eligibility_state"] == "ineligible"
    assert recommendation["state"] == "blocked"
    assert recommendation["reason_codes"] == (
        "blocked_governor_evaluation",
    )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("missing", "blocked_recommendation_review_boundary_missing"),
        ("invalid_contract", "blocked_recommendation_review_boundary_invalid"),
        ("actionable", "blocked_recommendation_review_boundary_invalid"),
        ("missing_reference", "blocked_recommendation_review_boundary_invalid"),
    ),
)
def test_recommendation_review_boundary_fails_closed(
    mutation: str,
    reason: str,
) -> None:
    case = _case()
    boundary = case["recommendation_review_boundary"]
    if mutation == "missing":
        case.pop("recommendation_review_boundary")
    elif mutation == "invalid_contract":
        boundary["contract_version"] = "unsupported"
    elif mutation == "actionable":
        boundary["non_actionable"] = False
    else:
        boundary["reference"] = ""

    recommendation = _evaluate(case).recommendation_state

    assert recommendation["eligibility_state"] == "ineligible"
    assert recommendation["state"] == "blocked"
    assert recommendation["reason_codes"] == (reason,)


def test_valid_but_non_reviewable_recommendation_review_is_insufficient() -> None:
    case = _case()
    case["recommendation_review_boundary"][
        "review_state"
    ] = "insufficient_evidence"

    recommendation = _evaluate(case).recommendation_state

    assert recommendation["state"] == "insufficient_evidence"
    assert recommendation["reason_codes"] == (
        "ineligible_recommendation_review_state",
    )


def test_missing_critical_score_is_not_interpreted_as_neutral() -> None:
    case = _case()
    case["factor_evidence"]["growth"].pop("score_inputs")
    evaluation = _evaluate(case)
    recommendation = evaluation.recommendation_state

    assert _factor(evaluation, FactorFamily.GROWTH).state is FactorState.EVALUABLE
    assert _factor(evaluation, FactorFamily.GROWTH).score is None
    assert recommendation["state"] == "insufficient_evidence"
    assert recommendation["blocking_factors"] == ("growth",)
    assert recommendation["reason_codes"] == (
        "ineligible_critical_factor_coverage",
    )


def test_critical_score_limitations_block_eligibility() -> None:
    case = _case()
    case["factor_evidence"]["growth"]["score_inputs"]["components"][0][
        "limitations"
    ] = ["one_off_period_effect"]

    recommendation = _evaluate(case).recommendation_state

    assert recommendation["state"] == "insufficient_evidence"
    assert recommendation["blocking_factors"] == ("growth",)
    assert recommendation["reason_codes"] == (
        "ineligible_critical_score_limitations",
    )
    assert "growth:one_off_period_effect" in recommendation["limitations"]


def test_low_data_confidence_is_an_explicit_gate_not_a_multiplier() -> None:
    case = _case()
    _set_values(case, "data_confidence", (0.7, 0.7, 0.7))
    evaluation = _evaluate(case)
    recommendation = evaluation.recommendation_state

    assert _factor(evaluation, FactorFamily.DATA_CONFIDENCE).score == 70.0
    assert _factor(evaluation, FactorFamily.FUNDAMENTALS).score == 81.67
    assert _factor(evaluation, FactorFamily.GROWTH).score == 73.33
    assert _factor(evaluation, FactorFamily.RISK).score == 80.0
    assert recommendation["state"] == "insufficient_evidence"
    assert recommendation["reason_codes"] == (
        "ineligible_data_confidence_below_threshold",
    )


def test_hard_conflict_blocks_before_incomplete_evaluation() -> None:
    case = _case()
    growth = case["factor_evidence"]["growth"]
    growth["conflicting_evidence_references"] = [
        growth["evidence_references"][0],
        growth["evidence_references"][1],
    ]
    evaluation = _evaluate(case)
    recommendation = evaluation.recommendation_state

    assert _factor(evaluation, FactorFamily.GROWTH).state is FactorState.PARTIAL
    assert recommendation["state"] == "blocked"
    assert recommendation["reason_codes"] == (
        "blocked_unresolved_hard_conflict",
    )
    assert recommendation["conflict_references"] == (
        "fixture://recommendation/growth/earnings",
        "fixture://recommendation/growth/revenue",
    )


def test_soft_conflict_remains_eligible_but_caps_favorable_state_at_watch() -> None:
    case = _case()
    references = case["factor_evidence"]["fundamentals"]["evidence_references"]
    case["factor_evidence"]["fundamentals"][
        "soft_conflicting_evidence_references"
    ] = [references[0], references[1]]
    evaluation = _evaluate(case)
    recommendation = evaluation.recommendation_state

    assert _factor(
        evaluation,
        FactorFamily.FUNDAMENTALS,
    ).state is FactorState.EVALUABLE
    assert recommendation["eligibility_state"] == "eligible"
    assert recommendation["state"] == "watch"
    assert "recommendation_limited_by_soft_conflict" in (
        recommendation["reason_codes"]
    )
    assert recommendation["conflict_references"] == (
        "fixture://recommendation/fundamentals/cash-flow",
        "fixture://recommendation/fundamentals/profitability",
    )


def test_moderately_favorable_pattern_maps_to_consider() -> None:
    evaluation = _evaluate(_moderate_case())

    assert evaluation.recommendation_state["state"] == "consider"
    assert evaluation.recommendation_state["reason_codes"][-1] == (
        "mapped_moderately_favorable_critical_factor_pattern"
    )


def test_preferred_thresholds_are_inclusive() -> None:
    case = _case()
    _set_values(case, "fundamentals", (0.2, 0.2, 0.1375))
    _set_values(case, "growth", (0.18, 0.22, 0.22))
    _set_values(case, "risk", (0.38, 1.5, 1.4))
    _set_values(case, "data_confidence", (0.85, 0.85, 0.85))

    evaluation = _evaluate(case)

    assert _factor(evaluation, FactorFamily.FUNDAMENTALS).score == 75.0
    assert _factor(evaluation, FactorFamily.GROWTH).score == 70.0
    assert _factor(evaluation, FactorFamily.RISK).score == 70.0
    assert _factor(evaluation, FactorFamily.DATA_CONFIDENCE).score == 85.0
    assert evaluation.recommendation_state["state"] == "preferred"


def test_consider_thresholds_are_inclusive() -> None:
    case = _case()
    _set_values(case, "fundamentals", (0.14, 0.14, 0.1))
    _set_values(case, "growth", (0.12, 0.13, 0.13))
    _set_values(case, "risk", (0.44, 2.0, 1.2))
    _set_values(case, "data_confidence", (0.8, 0.8, 0.8))

    evaluation = _evaluate(case)

    assert _factor(evaluation, FactorFamily.FUNDAMENTALS).score == 60.0
    assert _factor(evaluation, FactorFamily.GROWTH).score == 55.0
    assert _factor(evaluation, FactorFamily.RISK).score == 60.0
    assert _factor(evaluation, FactorFamily.DATA_CONFIDENCE).score == 80.0
    assert evaluation.recommendation_state["state"] == "consider"


def test_unfavorable_threshold_is_exclusive() -> None:
    boundary = _case()
    _set_values(boundary, "fundamentals", (0.06, 0.06, 0.05))
    below = deepcopy(boundary)
    _set_values(below, "fundamentals", (0.056, 0.056, 0.0475))

    boundary_evaluation = _evaluate(boundary)
    below_evaluation = _evaluate(below)

    assert _factor(
        boundary_evaluation,
        FactorFamily.FUNDAMENTALS,
    ).score == 40.0
    assert boundary_evaluation.recommendation_state["state"] == "watch"
    assert _factor(
        below_evaluation,
        FactorFamily.FUNDAMENTALS,
    ).score == 39.0
    assert below_evaluation.recommendation_state["state"] == "avoid"


def test_mixed_pattern_maps_to_watch_without_hidden_average() -> None:
    case = _moderate_case()
    _set_values(case, "growth", (0.1, 0.1, 0.1))
    evaluation = _evaluate(case)

    assert _factor(evaluation, FactorFamily.GROWTH).score == 50.0
    assert evaluation.recommendation_state["state"] == "watch"
    assert evaluation.overall_evaluation["score"] is None


def test_unfavorable_pattern_maps_to_avoid_without_execution_semantics() -> None:
    case = _case()
    _set_values(case, "growth", (-0.1, -0.2, -0.2))
    recommendation = _evaluate(case).recommendation_state

    assert recommendation["state"] == "avoid"
    assert recommendation["reason_codes"][-1] == (
        "mapped_unfavorable_critical_factor_pattern"
    )
    assert recommendation["actionable"] is False
    assert recommendation["decision_engine_ready"] is False


def test_risk_direction_and_guardrail_are_not_reversed() -> None:
    favorable = _evaluate()
    guarded_case = _case()
    _set_values(guarded_case, "risk", (0.5, 2.5, 1.0))
    guarded = _evaluate(guarded_case)
    adverse_case = _case()
    _set_values(adverse_case, "risk", (0.8, 5.0, 0.0))
    adverse = _evaluate(adverse_case)

    assert _factor(favorable, FactorFamily.RISK).score == 80.0
    assert favorable.recommendation_state["state"] == "preferred"
    assert _factor(guarded, FactorFamily.RISK).score == 50.0
    assert guarded.recommendation_state["state"] == "watch"
    assert "recommendation_limited_by_risk_guardrail" in (
        guarded.recommendation_state["reason_codes"]
    )
    assert _factor(adverse, FactorFamily.RISK).score == 0.0
    assert adverse.recommendation_state["state"] == "avoid"


def test_missing_risk_is_not_interpreted_as_safe() -> None:
    case = _case()
    case["factor_evidence"]["risk"]["level"] = "none"

    recommendation = _evaluate(case).recommendation_state

    assert recommendation["eligibility_state"] == "ineligible"
    assert recommendation["state"] == "insufficient_evidence"


def test_missing_valuation_remains_missing_and_blocks_complete_evaluation() -> None:
    case = _case()
    case["factor_evidence"].pop("valuation")
    evaluation = _evaluate(case)

    assert _factor(evaluation, FactorFamily.VALUATION).score is None
    assert (
        _factor(evaluation, FactorFamily.VALUATION).state
        is FactorState.UNAVAILABLE
    )
    assert evaluation.recommendation_state["state"] == "insufficient_evidence"
    assert evaluation.recommendation_state["reason_codes"] == (
        "ineligible_governor_evaluation_incomplete",
    )


def test_blocked_portfolio_fit_is_a_limitation_not_allocation_authority() -> None:
    payload = to_plain_dict(_evaluate())
    portfolio = next(
        item
        for item in payload["factor_evaluations"]
        if item["factor"] == "portfolio_fit"
    )

    assert portfolio["state"] == "blocked"
    assert portfolio["score"] is None
    assert payload["recommendation_state"]["state"] == "preferred"
    assert payload["recommendation_state"]["limitations"] == (
        "portfolio_fit_not_used_without_approved_context",
    )
    assert payload["authority_boundary"]["decision_ready"] is False


def test_complete_technical_context_cannot_replace_missing_fundamentals() -> None:
    case = _case()
    case["factor_evidence"]["fundamentals"]["level"] = "none"
    evaluation = _evaluate(case)

    assert _factor(
        evaluation,
        FactorFamily.TECHNICAL_SETUP,
    ).state is FactorState.EVALUABLE
    assert evaluation.recommendation_state["state"] == "insufficient_evidence"
    assert evaluation.buy_zone_explanation["state"] == "blocked"
    assert (
        evaluation.position_management_explanation["state"]
        == "no_position_context"
    )


def test_weights_overall_score_rank_and_reserved_authority_remain_unavailable() -> None:
    payload = to_plain_dict(_evaluate())

    assert all(
        factor["weight"] is None and factor["weighted_score"] is None
        for factor in payload["factor_evaluations"]
    )
    assert payload["overall_evaluation"]["score"] is None
    assert payload["overall_evaluation"]["weighted_score"] is None
    assert payload["overall_evaluation"]["rank"] is None
    assert payload["recommendation_state"]["actionable"] is False
    assert payload["recommendation_state"]["recommendation_state_ready"] is False
    assert payload["recommendation_state"]["decision_engine_ready"] is False
    for field in (
        "actionable",
        "actionable_review",
        "recommendation_state_ready",
        "decision_ready",
        "de_ready",
        "decision_engine_ready",
    ):
        assert payload["authority_boundary"][field] is False


def test_malformed_public_mapping_input_fails_closed_deterministically() -> None:
    boundary = _case()["recommendation_review_boundary"]
    first = map_recommendation_state(
        governor_contract_version="unsupported",
        evaluation_state="evaluation_completed_non_actionable",
        factor_evaluations={},
        recommendation_review_boundary=boundary,
    )
    second = map_recommendation_state(
        governor_contract_version="unsupported",
        evaluation_state="evaluation_completed_non_actionable",
        factor_evaluations={},
        recommendation_review_boundary=boundary,
    )

    assert first == second
    assert first.state is RecommendationState.BLOCKED
    assert first.reason_codes == ("blocked_invalid_governor_contract",)


def test_repeated_input_is_byte_deterministic() -> None:
    first = json.dumps(to_plain_dict(_evaluate(deepcopy(_case()))), sort_keys=True)
    second = json.dumps(to_plain_dict(_evaluate(deepcopy(_case()))), sort_keys=True)

    assert first == second
