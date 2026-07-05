from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from market_engine.governor.evaluation import (
    FactorFamily,
    FactorState,
    evaluate_governor_evidence,
    to_plain_dict,
)
from market_engine.governor.scoring import (
    GOVERNOR_FACTOR_SCORING_CONTRACT_VERSION,
    SCORE_PRECISION,
    SCORE_SCALE,
    score_factor,
)


FIXTURE_PATH = Path(
    "tests/fixtures/market_engine/governor/"
    "me_gv04_governor_scoring_cases.json"
)
EVALUATED_AT = "2026-07-05T14:00:00Z"
INPUT_REFERENCE = "fixture://me-gv04"


def _cases() -> dict[str, dict[str, object]]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return {item["ticker"]: item for item in payload["cases"]}


def _evaluate(case: dict[str, object]):
    return evaluate_governor_evidence(
        case,
        evaluation_timestamp=EVALUATED_AT,
        input_reference=INPUT_REFERENCE,
    )


def _factor(evaluation, family: FactorFamily):
    return next(
        item for item in evaluation.factor_evaluations if item.factor is family
    )


def test_scoring_contract_scale_direction_and_precision_are_exact() -> None:
    assert GOVERNOR_FACTOR_SCORING_CONTRACT_VERSION == (
        "market-engine-governor-factor-scoring-v1"
    )
    assert dict(SCORE_SCALE) == {
        "minimum": 0.0,
        "maximum": 100.0,
        "midpoint": 50.0,
        "contract_version": GOVERNOR_FACTOR_SCORING_CONTRACT_VERSION,
    }
    assert SCORE_PRECISION == 2


def test_broad_evidence_scores_only_supported_evaluable_factors() -> None:
    evaluation = _evaluate(_cases()["GS001"])

    expected = {
        FactorFamily.FUNDAMENTALS: 65.0,
        FactorFamily.GROWTH: 53.33,
        FactorFamily.RISK: 75.0,
        FactorFamily.DATA_CONFIDENCE: 90.0,
    }
    for family, score in expected.items():
        factor = _factor(evaluation, family)
        assert factor.state is FactorState.EVALUABLE
        assert factor.score == score
        assert factor.score_scale == dict(SCORE_SCALE)
        assert len(factor.score_components) == 3
        assert len(factor.score_evidence_references) == 3
        assert factor.weight is None
        assert factor.weighted_score is None

    trend = _factor(evaluation, FactorFamily.TREND)
    assert trend.state is FactorState.EVALUABLE
    assert trend.score is None
    assert trend.score_scale is None
    assert trend.score_limitations == (
        "factor_scoring_rule_not_implemented",
    )


@pytest.mark.parametrize(
    "family",
    (
        FactorFamily.VALUATION,
        FactorFamily.MOMENTUM,
        FactorFamily.TECHNICAL_SETUP,
        FactorFamily.PORTFOLIO_FIT,
    ),
)
def test_non_evaluable_factors_are_never_scored(family: FactorFamily) -> None:
    factor = _factor(_evaluate(_cases()["GS001"]), family)

    assert factor.state is not FactorState.EVALUABLE
    assert factor.score is None
    assert factor.score_scale is None
    assert factor.score_components == ()
    assert factor.weight is None
    assert factor.weighted_score is None


@pytest.mark.parametrize(
    "state",
    (
        "not_started",
        "blocked",
        "unavailable",
        "insufficient_evidence",
        "partial",
        "qualitative_only",
    ),
)
def test_every_non_evaluable_state_is_score_ineligible(state: str) -> None:
    result = score_factor(
        factor="fundamentals",
        state=state,
        factor_evidence={},
        evidence_references=(),
    )

    assert result.score is None
    assert result.score_scale is None
    assert result.score_components == ()
    assert result.score_limitations == (
        f"score_ineligible_factor_state:{state}",
    )


def test_missing_component_is_not_zero_filled() -> None:
    case = deepcopy(_cases()["GS001"])
    components = case["factor_evidence"]["fundamentals"]["score_inputs"][
        "components"
    ]
    components.pop()

    factor = _factor(_evaluate(case), FactorFamily.FUNDAMENTALS)

    assert factor.state is FactorState.EVALUABLE
    assert factor.score is None
    assert factor.score_limitations == (
        "required_score_components_incomplete",
    )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("missing_value", "score_input_invalid:profitability_margin"),
        (
            "bad_reference",
            "score_evidence_reference_invalid:profitability_margin",
        ),
        (
            "bad_rule",
            "normalization_rule_invalid:profitability_margin",
        ),
        ("bad_contract", "score_input_contract_not_approved"),
    ),
)
def test_malformed_or_unapproved_fundamental_inputs_fail_closed(
    mutation: str,
    reason: str,
) -> None:
    case = deepcopy(_cases()["GS001"])
    score_inputs = case["factor_evidence"]["fundamentals"]["score_inputs"]
    first = score_inputs["components"][0]
    if mutation == "missing_value":
        first["input_value"] = None
    elif mutation == "bad_reference":
        first["evidence_reference"] = "fixture://not-approved"
    elif mutation == "bad_rule":
        first["normalization_rule"] = "invented"
    else:
        score_inputs["contract_version"] = "unapproved"

    factor = _factor(_evaluate(case), FactorFamily.FUNDAMENTALS)

    assert factor.score is None
    assert factor.score_limitations == (reason,)


def test_malformed_component_limitations_fail_closed() -> None:
    case = deepcopy(_cases()["GS001"])
    case["factor_evidence"]["fundamentals"]["score_inputs"]["components"][0][
        "limitations"
    ] = "not-a-list"

    factor = _factor(_evaluate(case), FactorFamily.FUNDAMENTALS)

    assert factor.score is None
    assert factor.score_limitations == (
        "score_component_limitations_invalid:profitability_margin",
    )


def test_growth_requires_aligned_multi_period_evidence() -> None:
    case = deepcopy(_cases()["GS001"])
    score_inputs = case["factor_evidence"]["growth"]["score_inputs"]
    score_inputs["period_alignment"] = "mixed_periods"

    factor = _factor(_evaluate(case), FactorFamily.GROWTH)

    assert factor.state is FactorState.EVALUABLE
    assert factor.score is None
    assert factor.score_limitations == ("growth_period_alignment_invalid",)


def test_conflict_remains_visible_and_cannot_be_silently_averaged() -> None:
    factor = _factor(
        _evaluate(_cases()["GS002"]),
        FactorFamily.GROWTH,
    )

    assert factor.state is FactorState.PARTIAL
    assert factor.score is None
    assert factor.conflicting_evidence_references == (
        "fixture://scoring/conflict/revenue-a",
        "fixture://scoring/conflict/revenue-b",
    )
    assert factor.score_limitations == (
        "score_ineligible_factor_state:partial",
    )


def test_stale_blocked_and_unprovenanced_evidence_cannot_score() -> None:
    for gate in ("fresh", "provenance_valid", "consumable"):
        case = deepcopy(_cases()["GS001"])
        case["factor_evidence"]["fundamentals"][gate] = False
        factor = _factor(_evaluate(case), FactorFamily.FUNDAMENTALS)
        assert factor.state is FactorState.BLOCKED
        assert factor.score is None


def test_risk_score_direction_is_higher_for_lower_risk_profile() -> None:
    favorable = _factor(
        _evaluate(_cases()["GS001"]),
        FactorFamily.RISK,
    )
    adverse_case = deepcopy(_cases()["GS001"])
    inputs = adverse_case["factor_evidence"]["risk"]["score_inputs"][
        "components"
    ]
    inputs[0]["input_value"] = 0.8
    inputs[1]["input_value"] = 5.0
    inputs[2]["input_value"] = 0.0
    adverse = _factor(_evaluate(adverse_case), FactorFamily.RISK)

    assert favorable.score == 75.0
    assert adverse.score == 0.0
    assert favorable.score > adverse.score


def test_normalization_clamps_boundaries_and_rounds_deterministically() -> None:
    case = _cases()["GS001"]
    raw = deepcopy(case["factor_evidence"]["fundamentals"])
    raw["score_inputs"]["components"][0]["input_value"] = 99.0
    raw["score_inputs"]["components"][1]["input_value"] = -99.0
    raw["score_inputs"]["components"][2]["input_value"] = 0.075

    result = score_factor(
        factor="fundamentals",
        state="evaluable",
        factor_evidence=raw,
        evidence_references=raw["evidence_references"],
    )

    assert result.score == 50.0
    assert [
        item["normalized_value"] for item in result.score_components
    ] == [100.0, 0.0, 50.0]


def test_data_confidence_is_independent_and_not_a_hidden_multiplier() -> None:
    first_case = deepcopy(_cases()["GS001"])
    second_case = deepcopy(first_case)
    second_case["factor_evidence"]["data_confidence"]["score_inputs"][
        "components"
    ][0]["input_value"] = 0.0

    first = _evaluate(first_case)
    second = _evaluate(second_case)

    assert _factor(first, FactorFamily.DATA_CONFIDENCE).score == 90.0
    assert _factor(second, FactorFamily.DATA_CONFIDENCE).score == 60.0
    for family in (
        FactorFamily.FUNDAMENTALS,
        FactorFamily.GROWTH,
        FactorFamily.RISK,
    ):
        assert _factor(first, family).score == _factor(second, family).score


def test_overall_weight_recommendation_and_reserved_boundaries_remain_null() -> None:
    payload = to_plain_dict(_evaluate(_cases()["GS001"]))

    for factor in payload["factor_evaluations"]:
        assert factor["weight"] is None
        assert factor["weighted_score"] is None
        if factor["score"] is not None:
            assert factor["state"] == "evaluable"
            assert 0.0 <= factor["score"] <= 100.0
    assert payload["overall_evaluation"]["score"] is None
    assert payload["overall_evaluation"]["score_scale"] is None
    assert payload["overall_evaluation"]["weighted_score"] is None
    assert payload["overall_evaluation"]["rank"] is None
    recommendation = payload["recommendation_state"]
    assert recommendation["contract_version"] == (
        "market-engine-governor-recommendation-state-v1"
    )
    assert recommendation["eligibility_state"] == "ineligible"
    assert recommendation["state"] == "blocked"
    assert recommendation["reason_codes"] == (
        "blocked_recommendation_review_boundary_missing",
    )
    assert recommendation["actionable"] is False
    assert recommendation["recommendation_state_ready"] is False
    assert recommendation["decision_engine_ready"] is False
    assert payload["buy_zone_explanation"]["state"] == "blocked_not_authorized"
    assert (
        payload["position_management_explanation"]["state"]
        == "blocked_not_authorized"
    )


def test_same_evidence_produces_identical_serialized_output() -> None:
    first = to_plain_dict(_evaluate(deepcopy(_cases()["GS001"])))
    second = to_plain_dict(_evaluate(deepcopy(_cases()["GS001"])))

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
