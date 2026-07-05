from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from market_engine.governor.evaluation import (
    FactorFamily,
    evaluate_governor_evidence,
    to_plain_dict,
)
from market_engine.governor.explanation import (
    APPROVED_PORTFOLIO_CONTEXT_CONTRACT_VERSION,
    APPROVED_PRICE_SETUP_CONTEXT_CONTRACT_VERSION,
    GOVERNOR_EXPLANATION_CONTRACT_VERSION,
    BuyZoneState,
    ExplanationEligibilityState,
    PositionManagementState,
)


FIXTURE_PATH = Path(
    "tests/fixtures/market_engine/governor/"
    "me_gv05_governor_recommendation_case.json"
)
EVALUATED_AT = "2026-07-05T18:00:00Z"
INPUT_REFERENCE = "fixture://me-gv06"


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


def _with_position(
    case: dict[str, object],
    *,
    position_state: str,
) -> dict[str, object]:
    case["approved_portfolio_context"] = True
    case["factor_evidence"]["portfolio_fit"] = {
        "evidence_references": ["fixture://explanation/portfolio-fit"],
        "level": "complete",
    }
    case["position_context"] = {
        "contract_version": "market-engine-portfolio-context-v1",
        "fresh": True,
        "position_state": position_state,
        "provenance_valid": True,
        "reference": "fixture://explanation/position-context",
        "ticker": case["ticker"],
    }
    return case


def test_contract_versions_and_state_sets_are_exact() -> None:
    assert GOVERNOR_EXPLANATION_CONTRACT_VERSION == (
        "market-engine-governor-buy-zone-position-management-explanation-v1"
    )
    assert APPROVED_PRICE_SETUP_CONTEXT_CONTRACT_VERSION == (
        "market-engine-governor-approved-price-setup-context-v1"
    )
    assert APPROVED_PORTFOLIO_CONTEXT_CONTRACT_VERSION == (
        "market-engine-portfolio-context-v1"
    )
    assert tuple(item.value for item in ExplanationEligibilityState) == (
        "eligible",
        "ineligible",
    )
    assert tuple(item.value for item in BuyZoneState) == (
        "blocked",
        "insufficient_evidence",
        "wait_for_pullback",
        "wait_for_breakout_confirmation",
        "acceptable_zone_context",
        "extended_avoid_chasing",
        "no_favorable_zone_identified",
    )
    assert tuple(item.value for item in PositionManagementState) == (
        "blocked",
        "insufficient_evidence",
        "no_position_context",
        "hold_context",
        "add_review_context",
        "reduce_review_context",
        "exit_review_context",
        "monitor_context",
    )
    with pytest.raises(ValueError):
        BuyZoneState("buy_now")
    with pytest.raises(ValueError):
        PositionManagementState("sell_half")


def test_approved_zone_context_is_eligible_and_preserves_exact_levels() -> None:
    evaluation = _evaluate()
    explanation = evaluation.buy_zone_explanation

    assert explanation["eligibility_state"] == "eligible"
    assert explanation["state"] == "acceptable_zone_context"
    assert explanation["pullback_condition"]["lower_bound"] == 140.0
    assert explanation["pullback_condition"]["upper_bound"] == 143.0
    assert explanation["breakout_condition"]["level"] == 152.0
    assert explanation["invalidation_context"]["level"] == 139.0
    assert explanation["current_position_relative_to_zone"] == "unavailable"
    assert explanation["execution_authorized"] is False
    assert explanation["stop_order_authorized"] is False
    assert "fixture://recommendation/price/acceptable-zone" in (
        explanation["approved_price_references"]
    )
    assert explanation["limitations"] == (
        "valuation_score_unavailable_no_target_inference",
    )


def test_blocked_recommendation_blocks_buy_zone() -> None:
    case = _case()
    case.pop("recommendation_review_boundary")
    explanation = _evaluate(case).buy_zone_explanation

    assert explanation["eligibility_state"] == "ineligible"
    assert explanation["state"] == "blocked"
    assert explanation["reason_codes"] == (
        "blocked_recommendation_not_eligible",
    )


def test_missing_price_context_is_insufficient_and_invents_no_levels() -> None:
    case = _case()
    case.pop("price_setup_context")
    explanation = _evaluate(case).buy_zone_explanation

    assert explanation["state"] == "insufficient_evidence"
    assert explanation["reason_codes"] == ("missing_approved_price_context",)
    assert explanation["approved_price_references"] == ()
    assert explanation["pullback_condition"]["lower_bound"] is None
    assert explanation["breakout_condition"]["level"] is None


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("stale", "blocked_stale_price_context"),
        ("provenance", "blocked_invalid_price_context"),
        ("contract", "blocked_invalid_price_context"),
        ("structure", "blocked_invalid_price_context"),
        ("conflicts", "blocked_invalid_price_context"),
    ),
)
def test_invalid_or_stale_price_context_fails_closed(
    mutation: str,
    reason: str,
) -> None:
    case = _case()
    context = case["price_setup_context"]
    if mutation == "stale":
        context["fresh"] = False
    elif mutation == "provenance":
        context["provenance_valid"] = False
    elif mutation == "contract":
        context["contract_version"] = "unsupported"
    else:
        if mutation == "structure":
            context["structurally_valid"] = False
        else:
            context["hard_conflict_references"] = "not-a-list"

    explanation = _evaluate(case).buy_zone_explanation

    assert explanation["state"] == "blocked"
    assert explanation["reason_codes"] == (reason,)
    assert explanation["approved_price_references"] == ()


def test_misaligned_price_or_position_context_fails_closed() -> None:
    price_case = _case()
    price_case["price_setup_context"]["ticker"] = "OTHER"
    position_case = _with_position(_case(), position_state="held")
    position_case["position_context"]["ticker"] = "OTHER"

    assert _evaluate(price_case).buy_zone_explanation["state"] == "blocked"
    assert (
        _evaluate(position_case).position_management_explanation["state"]
        == "blocked"
    )


@pytest.mark.parametrize(
    ("condition_state", "expected_state"),
    (
        ("pullback_preferred", "wait_for_pullback"),
        (
            "breakout_confirmation_required",
            "wait_for_breakout_confirmation",
        ),
        ("acceptable_zone", "acceptable_zone_context"),
        ("extended", "extended_avoid_chasing"),
        ("no_favorable_zone", "no_favorable_zone_identified"),
    ),
)
def test_price_condition_states_map_deterministically(
    condition_state: str,
    expected_state: str,
) -> None:
    case = _case()
    case["price_setup_context"]["condition_state"] = condition_state
    if condition_state == "extended":
        case["price_setup_context"][
            "extension_reference"
        ] = "fixture://explanation/extended-context"

    explanation = _evaluate(case).buy_zone_explanation

    assert explanation["state"] == expected_state
    assert explanation["reason_codes"][:2] == (
        "buy_zone_explanation_eligible",
        f"mapped_price_context:{condition_state}",
    )


def test_malformed_price_bounds_are_not_surfaced() -> None:
    case = _case()
    case["price_setup_context"]["acceptable_zone"]["lower_bound"] = "146"
    explanation = _evaluate(case).buy_zone_explanation

    assert explanation["state"] == "blocked"
    assert explanation["reason_codes"] == ("blocked_invalid_price_context",)
    assert explanation["approved_price_references"] == ()


def test_hard_price_conflict_blocks_and_preserves_references() -> None:
    case = _case()
    case["price_setup_context"]["hard_conflict_references"] = [
        "fixture://explanation/conflict/a",
        "fixture://explanation/conflict/b",
    ]
    explanation = _evaluate(case).buy_zone_explanation

    assert explanation["state"] == "blocked"
    assert explanation["reason_codes"] == (
        "blocked_unresolved_hard_price_conflict",
    )
    assert explanation["conflict_references"] == (
        "fixture://explanation/conflict/a",
        "fixture://explanation/conflict/b",
    )


def test_soft_price_conflict_limits_zone_without_silent_average() -> None:
    case = _case()
    case["price_setup_context"]["soft_conflict_references"] = [
        "fixture://explanation/conflict/soft-a",
        "fixture://explanation/conflict/soft-b",
    ]
    explanation = _evaluate(case).buy_zone_explanation

    assert explanation["eligibility_state"] == "eligible"
    assert explanation["state"] == "no_favorable_zone_identified"
    assert "buy_zone_limited_by_soft_price_conflict" in (
        explanation["reason_codes"]
    )
    assert explanation["conflict_references"] == (
        "fixture://explanation/conflict/soft-a",
        "fixture://explanation/conflict/soft-b",
    )


def test_unfavorable_risk_blocks_zone_and_is_not_reversed() -> None:
    case = _case()
    _set_values(case, "risk", (0.8, 5.0, 0.0))
    evaluation = _evaluate(case)

    assert _factor(evaluation, FactorFamily.RISK).score == 0.0
    assert evaluation.recommendation_state["state"] == "avoid"
    assert evaluation.buy_zone_explanation["state"] == "blocked"
    assert evaluation.buy_zone_explanation["reason_codes"] == (
        "blocked_unfavorable_risk_context",
    )


def test_low_data_confidence_blocks_downstream_explanation() -> None:
    case = _case()
    _set_values(case, "data_confidence", (0.7, 0.7, 0.7))
    evaluation = _evaluate(case)

    assert _factor(evaluation, FactorFamily.DATA_CONFIDENCE).score == 70.0
    assert evaluation.recommendation_state["eligibility_state"] == "ineligible"
    assert evaluation.buy_zone_explanation["state"] == "blocked"
    assert evaluation.buy_zone_explanation["execution_authorized"] is False


def test_missing_risk_is_never_interpreted_as_safe() -> None:
    case = _case()
    case["factor_evidence"]["risk"]["level"] = "none"
    evaluation = _evaluate(case)

    assert _factor(evaluation, FactorFamily.RISK).score is None
    assert evaluation.recommendation_state["eligibility_state"] == "ineligible"
    assert evaluation.buy_zone_explanation["state"] == "blocked"


def test_invalidation_is_explanation_only() -> None:
    invalidation = _evaluate().buy_zone_explanation["invalidation_context"]

    assert invalidation["state"] == "intact"
    assert invalidation["evidence_reference"] == (
        "fixture://recommendation/price/invalidation"
    )
    assert invalidation["stop_order_authorized"] is False


def test_missing_position_context_is_ineligible_no_context() -> None:
    explanation = _evaluate().position_management_explanation

    assert explanation["eligibility_state"] == "ineligible"
    assert explanation["state"] == "no_position_context"
    assert explanation["reason_codes"] == (
        "missing_approved_position_context",
    )


def test_explicit_not_held_context_is_handled_without_position_action() -> None:
    case = _with_position(_case(), position_state="not_held")
    explanation = _evaluate(case).position_management_explanation

    assert explanation["eligibility_state"] == "eligible"
    assert explanation["state"] == "no_position_context"
    assert explanation["position_context_reference"] == (
        "fixture://explanation/position-context"
    )
    assert explanation["portfolio_mutation_authorized"] is False
    assert explanation["order_generation_authorized"] is False


def test_existing_favorable_intact_position_maps_to_hold_context() -> None:
    case = _with_position(_case(), position_state="held")
    explanation = _evaluate(case).position_management_explanation

    assert explanation["state"] == "hold_context"
    assert explanation["reason_codes"][-1] == (
        "mapped_intact_existing_position_context"
    )


def test_approved_additional_confirmation_maps_to_add_review_only() -> None:
    case = _with_position(_case(), position_state="held")
    context = case["price_setup_context"]
    context["additional_setup_confirmation"] = True
    context["additional_setup_confirmation_reference"] = (
        "fixture://explanation/additional-confirmation"
    )
    explanation = _evaluate(case).position_management_explanation

    assert explanation["state"] == "add_review_context"
    assert explanation["portfolio_mutation_authorized"] is False
    assert explanation["order_generation_authorized"] is False


def test_deteriorating_or_unfavorable_risk_maps_to_reduce_review() -> None:
    deteriorating = _with_position(_case(), position_state="held")
    deteriorating["price_setup_context"]["setup_state"] = "deteriorating"
    risk_limited = _with_position(_case(), position_state="held")
    _set_values(risk_limited, "risk", (0.5, 2.5, 1.0))

    assert (
        _evaluate(deteriorating).position_management_explanation["state"]
        == "reduce_review_context"
    )
    risk_evaluation = _evaluate(risk_limited)
    assert _factor(risk_evaluation, FactorFamily.RISK).score == 50.0
    assert (
        risk_evaluation.position_management_explanation["state"]
        == "reduce_review_context"
    )


def test_invalidated_unfavorable_position_maps_to_exit_review_only() -> None:
    case = _with_position(_case(), position_state="held")
    _set_values(case, "growth", (-0.1, -0.2, -0.2))
    case["price_setup_context"]["setup_state"] = "invalidated"
    case["price_setup_context"]["invalidation_context"][
        "state"
    ] = "invalidated"
    explanation = _evaluate(case).position_management_explanation

    assert explanation["supporting_recommendation_state"] == "avoid"
    assert explanation["state"] == "exit_review_context"
    assert explanation["portfolio_mutation_authorized"] is False
    assert explanation["order_generation_authorized"] is False


def test_soft_conflict_prevents_add_review_upgrade() -> None:
    case = _with_position(_case(), position_state="held")
    context = case["price_setup_context"]
    context["additional_setup_confirmation"] = True
    context["additional_setup_confirmation_reference"] = (
        "fixture://explanation/additional-confirmation"
    )
    context["soft_conflict_references"] = [
        "fixture://explanation/conflict/soft"
    ]
    explanation = _evaluate(case).position_management_explanation

    assert explanation["state"] == "hold_context"
    assert explanation["limitations"] == (
        "soft_price_conflict_limits_position_review",
    )


def test_stale_or_invalid_position_context_fails_closed() -> None:
    stale = _with_position(_case(), position_state="held")
    stale["position_context"]["fresh"] = False
    invalid = _with_position(_case(), position_state="held")
    invalid["position_context"]["provenance_valid"] = False

    assert (
        _evaluate(stale).position_management_explanation["state"] == "blocked"
    )
    assert (
        _evaluate(invalid).position_management_explanation["state"] == "blocked"
    )


@pytest.mark.parametrize("mutation", ("stale", "invalid"))
def test_existing_position_requires_current_approved_price_context(
    mutation: str,
) -> None:
    case = _with_position(_case(), position_state="held")
    if mutation == "stale":
        case["price_setup_context"]["fresh"] = False
        expected_reason = "position_blocked_stale_price_context"
    else:
        case["price_setup_context"]["provenance_valid"] = False
        expected_reason = "position_blocked_invalid_price_context"

    explanation = _evaluate(case).position_management_explanation

    assert explanation["eligibility_state"] == "ineligible"
    assert explanation["state"] == "blocked"
    assert explanation["reason_codes"] == (expected_reason,)


def test_all_authority_and_aggregation_fields_remain_unavailable() -> None:
    payload = to_plain_dict(
        _evaluate(_with_position(_case(), position_state="held"))
    )

    assert payload["buy_zone_explanation"]["execution_authorized"] is False
    assert payload["buy_zone_explanation"]["stop_order_authorized"] is False
    assert (
        payload["position_management_explanation"][
            "portfolio_mutation_authorized"
        ]
        is False
    )
    assert (
        payload["position_management_explanation"][
            "order_generation_authorized"
        ]
        is False
    )
    assert payload["recommendation_state"]["actionable"] is False
    assert payload["recommendation_state"]["decision_engine_ready"] is False
    assert all(
        factor["weight"] is None and factor["weighted_score"] is None
        for factor in payload["factor_evaluations"]
    )
    assert payload["overall_evaluation"]["score"] is None
    assert payload["overall_evaluation"]["weighted_score"] is None
    assert payload["overall_evaluation"]["rank"] is None


def test_repeated_input_is_byte_deterministic() -> None:
    first = json.dumps(to_plain_dict(_evaluate(deepcopy(_case()))), sort_keys=True)
    second = json.dumps(to_plain_dict(_evaluate(deepcopy(_case()))), sort_keys=True)

    assert first == second
