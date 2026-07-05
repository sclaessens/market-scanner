from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any


GOVERNOR_EXPLANATION_CONTRACT_VERSION = (
    "market-engine-governor-buy-zone-position-management-explanation-v1"
)
APPROVED_PRICE_SETUP_CONTEXT_CONTRACT_VERSION = (
    "market-engine-governor-approved-price-setup-context-v1"
)
APPROVED_PORTFOLIO_CONTEXT_CONTRACT_VERSION = (
    "market-engine-portfolio-context-v1"
)


class ExplanationEligibilityState(StrEnum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"


class BuyZoneState(StrEnum):
    BLOCKED = "blocked"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    WAIT_FOR_PULLBACK = "wait_for_pullback"
    WAIT_FOR_BREAKOUT_CONFIRMATION = "wait_for_breakout_confirmation"
    ACCEPTABLE_ZONE_CONTEXT = "acceptable_zone_context"
    EXTENDED_AVOID_CHASING = "extended_avoid_chasing"
    NO_FAVORABLE_ZONE_IDENTIFIED = "no_favorable_zone_identified"


class PositionManagementState(StrEnum):
    BLOCKED = "blocked"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NO_POSITION_CONTEXT = "no_position_context"
    HOLD_CONTEXT = "hold_context"
    ADD_REVIEW_CONTEXT = "add_review_context"
    REDUCE_REVIEW_CONTEXT = "reduce_review_context"
    EXIT_REVIEW_CONTEXT = "exit_review_context"
    MONITOR_CONTEXT = "monitor_context"


@dataclass(frozen=True)
class BuyZoneExplanationResult:
    contract_version: str
    eligibility_state: ExplanationEligibilityState
    state: BuyZoneState
    reason_codes: tuple[str, ...]
    approved_price_references: tuple[str, ...]
    pullback_condition: Mapping[str, Any]
    breakout_condition: Mapping[str, Any]
    invalidation_context: Mapping[str, Any]
    current_position_relative_to_zone: str
    conflict_references: tuple[str, ...]
    limitations: tuple[str, ...]
    execution_authorized: bool = False
    stop_order_authorized: bool = False
    decision_engine_ready: bool = False


@dataclass(frozen=True)
class PositionManagementExplanationResult:
    contract_version: str
    eligibility_state: ExplanationEligibilityState
    state: PositionManagementState
    reason_codes: tuple[str, ...]
    position_context_reference: str | None
    supporting_recommendation_state: str
    supporting_factor_scores: tuple[Mapping[str, Any], ...]
    invalidation_context: Mapping[str, Any]
    conflict_references: tuple[str, ...]
    limitations: tuple[str, ...]
    portfolio_mutation_authorized: bool = False
    order_generation_authorized: bool = False
    decision_engine_ready: bool = False


_BUY_ZONE_STATE_BY_CONTEXT = {
    "pullback_preferred": BuyZoneState.WAIT_FOR_PULLBACK,
    "breakout_confirmation_required": (
        BuyZoneState.WAIT_FOR_BREAKOUT_CONFIRMATION
    ),
    "acceptable_zone": BuyZoneState.ACCEPTABLE_ZONE_CONTEXT,
    "extended": BuyZoneState.EXTENDED_AVOID_CHASING,
    "no_favorable_zone": BuyZoneState.NO_FAVORABLE_ZONE_IDENTIFIED,
}
_REQUIRED_FACTOR_STATES = ("technical_setup", "trend", "momentum")


def evaluate_buy_zone_explanation(
    *,
    evaluation_ticker: str,
    evaluation_state: object,
    factor_evaluations: Sequence[object],
    recommendation_state: object,
    price_setup_context: object,
) -> BuyZoneExplanationResult:
    """Build conditional price-context explanation without execution authority."""
    factors = _factor_mapping(factor_evaluations)
    recommendation_eligibility = _value(
        recommendation_state,
        "eligibility_state",
    )
    recommendation_direction = _value(recommendation_state, "state")
    if (
        evaluation_state != "evaluation_completed_non_actionable"
        or recommendation_eligibility != "eligible"
    ):
        return _buy_ineligible(
            state=BuyZoneState.BLOCKED,
            reason="blocked_recommendation_not_eligible",
        )

    critical_factor_blockers = tuple(
        factor
        for factor in _REQUIRED_FACTOR_STATES
        if _factor_state(factors.get(factor)) != "evaluable"
    )
    if critical_factor_blockers:
        return _buy_ineligible(
            state=BuyZoneState.INSUFFICIENT_EVIDENCE,
            reason="ineligible_price_setup_factor_coverage",
            limitations=tuple(
                f"missing_evaluable_factor:{factor}"
                for factor in critical_factor_blockers
            ),
        )

    risk_score = _factor_score(factors.get("risk"))
    data_confidence_score = _factor_score(factors.get("data_confidence"))
    if risk_score is None:
        return _buy_ineligible(
            state=BuyZoneState.INSUFFICIENT_EVIDENCE,
            reason="ineligible_missing_risk_score",
        )
    if data_confidence_score is None or data_confidence_score < 75.0:
        return _buy_ineligible(
            state=BuyZoneState.BLOCKED,
            reason="blocked_insufficient_data_confidence",
        )
    if risk_score < 40.0:
        return _buy_ineligible(
            state=BuyZoneState.BLOCKED,
            reason="blocked_unfavorable_risk_context",
        )

    context_reason = _price_context_block_reason(
        price_setup_context,
        expected_ticker=evaluation_ticker,
    )
    if context_reason is not None:
        state = (
            BuyZoneState.INSUFFICIENT_EVIDENCE
            if context_reason == "missing_approved_price_context"
            else BuyZoneState.BLOCKED
        )
        return _buy_ineligible(state=state, reason=context_reason)
    assert isinstance(price_setup_context, Mapping)

    hard_conflicts = _text_references(
        price_setup_context.get("hard_conflict_references")
    )
    if hard_conflicts:
        return _buy_ineligible(
            state=BuyZoneState.BLOCKED,
            reason="blocked_unresolved_hard_price_conflict",
            conflict_references=hard_conflicts,
        )

    invalidation = _invalidation_payload(price_setup_context)
    if invalidation is None:
        return _buy_ineligible(
            state=BuyZoneState.INSUFFICIENT_EVIDENCE,
            reason="ineligible_missing_invalidation_context",
        )

    condition_state = price_setup_context.get("condition_state")
    state = _BUY_ZONE_STATE_BY_CONTEXT.get(condition_state)
    if state is None or not _condition_evidence_valid(
        condition_state,
        price_setup_context,
    ):
        return _buy_ineligible(
            state=BuyZoneState.INSUFFICIENT_EVIDENCE,
            reason="ineligible_price_condition_evidence",
            invalidation_context=invalidation,
        )

    reason_codes = ["buy_zone_explanation_eligible"]
    reason_codes.append(f"mapped_price_context:{condition_state}")
    limitations = list(_valuation_limitations(factors))
    soft_conflicts = _text_references(
        price_setup_context.get("soft_conflict_references")
    )
    if soft_conflicts and state in {
        BuyZoneState.ACCEPTABLE_ZONE_CONTEXT,
        BuyZoneState.WAIT_FOR_PULLBACK,
        BuyZoneState.WAIT_FOR_BREAKOUT_CONFIRMATION,
    }:
        state = BuyZoneState.NO_FAVORABLE_ZONE_IDENTIFIED
        reason_codes.append("buy_zone_limited_by_soft_price_conflict")
        limitations.append("soft_price_conflict_requires_cautious_review")
    if recommendation_direction == "avoid":
        state = BuyZoneState.NO_FAVORABLE_ZONE_IDENTIFIED
        reason_codes.append("buy_zone_limited_by_unfavorable_recommendation")
    elif (
        recommendation_direction == "watch"
        and state is BuyZoneState.ACCEPTABLE_ZONE_CONTEXT
    ):
        state = BuyZoneState.NO_FAVORABLE_ZONE_IDENTIFIED
        reason_codes.append("buy_zone_limited_by_watch_recommendation")
    if (
        price_setup_context.get("setup_state")
        in {"deteriorating", "invalidated"}
        and state
        in {
            BuyZoneState.ACCEPTABLE_ZONE_CONTEXT,
            BuyZoneState.WAIT_FOR_PULLBACK,
            BuyZoneState.WAIT_FOR_BREAKOUT_CONFIRMATION,
        }
    ):
        state = BuyZoneState.NO_FAVORABLE_ZONE_IDENTIFIED
        reason_codes.append("buy_zone_limited_by_setup_deterioration")
    if risk_score < 60.0:
        if state is BuyZoneState.ACCEPTABLE_ZONE_CONTEXT:
            state = BuyZoneState.NO_FAVORABLE_ZONE_IDENTIFIED
        reason_codes.append("buy_zone_limited_by_risk_guardrail")

    return BuyZoneExplanationResult(
        contract_version=GOVERNOR_EXPLANATION_CONTRACT_VERSION,
        eligibility_state=ExplanationEligibilityState.ELIGIBLE,
        state=state,
        reason_codes=tuple(reason_codes),
        approved_price_references=_approved_price_references(
            price_setup_context
        ),
        pullback_condition=_price_condition_payload(
            price_setup_context.get("support_zone"),
            condition_type="approved_support_zone",
        ),
        breakout_condition=_price_condition_payload(
            price_setup_context.get("breakout_trigger"),
            condition_type="approved_breakout_trigger",
        ),
        invalidation_context=invalidation,
        current_position_relative_to_zone=str(
            price_setup_context.get(
                "current_position_relative_to_zone",
                "unavailable",
            )
        ),
        conflict_references=soft_conflicts,
        limitations=tuple(sorted(set(limitations))),
    )


def evaluate_position_management_explanation(
    *,
    evaluation_ticker: str,
    factor_evaluations: Sequence[object],
    recommendation_state: object,
    buy_zone_explanation: BuyZoneExplanationResult,
    price_setup_context: object,
    position_context: object,
) -> PositionManagementExplanationResult:
    """Explain review context without mutating or sizing a position."""
    recommendation_direction = _value(recommendation_state, "state")
    supporting_scores = _supporting_scores(recommendation_state)
    invalidation = (
        buy_zone_explanation.invalidation_context
        if buy_zone_explanation.invalidation_context
        else _unavailable_invalidation()
    )
    context_reason = _position_context_block_reason(
        position_context,
        expected_ticker=evaluation_ticker,
    )
    if context_reason is not None:
        state = (
            PositionManagementState.NO_POSITION_CONTEXT
            if context_reason == "missing_approved_position_context"
            else PositionManagementState.BLOCKED
        )
        return _position_ineligible(
            state=state,
            reason=context_reason,
            recommendation_direction=recommendation_direction,
            supporting_scores=supporting_scores,
            invalidation=invalidation,
        )
    assert isinstance(position_context, Mapping)
    position_state = position_context.get("position_state")
    position_reference = str(position_context["reference"])
    if position_state == "not_held":
        return PositionManagementExplanationResult(
            contract_version=GOVERNOR_EXPLANATION_CONTRACT_VERSION,
            eligibility_state=ExplanationEligibilityState.ELIGIBLE,
            state=PositionManagementState.NO_POSITION_CONTEXT,
            reason_codes=("approved_context_confirms_position_not_held",),
            position_context_reference=position_reference,
            supporting_recommendation_state=recommendation_direction,
            supporting_factor_scores=supporting_scores,
            invalidation_context=invalidation,
            conflict_references=(),
            limitations=(),
        )
    if position_state != "held":
        return _position_ineligible(
            state=PositionManagementState.INSUFFICIENT_EVIDENCE,
            reason="ineligible_position_state_not_held_or_held",
            recommendation_direction=recommendation_direction,
            supporting_scores=supporting_scores,
            invalidation=invalidation,
            position_reference=position_reference,
        )
    if _value(recommendation_state, "eligibility_state") != "eligible":
        return _position_ineligible(
            state=PositionManagementState.BLOCKED,
            reason="blocked_recommendation_not_eligible",
            recommendation_direction=recommendation_direction,
            supporting_scores=supporting_scores,
            invalidation=invalidation,
            position_reference=position_reference,
        )
    price_context_reason = _price_context_block_reason(
        price_setup_context,
        expected_ticker=evaluation_ticker,
    )
    if price_context_reason is not None:
        state = (
            PositionManagementState.INSUFFICIENT_EVIDENCE
            if price_context_reason == "missing_approved_price_context"
            else PositionManagementState.BLOCKED
        )
        return _position_ineligible(
            state=state,
            reason=f"position_{price_context_reason}",
            recommendation_direction=recommendation_direction,
            supporting_scores=supporting_scores,
            invalidation=invalidation,
            position_reference=position_reference,
        )
    assert isinstance(price_setup_context, Mapping)
    hard_conflicts = _text_references(
        price_setup_context.get("hard_conflict_references")
    )
    if hard_conflicts:
        return _position_ineligible(
            state=PositionManagementState.BLOCKED,
            reason="blocked_unresolved_hard_price_conflict",
            recommendation_direction=recommendation_direction,
            supporting_scores=supporting_scores,
            invalidation=invalidation,
            position_reference=position_reference,
            conflict_references=hard_conflicts,
        )

    factors = _factor_mapping(factor_evaluations)
    risk_score = _factor_score(factors.get("risk"))
    if risk_score is None:
        return _position_ineligible(
            state=PositionManagementState.INSUFFICIENT_EVIDENCE,
            reason="ineligible_missing_risk_score",
            recommendation_direction=recommendation_direction,
            supporting_scores=supporting_scores,
            invalidation=invalidation,
            position_reference=position_reference,
        )

    setup_state = price_setup_context.get("setup_state")
    invalidation_state = invalidation.get("state")
    soft_conflicts = _text_references(
        price_setup_context.get("soft_conflict_references")
    )
    limitations: list[str] = []
    if soft_conflicts:
        limitations.append("soft_price_conflict_limits_position_review")

    if invalidation_state == "invalidated" and recommendation_direction == "avoid":
        state = PositionManagementState.EXIT_REVIEW_CONTEXT
        reason = "mapped_invalidated_unfavorable_existing_position"
    elif (
        setup_state == "deteriorating"
        or risk_score < 60.0
        or recommendation_direction == "avoid"
    ):
        state = PositionManagementState.REDUCE_REVIEW_CONTEXT
        reason = "mapped_deteriorating_existing_position_context"
    elif (
        price_setup_context.get("additional_setup_confirmation") is True
        and _unpadded_text(
            price_setup_context.get(
                "additional_setup_confirmation_reference"
            )
        )
        and recommendation_direction in {"preferred", "consider"}
        and risk_score >= 70.0
        and not soft_conflicts
    ):
        state = PositionManagementState.ADD_REVIEW_CONTEXT
        reason = "mapped_approved_additional_confirmation_context"
    elif setup_state == "intact" and recommendation_direction in {
        "preferred",
        "consider",
        "watch",
    }:
        state = PositionManagementState.HOLD_CONTEXT
        reason = "mapped_intact_existing_position_context"
    else:
        state = PositionManagementState.MONITOR_CONTEXT
        reason = "mapped_monitor_existing_position_context"

    return PositionManagementExplanationResult(
        contract_version=GOVERNOR_EXPLANATION_CONTRACT_VERSION,
        eligibility_state=ExplanationEligibilityState.ELIGIBLE,
        state=state,
        reason_codes=("position_management_explanation_eligible", reason),
        position_context_reference=position_reference,
        supporting_recommendation_state=recommendation_direction,
        supporting_factor_scores=supporting_scores,
        invalidation_context=invalidation,
        conflict_references=soft_conflicts,
        limitations=tuple(limitations),
    )


def _price_context_block_reason(
    context: object,
    *,
    expected_ticker: str,
) -> str | None:
    if not isinstance(context, Mapping):
        return "missing_approved_price_context"
    if (
        context.get("contract_version")
        != APPROVED_PRICE_SETUP_CONTEXT_CONTRACT_VERSION
        or context.get("provenance_valid") is not True
        or context.get("structurally_valid") is not True
        or not _unpadded_text(context.get("reference"))
        or context.get("ticker") != expected_ticker
        or not _price_context_structure_valid(context)
    ):
        return "blocked_invalid_price_context"
    if context.get("fresh") is not True:
        return "blocked_stale_price_context"
    return None


def _price_context_structure_valid(context: Mapping[str, Any]) -> bool:
    if not all(
        _valid_text_sequence(context.get(field, []))
        for field in ("hard_conflict_references", "soft_conflict_references")
    ):
        return False
    if context.get(
        "current_position_relative_to_zone",
        "unavailable",
    ) not in {
        "unavailable",
        "within_approved_zone",
        "above_approved_zone",
        "below_approved_zone",
        "extended",
    }:
        return False
    if (
        "additional_setup_confirmation" in context
        and not isinstance(context["additional_setup_confirmation"], bool)
    ):
        return False
    for key in ("support_zone", "acceptable_zone"):
        raw = context.get(key)
        if raw is None:
            continue
        if (
            not isinstance(raw, Mapping)
            or not _unpadded_text(raw.get("reference"))
            or not _finite_number(raw.get("lower_bound"))
            or not _finite_number(raw.get("upper_bound"))
            or float(raw["lower_bound"]) > float(raw["upper_bound"])
            or not _valid_text_sequence(raw.get("limitations", []))
        ):
            return False
    trigger = context.get("breakout_trigger")
    if trigger is not None and (
        not isinstance(trigger, Mapping)
        or not _unpadded_text(trigger.get("reference"))
        or not _finite_number(trigger.get("level"))
        or not _valid_text_sequence(trigger.get("limitations", []))
    ):
        return False
    return True


def _position_context_block_reason(
    context: object,
    *,
    expected_ticker: str,
) -> str | None:
    if not isinstance(context, Mapping):
        return "missing_approved_position_context"
    if (
        context.get("contract_version")
        != APPROVED_PORTFOLIO_CONTEXT_CONTRACT_VERSION
        or context.get("provenance_valid") is not True
        or not _unpadded_text(context.get("reference"))
        or context.get("ticker") != expected_ticker
    ):
        return "blocked_invalid_position_context"
    if context.get("fresh") is not True:
        return "blocked_stale_position_context"
    return None


def _condition_evidence_valid(
    condition_state: object,
    context: Mapping[str, Any],
) -> bool:
    required_mapping = {
        "pullback_preferred": "support_zone",
        "breakout_confirmation_required": "breakout_trigger",
        "acceptable_zone": "acceptable_zone",
    }.get(condition_state)
    if required_mapping is not None:
        condition = context.get(required_mapping)
        if not isinstance(condition, Mapping):
            return False
        if not _unpadded_text(condition.get("reference")):
            return False
        if required_mapping in {"support_zone", "acceptable_zone"}:
            lower = condition.get("lower_bound")
            upper = condition.get("upper_bound")
            return (
                _finite_number(lower)
                and _finite_number(upper)
                and float(lower) <= float(upper)
            )
        return _finite_number(condition.get("level"))
    if condition_state == "extended":
        return _unpadded_text(context.get("extension_reference"))
    return condition_state == "no_favorable_zone"


def _invalidation_payload(context: Mapping[str, Any]) -> Mapping[str, Any] | None:
    raw = context.get("invalidation_context")
    if (
        not isinstance(raw, Mapping)
        or raw.get("state") not in {"intact", "deteriorating", "invalidated"}
        or not _unpadded_text(raw.get("reference"))
        or not _unpadded_text(raw.get("reason"))
    ):
        return None
    level = raw.get("level")
    if level is not None and not _finite_number(level):
        return None
    return {
        "state": raw["state"],
        "evidence_reference": raw["reference"],
        "level": level,
        "reason": raw["reason"],
        "limitations": tuple(_text_values(raw.get("limitations"))),
        "stop_order_authorized": False,
    }


def _price_condition_payload(
    raw: object,
    *,
    condition_type: str,
) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping) or not _unpadded_text(raw.get("reference")):
        return {
            "state": "unavailable",
            "condition_type": condition_type,
            "evidence_reference": None,
            "lower_bound": None,
            "upper_bound": None,
            "level": None,
            "explanation": None,
            "limitations": (),
        }
    return {
        "state": "available",
        "condition_type": condition_type,
        "evidence_reference": raw["reference"],
        "lower_bound": raw.get("lower_bound"),
        "upper_bound": raw.get("upper_bound"),
        "level": raw.get("level"),
        "explanation": raw.get("explanation"),
        "limitations": tuple(_text_values(raw.get("limitations"))),
    }


def _approved_price_references(context: Mapping[str, Any]) -> tuple[str, ...]:
    candidates = [context.get("reference"), context.get("extension_reference")]
    for key in (
        "support_zone",
        "acceptable_zone",
        "breakout_trigger",
        "invalidation_context",
    ):
        value = context.get(key)
        if isinstance(value, Mapping):
            candidates.append(value.get("reference"))
    return tuple(sorted({item for item in candidates if _unpadded_text(item)}))


def _buy_ineligible(
    *,
    state: BuyZoneState,
    reason: str,
    limitations: tuple[str, ...] = (),
    conflict_references: tuple[str, ...] = (),
    invalidation_context: Mapping[str, Any] | None = None,
) -> BuyZoneExplanationResult:
    return BuyZoneExplanationResult(
        contract_version=GOVERNOR_EXPLANATION_CONTRACT_VERSION,
        eligibility_state=ExplanationEligibilityState.INELIGIBLE,
        state=state,
        reason_codes=(reason,),
        approved_price_references=(),
        pullback_condition=_price_condition_payload(
            None,
            condition_type="approved_support_zone",
        ),
        breakout_condition=_price_condition_payload(
            None,
            condition_type="approved_breakout_trigger",
        ),
        invalidation_context=(
            invalidation_context or _unavailable_invalidation()
        ),
        current_position_relative_to_zone="unavailable",
        conflict_references=conflict_references,
        limitations=limitations,
    )


def _position_ineligible(
    *,
    state: PositionManagementState,
    reason: str,
    recommendation_direction: str,
    supporting_scores: tuple[Mapping[str, Any], ...],
    invalidation: Mapping[str, Any],
    position_reference: str | None = None,
    conflict_references: tuple[str, ...] = (),
) -> PositionManagementExplanationResult:
    return PositionManagementExplanationResult(
        contract_version=GOVERNOR_EXPLANATION_CONTRACT_VERSION,
        eligibility_state=ExplanationEligibilityState.INELIGIBLE,
        state=state,
        reason_codes=(reason,),
        position_context_reference=position_reference,
        supporting_recommendation_state=recommendation_direction,
        supporting_factor_scores=supporting_scores,
        invalidation_context=invalidation,
        conflict_references=conflict_references,
        limitations=(),
    )


def _unavailable_invalidation() -> Mapping[str, Any]:
    return {
        "state": "unavailable",
        "evidence_reference": None,
        "level": None,
        "reason": None,
        "limitations": (),
        "stop_order_authorized": False,
    }


def _factor_mapping(factors: object) -> dict[str, object]:
    if not isinstance(factors, Sequence) or isinstance(factors, (str, bytes)):
        return {}
    result: dict[str, object] = {}
    for factor in factors:
        name = _value(factor, "factor")
        if not name or name in result:
            return {}
        result[name] = factor
    return result


def _factor_state(factor: object) -> str:
    return _value(factor, "state")


def _factor_score(factor: object) -> float | None:
    raw = _raw_value(factor, "score")
    if (
        isinstance(raw, bool)
        or not isinstance(raw, (int, float))
        or not isfinite(raw)
        or not 0.0 <= float(raw) <= 100.0
    ):
        return None
    return float(raw)


def _supporting_scores(
    recommendation: object,
) -> tuple[Mapping[str, Any], ...]:
    raw = _raw_value(recommendation, "supporting_factor_scores")
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(item for item in raw if isinstance(item, Mapping))


def _valuation_limitations(
    factors: Mapping[str, object],
) -> tuple[str, ...]:
    valuation = factors.get("valuation")
    if _factor_score(valuation) is None:
        return ("valuation_score_unavailable_no_target_inference",)
    return ()


def _raw_value(value: object, key: str) -> object:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _value(value: object, key: str) -> str:
    raw = _raw_value(value, key)
    enum_value = getattr(raw, "value", raw)
    return enum_value if isinstance(enum_value, str) else ""


def _text_references(value: object) -> tuple[str, ...]:
    return tuple(sorted(set(_text_values(value))))


def _text_values(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if _unpadded_text(item))


def _valid_text_sequence(value: object) -> bool:
    return isinstance(value, (list, tuple)) and all(
        _unpadded_text(item) for item in value
    )


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and isfinite(value)
    )


def _unpadded_text(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()
