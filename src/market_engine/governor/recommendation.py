from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any

from market_engine.governor.scoring import (
    GOVERNOR_FACTOR_SCORING_CONTRACT_VERSION,
)


GOVERNOR_RECOMMENDATION_STATE_CONTRACT_VERSION = (
    "market-engine-governor-recommendation-state-v1"
)
APPROVED_RECOMMENDATION_REVIEW_CONTRACT_VERSION = (
    "sec-companyfacts-recommendation-review-v1"
)
REQUIRED_EVALUATION_STATE = "evaluation_completed_non_actionable"
DATA_CONFIDENCE_ELIGIBILITY_THRESHOLD = 75.0


class RecommendationEligibilityState(StrEnum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"


class RecommendationState(StrEnum):
    BLOCKED = "blocked"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    AVOID = "avoid"
    WATCH = "watch"
    CONSIDER = "consider"
    PREFERRED = "preferred"


@dataclass(frozen=True)
class RecommendationResult:
    contract_version: str
    eligibility_state: RecommendationEligibilityState
    state: RecommendationState
    reason_codes: tuple[str, ...]
    supporting_factor_scores: tuple[Mapping[str, Any], ...]
    supporting_factor_states: tuple[Mapping[str, str], ...]
    blocking_factors: tuple[str, ...]
    conflict_references: tuple[str, ...]
    limitations: tuple[str, ...]
    actionable: bool = False
    recommendation_state_ready: bool = False
    decision_engine_ready: bool = False


_CRITICAL_FACTORS = (
    "fundamentals",
    "growth",
    "risk",
    "data_confidence",
)
_PREFERRED_THRESHOLDS = {
    "fundamentals": 75.0,
    "growth": 70.0,
    "risk": 70.0,
    "data_confidence": 85.0,
}
_CONSIDER_THRESHOLDS = {
    "fundamentals": 60.0,
    "growth": 55.0,
    "risk": 60.0,
    "data_confidence": 80.0,
}
_UNFAVORABLE_THRESHOLDS = {
    "fundamentals": 40.0,
    "growth": 40.0,
    "risk": 40.0,
}


def map_recommendation_state(
    *,
    governor_contract_version: object,
    evaluation_state: object,
    factor_evaluations: Sequence[object],
    recommendation_review_boundary: object,
) -> RecommendationResult:
    """Map eligible Governor factor evidence without action authority."""
    factor_by_name = _validated_factor_mapping(factor_evaluations)
    supporting_states = _supporting_states(factor_by_name)
    supporting_scores = _supporting_scores(factor_by_name)

    if governor_contract_version != (
        "market-engine-governor-investment-evaluation-v1"
    ):
        return _ineligible(
            state=RecommendationState.BLOCKED,
            reason="blocked_invalid_governor_contract",
            factor_by_name=factor_by_name,
            supporting_states=supporting_states,
            supporting_scores=supporting_scores,
        )
    if evaluation_state == "blocked":
        return _ineligible(
            state=RecommendationState.BLOCKED,
            reason="blocked_governor_evaluation",
            factor_by_name=factor_by_name,
            supporting_states=supporting_states,
            supporting_scores=supporting_scores,
        )

    boundary_reason = _recommendation_review_boundary_reason(
        recommendation_review_boundary
    )
    if boundary_reason is not None:
        boundary_state = (
            RecommendationState.INSUFFICIENT_EVIDENCE
            if boundary_reason == "ineligible_recommendation_review_state"
            else RecommendationState.BLOCKED
        )
        return _ineligible(
            state=boundary_state,
            reason=boundary_reason,
            factor_by_name=factor_by_name,
            supporting_states=supporting_states,
            supporting_scores=supporting_scores,
        )

    hard_conflicts = _hard_conflict_references(factor_by_name)
    if hard_conflicts:
        return _ineligible(
            state=RecommendationState.BLOCKED,
            reason="blocked_unresolved_hard_conflict",
            factor_by_name=factor_by_name,
            supporting_states=supporting_states,
            supporting_scores=supporting_scores,
            conflict_references=hard_conflicts,
        )
    if evaluation_state != REQUIRED_EVALUATION_STATE:
        return _ineligible(
            state=RecommendationState.INSUFFICIENT_EVIDENCE,
            reason="ineligible_governor_evaluation_incomplete",
            factor_by_name=factor_by_name,
            supporting_states=supporting_states,
            supporting_scores=supporting_scores,
        )

    critical_blockers = _critical_factor_blockers(factor_by_name)
    if critical_blockers:
        return _ineligible(
            state=RecommendationState.INSUFFICIENT_EVIDENCE,
            reason="ineligible_critical_factor_coverage",
            factor_by_name=factor_by_name,
            supporting_states=supporting_states,
            supporting_scores=supporting_scores,
            blocking_factors=critical_blockers,
        )

    score_limitation_blockers = _critical_score_limitation_blockers(
        factor_by_name
    )
    if score_limitation_blockers:
        return _ineligible(
            state=RecommendationState.INSUFFICIENT_EVIDENCE,
            reason="ineligible_critical_score_limitations",
            factor_by_name=factor_by_name,
            supporting_states=supporting_states,
            supporting_scores=supporting_scores,
            blocking_factors=score_limitation_blockers,
        )

    scores = {
        factor: float(getattr(factor_by_name[factor], "score"))
        for factor in _CRITICAL_FACTORS
    }
    if scores["data_confidence"] < DATA_CONFIDENCE_ELIGIBILITY_THRESHOLD:
        return _ineligible(
            state=RecommendationState.INSUFFICIENT_EVIDENCE,
            reason="ineligible_data_confidence_below_threshold",
            factor_by_name=factor_by_name,
            supporting_states=supporting_states,
            supporting_scores=supporting_scores,
            blocking_factors=("data_confidence",),
        )

    soft_conflicts = _soft_conflict_references(factor_by_name)
    state, mapping_reason = _directional_state(scores)
    limitations = _portfolio_limitations(factor_by_name)
    reason_codes = ["recommendation_eligible", mapping_reason]
    if soft_conflicts and state in {
        RecommendationState.CONSIDER,
        RecommendationState.PREFERRED,
    }:
        state = RecommendationState.WATCH
        reason_codes.append("recommendation_limited_by_soft_conflict")
        limitations = (
            *limitations,
            "soft_conflicting_evidence_requires_cautious_interpretation",
        )
    if (
        scores["risk"] < _CONSIDER_THRESHOLDS["risk"]
        and state is RecommendationState.WATCH
    ):
        reason_codes.append("recommendation_limited_by_risk_guardrail")

    return RecommendationResult(
        contract_version=GOVERNOR_RECOMMENDATION_STATE_CONTRACT_VERSION,
        eligibility_state=RecommendationEligibilityState.ELIGIBLE,
        state=state,
        reason_codes=tuple(reason_codes),
        supporting_factor_scores=supporting_scores,
        supporting_factor_states=supporting_states,
        blocking_factors=(),
        conflict_references=soft_conflicts,
        limitations=tuple(sorted(set(limitations))),
    )


def _directional_state(
    scores: Mapping[str, float],
) -> tuple[RecommendationState, str]:
    if any(
        scores[factor] < threshold
        for factor, threshold in _UNFAVORABLE_THRESHOLDS.items()
    ):
        return (
            RecommendationState.AVOID,
            "mapped_unfavorable_critical_factor_pattern",
        )
    if all(
        scores[factor] >= threshold
        for factor, threshold in _PREFERRED_THRESHOLDS.items()
    ):
        return (
            RecommendationState.PREFERRED,
            "mapped_favorable_critical_factor_pattern",
        )
    if all(
        scores[factor] >= threshold
        for factor, threshold in _CONSIDER_THRESHOLDS.items()
    ):
        return (
            RecommendationState.CONSIDER,
            "mapped_moderately_favorable_critical_factor_pattern",
        )
    return RecommendationState.WATCH, "mapped_mixed_critical_factor_pattern"


def _validated_factor_mapping(
    factor_evaluations: object,
) -> dict[str, object]:
    if not isinstance(factor_evaluations, Sequence) or isinstance(
        factor_evaluations,
        (str, bytes),
    ):
        return {}
    result: dict[str, object] = {}
    for factor in factor_evaluations:
        name = getattr(getattr(factor, "factor", None), "value", None)
        if not isinstance(name, str) or name in result:
            return {}
        result[name] = factor
    return result


def _recommendation_review_boundary_reason(boundary: object) -> str | None:
    if not isinstance(boundary, Mapping):
        return "blocked_recommendation_review_boundary_missing"
    if (
        boundary.get("contract_version")
        != APPROVED_RECOMMENDATION_REVIEW_CONTRACT_VERSION
        or boundary.get("non_actionable") is not True
        or not _unpadded_text(boundary.get("reference"))
    ):
        return "blocked_recommendation_review_boundary_invalid"
    if boundary.get("review_state") != "human_review_required":
        return "ineligible_recommendation_review_state"
    return None


def _critical_factor_blockers(
    factor_by_name: Mapping[str, object],
) -> tuple[str, ...]:
    blockers: list[str] = []
    for factor_name in _CRITICAL_FACTORS:
        factor = factor_by_name.get(factor_name)
        state = getattr(getattr(factor, "state", None), "value", None)
        score = getattr(factor, "score", None)
        score_scale = getattr(factor, "score_scale", None)
        if (
            state != "evaluable"
            or not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not isfinite(score)
            or not 0.0 <= float(score) <= 100.0
            or not isinstance(score_scale, Mapping)
            or score_scale.get("contract_version")
            != GOVERNOR_FACTOR_SCORING_CONTRACT_VERSION
        ):
            blockers.append(factor_name)
    return tuple(blockers)


def _supporting_states(
    factor_by_name: Mapping[str, object],
) -> tuple[Mapping[str, str], ...]:
    return tuple(
        {
            "factor": name,
            "state": str(getattr(getattr(factor, "state", None), "value", "")),
        }
        for name, factor in factor_by_name.items()
    )


def _supporting_scores(
    factor_by_name: Mapping[str, object],
) -> tuple[Mapping[str, Any], ...]:
    result: list[Mapping[str, Any]] = []
    for name, factor in factor_by_name.items():
        score = getattr(factor, "score", None)
        if score is None:
            continue
        scale = getattr(factor, "score_scale", None)
        result.append(
            {
                "factor": name,
                "score": score,
                "score_contract_version": (
                    scale.get("contract_version")
                    if isinstance(scale, Mapping)
                    else None
                ),
            }
        )
    return tuple(result)


def _critical_score_limitation_blockers(
    factor_by_name: Mapping[str, object],
) -> tuple[str, ...]:
    blockers: list[str] = []
    for factor_name in _CRITICAL_FACTORS:
        limitations = getattr(
            factor_by_name.get(factor_name),
            "score_limitations",
            (),
        )
        if (
            not isinstance(limitations, (list, tuple))
            or any(not isinstance(item, str) for item in limitations)
            or limitations
        ):
            blockers.append(factor_name)
    return tuple(blockers)


def _hard_conflict_references(
    factor_by_name: Mapping[str, object],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                reference
                for factor in factor_by_name.values()
                for reference in getattr(
                    factor,
                    "conflicting_evidence_references",
                    (),
                )
            }
        )
    )


def _soft_conflict_references(
    factor_by_name: Mapping[str, object],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                reference
                for factor in factor_by_name.values()
                for reference in getattr(
                    factor,
                    "soft_conflicting_evidence_references",
                    (),
                )
            }
        )
    )


def _portfolio_limitations(
    factor_by_name: Mapping[str, object],
) -> tuple[str, ...]:
    portfolio_fit = factor_by_name.get("portfolio_fit")
    if (
        getattr(getattr(portfolio_fit, "state", None), "value", None)
        == "blocked"
        and "blocked_missing_approved_portfolio_context"
        in getattr(portfolio_fit, "blocked_reasons", ())
    ):
        return ("portfolio_fit_not_used_without_approved_context",)
    return ()


def _recommendation_limitations(
    factor_by_name: Mapping[str, object],
) -> tuple[str, ...]:
    limitations = list(_portfolio_limitations(factor_by_name))
    for factor_name in _CRITICAL_FACTORS:
        factor_limitations = getattr(
            factor_by_name.get(factor_name),
            "score_limitations",
            (),
        )
        if not isinstance(factor_limitations, (list, tuple)):
            limitations.append(f"{factor_name}:score_limitations_malformed")
            continue
        limitations.extend(
            f"{factor_name}:{limitation}"
            for limitation in factor_limitations
            if isinstance(limitation, str)
        )
    return tuple(sorted(set(limitations)))


def _ineligible(
    *,
    state: RecommendationState,
    reason: str,
    factor_by_name: Mapping[str, object],
    supporting_states: tuple[Mapping[str, str], ...],
    supporting_scores: tuple[Mapping[str, Any], ...],
    blocking_factors: tuple[str, ...] = (),
    conflict_references: tuple[str, ...] = (),
) -> RecommendationResult:
    limitations = _recommendation_limitations(factor_by_name)
    return RecommendationResult(
        contract_version=GOVERNOR_RECOMMENDATION_STATE_CONTRACT_VERSION,
        eligibility_state=RecommendationEligibilityState.INELIGIBLE,
        state=state,
        reason_codes=(reason,),
        supporting_factor_scores=supporting_scores,
        supporting_factor_states=supporting_states,
        blocking_factors=blocking_factors,
        conflict_references=conflict_references,
        limitations=limitations,
    )


def _unpadded_text(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()
