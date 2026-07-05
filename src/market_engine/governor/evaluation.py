from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


GOVERNOR_INVESTMENT_EVALUATION_CONTRACT_VERSION = (
    "market-engine-governor-investment-evaluation-v1"
)
GOVERNOR_FACTOR_TAXONOMY_VERSION = (
    "market-engine-governor-factor-taxonomy-v1"
)
APPROVED_GOVERNOR_EVIDENCE_CONTRACT_VERSION = (
    "market-engine-governor-approved-evidence-v1"
)


class GovernorEvaluationError(ValueError):
    """Raised when Governor evidence cannot be evaluated deterministically."""


class FactorFamily(StrEnum):
    FUNDAMENTALS = "fundamentals"
    GROWTH = "growth"
    VALUATION = "valuation"
    TREND = "trend"
    MOMENTUM = "momentum"
    RISK = "risk"
    TECHNICAL_SETUP = "technical_setup"
    PORTFOLIO_FIT = "portfolio_fit"
    DATA_CONFIDENCE = "data_confidence"


class FactorState(StrEnum):
    NOT_STARTED = "not_started"
    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PARTIAL = "partial"
    QUALITATIVE_ONLY = "qualitative_only"
    EVALUABLE = "evaluable"


class EvaluationState(StrEnum):
    NOT_STARTED = "not_started"
    BLOCKED = "blocked"
    DESCRIPTIVE_ONLY = "descriptive_only"
    PARTIAL_EVALUATION = "partial_evaluation"
    EVALUATION_READY = "evaluation_ready"
    EVALUATION_COMPLETED_NON_ACTIONABLE = (
        "evaluation_completed_non_actionable"
    )


class EvidenceLevel(StrEnum):
    NONE = "none"
    DESCRIPTIVE = "descriptive"
    LIMITED = "limited"
    COMPLETE = "complete"


@dataclass(frozen=True)
class FactorEvaluation:
    factor: FactorFamily
    state: FactorState
    evidence_references: tuple[str, ...]
    evidence_requirements: Mapping[str, str]
    missing_evidence: tuple[str, ...]
    blocked_reasons: tuple[str, ...]
    limitations: tuple[str, ...]
    conflicting_evidence_references: tuple[str, ...]
    qualitative_summary: str
    score: None = None
    score_scale: None = None
    weight: None = None
    weighted_score: None = None
    provenance: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class GovernorEvaluation:
    contract_version: str
    evaluation_id: str
    ticker: str
    market: str
    company_name: str
    input_references: Mapping[str, str]
    evidence_readiness: Mapping[str, Any]
    evaluation_state: EvaluationState
    factor_evaluations: tuple[FactorEvaluation, ...]
    overall_evaluation: Mapping[str, Any]
    recommendation_state: Mapping[str, Any]
    buy_zone_explanation: Mapping[str, Any]
    position_management_explanation: Mapping[str, Any]
    risk_and_limitations: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    blocked_reasons: tuple[str, ...]
    authority_boundary: Mapping[str, Any]
    provenance: Mapping[str, Any]


_FACTOR_ORDER = tuple(FactorFamily)
_CORE_FACTORS = tuple(
    factor
    for factor in _FACTOR_ORDER
    if factor not in {
        FactorFamily.PORTFOLIO_FIT,
        FactorFamily.DATA_CONFIDENCE,
    }
)
_HARD_BLOCKERS = frozenset(
    {
        "evidence_contract_not_approved",
        "invalid_manifest",
        "missing_provenance",
        "evidence_not_consumable",
        "malformed_evidence",
    }
)
_DEFAULT_GATES = (
    "manifest_valid",
    "provenance_valid",
    "fresh",
    "consumable",
    "structurally_valid",
)
_EVIDENCE_REQUIREMENTS: dict[FactorFamily, Mapping[str, str]] = {
    FactorFamily.FUNDAMENTALS: {
        "partial": "one approved fundamental evidence family",
        "evaluable": "multiple approved financial dimensions",
    },
    FactorFamily.GROWTH: {
        "partial": "one valid growth series",
        "evaluable": "multiple aligned growth dimensions",
    },
    FactorFamily.VALUATION: {
        "partial": "complete supporting inputs are required",
        "evaluable": "approved valuation evidence with complete inputs",
    },
    FactorFamily.TREND: {
        "partial": "limited approved trend context",
        "evaluable": "complete approved trend context",
    },
    FactorFamily.MOMENTUM: {
        "partial": "one approved momentum evidence item",
        "evaluable": "complete approved momentum context",
    },
    FactorFamily.RISK: {
        "partial": "validated explicit risk evidence with gaps",
        "evaluable": "broad approved risk evidence",
    },
    FactorFamily.TECHNICAL_SETUP: {
        "partial": "detected setup with incomplete confirmation",
        "evaluable": "complete approved setup evidence",
    },
    FactorFamily.PORTFOLIO_FIT: {
        "partial": "approved portfolio context",
        "evaluable": "approved complete portfolio context",
    },
    FactorFamily.DATA_CONFIDENCE: {
        "partial": "inspectable but incomplete evidence trust context",
        "evaluable": "fresh, provenanced, consumable, complete evidence",
    },
}


def evaluate_governor_evidence(
    evidence: Mapping[str, Any],
    *,
    evaluation_timestamp: str,
    input_reference: str,
) -> GovernorEvaluation:
    payload = _validated_evidence(evidence)
    gates = _validated_gates(payload.get("evidence_readiness"))
    evidence_contract_version = _required_text(
        payload,
        "evidence_contract_version",
    )
    contract_approved = (
        evidence_contract_version
        == APPROVED_GOVERNOR_EVIDENCE_CONTRACT_VERSION
    )
    factor_payloads = _validated_factor_payloads(payload.get("factor_evidence"))
    portfolio_context_approved = _required_bool(
        payload,
        "approved_portfolio_context",
    )
    factor_evaluations = tuple(
        _evaluate_factor(
            factor=factor,
            raw=factor_payloads.get(factor, {}),
            global_gates=gates,
            contract_approved=contract_approved,
            portfolio_context_approved=portfolio_context_approved,
        )
        for factor in _FACTOR_ORDER
    )
    evaluation_state = _evaluation_state(
        factor_evaluations=factor_evaluations,
        global_gates=gates,
        contract_approved=contract_approved,
    )
    missing_evidence = _unique_sorted(
        item
        for factor in factor_evaluations
        for item in factor.missing_evidence
    )
    blocked_reasons = _unique_sorted(
        item
        for factor in factor_evaluations
        for item in factor.blocked_reasons
    )
    risk_and_limitations = _unique_sorted(
        item
        for factor in factor_evaluations
        for item in factor.limitations
    )
    state_counts = Counter(item.state.value for item in factor_evaluations)
    return GovernorEvaluation(
        contract_version=GOVERNOR_INVESTMENT_EVALUATION_CONTRACT_VERSION,
        evaluation_id=_required_text(payload, "evaluation_id"),
        ticker=_required_text(payload, "ticker"),
        market=_required_text(payload, "market"),
        company_name=_required_text(payload, "company_name"),
        input_references=_validated_text_mapping(
            payload.get("input_references"),
            "input_references",
        ),
        evidence_readiness={
            "evidence_contract_version": evidence_contract_version,
            "contract_approved": contract_approved,
            **gates,
            "factor_state_counts": dict(sorted(state_counts.items())),
        },
        evaluation_state=evaluation_state,
        factor_evaluations=factor_evaluations,
        overall_evaluation={
            "state": evaluation_state.value,
            "score": None,
            "score_scale": None,
            "weighted_score": None,
            "rank": None,
            "summary": (
                "Evidence sufficiency classification only; investment quality "
                "and scoring are not evaluated in ME-GV03."
            ),
        },
        recommendation_state={
            "state": "blocked_not_authorized",
            "reason": (
                "governor_recommendation_state_mapping_not_implemented_or_"
                "not_authorized"
            ),
            "actionable": False,
            "recommendation_state_ready": False,
            "decision_engine_ready": False,
        },
        buy_zone_explanation={
            "state": "blocked_not_authorized",
            "reason": "governor_buy_zone_not_implemented_or_not_authorized",
        },
        position_management_explanation={
            "state": "blocked_not_authorized",
            "reason": (
                "governor_position_management_not_implemented_or_not_authorized"
            ),
        },
        risk_and_limitations=risk_and_limitations,
        missing_evidence=missing_evidence,
        blocked_reasons=blocked_reasons,
        authority_boundary={
            "non_actionable": True,
            "scoring_authorized": False,
            "recommendation_mapping_authorized": False,
            "buy_zone_authorized": False,
            "position_management_authorized": False,
            "actionable": False,
            "actionable_review": False,
            "recommendation_state_ready": False,
            "decision_ready": False,
            "de_ready": False,
            "decision_engine_ready": False,
        },
        provenance={
            "input_reference": _required_unpadded_text(
                input_reference,
                "input_reference",
            ),
            "evaluation_timestamp": _required_unpadded_text(
                evaluation_timestamp,
                "evaluation_timestamp",
            ),
            "factor_taxonomy_version": GOVERNOR_FACTOR_TAXONOMY_VERSION,
            "deterministic": True,
        },
    )


def to_plain_dict(evaluation: GovernorEvaluation) -> dict[str, Any]:
    return asdict(evaluation)


def _evaluate_factor(
    *,
    factor: FactorFamily,
    raw: Mapping[str, Any],
    global_gates: Mapping[str, bool],
    contract_approved: bool,
    portfolio_context_approved: bool,
) -> FactorEvaluation:
    level = _evidence_level(raw)
    references = _text_sequence(raw.get("evidence_references", ()), "references")
    missing = list(_text_sequence(raw.get("missing_evidence", ()), "missing"))
    limitations = list(_text_sequence(raw.get("limitations", ()), "limitations"))
    conflicts = _text_sequence(
        raw.get("conflicting_evidence_references", ()),
        "conflicting evidence references",
    )
    gates = {
        name: _optional_bool(raw, name, global_gates[name])
        for name in _DEFAULT_GATES
    }
    blocked_reasons = _gate_blockers(
        contract_approved=contract_approved,
        gates=gates,
        freshness_blocks=factor is not FactorFamily.DATA_CONFIDENCE,
    )

    if factor is FactorFamily.PORTFOLIO_FIT and not portfolio_context_approved:
        blocked_reasons.append("blocked_missing_approved_portfolio_context")
        state = FactorState.BLOCKED
        missing.append("approved_portfolio_context")
    elif blocked_reasons:
        state = FactorState.BLOCKED
    elif level is EvidenceLevel.NONE:
        state = FactorState.UNAVAILABLE
        missing.append(f"{factor.value}_evidence")
    else:
        state = _state_for_level(factor=factor, level=level)

    if (
        factor is FactorFamily.DATA_CONFIDENCE
        and not gates["fresh"]
        and state not in {FactorState.BLOCKED, FactorState.UNAVAILABLE}
    ):
        state = FactorState.PARTIAL
        limitations.append("stale_evidence_remains_inspectable")

    if conflicts:
        limitations.append("conflicting_evidence_preserved_without_averaging")
        if state is FactorState.EVALUABLE:
            state = FactorState.PARTIAL

    if level is not EvidenceLevel.NONE and not references:
        state = FactorState.BLOCKED
        blocked_reasons.append("missing_deterministic_evidence_reference")

    return FactorEvaluation(
        factor=factor,
        state=state,
        evidence_references=references,
        evidence_requirements=_EVIDENCE_REQUIREMENTS[factor],
        missing_evidence=_unique_sorted(missing),
        blocked_reasons=_unique_sorted(blocked_reasons),
        limitations=_unique_sorted(limitations),
        conflicting_evidence_references=conflicts,
        qualitative_summary=_factor_summary(state),
        provenance={
            **gates,
            "contract_approved": contract_approved,
            "evidence_level": level.value,
        },
    )


def _state_for_level(
    *,
    factor: FactorFamily,
    level: EvidenceLevel,
) -> FactorState:
    if level is EvidenceLevel.DESCRIPTIVE:
        return FactorState.QUALITATIVE_ONLY
    if level is EvidenceLevel.COMPLETE:
        return FactorState.EVALUABLE
    if factor is FactorFamily.VALUATION:
        return FactorState.INSUFFICIENT_EVIDENCE
    return FactorState.PARTIAL


def _evaluation_state(
    *,
    factor_evaluations: Sequence[FactorEvaluation],
    global_gates: Mapping[str, bool],
    contract_approved: bool,
) -> EvaluationState:
    all_blockers = {
        reason
        for factor in factor_evaluations
        for reason in factor.blocked_reasons
    }
    if not contract_approved or not all(global_gates.values()):
        return EvaluationState.BLOCKED
    if all_blockers & _HARD_BLOCKERS:
        return EvaluationState.BLOCKED
    by_factor = {item.factor: item.state for item in factor_evaluations}
    core_states = tuple(by_factor[factor] for factor in _CORE_FACTORS)
    if (
        all(state is FactorState.EVALUABLE for state in core_states)
        and by_factor[FactorFamily.DATA_CONFIDENCE] is FactorState.EVALUABLE
    ):
        return EvaluationState.EVALUATION_COMPLETED_NON_ACTIONABLE
    if any(
        state in {
            FactorState.PARTIAL,
            FactorState.INSUFFICIENT_EVIDENCE,
            FactorState.EVALUABLE,
        }
        for state in core_states
    ):
        return EvaluationState.PARTIAL_EVALUATION
    if any(state is FactorState.QUALITATIVE_ONLY for state in core_states):
        return EvaluationState.DESCRIPTIVE_ONLY
    return EvaluationState.BLOCKED


def _gate_blockers(
    *,
    contract_approved: bool,
    gates: Mapping[str, bool],
    freshness_blocks: bool,
) -> list[str]:
    blockers: list[str] = []
    if not contract_approved:
        blockers.append("evidence_contract_not_approved")
    if not gates["manifest_valid"]:
        blockers.append("invalid_manifest")
    if not gates["provenance_valid"]:
        blockers.append("missing_provenance")
    if freshness_blocks and not gates["fresh"]:
        blockers.append("stale_evidence")
    if not gates["consumable"]:
        blockers.append("evidence_not_consumable")
    if not gates["structurally_valid"]:
        blockers.append("malformed_evidence")
    return blockers


def _factor_summary(state: FactorState) -> str:
    messages = {
        FactorState.NOT_STARTED: "Factor evidence evaluation was not started.",
        FactorState.BLOCKED: "Evidence gates block factor evaluation.",
        FactorState.UNAVAILABLE: "No approved factor evidence is available.",
        FactorState.INSUFFICIENT_EVIDENCE: (
            "Approved evidence exists but does not satisfy minimum requirements."
        ),
        FactorState.PARTIAL: (
            "Approved evidence supports a limited non-actionable classification."
        ),
        FactorState.QUALITATIVE_ONLY: (
            "Approved descriptive evidence supports qualitative context only."
        ),
        FactorState.EVALUABLE: (
            "Evidence gates are sufficient for future evaluation; no score "
            "or investment-quality conclusion is produced."
        ),
    }
    return messages[state]


def _validated_evidence(evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(evidence, Mapping):
        raise GovernorEvaluationError("Governor evidence must be a mapping")
    return evidence


def _validated_gates(value: object) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        raise GovernorEvaluationError("evidence_readiness must be a mapping")
    return {
        name: _required_bool(value, name)
        for name in _DEFAULT_GATES
    }


def _validated_factor_payloads(
    value: object,
) -> dict[FactorFamily, Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise GovernorEvaluationError("factor_evidence must be a mapping")
    result: dict[FactorFamily, Mapping[str, Any]] = {}
    for raw_factor, raw_payload in value.items():
        try:
            factor = FactorFamily(raw_factor)
        except (TypeError, ValueError) as exc:
            raise GovernorEvaluationError(
                f"unsupported Governor factor family: {raw_factor}"
            ) from exc
        if not isinstance(raw_payload, Mapping):
            raise GovernorEvaluationError(
                f"factor evidence for {factor.value} must be a mapping"
            )
        result[factor] = raw_payload
    return result


def _evidence_level(raw: Mapping[str, Any]) -> EvidenceLevel:
    value = raw.get("level", EvidenceLevel.NONE.value)
    try:
        return EvidenceLevel(value)
    except (TypeError, ValueError) as exc:
        raise GovernorEvaluationError(
            f"unsupported Governor evidence level: {value}"
        ) from exc


def _required_text(mapping: Mapping[str, Any], field_name: str) -> str:
    return _required_unpadded_text(mapping.get(field_name), field_name)


def _required_unpadded_text(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
    ):
        raise GovernorEvaluationError(
            f"{field_name} must be non-empty text without padding"
        )
    return value


def _required_bool(mapping: Mapping[str, Any], field_name: str) -> bool:
    value = mapping.get(field_name)
    if not isinstance(value, bool):
        raise GovernorEvaluationError(f"{field_name} must be a boolean")
    return value


def _optional_bool(
    mapping: Mapping[str, Any],
    field_name: str,
    default: bool,
) -> bool:
    value = mapping.get(field_name, default)
    if not isinstance(value, bool):
        raise GovernorEvaluationError(f"{field_name} must be a boolean")
    return value


def _text_sequence(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item and item == item.strip()
        for item in value
    ):
        raise GovernorEvaluationError(
            f"{field_name} must contain unpadded text values"
        )
    return tuple(value)


def _validated_text_mapping(
    value: object,
    field_name: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise GovernorEvaluationError(f"{field_name} must be a non-empty mapping")
    result: dict[str, str] = {}
    for key, item in value.items():
        result[
            _required_unpadded_text(key, f"{field_name} key")
        ] = _required_unpadded_text(item, f"{field_name} value")
    return dict(sorted(result.items()))


def _unique_sorted(values: Sequence[str] | Any) -> tuple[str, ...]:
    return tuple(sorted(set(values)))
