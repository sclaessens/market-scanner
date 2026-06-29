from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


ANALYSIS_CONTEXT_READINESS_FORMAT_VERSION = (
    "market-engine-analysis-context-readiness-v1"
)
ANALYSIS_CONTEXT_READINESS_BOUNDARY = (
    "Analysis-context readiness classifies evidence completeness only and does "
    "not create recommendation, action, allocation, execution, delivery, or "
    "Decision Engine authority."
)

COMPANY_PROFILE_ONLY_CONTEXT_NON_ACTIONABLE = (
    "company_profile_only_context_non_actionable"
)
INSUFFICIENT_ANALYSIS_CONTEXT = "insufficient_analysis_context"
MISSING_FUNDAMENTAL_EVIDENCE = "missing_fundamental_evidence"
MISSING_SETUP_OR_PRICE_CONTEXT = "missing_setup_or_price_context"
STALE_OR_UNPROVENANCED_ANALYSIS_CONTEXT = (
    "stale_or_unprovenanced_analysis_context"
)


class AnalysisContextReadinessLevel(str, Enum):
    DESCRIPTIVE_ONLY = "descriptive_only"
    PARTIAL_ANALYSIS = "partial_analysis"
    RECOMMENDATION_ELIGIBLE = "recommendation_eligible"
    ACTIONABLE_REVIEW = "actionable_review"
    DECISION_READY = "decision_ready"


class AnalysisContextEvidenceFamily(str, Enum):
    COMPANY_PROFILE = "company_profile"
    FUNDAMENTALS = "fundamentals"
    VALUATION = "valuation"
    SETUP_PRICE_MARKET = "setup_price_market"
    PORTFOLIO_CONTEXT = "portfolio_context"
    PROVENANCE_MANIFEST_STALENESS = "provenance_manifest_staleness"
    DELIVERY_REPORTING_HANDOFF = "delivery_reporting_handoff"


_EVIDENCE_FAMILY_ORDER = tuple(AnalysisContextEvidenceFamily)
_PROHIBITED_INFERENCES = (
    "company_profile_as_analytical_evidence",
    "missing_values_as_numeric_zero",
    "source_family_presence_as_recommendation",
    "readiness_as_action_or_allocation_authority",
    "handoff_readiness_as_decision",
    "delivery_context_as_upstream_evidence",
)
_SAFETY_NOTES = (
    "Recommendation eligibility permits evaluation only; it is not a recommendation.",
    "Actionable review and decision ready are reserved under current governance.",
    "Decision Engine remains the only allocation authority.",
)


@dataclass(frozen=True)
class AnalysisContextReadinessResult:
    readiness_level: AnalysisContextReadinessLevel
    evidence_families_present: tuple[AnalysisContextEvidenceFamily, ...]
    evidence_families_missing: tuple[AnalysisContextEvidenceFamily, ...]
    blocked_reasons: tuple[str, ...]
    recommendation_review_eligible: bool
    actionable_review_allowed: bool
    decision_engine_ready: bool
    provenance_valid: bool
    context_stale: bool
    unknown_evidence_families: tuple[str, ...] = field(default_factory=tuple)
    input_notes: tuple[str, ...] = field(default_factory=tuple)
    prohibited_inferences: tuple[str, ...] = _PROHIBITED_INFERENCES
    safety_notes: tuple[str, ...] = _SAFETY_NOTES
    readiness_format_version: str = ANALYSIS_CONTEXT_READINESS_FORMAT_VERSION
    non_authority_boundary: str = ANALYSIS_CONTEXT_READINESS_BOUNDARY

    def to_payload(self) -> dict[str, Any]:
        return {
            "readiness_format_version": self.readiness_format_version,
            "readiness_level": self.readiness_level.value,
            "evidence_families_present": [
                family.value for family in self.evidence_families_present
            ],
            "evidence_families_missing": [
                family.value for family in self.evidence_families_missing
            ],
            "blocked_reasons": list(self.blocked_reasons),
            "recommendation_review_eligible": (
                self.recommendation_review_eligible
            ),
            "actionable_review_allowed": self.actionable_review_allowed,
            "decision_engine_ready": self.decision_engine_ready,
            "provenance_valid": self.provenance_valid,
            "context_stale": self.context_stale,
            "unknown_evidence_families": list(self.unknown_evidence_families),
            "input_notes": list(self.input_notes),
            "prohibited_inferences": list(self.prohibited_inferences),
            "safety_notes": list(self.safety_notes),
            "non_authority_boundary": self.non_authority_boundary,
        }


def classify_analysis_context_readiness(
    evidence_families: Iterable[AnalysisContextEvidenceFamily] | None,
    *,
    provenance_valid: bool = False,
    context_stale: bool = False,
    valuation_required: bool = False,
) -> AnalysisContextReadinessResult:
    (
        present,
        unknown,
        evidence_input_malformed,
    ) = _normalize_evidence_families(evidence_families)
    flags_malformed = not all(
        isinstance(value, bool)
        for value in (provenance_valid, context_stale, valuation_required)
    )
    input_malformed = evidence_input_malformed or flags_malformed
    safe_provenance_valid = (
        provenance_valid if isinstance(provenance_valid, bool) else False
    )
    safe_context_stale = context_stale if isinstance(context_stale, bool) else True
    safe_valuation_required = (
        valuation_required if isinstance(valuation_required, bool) else True
    )

    missing = _missing_evidence_families(
        present,
        valuation_required=safe_valuation_required,
    )
    input_notes = _input_notes(
        input_malformed=input_malformed,
        unknown=unknown,
        flags_malformed=flags_malformed,
    )

    if input_malformed:
        return _result(
            level=AnalysisContextReadinessLevel.DESCRIPTIVE_ONLY,
            present=present,
            missing=missing,
            blocked_reasons=(INSUFFICIENT_ANALYSIS_CONTEXT,),
            provenance_valid=safe_provenance_valid,
            context_stale=safe_context_stale,
            unknown=unknown,
            input_notes=input_notes,
        )

    has_fundamentals = AnalysisContextEvidenceFamily.FUNDAMENTALS in present
    has_setup = AnalysisContextEvidenceFamily.SETUP_PRICE_MARKET in present
    has_profile = AnalysisContextEvidenceFamily.COMPANY_PROFILE in present
    has_valuation = AnalysisContextEvidenceFamily.VALUATION in present
    provenance_gate_valid = (
        AnalysisContextEvidenceFamily.PROVENANCE_MANIFEST_STALENESS in present
        and safe_provenance_valid
        and not safe_context_stale
    )

    if not has_fundamentals and not has_setup:
        profile_only = has_profile and present.issubset(
            {
                AnalysisContextEvidenceFamily.COMPANY_PROFILE,
                AnalysisContextEvidenceFamily.PROVENANCE_MANIFEST_STALENESS,
            }
        )
        blocked_reasons = []
        if present and not provenance_gate_valid:
            blocked_reasons.append(STALE_OR_UNPROVENANCED_ANALYSIS_CONTEXT)
        blocked_reasons.append(
            COMPANY_PROFILE_ONLY_CONTEXT_NON_ACTIONABLE
            if profile_only
            else INSUFFICIENT_ANALYSIS_CONTEXT
        )
        return _result(
            level=AnalysisContextReadinessLevel.DESCRIPTIVE_ONLY,
            present=present,
            missing=missing,
            blocked_reasons=tuple(blocked_reasons),
            provenance_valid=safe_provenance_valid,
            context_stale=safe_context_stale,
            unknown=unknown,
            input_notes=input_notes,
        )

    blocked_reasons = []
    if not provenance_gate_valid:
        blocked_reasons.append(STALE_OR_UNPROVENANCED_ANALYSIS_CONTEXT)
    if not has_fundamentals:
        blocked_reasons.append(MISSING_FUNDAMENTAL_EVIDENCE)
    if not has_setup:
        blocked_reasons.append(MISSING_SETUP_OR_PRICE_CONTEXT)
    if safe_valuation_required and not has_valuation:
        blocked_reasons.append(INSUFFICIENT_ANALYSIS_CONTEXT)

    if blocked_reasons:
        return _result(
            level=AnalysisContextReadinessLevel.PARTIAL_ANALYSIS,
            present=present,
            missing=missing,
            blocked_reasons=tuple(blocked_reasons),
            provenance_valid=safe_provenance_valid,
            context_stale=safe_context_stale,
            unknown=unknown,
            input_notes=input_notes,
        )

    return _result(
        level=AnalysisContextReadinessLevel.RECOMMENDATION_ELIGIBLE,
        present=present,
        missing=missing,
        blocked_reasons=(),
        provenance_valid=safe_provenance_valid,
        context_stale=safe_context_stale,
        unknown=unknown,
        input_notes=input_notes,
    )


def _normalize_evidence_families(
    evidence_families: Iterable[AnalysisContextEvidenceFamily] | None,
) -> tuple[set[AnalysisContextEvidenceFamily], tuple[str, ...], bool]:
    if evidence_families is None:
        return set(), (), False
    if isinstance(evidence_families, (str, bytes)):
        return set(), (_unknown_evidence_label(evidence_families),), True

    try:
        values = tuple(evidence_families)
    except TypeError:
        return set(), (_unknown_evidence_label(evidence_families),), True

    present = {
        value for value in values if isinstance(value, AnalysisContextEvidenceFamily)
    }
    unknown = tuple(
        dict.fromkeys(
            _unknown_evidence_label(value)
            for value in values
            if not isinstance(value, AnalysisContextEvidenceFamily)
        )
    )
    return present, unknown, bool(unknown)


def _unknown_evidence_label(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    return type(value).__name__


def _missing_evidence_families(
    present: set[AnalysisContextEvidenceFamily],
    *,
    valuation_required: bool,
) -> tuple[AnalysisContextEvidenceFamily, ...]:
    required = {
        AnalysisContextEvidenceFamily.FUNDAMENTALS,
        AnalysisContextEvidenceFamily.SETUP_PRICE_MARKET,
        AnalysisContextEvidenceFamily.PROVENANCE_MANIFEST_STALENESS,
    }
    if valuation_required:
        required.add(AnalysisContextEvidenceFamily.VALUATION)

    missing = required - present
    return tuple(family for family in _EVIDENCE_FAMILY_ORDER if family in missing)


def _input_notes(
    *,
    input_malformed: bool,
    unknown: tuple[str, ...],
    flags_malformed: bool,
) -> tuple[str, ...]:
    notes = []
    if unknown:
        notes.append("Unknown evidence families were rejected.")
    if flags_malformed:
        notes.append("Readiness gate flags must be booleans.")
    if input_malformed:
        notes.append("Malformed readiness input failed closed.")
    return tuple(notes)


def _result(
    *,
    level: AnalysisContextReadinessLevel,
    present: set[AnalysisContextEvidenceFamily],
    missing: tuple[AnalysisContextEvidenceFamily, ...],
    blocked_reasons: tuple[str, ...],
    provenance_valid: bool,
    context_stale: bool,
    unknown: tuple[str, ...],
    input_notes: tuple[str, ...],
) -> AnalysisContextReadinessResult:
    ordered_present = tuple(
        family for family in _EVIDENCE_FAMILY_ORDER if family in present
    )
    return AnalysisContextReadinessResult(
        readiness_level=level,
        evidence_families_present=ordered_present,
        evidence_families_missing=missing,
        blocked_reasons=blocked_reasons,
        recommendation_review_eligible=(
            level == AnalysisContextReadinessLevel.RECOMMENDATION_ELIGIBLE
        ),
        actionable_review_allowed=False,
        decision_engine_ready=False,
        provenance_valid=provenance_valid,
        context_stale=context_stale,
        unknown_evidence_families=unknown,
        input_notes=input_notes,
    )
