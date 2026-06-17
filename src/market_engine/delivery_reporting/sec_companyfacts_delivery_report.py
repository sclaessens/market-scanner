from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from market_engine.decision_engine_handoff.sec_companyfacts_handoff import (
    MARKET_ENGINE_DECISION_ENGINE_HANDOFF_FORMAT_VERSION,
    MarketEngineDecisionEngineHandoff,
    MarketEngineDecisionEngineHandoffReadinessState,
)


MARKET_ENGINE_DELIVERY_REPORT_FORMAT_VERSION = "market-engine-delivery-report-v1"

MARKET_ENGINE_DELIVERY_REPORT_BOUNDARY = (
    "Delivery / Reporting presents upstream review evidence only; it does not "
    "create decisions, channel delivery, portfolio changes, or external instructions."
)

FORBIDDEN_DELIVERY_LANGUAGE = (
    "buy",
    "sell",
    "hold",
    "target price",
    "allocation",
    "position size",
    "ranking",
    "conviction",
    "urgency",
    "execute",
    "order",
    "broker-ready",
    "best pick",
)


class MarketEngineDeliveryReportState(str, Enum):
    READY_FOR_USER_REVIEW = "ready_for_user_review"
    BLOCKED_UPSTREAM = "blocked_upstream"
    INSUFFICIENT_DATA = "insufficient_data"
    STALE_DATA = "stale_data"
    UNSUPPORTED_INPUT = "unsupported_input"
    CONTRACT_VIOLATION = "contract_violation"


class MarketEngineDeliveryReportCategory(str, Enum):
    FACTUAL_SUMMARY = "factual_summary"
    EVIDENCE_SUMMARY = "evidence_summary"
    UPSTREAM_REVIEW_SUMMARY = "upstream_review_summary"
    PORTFOLIO_CONTEXT_SUMMARY = "portfolio_context_summary"
    RISK_LIMITATION_SUMMARY = "risk_limitation_summary"
    MISSING_DATA_SUMMARY = "missing_data_summary"
    STALE_DATA_SUMMARY = "stale_data_summary"
    REQUIRES_HUMAN_REVIEW_NOTE = "requires_human_review_note"


@dataclass(frozen=True)
class MarketEngineDeliveryReportDisplaySection:
    category: MarketEngineDeliveryReportCategory
    title: str
    body: str
    source_references: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketEngineDeliveryReport:
    report_format_version: str
    report_id: str
    generated_at: str | None
    source_handoff_format_version: str | None
    source_handoff_run_id: str | None
    ticker: str
    cik: str
    provider_name: str
    delivery_state: MarketEngineDeliveryReportState
    display_sections: tuple[MarketEngineDeliveryReportDisplaySection, ...]
    blocked_unavailable_reasons: tuple[str, ...]
    upstream_provenance_summary: dict[str, Any]
    missing_data_summary: tuple[str, ...]
    stale_data_summary: tuple[str, ...]
    numeric_zero_evidence: dict[str, Any]
    allowed_user_facing_language_categories: tuple[str, ...]
    forbidden_language_guardrails: tuple[str, ...]
    audit_metadata: dict[str, Any]
    non_execution_boundary: str = MARKET_ENGINE_DELIVERY_REPORT_BOUNDARY
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["delivery_state"] = self.delivery_state.value
        payload["display_sections"] = [
            {
                **asdict(section),
                "category": section.category.value,
            }
            for section in self.display_sections
        ]
        return payload


def build_market_engine_delivery_report(
    decision_engine_handoff: MarketEngineDecisionEngineHandoff | dict[str, Any] | None,
    *,
    report_id: str,
    generated_at: str | None = None,
) -> MarketEngineDeliveryReport:
    handoff_payload = _handoff_payload(decision_engine_handoff)
    validation_reasons = _contract_validation_reasons(handoff_payload)
    delivery_state = _delivery_state(handoff_payload, validation_reasons)
    blocked_reasons = _blocked_reasons(handoff_payload, validation_reasons)
    display_sections = _display_sections(handoff_payload, delivery_state)

    return MarketEngineDeliveryReport(
        report_format_version=MARKET_ENGINE_DELIVERY_REPORT_FORMAT_VERSION,
        report_id=report_id,
        generated_at=generated_at,
        source_handoff_format_version=handoff_payload.get("handoff_format_version"),
        source_handoff_run_id=handoff_payload.get("handoff_run_id"),
        ticker=handoff_payload.get("ticker") or "",
        cik=handoff_payload.get("cik") or "",
        provider_name=handoff_payload.get("provider_name") or "",
        delivery_state=delivery_state,
        display_sections=display_sections,
        blocked_unavailable_reasons=blocked_reasons,
        upstream_provenance_summary=_upstream_provenance_summary(handoff_payload),
        missing_data_summary=tuple(handoff_payload.get("missing_data_markers") or ()),
        stale_data_summary=tuple(handoff_payload.get("stale_data_markers") or ()),
        numeric_zero_evidence=_numeric_zero_evidence(handoff_payload),
        allowed_user_facing_language_categories=tuple(
            category.value for category in MarketEngineDeliveryReportCategory
        ),
        forbidden_language_guardrails=FORBIDDEN_DELIVERY_LANGUAGE,
        audit_metadata=_audit_metadata(handoff_payload, validation_reasons),
        warnings=(),
    )


def _handoff_payload(
    decision_engine_handoff: MarketEngineDecisionEngineHandoff | dict[str, Any] | None,
) -> dict[str, Any]:
    if decision_engine_handoff is None:
        return {}
    if isinstance(decision_engine_handoff, MarketEngineDecisionEngineHandoff):
        return decision_engine_handoff.to_payload()
    if isinstance(decision_engine_handoff, dict):
        return decision_engine_handoff
    return {
        "contract_violation": (
            "Decision Engine handoff input must be a handoff object or payload."
        )
    }


def _contract_validation_reasons(handoff_payload: dict[str, Any]) -> tuple[str, ...]:
    reasons: list[str] = []
    if not handoff_payload:
        return ("Decision Engine handoff input is missing.",)
    if "contract_violation" in handoff_payload:
        return (handoff_payload["contract_violation"],)
    if (
        handoff_payload.get("handoff_format_version")
        != MARKET_ENGINE_DECISION_ENGINE_HANDOFF_FORMAT_VERSION
    ):
        reasons.append("Decision Engine handoff format is unsupported.")
    if _is_missing_text(handoff_payload.get("handoff_readiness_state")):
        reasons.append("Decision Engine handoff readiness state is missing.")
    if _is_missing_text(handoff_payload.get("ticker")):
        reasons.append("Ticker identity is missing.")
    if not isinstance(handoff_payload.get("audit_provenance"), dict):
        reasons.append("Audit provenance is missing or malformed.")
    return tuple(reasons)


def _delivery_state(
    handoff_payload: dict[str, Any],
    validation_reasons: tuple[str, ...],
) -> MarketEngineDeliveryReportState:
    if not handoff_payload or any("must be" in reason for reason in validation_reasons):
        return MarketEngineDeliveryReportState.CONTRACT_VIOLATION
    if any("unsupported" in reason for reason in validation_reasons):
        return MarketEngineDeliveryReportState.UNSUPPORTED_INPUT
    if validation_reasons:
        return MarketEngineDeliveryReportState.CONTRACT_VIOLATION

    handoff_state = handoff_payload.get("handoff_readiness_state")
    if handoff_state == (
        MarketEngineDecisionEngineHandoffReadinessState.READY_FOR_DECISION_ENGINE_REVIEW.value
    ):
        if handoff_payload.get("stale_data_markers"):
            return MarketEngineDeliveryReportState.STALE_DATA
        if handoff_payload.get("missing_data_markers"):
            return MarketEngineDeliveryReportState.INSUFFICIENT_DATA
        return MarketEngineDeliveryReportState.READY_FOR_USER_REVIEW

    if handoff_state == (
        MarketEngineDecisionEngineHandoffReadinessState.BLOCKED_STALE_PORTFOLIO_CONTEXT.value
    ) or handoff_payload.get("stale_data_markers"):
        return MarketEngineDeliveryReportState.STALE_DATA

    if handoff_state in {
        MarketEngineDecisionEngineHandoffReadinessState.BLOCKED_INCOMPLETE_PROVENANCE.value,
        MarketEngineDecisionEngineHandoffReadinessState.BLOCKED_INSUFFICIENT_EVIDENCE.value,
        MarketEngineDecisionEngineHandoffReadinessState.BLOCKED_MISSING_PORTFOLIO_CONTEXT.value,
    }:
        return MarketEngineDeliveryReportState.INSUFFICIENT_DATA

    if handoff_state == (
        MarketEngineDecisionEngineHandoffReadinessState.BLOCKED_INVALID_PORTFOLIO_REVIEW_CONTRACT.value
    ):
        return MarketEngineDeliveryReportState.UNSUPPORTED_INPUT

    return MarketEngineDeliveryReportState.BLOCKED_UPSTREAM


def _blocked_reasons(
    handoff_payload: dict[str, Any],
    validation_reasons: tuple[str, ...],
) -> tuple[str, ...]:
    reasons = list(validation_reasons)
    reasons.extend(handoff_payload.get("blocked_reasons") or ())
    if (
        not reasons
        and handoff_payload.get("handoff_readiness_state")
        != MarketEngineDecisionEngineHandoffReadinessState.READY_FOR_DECISION_ENGINE_REVIEW.value
    ):
        reasons.append("Decision Engine handoff is blocked upstream.")
    return tuple(dict.fromkeys(reasons))


def _display_sections(
    handoff_payload: dict[str, Any],
    delivery_state: MarketEngineDeliveryReportState,
) -> tuple[MarketEngineDeliveryReportDisplaySection, ...]:
    ticker = handoff_payload.get("ticker") or "the reviewed entity"
    sections = [
        MarketEngineDeliveryReportDisplaySection(
            category=MarketEngineDeliveryReportCategory.FACTUAL_SUMMARY,
            title="Factual summary",
            body=f"{ticker} has a Market Engine handoff state of {handoff_payload.get('handoff_readiness_state', 'unavailable')}.",
            source_references={"handoff_run_id": handoff_payload.get("handoff_run_id")},
        ),
        MarketEngineDeliveryReportDisplaySection(
            category=MarketEngineDeliveryReportCategory.UPSTREAM_REVIEW_SUMMARY,
            title="Upstream review summary",
            body="The report preserves upstream review evidence and does not add new conclusions.",
            source_references=handoff_payload.get("portfolio_review_reference") or {},
        ),
        MarketEngineDeliveryReportDisplaySection(
            category=MarketEngineDeliveryReportCategory.PORTFOLIO_CONTEXT_SUMMARY,
            title="Portfolio context summary",
            body="Portfolio context is shown only as upstream context from the handoff payload.",
            source_references=handoff_payload.get("portfolio_context_reference") or {},
        ),
    ]
    if delivery_state != MarketEngineDeliveryReportState.READY_FOR_USER_REVIEW:
        sections.append(
            MarketEngineDeliveryReportDisplaySection(
                category=MarketEngineDeliveryReportCategory.RISK_LIMITATION_SUMMARY,
                title="Limitation summary",
                body="The report preserves upstream limitations and remains unavailable for action-oriented use.",
                source_references={
                    "blocked_reasons": handoff_payload.get("blocked_reasons") or ()
                },
            )
        )
    if handoff_payload.get("missing_data_markers"):
        sections.append(
            MarketEngineDeliveryReportDisplaySection(
                category=MarketEngineDeliveryReportCategory.MISSING_DATA_SUMMARY,
                title="Missing data summary",
                body="Missing upstream evidence remains visible in this report.",
                source_references={
                    "missing_data_markers": handoff_payload.get("missing_data_markers")
                },
            )
        )
    if handoff_payload.get("stale_data_markers"):
        sections.append(
            MarketEngineDeliveryReportDisplaySection(
                category=MarketEngineDeliveryReportCategory.STALE_DATA_SUMMARY,
                title="Stale data summary",
                body="Stale upstream evidence remains visible in this report.",
                source_references={
                    "stale_data_markers": handoff_payload.get("stale_data_markers")
                },
            )
        )
    sections.append(
        MarketEngineDeliveryReportDisplaySection(
            category=MarketEngineDeliveryReportCategory.REQUIRES_HUMAN_REVIEW_NOTE,
            title="Human review note",
            body="This report is for human review and has no external delivery authority.",
            source_references={},
        )
    )
    _validate_display_language(sections)
    return tuple(sections)


def _validate_display_language(
    sections: list[MarketEngineDeliveryReportDisplaySection],
) -> None:
    display_text = " ".join(
        f"{section.title} {section.body}" for section in sections
    ).lower()
    forbidden_hits = [
        term for term in FORBIDDEN_DELIVERY_LANGUAGE if term in display_text
    ]
    if forbidden_hits:
        raise ValueError(
            "Delivery report display text contains forbidden language: "
            f"{', '.join(forbidden_hits)}"
        )


def _upstream_provenance_summary(handoff_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "portfolio_review": handoff_payload.get("portfolio_review_reference") or {},
        "portfolio_context": handoff_payload.get("portfolio_context_reference") or {},
        "recommendation_review": (
            handoff_payload.get("recommendation_review_reference") or {}
        ),
        "analysis_review": handoff_payload.get("analysis_review_reference") or {},
        "setup_detection": handoff_payload.get("setup_detection_reference") or {},
        "source_context": handoff_payload.get("source_context_references") or {},
    }


def _numeric_zero_evidence(handoff_payload: dict[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    _collect_zero_values("portfolio_context", handoff_payload.get("portfolio_context_reference"), evidence)
    _collect_zero_values("portfolio_review", handoff_payload.get("portfolio_review_reference"), evidence)
    return evidence


def _collect_zero_values(prefix: str, value: Any, evidence: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            _collect_zero_values(f"{prefix}.{key}", nested_value, evidence)
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0:
        evidence[prefix] = value


def _audit_metadata(
    handoff_payload: dict[str, Any],
    validation_reasons: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "source_handoff_format_version": handoff_payload.get("handoff_format_version"),
        "source_handoff_run_id": handoff_payload.get("handoff_run_id"),
        "source_handoff_state": handoff_payload.get("handoff_readiness_state"),
        "validation_reasons": validation_reasons,
        "non_execution_boundary": MARKET_ENGINE_DELIVERY_REPORT_BOUNDARY,
    }


def _is_missing_text(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")
