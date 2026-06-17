from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from market_engine.portfolio_review.sec_companyfacts_portfolio_review import (
    MARKET_ENGINE_PORTFOLIO_CONTEXT_FORMAT_VERSION,
    SEC_COMPANYFACTS_PORTFOLIO_REVIEW_FORMAT_VERSION,
    SecCompanyFactsPortfolioReview,
    SecCompanyFactsPortfolioReviewCategory,
    SecCompanyFactsPortfolioReviewItem,
    SecCompanyFactsPortfolioReviewState,
)


MARKET_ENGINE_DECISION_ENGINE_HANDOFF_FORMAT_VERSION = (
    "market-engine-decision-engine-handoff-v1"
)

MARKET_ENGINE_DECISION_ENGINE_HANDOFF_BOUNDARY = (
    "Decision Engine handoff preserves review evidence and readiness only; "
    "it does not grant trade, allocation, order, execution, delivery, or broker authority."
)


class MarketEngineDecisionEngineHandoffReadinessState(str, Enum):
    READY_FOR_DECISION_ENGINE_REVIEW = "ready_for_decision_engine_review"
    BLOCKED_MISSING_PORTFOLIO_REVIEW = "blocked_missing_portfolio_review"
    BLOCKED_INVALID_PORTFOLIO_REVIEW_CONTRACT = (
        "blocked_invalid_portfolio_review_contract"
    )
    BLOCKED_UNAPPROVED_PORTFOLIO_REVIEW = "blocked_unapproved_portfolio_review"
    BLOCKED_MISSING_PORTFOLIO_CONTEXT = "blocked_missing_portfolio_context"
    BLOCKED_STALE_PORTFOLIO_CONTEXT = "blocked_stale_portfolio_context"
    BLOCKED_INCOMPLETE_PROVENANCE = "blocked_incomplete_provenance"
    BLOCKED_TICKER_MISMATCH = "blocked_ticker_mismatch"
    BLOCKED_INSUFFICIENT_EVIDENCE = "blocked_insufficient_evidence"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class MarketEngineDecisionEngineHandoff:
    ticker: str
    cik: str
    provider_name: str
    handoff_format_version: str
    handoff_run_id: str
    created_at: str | None
    portfolio_review_format_version: str | None
    portfolio_review_run_id: str | None
    portfolio_review_state_summary: str | None
    portfolio_review_category_summary: str | None
    portfolio_context_format_version: str | None
    portfolio_context_run_id: str | None
    portfolio_context_state_summary: str | None
    recommendation_review_reference: dict[str, Any]
    analysis_review_reference: dict[str, Any]
    setup_detection_reference: dict[str, Any]
    source_context_references: dict[str, Any]
    portfolio_review_reference: dict[str, Any]
    portfolio_context_reference: dict[str, Any]
    portfolio_review_item_references: tuple[dict[str, Any], ...]
    missing_data_markers: tuple[str, ...]
    stale_data_markers: tuple[str, ...]
    blocked_reasons: tuple[str, ...]
    handoff_readiness_state: MarketEngineDecisionEngineHandoffReadinessState
    audit_provenance: dict[str, Any]
    authority_boundary: str = MARKET_ENGINE_DECISION_ENGINE_HANDOFF_BOUNDARY
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["handoff_readiness_state"] = self.handoff_readiness_state.value
        return payload


def build_market_engine_decision_engine_handoff(
    portfolio_review: SecCompanyFactsPortfolioReview | None,
    *,
    handoff_run_id: str,
    created_at: str | None = None,
) -> MarketEngineDecisionEngineHandoff:
    if portfolio_review is None:
        return _blocked_handoff(
            portfolio_review=None,
            handoff_run_id=handoff_run_id,
            created_at=created_at,
            state=(
                MarketEngineDecisionEngineHandoffReadinessState.BLOCKED_MISSING_PORTFOLIO_REVIEW
            ),
            blocked_reasons=("Portfolio Review input is missing.",),
        )

    blocked_reasons = _blocked_reasons(portfolio_review)
    readiness_state = _readiness_state(portfolio_review, blocked_reasons)

    return MarketEngineDecisionEngineHandoff(
        ticker=portfolio_review.ticker,
        cik=portfolio_review.cik,
        provider_name=portfolio_review.provider_name,
        handoff_format_version=MARKET_ENGINE_DECISION_ENGINE_HANDOFF_FORMAT_VERSION,
        handoff_run_id=handoff_run_id,
        created_at=created_at,
        portfolio_review_format_version=portfolio_review.portfolio_review_format_version,
        portfolio_review_run_id=portfolio_review.portfolio_review_run_id,
        portfolio_review_state_summary=_enum_value(portfolio_review.review_state),
        portfolio_review_category_summary=_enum_value(portfolio_review.review_category),
        portfolio_context_format_version=portfolio_review.portfolio_context_format_version,
        portfolio_context_run_id=portfolio_review.portfolio_context_run_id,
        portfolio_context_state_summary=portfolio_review.position_state,
        recommendation_review_reference=portfolio_review.recommendation_review_provenance,
        analysis_review_reference=_analysis_review_reference(portfolio_review),
        setup_detection_reference=portfolio_review.setup_aware_provenance,
        source_context_references=_source_context_references(portfolio_review),
        portfolio_review_reference=_portfolio_review_reference(portfolio_review),
        portfolio_context_reference=portfolio_review.portfolio_context_provenance,
        portfolio_review_item_references=tuple(
            _portfolio_review_item_reference(item)
            for item in portfolio_review.portfolio_review_items
        ),
        missing_data_markers=_missing_data_markers(portfolio_review),
        stale_data_markers=tuple(portfolio_review.stale_portfolio_context_fields),
        blocked_reasons=tuple(blocked_reasons),
        handoff_readiness_state=readiness_state,
        audit_provenance=_audit_provenance(portfolio_review),
        warnings=(),
    )


def _blocked_handoff(
    *,
    portfolio_review: SecCompanyFactsPortfolioReview | None,
    handoff_run_id: str,
    created_at: str | None,
    state: MarketEngineDecisionEngineHandoffReadinessState,
    blocked_reasons: tuple[str, ...],
) -> MarketEngineDecisionEngineHandoff:
    return MarketEngineDecisionEngineHandoff(
        ticker=portfolio_review.ticker if portfolio_review is not None else "",
        cik=portfolio_review.cik if portfolio_review is not None else "",
        provider_name=portfolio_review.provider_name if portfolio_review is not None else "",
        handoff_format_version=MARKET_ENGINE_DECISION_ENGINE_HANDOFF_FORMAT_VERSION,
        handoff_run_id=handoff_run_id,
        created_at=created_at,
        portfolio_review_format_version=(
            portfolio_review.portfolio_review_format_version
            if portfolio_review is not None
            else None
        ),
        portfolio_review_run_id=(
            portfolio_review.portfolio_review_run_id
            if portfolio_review is not None
            else None
        ),
        portfolio_review_state_summary=(
            _enum_value(portfolio_review.review_state)
            if portfolio_review is not None
            else None
        ),
        portfolio_review_category_summary=(
            _enum_value(portfolio_review.review_category)
            if portfolio_review is not None
            else None
        ),
        portfolio_context_format_version=(
            portfolio_review.portfolio_context_format_version
            if portfolio_review is not None
            else None
        ),
        portfolio_context_run_id=(
            portfolio_review.portfolio_context_run_id
            if portfolio_review is not None
            else None
        ),
        portfolio_context_state_summary=(
            portfolio_review.position_state if portfolio_review is not None else None
        ),
        recommendation_review_reference=(
            portfolio_review.recommendation_review_provenance
            if portfolio_review is not None
            else {}
        ),
        analysis_review_reference=(
            _analysis_review_reference(portfolio_review)
            if portfolio_review is not None
            else {}
        ),
        setup_detection_reference=(
            portfolio_review.setup_aware_provenance
            if portfolio_review is not None
            else {}
        ),
        source_context_references=(
            _source_context_references(portfolio_review)
            if portfolio_review is not None
            else {}
        ),
        portfolio_review_reference=(
            _portfolio_review_reference(portfolio_review)
            if portfolio_review is not None
            else {}
        ),
        portfolio_context_reference=(
            portfolio_review.portfolio_context_provenance
            if portfolio_review is not None
            else {}
        ),
        portfolio_review_item_references=(
            tuple(
                _portfolio_review_item_reference(item)
                for item in portfolio_review.portfolio_review_items
            )
            if portfolio_review is not None
            else ()
        ),
        missing_data_markers=(
            _missing_data_markers(portfolio_review)
            if portfolio_review is not None
            else ("portfolio_review",)
        ),
        stale_data_markers=(
            tuple(portfolio_review.stale_portfolio_context_fields)
            if portfolio_review is not None
            else ()
        ),
        blocked_reasons=blocked_reasons,
        handoff_readiness_state=state,
        audit_provenance=(
            _audit_provenance(portfolio_review) if portfolio_review is not None else {}
        ),
        warnings=(),
    )


def _blocked_reasons(
    portfolio_review: SecCompanyFactsPortfolioReview,
) -> list[str]:
    reasons: list[str] = []

    if (
        portfolio_review.portfolio_review_format_version
        != SEC_COMPANYFACTS_PORTFOLIO_REVIEW_FORMAT_VERSION
    ):
        reasons.append("Portfolio Review contract is unsupported.")

    if _is_missing_text(portfolio_review.ticker):
        reasons.append("Ticker identity is missing.")

    if _portfolio_context_ticker_mismatch(portfolio_review):
        reasons.append("Portfolio context ticker does not match Portfolio Review ticker.")

    if (
        portfolio_review.portfolio_context_format_version
        != MARKET_ENGINE_PORTFOLIO_CONTEXT_FORMAT_VERSION
    ):
        reasons.append("Portfolio context contract is missing or unsupported.")

    if not portfolio_review.portfolio_context_provenance:
        reasons.append("Portfolio context provenance is missing.")

    if not portfolio_review.recommendation_review_provenance:
        reasons.append("Recommendation Review provenance is missing.")

    if portfolio_review.stale_portfolio_context_fields or _state_is(
        portfolio_review.review_state,
        SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_STALE,
    ):
        reasons.append("Portfolio context is stale.")

    if portfolio_review.missing_portfolio_context_fields or _state_is(
        portfolio_review.review_state,
        SecCompanyFactsPortfolioReviewState.BLOCKED_BY_MISSING_PORTFOLIO_CONTEXT,
    ):
        reasons.append("Portfolio context is missing or incomplete.")

    if _state_is(
        portfolio_review.review_state,
        SecCompanyFactsPortfolioReviewState.BLOCKED_BY_INVALID_INPUT,
    ) or _category_is(
        portfolio_review.review_category,
        SecCompanyFactsPortfolioReviewCategory.INPUT_CONTRACT_INVALID,
    ):
        reasons.append("Portfolio Review input is invalid or not approved.")

    if not _has_handoff_ready_item(portfolio_review):
        reasons.append("Portfolio Review handoff-readiness evidence is missing.")

    if _contains_invalid_review_value(portfolio_review):
        reasons.append("Portfolio Review state or category is unsupported.")

    return reasons


def _readiness_state(
    portfolio_review: SecCompanyFactsPortfolioReview,
    blocked_reasons: list[str],
) -> MarketEngineDecisionEngineHandoffReadinessState:
    if not blocked_reasons:
        return (
            MarketEngineDecisionEngineHandoffReadinessState.READY_FOR_DECISION_ENGINE_REVIEW
        )

    reason_text = " ".join(blocked_reasons).lower()
    if "contract is unsupported" in reason_text:
        return (
            MarketEngineDecisionEngineHandoffReadinessState.BLOCKED_INVALID_PORTFOLIO_REVIEW_CONTRACT
        )
    if "ticker" in reason_text:
        return MarketEngineDecisionEngineHandoffReadinessState.BLOCKED_TICKER_MISMATCH
    if "stale" in reason_text:
        return (
            MarketEngineDecisionEngineHandoffReadinessState.BLOCKED_STALE_PORTFOLIO_CONTEXT
        )
    if "portfolio context" in reason_text and (
        "missing" in reason_text or "incomplete" in reason_text
    ):
        return (
            MarketEngineDecisionEngineHandoffReadinessState.BLOCKED_MISSING_PORTFOLIO_CONTEXT
        )
    if "provenance" in reason_text:
        return (
            MarketEngineDecisionEngineHandoffReadinessState.BLOCKED_INCOMPLETE_PROVENANCE
        )
    if _state_is(
        portfolio_review.review_state,
        SecCompanyFactsPortfolioReviewState.BLOCKED_BY_INVALID_INPUT,
    ):
        return (
            MarketEngineDecisionEngineHandoffReadinessState.BLOCKED_UNAPPROVED_PORTFOLIO_REVIEW
        )
    if "handoff-readiness evidence" in reason_text:
        return (
            MarketEngineDecisionEngineHandoffReadinessState.BLOCKED_INSUFFICIENT_EVIDENCE
        )
    return (
        MarketEngineDecisionEngineHandoffReadinessState.BLOCKED_UNAPPROVED_PORTFOLIO_REVIEW
    )


def _has_handoff_ready_item(portfolio_review: SecCompanyFactsPortfolioReview) -> bool:
    return any(
        _category_is(
            item.category,
            SecCompanyFactsPortfolioReviewCategory.DOWNSTREAM_HANDOFF_READINESS_REVIEW,
        )
        and _state_is(
            item.state,
            SecCompanyFactsPortfolioReviewState.READY_FOR_DECISION_ENGINE_HANDOFF_REVIEW,
        )
        for item in portfolio_review.portfolio_review_items
    )


def _contains_invalid_review_value(
    portfolio_review: SecCompanyFactsPortfolioReview,
) -> bool:
    if not _is_known_enum_value(
        portfolio_review.review_state,
        SecCompanyFactsPortfolioReviewState,
    ):
        return True
    if not _is_known_enum_value(
        portfolio_review.review_category,
        SecCompanyFactsPortfolioReviewCategory,
    ):
        return True
    return any(
        not _is_known_enum_value(item.state, SecCompanyFactsPortfolioReviewState)
        or not _is_known_enum_value(item.category, SecCompanyFactsPortfolioReviewCategory)
        for item in portfolio_review.portfolio_review_items
    )


def _portfolio_context_ticker_mismatch(
    portfolio_review: SecCompanyFactsPortfolioReview,
) -> bool:
    context_ticker = portfolio_review.portfolio_context_provenance.get("ticker")
    if _is_missing_text(context_ticker):
        return False
    return str(context_ticker) != portfolio_review.ticker


def _portfolio_review_reference(
    portfolio_review: SecCompanyFactsPortfolioReview,
) -> dict[str, Any]:
    return {
        "portfolio_review_format_version": (
            portfolio_review.portfolio_review_format_version
        ),
        "portfolio_review_run_id": portfolio_review.portfolio_review_run_id,
        "ticker": portfolio_review.ticker,
        "cik": portfolio_review.cik,
        "provider_name": portfolio_review.provider_name,
        "review_state": _enum_value(portfolio_review.review_state),
        "review_category": _enum_value(portfolio_review.review_category),
        "non_actionable_boundary": portfolio_review.non_actionable_boundary,
    }


def _portfolio_review_item_reference(
    item: SecCompanyFactsPortfolioReviewItem,
) -> dict[str, Any]:
    return {
        "category": _enum_value(item.category),
        "state": _enum_value(item.state),
        "message": item.message,
        "recommendation_review_references": item.recommendation_review_references,
        "portfolio_context_references": item.portfolio_context_references,
        "missing_portfolio_context_fields": item.missing_portfolio_context_fields,
        "stale_portfolio_context_fields": item.stale_portfolio_context_fields,
        "setup_aware_provenance": item.setup_aware_provenance,
        "boundary_notes": item.boundary_notes,
        "non_actionable_boundary": item.non_actionable_boundary,
    }


def _analysis_review_reference(
    portfolio_review: SecCompanyFactsPortfolioReview,
) -> dict[str, Any]:
    return {
        "review_items": [
            item.get("analysis_review_references", {})
            for item in portfolio_review.recommendation_review_provenance.get(
                "review_items", []
            )
            if item.get("analysis_review_references")
        ]
    }


def _source_context_references(
    portfolio_review: SecCompanyFactsPortfolioReview,
) -> dict[str, Any]:
    recommendation_provenance = portfolio_review.recommendation_review_provenance
    input_provenance = recommendation_provenance.get("input_provenance", {})
    return {
        "source_context_format_version": input_provenance.get(
            "source_context_format_version"
        ),
        "source_context_state": input_provenance.get("source_context_state"),
        "source_refresh_snapshot_id": input_provenance.get(
            "source_refresh_snapshot_id"
        ),
    }


def _missing_data_markers(
    portfolio_review: SecCompanyFactsPortfolioReview,
) -> tuple[str, ...]:
    markers = list(portfolio_review.missing_portfolio_context_fields)
    for item in portfolio_review.portfolio_review_items:
        markers.extend(item.missing_portfolio_context_fields)
    return tuple(dict.fromkeys(markers))


def _audit_provenance(
    portfolio_review: SecCompanyFactsPortfolioReview,
) -> dict[str, Any]:
    return {
        "portfolio_review": _portfolio_review_reference(portfolio_review),
        "portfolio_context": portfolio_review.portfolio_context_provenance,
        "recommendation_review": portfolio_review.recommendation_review_provenance,
        "setup_detection": portfolio_review.setup_aware_provenance,
    }


def _state_is(value: Any, expected: SecCompanyFactsPortfolioReviewState) -> bool:
    return _enum_value(value) == expected.value


def _category_is(value: Any, expected: SecCompanyFactsPortfolioReviewCategory) -> bool:
    return _enum_value(value) == expected.value


def _is_known_enum_value(value: Any, enum_type: type[Enum]) -> bool:
    value = _enum_value(value)
    return any(value == enum_member.value for enum_member in enum_type)


def _enum_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def _is_missing_text(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")
