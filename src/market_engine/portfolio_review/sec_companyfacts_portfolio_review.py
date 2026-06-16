from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from market_engine.recommendation_review.sec_companyfacts_recommendation_review import (
    SEC_COMPANYFACTS_RECOMMENDATION_REVIEW_FORMAT_VERSION,
    SecCompanyFactsRecommendationReview,
    SecCompanyFactsRecommendationReviewCategory,
    SecCompanyFactsRecommendationReviewItem,
    SecCompanyFactsRecommendationReviewState,
)


SEC_COMPANYFACTS_PORTFOLIO_REVIEW_FORMAT_VERSION = (
    "sec-companyfacts-portfolio-review-v1"
)
MARKET_ENGINE_PORTFOLIO_CONTEXT_FORMAT_VERSION = "market-engine-portfolio-context-v1"
SEC_COMPANYFACTS_PORTFOLIO_REVIEW_ROOT = Path(
    "data/market_engine/portfolio_reviews"
)

NON_ACTIONABLE_PORTFOLIO_REVIEW_BOUNDARY = (
    "Portfolio Review creates non-actionable portfolio-context review only and "
    "does not create trade, allocation, delivery, or Decision Engine authority."
)

FORBIDDEN_PORTFOLIO_REVIEW_ACTIONS = (
    "BUY",
    "SELL",
    "HOLD_ACTION",
    "ALLOCATION_ADVICE",
    "POSITION_SIZING",
    "EXECUTION",
    "ORDER_GENERATION",
    "PORTFOLIO_MUTATION",
    "WATCHLIST_MUTATION",
    "TELEGRAM",
    "REPORTING",
    "DELIVERY",
    "DECISION_ENGINE_ACTION",
    "SCORE",
    "RANKING",
    "CONVICTION",
    "URGENCY",
    "TRADEABILITY",
)


class MarketEnginePortfolioPositionState(str, Enum):
    NOT_HELD = "not_held"
    HELD = "held"
    PARTIALLY_KNOWN = "partially_known"
    UNKNOWN = "unknown"
    STALE = "stale"
    INVALID = "invalid"


class SecCompanyFactsPortfolioReviewCategory(str, Enum):
    POSITION_CONTEXT_REVIEW = "position_context_review"
    EXPOSURE_CONTEXT_REVIEW = "exposure_context_review"
    CONCENTRATION_CONTEXT_REVIEW = "concentration_context_review"
    PORTFOLIO_FIT_CONTEXT_REVIEW = "portfolio_fit_context_review"
    PORTFOLIO_DATA_LIMITATION_REVIEW = "portfolio_data_limitation_review"
    DOWNSTREAM_HANDOFF_READINESS_REVIEW = "downstream_handoff_readiness_review"
    INPUT_CONTRACT_INVALID = "input_contract_invalid"


class SecCompanyFactsPortfolioReviewState(str, Enum):
    PORTFOLIO_REVIEW_REQUIRED = "portfolio_review_required"
    PORTFOLIO_CONTEXT_SUPPORTED = "portfolio_context_supported"
    PORTFOLIO_CONTEXT_PARTIAL = "portfolio_context_partial"
    PORTFOLIO_CONTEXT_MISSING = "portfolio_context_missing"
    PORTFOLIO_CONTEXT_STALE = "portfolio_context_stale"
    PORTFOLIO_CONTEXT_INVALID = "portfolio_context_invalid"
    POSITION_ALREADY_HELD = "position_already_held"
    POSITION_NOT_HELD = "position_not_held"
    POSITION_UNKNOWN = "position_unknown"
    EXPOSURE_KNOWN = "exposure_known"
    EXPOSURE_MISSING = "exposure_missing"
    CONCENTRATION_WITHIN_CONTEXT = "concentration_within_context"
    CONCENTRATION_REQUIRES_REVIEW = "concentration_requires_review"
    BLOCKED_BY_MISSING_PORTFOLIO_CONTEXT = "blocked_by_missing_portfolio_context"
    BLOCKED_BY_INVALID_INPUT = "blocked_by_invalid_input"
    READY_FOR_DECISION_ENGINE_HANDOFF_REVIEW = (
        "ready_for_decision_engine_handoff_review"
    )
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class MarketEnginePortfolioContext:
    portfolio_context_format_version: str
    portfolio_context_run_id: str
    portfolio_snapshot_timestamp: str
    portfolio_base_currency: str
    ticker: str
    position_state: MarketEnginePortfolioPositionState | str
    current_quantity: float | int | None
    current_market_value: float | int | None
    portfolio_total_value: float | int | None
    current_ticker_exposure_pct: float | int | None
    exposure_buckets: dict[str, Any] = field(default_factory=dict)
    concentration_thresholds: dict[str, Any] = field(default_factory=dict)
    policy_constraints: dict[str, Any] = field(default_factory=dict)
    missing_portfolio_context_fields: tuple[str, ...] = field(default_factory=tuple)
    stale_portfolio_context_fields: tuple[str, ...] = field(default_factory=tuple)
    context_provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SecCompanyFactsPortfolioReviewItem:
    category: SecCompanyFactsPortfolioReviewCategory
    state: SecCompanyFactsPortfolioReviewState
    message: str
    recommendation_review_references: dict[str, Any]
    portfolio_context_references: dict[str, Any]
    missing_portfolio_context_fields: tuple[str, ...]
    stale_portfolio_context_fields: tuple[str, ...]
    setup_aware_provenance: dict[str, Any] = field(default_factory=dict)
    boundary_notes: tuple[str, ...] = field(default_factory=tuple)
    forbidden_actions: tuple[str, ...] = FORBIDDEN_PORTFOLIO_REVIEW_ACTIONS
    non_actionable_boundary: str = NON_ACTIONABLE_PORTFOLIO_REVIEW_BOUNDARY


@dataclass(frozen=True)
class SecCompanyFactsPortfolioReview:
    ticker: str
    cik: str
    provider_name: str
    portfolio_review_format_version: str
    portfolio_review_run_id: str
    created_at: str | None
    recommendation_review_format_version: str | None
    recommendation_review_run_id: str | None
    portfolio_context_format_version: str | None
    portfolio_context_run_id: str | None
    portfolio_snapshot_timestamp: str | None
    portfolio_base_currency: str | None
    position_state: str
    review_state: SecCompanyFactsPortfolioReviewState
    review_category: SecCompanyFactsPortfolioReviewCategory
    portfolio_review_items: tuple[SecCompanyFactsPortfolioReviewItem, ...]
    missing_portfolio_context_fields: tuple[str, ...]
    stale_portfolio_context_fields: tuple[str, ...]
    recommendation_review_provenance: dict[str, Any]
    setup_aware_provenance: dict[str, Any]
    portfolio_context_provenance: dict[str, Any]
    forbidden_actions: tuple[str, ...] = FORBIDDEN_PORTFOLIO_REVIEW_ACTIONS
    non_actionable_boundary: str = NON_ACTIONABLE_PORTFOLIO_REVIEW_BOUNDARY
    warnings: tuple[str, ...] = field(default_factory=tuple)


def build_sec_companyfacts_portfolio_review(
    recommendation_review: SecCompanyFactsRecommendationReview | None,
    portfolio_context: MarketEnginePortfolioContext | None,
    *,
    portfolio_review_run_id: str,
    created_at: str | None = None,
) -> SecCompanyFactsPortfolioReview:
    if recommendation_review is None:
        return _invalid_input_portfolio_review(
            portfolio_context=portfolio_context,
            portfolio_review_run_id=portfolio_review_run_id,
            created_at=created_at,
            reason="Recommendation Review input is missing.",
        )

    _validate_recommendation_review_contract(recommendation_review)

    if _recommendation_review_not_reviewable(recommendation_review):
        return _invalid_input_portfolio_review(
            recommendation_review=recommendation_review,
            portfolio_context=portfolio_context,
            portfolio_review_run_id=portfolio_review_run_id,
            created_at=created_at,
            reason="Recommendation Review input is not reviewable for Portfolio Review.",
        )

    if portfolio_context is None:
        return _missing_portfolio_context_review(
            recommendation_review=recommendation_review,
            portfolio_review_run_id=portfolio_review_run_id,
            created_at=created_at,
        )

    if (
        portfolio_context.portfolio_context_format_version
        != MARKET_ENGINE_PORTFOLIO_CONTEXT_FORMAT_VERSION
    ):
        return _invalid_input_portfolio_review(
            recommendation_review=recommendation_review,
            portfolio_context=portfolio_context,
            portfolio_review_run_id=portfolio_review_run_id,
            created_at=created_at,
            reason="Portfolio context contract is unsupported.",
        )

    if portfolio_context.ticker != recommendation_review.ticker:
        return _invalid_input_portfolio_review(
            recommendation_review=recommendation_review,
            portfolio_context=portfolio_context,
            portfolio_review_run_id=portfolio_review_run_id,
            created_at=created_at,
            reason="Portfolio context ticker does not match Recommendation Review ticker.",
        )

    context_state = _portfolio_context_state(portfolio_context)
    review_items = [
        _position_context_review(recommendation_review, portfolio_context),
        _exposure_context_review(recommendation_review, portfolio_context),
        _concentration_context_review(recommendation_review, portfolio_context),
        _portfolio_fit_context_review(
            recommendation_review,
            portfolio_context,
            context_state,
        ),
        _downstream_handoff_readiness_review(
            recommendation_review,
            portfolio_context,
            context_state,
        ),
    ]

    limitation_review = _portfolio_data_limitation_review(
        recommendation_review,
        portfolio_context,
        context_state,
    )
    if limitation_review is not None:
        review_items.append(limitation_review)

    review_state, review_category = _top_level_state_and_category(context_state)

    return SecCompanyFactsPortfolioReview(
        ticker=recommendation_review.ticker,
        cik=recommendation_review.cik,
        provider_name=recommendation_review.provider_name,
        portfolio_review_format_version=SEC_COMPANYFACTS_PORTFOLIO_REVIEW_FORMAT_VERSION,
        portfolio_review_run_id=portfolio_review_run_id,
        created_at=created_at,
        recommendation_review_format_version=(
            recommendation_review.recommendation_review_format_version
        ),
        recommendation_review_run_id=recommendation_review.recommendation_review_run_id,
        portfolio_context_format_version=(
            portfolio_context.portfolio_context_format_version
        ),
        portfolio_context_run_id=portfolio_context.portfolio_context_run_id,
        portfolio_snapshot_timestamp=portfolio_context.portfolio_snapshot_timestamp,
        portfolio_base_currency=portfolio_context.portfolio_base_currency,
        position_state=_position_state_value(portfolio_context.position_state),
        review_state=review_state,
        review_category=review_category,
        portfolio_review_items=tuple(review_items),
        missing_portfolio_context_fields=(
            portfolio_context.missing_portfolio_context_fields
        ),
        stale_portfolio_context_fields=portfolio_context.stale_portfolio_context_fields,
        recommendation_review_provenance=_recommendation_review_reference(
            recommendation_review
        ),
        setup_aware_provenance=_setup_aware_provenance(recommendation_review),
        portfolio_context_provenance=_portfolio_context_reference(portfolio_context),
    )


def persist_sec_companyfacts_portfolio_review(
    portfolio_review: SecCompanyFactsPortfolioReview,
    *,
    run_id: str,
    root_dir: Path | None = None,
) -> Path:
    root = root_dir or SEC_COMPANYFACTS_PORTFOLIO_REVIEW_ROOT
    portfolio_review_dir = root / run_id / portfolio_review.ticker
    portfolio_review_dir.mkdir(parents=True, exist_ok=True)
    portfolio_review_path = portfolio_review_dir / "portfolio_review.json"

    if portfolio_review_path.exists():
        raise FileExistsError(
            "refusing to overwrite existing SEC CompanyFacts Portfolio Review: "
            f"{portfolio_review_path}"
        )

    portfolio_review_path.write_text(
        json.dumps(_to_jsonable(portfolio_review), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return portfolio_review_path


def _validate_recommendation_review_contract(
    recommendation_review: SecCompanyFactsRecommendationReview,
) -> None:
    if (
        recommendation_review.recommendation_review_format_version
        != SEC_COMPANYFACTS_RECOMMENDATION_REVIEW_FORMAT_VERSION
    ):
        raise ValueError(
            "unsupported SEC CompanyFacts Recommendation Review contract: "
            f"{recommendation_review.recommendation_review_format_version}"
        )


def _recommendation_review_not_reviewable(
    recommendation_review: SecCompanyFactsRecommendationReview,
) -> bool:
    return (
        recommendation_review.review_state
        == SecCompanyFactsRecommendationReviewState.NOT_APPLICABLE
        or recommendation_review.review_category
        == SecCompanyFactsRecommendationReviewCategory.INPUT_CONTRACT_INVALID
    )


def _portfolio_context_state(
    portfolio_context: MarketEnginePortfolioContext,
) -> SecCompanyFactsPortfolioReviewState:
    position_state = _position_state_value(portfolio_context.position_state)
    if position_state == MarketEnginePortfolioPositionState.INVALID.value:
        return SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_INVALID
    if portfolio_context.stale_portfolio_context_fields or position_state == (
        MarketEnginePortfolioPositionState.STALE.value
    ):
        return SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_STALE
    if portfolio_context.missing_portfolio_context_fields or position_state == (
        MarketEnginePortfolioPositionState.PARTIALLY_KNOWN.value
    ):
        return SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_PARTIAL
    return SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_SUPPORTED


def _top_level_state_and_category(
    context_state: SecCompanyFactsPortfolioReviewState,
) -> tuple[
    SecCompanyFactsPortfolioReviewState,
    SecCompanyFactsPortfolioReviewCategory,
]:
    if context_state == SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_INVALID:
        return (
            SecCompanyFactsPortfolioReviewState.BLOCKED_BY_INVALID_INPUT,
            SecCompanyFactsPortfolioReviewCategory.INPUT_CONTRACT_INVALID,
        )
    if context_state == SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_STALE:
        return (
            SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_STALE,
            SecCompanyFactsPortfolioReviewCategory.PORTFOLIO_DATA_LIMITATION_REVIEW,
        )
    if context_state == SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_PARTIAL:
        return (
            SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_PARTIAL,
            SecCompanyFactsPortfolioReviewCategory.PORTFOLIO_DATA_LIMITATION_REVIEW,
        )
    return (
        SecCompanyFactsPortfolioReviewState.PORTFOLIO_REVIEW_REQUIRED,
        SecCompanyFactsPortfolioReviewCategory.PORTFOLIO_FIT_CONTEXT_REVIEW,
    )


def _position_context_review(
    recommendation_review: SecCompanyFactsRecommendationReview,
    portfolio_context: MarketEnginePortfolioContext,
) -> SecCompanyFactsPortfolioReviewItem:
    position_state = _position_state_value(portfolio_context.position_state)
    if position_state == MarketEnginePortfolioPositionState.HELD.value:
        state = SecCompanyFactsPortfolioReviewState.POSITION_ALREADY_HELD
        message = "The ticker is already held according to approved portfolio context."
    elif position_state == MarketEnginePortfolioPositionState.NOT_HELD.value:
        state = SecCompanyFactsPortfolioReviewState.POSITION_NOT_HELD
        message = "The ticker is not held according to approved portfolio context."
    elif position_state == MarketEnginePortfolioPositionState.STALE.value:
        state = SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_STALE
        message = "Position context is stale according to approved portfolio context."
    elif position_state == MarketEnginePortfolioPositionState.INVALID.value:
        state = SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_INVALID
        message = "Position context is invalid according to approved portfolio context."
    else:
        state = SecCompanyFactsPortfolioReviewState.POSITION_UNKNOWN
        message = "Position state cannot be determined from approved portfolio context."

    return _portfolio_review_item(
        recommendation_review,
        portfolio_context,
        category=SecCompanyFactsPortfolioReviewCategory.POSITION_CONTEXT_REVIEW,
        state=state,
        message=message,
    )


def _exposure_context_review(
    recommendation_review: SecCompanyFactsRecommendationReview,
    portfolio_context: MarketEnginePortfolioContext,
) -> SecCompanyFactsPortfolioReviewItem:
    if _is_missing(portfolio_context.current_ticker_exposure_pct):
        state = SecCompanyFactsPortfolioReviewState.EXPOSURE_MISSING
        message = "Ticker exposure data is missing from approved portfolio context."
    else:
        state = SecCompanyFactsPortfolioReviewState.EXPOSURE_KNOWN
        message = "Ticker exposure data is explicitly available and remains non-actionable."

    return _portfolio_review_item(
        recommendation_review,
        portfolio_context,
        category=SecCompanyFactsPortfolioReviewCategory.EXPOSURE_CONTEXT_REVIEW,
        state=state,
        message=message,
    )


def _concentration_context_review(
    recommendation_review: SecCompanyFactsRecommendationReview,
    portfolio_context: MarketEnginePortfolioContext,
) -> SecCompanyFactsPortfolioReviewItem:
    threshold = portfolio_context.concentration_thresholds.get(
        "max_ticker_exposure_pct"
    )
    exposure = portfolio_context.current_ticker_exposure_pct
    if _is_missing(threshold) or _is_missing(exposure):
        state = SecCompanyFactsPortfolioReviewState.CONCENTRATION_REQUIRES_REVIEW
        message = "Concentration context requires review because exposure or threshold context is incomplete."
    elif float(exposure) <= float(threshold):
        state = SecCompanyFactsPortfolioReviewState.CONCENTRATION_WITHIN_CONTEXT
        message = "Exposure is within explicitly supplied concentration context."
    else:
        state = SecCompanyFactsPortfolioReviewState.CONCENTRATION_REQUIRES_REVIEW
        message = "Exposure exceeds or requires review against explicitly supplied concentration context."

    return _portfolio_review_item(
        recommendation_review,
        portfolio_context,
        category=SecCompanyFactsPortfolioReviewCategory.CONCENTRATION_CONTEXT_REVIEW,
        state=state,
        message=message,
    )


def _portfolio_fit_context_review(
    recommendation_review: SecCompanyFactsRecommendationReview,
    portfolio_context: MarketEnginePortfolioContext,
    context_state: SecCompanyFactsPortfolioReviewState,
) -> SecCompanyFactsPortfolioReviewItem:
    if context_state == SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_SUPPORTED:
        state = SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_SUPPORTED
        message = "Portfolio context is available for non-actionable review."
    elif context_state == SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_STALE:
        state = SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_STALE
        message = "Portfolio context is stale and requires review."
    elif context_state == SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_INVALID:
        state = SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_INVALID
        message = "Portfolio context is invalid and blocks review."
    else:
        state = SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_PARTIAL
        message = "Portfolio context is partial and requires review."

    return _portfolio_review_item(
        recommendation_review,
        portfolio_context,
        category=SecCompanyFactsPortfolioReviewCategory.PORTFOLIO_FIT_CONTEXT_REVIEW,
        state=state,
        message=message,
    )


def _downstream_handoff_readiness_review(
    recommendation_review: SecCompanyFactsRecommendationReview,
    portfolio_context: MarketEnginePortfolioContext,
    context_state: SecCompanyFactsPortfolioReviewState,
) -> SecCompanyFactsPortfolioReviewItem:
    if context_state == SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_SUPPORTED:
        state = (
            SecCompanyFactsPortfolioReviewState.READY_FOR_DECISION_ENGINE_HANDOFF_REVIEW
        )
        message = "Output is structurally ready for a later Decision Engine handoff review."
    else:
        state = SecCompanyFactsPortfolioReviewState.NOT_APPLICABLE
        message = "Decision Engine handoff review is not applicable until portfolio context limitations are resolved."

    return _portfolio_review_item(
        recommendation_review,
        portfolio_context,
        category=(
            SecCompanyFactsPortfolioReviewCategory.DOWNSTREAM_HANDOFF_READINESS_REVIEW
        ),
        state=state,
        message=message,
    )


def _portfolio_data_limitation_review(
    recommendation_review: SecCompanyFactsRecommendationReview,
    portfolio_context: MarketEnginePortfolioContext,
    context_state: SecCompanyFactsPortfolioReviewState,
) -> SecCompanyFactsPortfolioReviewItem | None:
    if context_state == SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_SUPPORTED:
        return None
    if context_state == SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_STALE:
        state = SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_STALE
        message = "Portfolio Review preserves stale portfolio-context fields."
    elif context_state == SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_INVALID:
        state = SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_INVALID
        message = "Portfolio Review is limited by invalid portfolio context."
    else:
        state = SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_PARTIAL
        message = "Portfolio Review preserves partial or missing portfolio-context fields."

    return _portfolio_review_item(
        recommendation_review,
        portfolio_context,
        category=(
            SecCompanyFactsPortfolioReviewCategory.PORTFOLIO_DATA_LIMITATION_REVIEW
        ),
        state=state,
        message=message,
    )


def _missing_portfolio_context_review(
    *,
    recommendation_review: SecCompanyFactsRecommendationReview,
    portfolio_review_run_id: str,
    created_at: str | None,
) -> SecCompanyFactsPortfolioReview:
    item = SecCompanyFactsPortfolioReviewItem(
        category=SecCompanyFactsPortfolioReviewCategory.PORTFOLIO_DATA_LIMITATION_REVIEW,
        state=SecCompanyFactsPortfolioReviewState.BLOCKED_BY_MISSING_PORTFOLIO_CONTEXT,
        message="Portfolio Review is blocked because approved portfolio context is missing.",
        recommendation_review_references=_recommendation_review_reference(
            recommendation_review
        ),
        portfolio_context_references={},
        missing_portfolio_context_fields=("portfolio_context",),
        stale_portfolio_context_fields=(),
        setup_aware_provenance=_setup_aware_provenance(recommendation_review),
        boundary_notes=_standard_boundary_notes(),
    )
    return SecCompanyFactsPortfolioReview(
        ticker=recommendation_review.ticker,
        cik=recommendation_review.cik,
        provider_name=recommendation_review.provider_name,
        portfolio_review_format_version=SEC_COMPANYFACTS_PORTFOLIO_REVIEW_FORMAT_VERSION,
        portfolio_review_run_id=portfolio_review_run_id,
        created_at=created_at,
        recommendation_review_format_version=(
            recommendation_review.recommendation_review_format_version
        ),
        recommendation_review_run_id=recommendation_review.recommendation_review_run_id,
        portfolio_context_format_version=None,
        portfolio_context_run_id=None,
        portfolio_snapshot_timestamp=None,
        portfolio_base_currency=None,
        position_state=MarketEnginePortfolioPositionState.UNKNOWN.value,
        review_state=(
            SecCompanyFactsPortfolioReviewState.BLOCKED_BY_MISSING_PORTFOLIO_CONTEXT
        ),
        review_category=(
            SecCompanyFactsPortfolioReviewCategory.PORTFOLIO_DATA_LIMITATION_REVIEW
        ),
        portfolio_review_items=(item,),
        missing_portfolio_context_fields=("portfolio_context",),
        stale_portfolio_context_fields=(),
        recommendation_review_provenance=_recommendation_review_reference(
            recommendation_review
        ),
        setup_aware_provenance=_setup_aware_provenance(recommendation_review),
        portfolio_context_provenance={},
    )


def _invalid_input_portfolio_review(
    *,
    portfolio_review_run_id: str,
    created_at: str | None,
    reason: str,
    recommendation_review: SecCompanyFactsRecommendationReview | None = None,
    portfolio_context: MarketEnginePortfolioContext | None = None,
) -> SecCompanyFactsPortfolioReview:
    ticker = (
        recommendation_review.ticker
        if recommendation_review is not None
        else portfolio_context.ticker
        if portfolio_context is not None
        else ""
    )
    cik = recommendation_review.cik if recommendation_review is not None else ""
    provider_name = (
        recommendation_review.provider_name if recommendation_review is not None else ""
    )
    item = SecCompanyFactsPortfolioReviewItem(
        category=SecCompanyFactsPortfolioReviewCategory.INPUT_CONTRACT_INVALID,
        state=SecCompanyFactsPortfolioReviewState.BLOCKED_BY_INVALID_INPUT,
        message=reason,
        recommendation_review_references=(
            _recommendation_review_reference(recommendation_review)
            if recommendation_review is not None
            else {}
        ),
        portfolio_context_references=(
            _portfolio_context_reference(portfolio_context)
            if portfolio_context is not None
            else {}
        ),
        missing_portfolio_context_fields=(
            portfolio_context.missing_portfolio_context_fields
            if portfolio_context is not None
            else ()
        ),
        stale_portfolio_context_fields=(
            portfolio_context.stale_portfolio_context_fields
            if portfolio_context is not None
            else ()
        ),
        setup_aware_provenance=(
            _setup_aware_provenance(recommendation_review)
            if recommendation_review is not None
            else {}
        ),
        boundary_notes=_standard_boundary_notes(),
    )
    return SecCompanyFactsPortfolioReview(
        ticker=ticker,
        cik=cik,
        provider_name=provider_name,
        portfolio_review_format_version=SEC_COMPANYFACTS_PORTFOLIO_REVIEW_FORMAT_VERSION,
        portfolio_review_run_id=portfolio_review_run_id,
        created_at=created_at,
        recommendation_review_format_version=(
            recommendation_review.recommendation_review_format_version
            if recommendation_review is not None
            else None
        ),
        recommendation_review_run_id=(
            recommendation_review.recommendation_review_run_id
            if recommendation_review is not None
            else None
        ),
        portfolio_context_format_version=(
            portfolio_context.portfolio_context_format_version
            if portfolio_context is not None
            else None
        ),
        portfolio_context_run_id=(
            portfolio_context.portfolio_context_run_id
            if portfolio_context is not None
            else None
        ),
        portfolio_snapshot_timestamp=(
            portfolio_context.portfolio_snapshot_timestamp
            if portfolio_context is not None
            else None
        ),
        portfolio_base_currency=(
            portfolio_context.portfolio_base_currency
            if portfolio_context is not None
            else None
        ),
        position_state=(
            _position_state_value(portfolio_context.position_state)
            if portfolio_context is not None
            else MarketEnginePortfolioPositionState.UNKNOWN.value
        ),
        review_state=SecCompanyFactsPortfolioReviewState.BLOCKED_BY_INVALID_INPUT,
        review_category=SecCompanyFactsPortfolioReviewCategory.INPUT_CONTRACT_INVALID,
        portfolio_review_items=(item,),
        missing_portfolio_context_fields=(
            portfolio_context.missing_portfolio_context_fields
            if portfolio_context is not None
            else ()
        ),
        stale_portfolio_context_fields=(
            portfolio_context.stale_portfolio_context_fields
            if portfolio_context is not None
            else ()
        ),
        recommendation_review_provenance=(
            _recommendation_review_reference(recommendation_review)
            if recommendation_review is not None
            else {}
        ),
        setup_aware_provenance=(
            _setup_aware_provenance(recommendation_review)
            if recommendation_review is not None
            else {}
        ),
        portfolio_context_provenance=(
            _portfolio_context_reference(portfolio_context)
            if portfolio_context is not None
            else {}
        ),
    )


def _portfolio_review_item(
    recommendation_review: SecCompanyFactsRecommendationReview,
    portfolio_context: MarketEnginePortfolioContext,
    *,
    category: SecCompanyFactsPortfolioReviewCategory,
    state: SecCompanyFactsPortfolioReviewState,
    message: str,
) -> SecCompanyFactsPortfolioReviewItem:
    return SecCompanyFactsPortfolioReviewItem(
        category=category,
        state=state,
        message=message,
        recommendation_review_references=_recommendation_review_reference(
            recommendation_review
        ),
        portfolio_context_references=_portfolio_context_reference(portfolio_context),
        missing_portfolio_context_fields=(
            portfolio_context.missing_portfolio_context_fields
        ),
        stale_portfolio_context_fields=portfolio_context.stale_portfolio_context_fields,
        setup_aware_provenance=_setup_aware_provenance(recommendation_review),
        boundary_notes=_standard_boundary_notes(),
    )


def _recommendation_review_reference(
    recommendation_review: SecCompanyFactsRecommendationReview,
) -> dict[str, Any]:
    return {
        "recommendation_review_format_version": (
            recommendation_review.recommendation_review_format_version
        ),
        "recommendation_review_run_id": (
            recommendation_review.recommendation_review_run_id
        ),
        "ticker": recommendation_review.ticker,
        "cik": recommendation_review.cik,
        "provider_name": recommendation_review.provider_name,
        "review_state": recommendation_review.review_state.value,
        "review_category": recommendation_review.review_category.value,
        "input_provenance": recommendation_review.input_provenance,
        "review_items": [
            _recommendation_review_item_reference(review_item)
            for review_item in recommendation_review.review_items
        ],
        "non_actionable_boundary": recommendation_review.non_actionable_boundary,
    }


def _recommendation_review_item_reference(
    review_item: SecCompanyFactsRecommendationReviewItem,
) -> dict[str, Any]:
    return {
        "category": review_item.category.value,
        "state": review_item.state.value,
        "message": review_item.message,
        "supporting_analysis_review_categories": (
            review_item.supporting_analysis_review_categories
        ),
        "blocking_analysis_review_categories": (
            review_item.blocking_analysis_review_categories
        ),
        "missing_data": review_item.missing_data,
        "analysis_review_references": review_item.analysis_review_references,
        "setup_aware_analysis_review_references": (
            review_item.setup_aware_analysis_review_references
        ),
        "setup_categories": review_item.setup_categories,
        "setup_states": review_item.setup_states,
        "setup_evidence": review_item.setup_evidence,
        "setup_limitations": review_item.setup_limitations,
        "missing_setup_observations": review_item.missing_setup_observations,
        "non_actionable_boundary": review_item.non_actionable_boundary,
    }


def _setup_aware_provenance(
    recommendation_review: SecCompanyFactsRecommendationReview,
) -> dict[str, Any]:
    return {
        "setup_detection_format_version": (
            recommendation_review.setup_detection_format_version
        ),
        "setup_detection_run_id": recommendation_review.setup_detection_run_id,
        "review_items": {
            index: {
                "setup_aware_analysis_review_references": (
                    review_item.setup_aware_analysis_review_references
                ),
                "setup_categories": review_item.setup_categories,
                "setup_states": review_item.setup_states,
                "setup_evidence": review_item.setup_evidence,
                "setup_limitations": review_item.setup_limitations,
                "missing_setup_observations": review_item.missing_setup_observations,
            }
            for index, review_item in enumerate(recommendation_review.review_items)
            if (
                review_item.setup_aware_analysis_review_references
                or review_item.setup_categories
                or review_item.setup_states
                or review_item.setup_evidence
                or review_item.setup_limitations
                or review_item.missing_setup_observations
            )
        },
    }


def _portfolio_context_reference(
    portfolio_context: MarketEnginePortfolioContext,
) -> dict[str, Any]:
    return {
        "portfolio_context_format_version": (
            portfolio_context.portfolio_context_format_version
        ),
        "portfolio_context_run_id": portfolio_context.portfolio_context_run_id,
        "portfolio_snapshot_timestamp": portfolio_context.portfolio_snapshot_timestamp,
        "portfolio_base_currency": portfolio_context.portfolio_base_currency,
        "ticker": portfolio_context.ticker,
        "position_state": _position_state_value(portfolio_context.position_state),
        "current_quantity": portfolio_context.current_quantity,
        "current_market_value": portfolio_context.current_market_value,
        "portfolio_total_value": portfolio_context.portfolio_total_value,
        "current_ticker_exposure_pct": portfolio_context.current_ticker_exposure_pct,
        "exposure_buckets": portfolio_context.exposure_buckets,
        "concentration_thresholds": portfolio_context.concentration_thresholds,
        "policy_constraints": portfolio_context.policy_constraints,
        "missing_portfolio_context_fields": (
            portfolio_context.missing_portfolio_context_fields
        ),
        "stale_portfolio_context_fields": portfolio_context.stale_portfolio_context_fields,
        "context_provenance": portfolio_context.context_provenance,
    }


def _position_state_value(position_state: MarketEnginePortfolioPositionState | str) -> str:
    if isinstance(position_state, MarketEnginePortfolioPositionState):
        return position_state.value
    return position_state


def _is_missing(value: Any) -> bool:
    return value is None


def _standard_boundary_notes() -> tuple[str, ...]:
    return (
        "Portfolio Review is non-actionable.",
        "Decision Engine behavior is outside ME-PR authority.",
        "Delivery and reporting are outside ME-PR authority.",
        "Portfolio Review preserves context and does not mutate portfolio state.",
    )


def _to_jsonable(portfolio_review: SecCompanyFactsPortfolioReview) -> dict[str, Any]:
    payload = asdict(portfolio_review)
    payload["review_state"] = portfolio_review.review_state.value
    payload["review_category"] = portfolio_review.review_category.value
    payload["portfolio_review_items"] = [
        {
            **asdict(review_item),
            "category": review_item.category.value,
            "state": review_item.state.value,
        }
        for review_item in portfolio_review.portfolio_review_items
    ]
    return payload
