from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from market_engine.analysis_review.sec_companyfacts_analysis_review import (
    SEC_COMPANYFACTS_ANALYSIS_REVIEW_FORMAT_VERSION,
    SecCompanyFactsAnalysisReview,
    SecCompanyFactsAnalysisReviewCategory,
    SecCompanyFactsAnalysisReviewItem,
    SecCompanyFactsAnalysisReviewState,
)


SEC_COMPANYFACTS_RECOMMENDATION_REVIEW_FORMAT_VERSION = (
    "sec-companyfacts-recommendation-review-v1"
)
SEC_COMPANYFACTS_RECOMMENDATION_REVIEW_ROOT = Path(
    "data/market_engine/recommendation_reviews"
)

NON_ACTIONABLE_RECOMMENDATION_REVIEW_BOUNDARY = (
    "Recommendation Review creates a non-actionable human-review candidate only "
    "and does not create trade, portfolio, delivery, or Decision Engine authority."
)

REQUIRED_ANALYSIS_REVIEW_FORMAT_VERSION = SEC_COMPANYFACTS_ANALYSIS_REVIEW_FORMAT_VERSION

FORBIDDEN_RECOMMENDATION_REVIEW_ACTIONS = (
    "BUY",
    "SELL",
    "HOLD",
    "ALLOCATION",
    "POSITION_SIZING",
    "EXECUTION",
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


class SecCompanyFactsRecommendationReviewCategory(str, Enum):
    ANALYSIS_SUPPORTIVE_BUT_NOT_ACTIONABLE = "analysis_supportive_but_not_actionable"
    ANALYSIS_MIXED_OR_CONFLICTED = "analysis_mixed_or_conflicted"
    ANALYSIS_BLOCKED_BY_MISSING_DATA = "analysis_blocked_by_missing_data"
    ANALYSIS_NOT_SUPPORTED = "analysis_not_supported"
    INPUT_CONTRACT_INVALID = "input_contract_invalid"


class SecCompanyFactsRecommendationReviewState(str, Enum):
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    BLOCKED_BY_MISSING_DATA = "blocked_by_missing_data"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class SecCompanyFactsRecommendationReviewItem:
    category: SecCompanyFactsRecommendationReviewCategory
    state: SecCompanyFactsRecommendationReviewState
    message: str
    supporting_analysis_review_categories: tuple[str, ...]
    blocking_analysis_review_categories: tuple[str, ...]
    missing_data: tuple[str, ...]
    analysis_review_references: dict[str, dict[str, Any]]
    boundary_notes: tuple[str, ...]
    forbidden_actions: tuple[str, ...] = FORBIDDEN_RECOMMENDATION_REVIEW_ACTIONS
    non_actionable_boundary: str = NON_ACTIONABLE_RECOMMENDATION_REVIEW_BOUNDARY


@dataclass(frozen=True)
class SecCompanyFactsRecommendationReview:
    ticker: str
    cik: str
    provider_name: str
    recommendation_review_format_version: str
    analysis_review_format_version: str
    source_context_format_version: str
    source_context_state: str
    source_refresh_snapshot_id: str
    source_refresh_fetched_at: str
    source_refresh_payload_format_version: str
    recommendation_review_run_id: str
    input_contract: str
    review_state: SecCompanyFactsRecommendationReviewState
    review_category: SecCompanyFactsRecommendationReviewCategory
    review_items: tuple[SecCompanyFactsRecommendationReviewItem, ...]
    input_provenance: dict[str, Any]
    forbidden_actions: tuple[str, ...] = FORBIDDEN_RECOMMENDATION_REVIEW_ACTIONS
    non_actionable_boundary: str = NON_ACTIONABLE_RECOMMENDATION_REVIEW_BOUNDARY
    warnings: tuple[str, ...] = field(default_factory=tuple)


def build_sec_companyfacts_recommendation_review(
    analysis_review: SecCompanyFactsAnalysisReview,
    *,
    recommendation_review_run_id: str,
) -> SecCompanyFactsRecommendationReview:
    _validate_analysis_review_contract(analysis_review)

    analysis_items_by_category = _analysis_review_items_by_category(analysis_review)
    limited_items = _limited_analysis_review_items(analysis_review)
    human_review_item = analysis_items_by_category.get(
        SecCompanyFactsAnalysisReviewCategory.HUMAN_REVIEW_REQUIREMENT
    )

    if limited_items or human_review_item is not None:
        review_state = SecCompanyFactsRecommendationReviewState.BLOCKED_BY_MISSING_DATA
        review_category = (
            SecCompanyFactsRecommendationReviewCategory.ANALYSIS_BLOCKED_BY_MISSING_DATA
        )
        review_item = _blocked_by_missing_data_review_item(
            analysis_review=analysis_review,
            limited_items=limited_items,
            human_review_item=human_review_item,
        )
    elif _has_supportive_analysis_review(analysis_review):
        review_state = SecCompanyFactsRecommendationReviewState.HUMAN_REVIEW_REQUIRED
        review_category = (
            SecCompanyFactsRecommendationReviewCategory.ANALYSIS_SUPPORTIVE_BUT_NOT_ACTIONABLE
        )
        review_item = _supportive_but_not_actionable_review_item(
            analysis_review=analysis_review
        )
    elif analysis_review.review_items:
        review_state = SecCompanyFactsRecommendationReviewState.INSUFFICIENT_EVIDENCE
        review_category = SecCompanyFactsRecommendationReviewCategory.ANALYSIS_NOT_SUPPORTED
        review_item = _insufficient_evidence_review_item(analysis_review=analysis_review)
    else:
        review_state = SecCompanyFactsRecommendationReviewState.NOT_APPLICABLE
        review_category = SecCompanyFactsRecommendationReviewCategory.INPUT_CONTRACT_INVALID
        review_item = _not_applicable_review_item(analysis_review=analysis_review)

    return SecCompanyFactsRecommendationReview(
        ticker=analysis_review.ticker,
        cik=analysis_review.cik,
        provider_name=analysis_review.provider_name,
        recommendation_review_format_version=(
            SEC_COMPANYFACTS_RECOMMENDATION_REVIEW_FORMAT_VERSION
        ),
        analysis_review_format_version=analysis_review.analysis_review_format_version,
        source_context_format_version=analysis_review.source_context_format_version,
        source_context_state=analysis_review.source_context_state,
        source_refresh_snapshot_id=analysis_review.source_refresh_snapshot_id,
        source_refresh_fetched_at=analysis_review.source_refresh_fetched_at,
        source_refresh_payload_format_version=(
            analysis_review.source_refresh_payload_format_version
        ),
        recommendation_review_run_id=recommendation_review_run_id,
        input_contract=REQUIRED_ANALYSIS_REVIEW_FORMAT_VERSION,
        review_state=review_state,
        review_category=review_category,
        review_items=(review_item,),
        input_provenance=_input_provenance(analysis_review),
    )


def persist_sec_companyfacts_recommendation_review(
    recommendation_review: SecCompanyFactsRecommendationReview,
    *,
    run_id: str,
    root_dir: Path | None = None,
) -> Path:
    root = root_dir or SEC_COMPANYFACTS_RECOMMENDATION_REVIEW_ROOT
    recommendation_review_dir = root / run_id / recommendation_review.ticker
    recommendation_review_dir.mkdir(parents=True, exist_ok=True)
    recommendation_review_path = recommendation_review_dir / "recommendation_review.json"

    if recommendation_review_path.exists():
        raise FileExistsError(
            "refusing to overwrite existing SEC CompanyFacts Recommendation Review: "
            f"{recommendation_review_path}"
        )

    recommendation_review_path.write_text(
        json.dumps(_to_jsonable(recommendation_review), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return recommendation_review_path


def _validate_analysis_review_contract(
    analysis_review: SecCompanyFactsAnalysisReview,
) -> None:
    if (
        analysis_review.analysis_review_format_version
        != REQUIRED_ANALYSIS_REVIEW_FORMAT_VERSION
    ):
        raise ValueError(
            "unsupported SEC CompanyFacts Analysis Review contract: "
            f"{analysis_review.analysis_review_format_version}"
        )


def _analysis_review_items_by_category(
    analysis_review: SecCompanyFactsAnalysisReview,
) -> dict[SecCompanyFactsAnalysisReviewCategory, SecCompanyFactsAnalysisReviewItem]:
    return {review_item.category: review_item for review_item in analysis_review.review_items}


def _limited_analysis_review_items(
    analysis_review: SecCompanyFactsAnalysisReview,
) -> tuple[SecCompanyFactsAnalysisReviewItem, ...]:
    limited_states = {
        SecCompanyFactsAnalysisReviewState.SOURCE_LIMITED,
        SecCompanyFactsAnalysisReviewState.OBSERVATIONS_LIMITED,
        SecCompanyFactsAnalysisReviewState.DATA_LIMITED,
        SecCompanyFactsAnalysisReviewState.REQUIRES_HUMAN_REVIEW,
        SecCompanyFactsAnalysisReviewState.NOT_ASSESSED,
    }

    return tuple(
        review_item
        for review_item in analysis_review.review_items
        if review_item.state in limited_states
    )


def _has_supportive_analysis_review(
    analysis_review: SecCompanyFactsAnalysisReview,
) -> bool:
    supportive_states = {
        SecCompanyFactsAnalysisReviewState.SOURCE_HEALTHY,
        SecCompanyFactsAnalysisReviewState.OBSERVATIONS_COMPLETE,
        SecCompanyFactsAnalysisReviewState.CASH_GENERATION_POSITIVE,
        SecCompanyFactsAnalysisReviewState.CASH_GENERATION_NEUTRAL,
    }

    return any(review_item.state in supportive_states for review_item in analysis_review.review_items)


def _blocked_by_missing_data_review_item(
    *,
    analysis_review: SecCompanyFactsAnalysisReview,
    limited_items: tuple[SecCompanyFactsAnalysisReviewItem, ...],
    human_review_item: SecCompanyFactsAnalysisReviewItem | None,
) -> SecCompanyFactsRecommendationReviewItem:
    blocking_items = limited_items
    if human_review_item is not None and human_review_item not in blocking_items:
        blocking_items = blocking_items + (human_review_item,)

    missing_data = tuple(
        sorted(
            {
                missing_observation
                for review_item in blocking_items
                for missing_observation in review_item.missing_observations
            }
        )
    )

    if not missing_data and blocking_items:
        missing_data = tuple(review_item.category.value for review_item in blocking_items)

    return SecCompanyFactsRecommendationReviewItem(
        category=(
            SecCompanyFactsRecommendationReviewCategory.ANALYSIS_BLOCKED_BY_MISSING_DATA
        ),
        state=SecCompanyFactsRecommendationReviewState.BLOCKED_BY_MISSING_DATA,
        message=(
            "Recommendation Review is blocked because the upstream Analysis Review "
            "contains limited, missing, or human-review-required evidence."
        ),
        supporting_analysis_review_categories=_supporting_category_names(analysis_review),
        blocking_analysis_review_categories=tuple(
            review_item.category.value for review_item in blocking_items
        ),
        missing_data=missing_data,
        analysis_review_references=_analysis_review_references(blocking_items),
        boundary_notes=_standard_boundary_notes(),
    )


def _supportive_but_not_actionable_review_item(
    *,
    analysis_review: SecCompanyFactsAnalysisReview,
) -> SecCompanyFactsRecommendationReviewItem:
    supportive_items = tuple(
        review_item
        for review_item in analysis_review.review_items
        if review_item.state
        in {
            SecCompanyFactsAnalysisReviewState.SOURCE_HEALTHY,
            SecCompanyFactsAnalysisReviewState.OBSERVATIONS_COMPLETE,
            SecCompanyFactsAnalysisReviewState.CASH_GENERATION_POSITIVE,
            SecCompanyFactsAnalysisReviewState.CASH_GENERATION_NEUTRAL,
        }
    )

    return SecCompanyFactsRecommendationReviewItem(
        category=(
            SecCompanyFactsRecommendationReviewCategory.ANALYSIS_SUPPORTIVE_BUT_NOT_ACTIONABLE
        ),
        state=SecCompanyFactsRecommendationReviewState.HUMAN_REVIEW_REQUIRED,
        message=(
            "Analysis Review supports a non-actionable human-review candidate. "
            "Portfolio Review and Decision Engine checks are required before any action."
        ),
        supporting_analysis_review_categories=tuple(
            review_item.category.value for review_item in supportive_items
        ),
        blocking_analysis_review_categories=(),
        missing_data=(),
        analysis_review_references=_analysis_review_references(supportive_items),
        boundary_notes=_standard_boundary_notes(),
    )


def _insufficient_evidence_review_item(
    *,
    analysis_review: SecCompanyFactsAnalysisReview,
) -> SecCompanyFactsRecommendationReviewItem:
    return SecCompanyFactsRecommendationReviewItem(
        category=SecCompanyFactsRecommendationReviewCategory.ANALYSIS_NOT_SUPPORTED,
        state=SecCompanyFactsRecommendationReviewState.INSUFFICIENT_EVIDENCE,
        message=(
            "Analysis Review does not provide enough evidence to create a useful "
            "Recommendation Review candidate."
        ),
        supporting_analysis_review_categories=_supporting_category_names(analysis_review),
        blocking_analysis_review_categories=(),
        missing_data=(),
        analysis_review_references=_analysis_review_references(analysis_review.review_items),
        boundary_notes=_standard_boundary_notes(),
    )


def _not_applicable_review_item(
    *,
    analysis_review: SecCompanyFactsAnalysisReview,
) -> SecCompanyFactsRecommendationReviewItem:
    return SecCompanyFactsRecommendationReviewItem(
        category=SecCompanyFactsRecommendationReviewCategory.INPUT_CONTRACT_INVALID,
        state=SecCompanyFactsRecommendationReviewState.NOT_APPLICABLE,
        message=(
            "Recommendation Review is not applicable because no Analysis Review items "
            "are available for review."
        ),
        supporting_analysis_review_categories=(),
        blocking_analysis_review_categories=(),
        missing_data=(),
        analysis_review_references={},
        boundary_notes=_standard_boundary_notes(),
    )


def _supporting_category_names(
    analysis_review: SecCompanyFactsAnalysisReview,
) -> tuple[str, ...]:
    supportive_states = {
        SecCompanyFactsAnalysisReviewState.SOURCE_HEALTHY,
        SecCompanyFactsAnalysisReviewState.OBSERVATIONS_COMPLETE,
        SecCompanyFactsAnalysisReviewState.CASH_GENERATION_POSITIVE,
        SecCompanyFactsAnalysisReviewState.CASH_GENERATION_NEUTRAL,
    }

    return tuple(
        review_item.category.value
        for review_item in analysis_review.review_items
        if review_item.state in supportive_states
    )


def _analysis_review_references(
    review_items: tuple[SecCompanyFactsAnalysisReviewItem, ...],
) -> dict[str, dict[str, Any]]:
    return {
        review_item.category.value: _analysis_review_item_reference(review_item)
        for review_item in review_items
    }


def _analysis_review_item_reference(
    review_item: SecCompanyFactsAnalysisReviewItem,
) -> dict[str, Any]:
    return {
        "category": review_item.category.value,
        "state": review_item.state.value,
        "message": review_item.message,
        "input_observation_families": review_item.input_observation_families,
        "required_observations": review_item.required_observations,
        "missing_observations": review_item.missing_observations,
        "source_observation_references": review_item.source_observation_references,
        "derived_observation_references": review_item.derived_observation_references,
        "non_recommendation_boundary": review_item.non_recommendation_boundary,
    }


def _input_provenance(analysis_review: SecCompanyFactsAnalysisReview) -> dict[str, Any]:
    return {
        "input_contract": analysis_review.analysis_review_format_version,
        "ticker": analysis_review.ticker,
        "cik": analysis_review.cik,
        "provider_name": analysis_review.provider_name,
        "source_context_format_version": analysis_review.source_context_format_version,
        "source_context_state": analysis_review.source_context_state,
        "source_refresh_snapshot_id": analysis_review.source_refresh_snapshot_id,
        "source_refresh_fetched_at": analysis_review.source_refresh_fetched_at,
        "source_refresh_payload_format_version": (
            analysis_review.source_refresh_payload_format_version
        ),
    }


def _standard_boundary_notes() -> tuple[str, ...]:
    return (
        "Recommendation Review is non-actionable.",
        "Portfolio Review is outside ME-RR authority.",
        "Decision Engine behavior is outside ME-RR authority.",
        "External communication channels are outside ME-RR authority.",
        "Restricted action-authority outputs are represented only as structured forbidden actions, not as review guidance.",
    )


def _to_jsonable(
    recommendation_review: SecCompanyFactsRecommendationReview,
) -> dict[str, Any]:
    payload = asdict(recommendation_review)
    payload["review_state"] = recommendation_review.review_state.value
    payload["review_category"] = recommendation_review.review_category.value
    payload["review_items"] = [
        {
            **asdict(review_item),
            "category": review_item.category.value,
            "state": review_item.state.value,
        }
        for review_item in recommendation_review.review_items
    ]
    return payload