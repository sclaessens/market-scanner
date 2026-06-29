from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from market_engine.analysis_review.company_profile_analysis_context import (
    COMPANY_PROFILE_ANALYSIS_CONTEXT_FORMAT_VERSION,
    CompanyProfileAnalysisContext,
)
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
COMPANY_PROFILE_ONLY_CONTEXT_NON_ACTIONABLE = (
    "company_profile_only_context_non_actionable"
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
    COMPANY_PROFILE_ONLY_CONTEXT_NON_ACTIONABLE = (
        COMPANY_PROFILE_ONLY_CONTEXT_NON_ACTIONABLE
    )
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
    setup_aware_analysis_review_references: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    setup_categories: tuple[str, ...] = field(default_factory=tuple)
    setup_states: tuple[str, ...] = field(default_factory=tuple)
    setup_evidence: dict[str, Any] = field(default_factory=dict)
    setup_limitations: tuple[str, ...] = field(default_factory=tuple)
    missing_setup_observations: tuple[str, ...] = field(default_factory=tuple)
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
    setup_detection_format_version: str | None
    setup_detection_run_id: str | None
    review_state: SecCompanyFactsRecommendationReviewState
    review_category: SecCompanyFactsRecommendationReviewCategory
    review_items: tuple[SecCompanyFactsRecommendationReviewItem, ...]
    input_provenance: dict[str, Any]
    forbidden_actions: tuple[str, ...] = FORBIDDEN_RECOMMENDATION_REVIEW_ACTIONS
    non_actionable_boundary: str = NON_ACTIONABLE_RECOMMENDATION_REVIEW_BOUNDARY
    blocked_reasons: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


def build_sec_companyfacts_recommendation_review(
    analysis_review: SecCompanyFactsAnalysisReview,
    *,
    recommendation_review_run_id: str,
    company_profile_context: CompanyProfileAnalysisContext | None = None,
) -> SecCompanyFactsRecommendationReview:
    _validate_analysis_review_contract(analysis_review)
    if company_profile_context is not None:
        _validate_company_profile_context(
            company_profile_context,
            expected_ticker=analysis_review.ticker,
        )

    analysis_items_by_category = _analysis_review_items_by_category(analysis_review)
    setup_aware_items = _setup_aware_analysis_review_items(analysis_review)
    limited_items = _limited_analysis_review_items(analysis_review)
    human_review_item = analysis_items_by_category.get(
        SecCompanyFactsAnalysisReviewCategory.HUMAN_REVIEW_REQUIREMENT
    )

    if setup_aware_items:
        review_state, review_category = _setup_aware_review_state_and_category(
            setup_aware_items
        )
        review_item = _setup_aware_recommendation_review_item(
            analysis_review=analysis_review,
            setup_aware_items=setup_aware_items,
            review_state=review_state,
            review_category=review_category,
        )
    elif limited_items or human_review_item is not None:
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

    input_provenance = _input_provenance(analysis_review)
    if company_profile_context is not None:
        input_provenance["company_profile_context"] = (
            _company_profile_analysis_context_reference(company_profile_context)
        )

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
        setup_detection_format_version=getattr(
            analysis_review,
            "setup_detection_format_version",
            None,
        ),
        setup_detection_run_id=getattr(analysis_review, "setup_detection_run_id", None),
        review_state=review_state,
        review_category=review_category,
        review_items=(review_item,),
        input_provenance=input_provenance,
    )


def build_company_profile_only_recommendation_review(
    analysis_context: CompanyProfileAnalysisContext,
    *,
    recommendation_review_run_id: str,
) -> SecCompanyFactsRecommendationReview:
    _validate_company_profile_context(
        analysis_context,
        expected_ticker=analysis_context.ticker,
    )
    review_category = (
        SecCompanyFactsRecommendationReviewCategory.COMPANY_PROFILE_ONLY_CONTEXT_NON_ACTIONABLE
    )
    review_state = SecCompanyFactsRecommendationReviewState.BLOCKED_BY_MISSING_DATA
    review_item = SecCompanyFactsRecommendationReviewItem(
        category=review_category,
        state=review_state,
        message=(
            "Recommendation Review is non-actionable because company profile "
            "context is descriptive only and contains no sufficient fundamental, "
            "financial-market, valuation, or setup evidence."
        ),
        supporting_analysis_review_categories=(),
        blocking_analysis_review_categories=("company_profile_only_context",),
        missing_data=(
            "fundamental_financial_evidence",
            "financial_market_evidence",
            "setup_evidence",
        ),
        analysis_review_references={
            "company_profile_context": (
                _company_profile_analysis_context_reference(analysis_context)
            )
        },
        boundary_notes=(
            *_standard_boundary_notes(),
            "Company profile metadata is descriptive context, not recommendation evidence.",
            "Downstream handoff remains blocked until sufficient non-profile analysis evidence exists.",
        ),
    )
    return SecCompanyFactsRecommendationReview(
        ticker=analysis_context.ticker,
        cik="",
        provider_name=analysis_context.provider_name,
        recommendation_review_format_version=(
            SEC_COMPANYFACTS_RECOMMENDATION_REVIEW_FORMAT_VERSION
        ),
        analysis_review_format_version=(
            analysis_context.analysis_review_format_version
        ),
        source_context_format_version=(
            "market-engine-company-profile-source-context-v1"
        ),
        source_context_state="consumed",
        source_refresh_snapshot_id=analysis_context.source_refresh_snapshot_id,
        source_refresh_fetched_at=analysis_context.source_refresh_fetched_at or "",
        source_refresh_payload_format_version=(
            analysis_context.source_refresh_payload_format_version
        ),
        recommendation_review_run_id=recommendation_review_run_id,
        input_contract=COMPANY_PROFILE_ANALYSIS_CONTEXT_FORMAT_VERSION,
        setup_detection_format_version=analysis_context.setup_boundary_format_version,
        setup_detection_run_id=None,
        review_state=review_state,
        review_category=review_category,
        review_items=(review_item,),
        input_provenance=_company_profile_analysis_context_reference(
            analysis_context
        ),
        blocked_reasons=(COMPANY_PROFILE_ONLY_CONTEXT_NON_ACTIONABLE,),
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


def _validate_company_profile_context(
    analysis_context: CompanyProfileAnalysisContext,
    *,
    expected_ticker: str,
) -> None:
    if (
        analysis_context.analysis_review_format_version
        != COMPANY_PROFILE_ANALYSIS_CONTEXT_FORMAT_VERSION
    ):
        raise ValueError(
            "unsupported Company Profile Analysis Review contract: "
            f"{analysis_context.analysis_review_format_version}"
        )
    if analysis_context.input_family != "company_profile":
        raise ValueError(
            "Company Profile Recommendation Review requires company_profile input."
        )
    if analysis_context.context_state != "descriptive_context_available":
        raise ValueError(
            "Company Profile Recommendation Review requires available descriptive context."
        )
    if analysis_context.ticker.upper() != expected_ticker.upper():
        raise ValueError(
            "Company Profile Recommendation Review ticker alignment failed."
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


def _setup_aware_analysis_review_items(
    analysis_review: SecCompanyFactsAnalysisReview,
) -> tuple[SecCompanyFactsAnalysisReviewItem, ...]:
    setup_categories = {
        SecCompanyFactsAnalysisReviewCategory.SETUP_DETECTION_REVIEW,
        SecCompanyFactsAnalysisReviewCategory.SETUP_EVIDENCE_COMPLETENESS_REVIEW,
        SecCompanyFactsAnalysisReviewCategory.SETUP_LIMITATION_REVIEW,
        SecCompanyFactsAnalysisReviewCategory.SETUP_HUMAN_REVIEW_REQUIREMENT,
    }
    setup_states = {
        SecCompanyFactsAnalysisReviewState.SETUP_DETECTED,
        SecCompanyFactsAnalysisReviewState.SETUP_PARTIALLY_DETECTED,
        SecCompanyFactsAnalysisReviewState.SETUP_NOT_DETECTED,
        SecCompanyFactsAnalysisReviewState.SETUP_CONFLICTED,
        SecCompanyFactsAnalysisReviewState.SETUP_BLOCKED_BY_MISSING_DATA,
        SecCompanyFactsAnalysisReviewState.SETUP_NOT_ASSESSED,
        SecCompanyFactsAnalysisReviewState.SETUP_REQUIRES_HUMAN_REVIEW,
    }
    return tuple(
        review_item
        for review_item in analysis_review.review_items
        if review_item.category in setup_categories or review_item.state in setup_states
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


def _setup_aware_review_state_and_category(
    setup_aware_items: tuple[SecCompanyFactsAnalysisReviewItem, ...],
) -> tuple[
    SecCompanyFactsRecommendationReviewState,
    SecCompanyFactsRecommendationReviewCategory,
]:
    states = {review_item.state for review_item in setup_aware_items}

    if SecCompanyFactsAnalysisReviewState.SETUP_BLOCKED_BY_MISSING_DATA in states:
        return (
            SecCompanyFactsRecommendationReviewState.BLOCKED_BY_MISSING_DATA,
            SecCompanyFactsRecommendationReviewCategory.ANALYSIS_BLOCKED_BY_MISSING_DATA,
        )

    if states & {
        SecCompanyFactsAnalysisReviewState.SETUP_DETECTED,
        SecCompanyFactsAnalysisReviewState.SETUP_PARTIALLY_DETECTED,
        SecCompanyFactsAnalysisReviewState.SETUP_CONFLICTED,
        SecCompanyFactsAnalysisReviewState.SETUP_REQUIRES_HUMAN_REVIEW,
    }:
        if SecCompanyFactsAnalysisReviewState.SETUP_DETECTED in states and not (
            states
            & {
                SecCompanyFactsAnalysisReviewState.SETUP_PARTIALLY_DETECTED,
                SecCompanyFactsAnalysisReviewState.SETUP_CONFLICTED,
                SecCompanyFactsAnalysisReviewState.SETUP_REQUIRES_HUMAN_REVIEW,
            }
        ):
            return (
                SecCompanyFactsRecommendationReviewState.HUMAN_REVIEW_REQUIRED,
                SecCompanyFactsRecommendationReviewCategory.ANALYSIS_SUPPORTIVE_BUT_NOT_ACTIONABLE,
            )
        return (
            SecCompanyFactsRecommendationReviewState.HUMAN_REVIEW_REQUIRED,
            SecCompanyFactsRecommendationReviewCategory.ANALYSIS_MIXED_OR_CONFLICTED,
        )

    if states & {
        SecCompanyFactsAnalysisReviewState.SETUP_NOT_DETECTED,
        SecCompanyFactsAnalysisReviewState.SETUP_NOT_ASSESSED,
    }:
        return (
            SecCompanyFactsRecommendationReviewState.INSUFFICIENT_EVIDENCE,
            SecCompanyFactsRecommendationReviewCategory.ANALYSIS_NOT_SUPPORTED,
        )

    return (
        SecCompanyFactsRecommendationReviewState.INSUFFICIENT_EVIDENCE,
        SecCompanyFactsRecommendationReviewCategory.ANALYSIS_NOT_SUPPORTED,
    )


def _setup_aware_recommendation_review_item(
    *,
    analysis_review: SecCompanyFactsAnalysisReview,
    setup_aware_items: tuple[SecCompanyFactsAnalysisReviewItem, ...],
    review_state: SecCompanyFactsRecommendationReviewState,
    review_category: SecCompanyFactsRecommendationReviewCategory,
) -> SecCompanyFactsRecommendationReviewItem:
    missing_setup_observations = _missing_setup_observations(setup_aware_items)
    return SecCompanyFactsRecommendationReviewItem(
        category=review_category,
        state=review_state,
        message=_setup_aware_recommendation_message(review_state, review_category),
        supporting_analysis_review_categories=_supporting_category_names(analysis_review),
        blocking_analysis_review_categories=_setup_blocking_category_names(
            setup_aware_items
        ),
        missing_data=missing_setup_observations,
        analysis_review_references=_analysis_review_references(setup_aware_items),
        boundary_notes=_standard_boundary_notes(),
        setup_aware_analysis_review_references=_analysis_review_references(
            setup_aware_items
        ),
        setup_categories=_setup_categories(setup_aware_items),
        setup_states=_setup_states(setup_aware_items),
        setup_evidence=_setup_evidence(setup_aware_items),
        setup_limitations=_setup_limitations(setup_aware_items),
        missing_setup_observations=missing_setup_observations,
    )


def _setup_aware_recommendation_message(
    review_state: SecCompanyFactsRecommendationReviewState,
    review_category: SecCompanyFactsRecommendationReviewCategory,
) -> str:
    if (
        review_state
        == SecCompanyFactsRecommendationReviewState.BLOCKED_BY_MISSING_DATA
    ):
        return (
            "Recommendation Review is blocked by missing setup evidence from Setup-aware Analysis Review."
        )
    if (
        review_category
        == SecCompanyFactsRecommendationReviewCategory.ANALYSIS_SUPPORTIVE_BUT_NOT_ACTIONABLE
    ):
        return (
            "Setup-aware Analysis Review supports human review and remains non-actionable."
        )
    if (
        review_category
        == SecCompanyFactsRecommendationReviewCategory.ANALYSIS_MIXED_OR_CONFLICTED
    ):
        return (
            "Setup-aware Analysis Review is partial, conflicted, or requires human review."
        )
    return (
        "Setup-aware Analysis Review does not provide enough setup evidence for a useful Recommendation Review candidate."
    )


def _setup_blocking_category_names(
    setup_aware_items: tuple[SecCompanyFactsAnalysisReviewItem, ...],
) -> tuple[str, ...]:
    blocking_states = {
        SecCompanyFactsAnalysisReviewState.SETUP_PARTIALLY_DETECTED,
        SecCompanyFactsAnalysisReviewState.SETUP_CONFLICTED,
        SecCompanyFactsAnalysisReviewState.SETUP_BLOCKED_BY_MISSING_DATA,
        SecCompanyFactsAnalysisReviewState.SETUP_NOT_ASSESSED,
        SecCompanyFactsAnalysisReviewState.SETUP_REQUIRES_HUMAN_REVIEW,
    }
    return tuple(
        review_item.category.value
        for review_item in setup_aware_items
        if review_item.state in blocking_states or review_item.missing_observations
    )


def _missing_setup_observations(
    setup_aware_items: tuple[SecCompanyFactsAnalysisReviewItem, ...],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                missing_observation
                for review_item in setup_aware_items
                for missing_observation in (
                    tuple(review_item.missing_observations)
                    + tuple(getattr(review_item, "missing_setup_observations", ()))
                )
            }
        )
    )


def _setup_categories(
    setup_aware_items: tuple[SecCompanyFactsAnalysisReviewItem, ...],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            setup_category
            for review_item in setup_aware_items
            for setup_category in review_item.setup_categories
        )
    )


def _setup_states(
    setup_aware_items: tuple[SecCompanyFactsAnalysisReviewItem, ...],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            setup_state
            for review_item in setup_aware_items
            for setup_state in review_item.setup_states
        )
    )


def _setup_evidence(
    setup_aware_items: tuple[SecCompanyFactsAnalysisReviewItem, ...],
) -> dict[str, Any]:
    return {
        setup_category or review_item.category.value: review_item.setup_evidence
        for review_item in setup_aware_items
        for setup_category in (review_item.setup_categories or (review_item.category.value,))
        if review_item.setup_evidence
    }


def _setup_limitations(
    setup_aware_items: tuple[SecCompanyFactsAnalysisReviewItem, ...],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            limitation
            for review_item in setup_aware_items
            for limitation in review_item.setup_limitations
        )
    )


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
        "setup_detection_references": getattr(
            review_item,
            "setup_detection_references",
            {},
        ),
        "setup_categories": getattr(review_item, "setup_categories", ()),
        "setup_states": getattr(review_item, "setup_states", ()),
        "setup_evidence": getattr(review_item, "setup_evidence", {}),
        "setup_limitations": getattr(review_item, "setup_limitations", ()),
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
        "setup_detection_format_version": getattr(
            analysis_review,
            "setup_detection_format_version",
            None,
        ),
        "setup_detection_run_id": getattr(analysis_review, "setup_detection_run_id", None),
        "setup_detection_non_actionable_boundary": getattr(
            analysis_review,
            "setup_detection_non_actionable_boundary",
            None,
        ),
    }


def _company_profile_analysis_context_reference(
    analysis_context: CompanyProfileAnalysisContext,
) -> dict[str, Any]:
    return {
        "input_contract": analysis_context.analysis_review_format_version,
        "analysis_review_run_id": analysis_context.analysis_review_run_id,
        "input_family": analysis_context.input_family,
        "context_state": analysis_context.context_state,
        "ticker": analysis_context.ticker,
        "symbol": analysis_context.symbol,
        "provider_name": analysis_context.provider_name,
        "source_observation_format_version": (
            analysis_context.source_observation_format_version
        ),
        "source_bridge_format_version": analysis_context.source_bridge_format_version,
        "setup_boundary_format_version": (
            analysis_context.setup_boundary_format_version
        ),
        "source_refresh_snapshot_id": analysis_context.source_refresh_snapshot_id,
        "source_refresh_fetched_at": analysis_context.source_refresh_fetched_at,
        "source_refresh_payload_format_version": (
            analysis_context.source_refresh_payload_format_version
        ),
        "as_of": analysis_context.as_of,
        "provenance": dict(analysis_context.provenance),
        "descriptive_context": tuple(
            asdict(item) for item in analysis_context.descriptive_context
        ),
        "non_advisory_boundary": analysis_context.non_advisory_boundary,
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
