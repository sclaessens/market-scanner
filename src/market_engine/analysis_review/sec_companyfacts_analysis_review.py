from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from market_engine.derived_observations.sec_companyfacts_cash_generation import (
    SecCompanyFactsDerivedCashGenerationCategory,
    SecCompanyFactsDerivedCashGenerationObservation,
    SecCompanyFactsDerivedCashGenerationObservationSet,
    SecCompanyFactsDerivedCashGenerationState,
)
from market_engine.fundamental_observations.sec_companyfacts_observations import (
    SecCompanyFactsFundamentalObservation,
    SecCompanyFactsFundamentalObservationCategory,
    SecCompanyFactsFundamentalObservationSet,
    SecCompanyFactsFundamentalObservationState,
)
from market_engine.setup_detection.sec_companyfacts_setup_detection import (
    SEC_COMPANYFACTS_SETUP_DETECTION_FORMAT_VERSION,
    SecCompanyFactsSetupCategory,
    SecCompanyFactsSetupDetection,
    SecCompanyFactsSetupDetectionItem,
    SecCompanyFactsSetupState,
)


SEC_COMPANYFACTS_ANALYSIS_REVIEW_FORMAT_VERSION = "sec-companyfacts-analysis-review-v1"
SEC_COMPANYFACTS_ANALYSIS_REVIEW_ROOT = Path("data/market_engine/analysis_reviews")

NON_RECOMMENDATION_ANALYSIS_REVIEW_BOUNDARY = (
    "Analysis Review describes approved upstream observations only and does not create action authority."
)


class SecCompanyFactsAnalysisReviewCategory(str, Enum):
    SOURCE_AVAILABILITY_REVIEW = "SOURCE_AVAILABILITY_REVIEW"
    FUNDAMENTAL_OBSERVATION_COMPLETENESS_REVIEW = (
        "FUNDAMENTAL_OBSERVATION_COMPLETENESS_REVIEW"
    )
    CASH_GENERATION_REVIEW = "CASH_GENERATION_REVIEW"
    FREE_CASH_FLOW_REVIEW = "FREE_CASH_FLOW_REVIEW"
    DATA_LIMITATION_REVIEW = "DATA_LIMITATION_REVIEW"
    HUMAN_REVIEW_REQUIREMENT = "HUMAN_REVIEW_REQUIREMENT"
    SETUP_DETECTION_REVIEW = "SETUP_DETECTION_REVIEW"
    SETUP_EVIDENCE_COMPLETENESS_REVIEW = "SETUP_EVIDENCE_COMPLETENESS_REVIEW"
    SETUP_LIMITATION_REVIEW = "SETUP_LIMITATION_REVIEW"
    SETUP_HUMAN_REVIEW_REQUIREMENT = "SETUP_HUMAN_REVIEW_REQUIREMENT"


class SecCompanyFactsAnalysisReviewState(str, Enum):
    SOURCE_HEALTHY = "SOURCE_HEALTHY"
    SOURCE_LIMITED = "SOURCE_LIMITED"
    OBSERVATIONS_COMPLETE = "OBSERVATIONS_COMPLETE"
    OBSERVATIONS_LIMITED = "OBSERVATIONS_LIMITED"
    CASH_GENERATION_POSITIVE = "CASH_GENERATION_POSITIVE"
    CASH_GENERATION_NEGATIVE = "CASH_GENERATION_NEGATIVE"
    CASH_GENERATION_NEUTRAL = "CASH_GENERATION_NEUTRAL"
    DATA_LIMITED = "DATA_LIMITED"
    REQUIRES_HUMAN_REVIEW = "REQUIRES_HUMAN_REVIEW"
    NOT_ASSESSED = "NOT_ASSESSED"
    SETUP_DETECTED = "SETUP_DETECTED"
    SETUP_PARTIALLY_DETECTED = "SETUP_PARTIALLY_DETECTED"
    SETUP_NOT_DETECTED = "SETUP_NOT_DETECTED"
    SETUP_CONFLICTED = "SETUP_CONFLICTED"
    SETUP_BLOCKED_BY_MISSING_DATA = "SETUP_BLOCKED_BY_MISSING_DATA"
    SETUP_NOT_ASSESSED = "SETUP_NOT_ASSESSED"
    SETUP_REQUIRES_HUMAN_REVIEW = "SETUP_REQUIRES_HUMAN_REVIEW"


@dataclass(frozen=True)
class SecCompanyFactsAnalysisReviewItem:
    category: SecCompanyFactsAnalysisReviewCategory
    state: SecCompanyFactsAnalysisReviewState
    message: str
    input_observation_families: tuple[str, ...]
    required_observations: tuple[str, ...]
    missing_observations: tuple[str, ...]
    source_observation_references: dict[str, dict[str, Any]]
    derived_observation_references: dict[str, dict[str, Any]]
    setup_detection_references: dict[str, dict[str, Any]] = field(default_factory=dict)
    setup_categories: tuple[str, ...] = field(default_factory=tuple)
    setup_states: tuple[str, ...] = field(default_factory=tuple)
    setup_evidence: dict[str, Any] = field(default_factory=dict)
    setup_limitations: tuple[str, ...] = field(default_factory=tuple)
    non_recommendation_boundary: str = NON_RECOMMENDATION_ANALYSIS_REVIEW_BOUNDARY


@dataclass(frozen=True)
class SecCompanyFactsAnalysisReview:
    ticker: str
    cik: str
    provider_name: str
    analysis_review_format_version: str
    fundamental_observation_format_version: str
    derived_observation_format_version: str
    source_context_format_version: str
    source_context_state: str
    source_refresh_snapshot_id: str
    source_refresh_fetched_at: str
    source_refresh_payload_format_version: str
    review_items: tuple[SecCompanyFactsAnalysisReviewItem, ...]
    setup_detection_format_version: str | None = None
    setup_detection_run_id: str | None = None
    setup_detection_non_actionable_boundary: str | None = None
    non_recommendation_boundary: str = NON_RECOMMENDATION_ANALYSIS_REVIEW_BOUNDARY
    warnings: tuple[str, ...] = field(default_factory=tuple)


def build_sec_companyfacts_analysis_review(
    fundamental_observation_set: SecCompanyFactsFundamentalObservationSet,
    derived_cash_generation_observation_set: SecCompanyFactsDerivedCashGenerationObservationSet,
    setup_detection: SecCompanyFactsSetupDetection | None = None,
) -> SecCompanyFactsAnalysisReview:
    _validate_observation_set_alignment(
        fundamental_observation_set,
        derived_cash_generation_observation_set,
    )
    if setup_detection is not None:
        _validate_setup_detection_alignment(
            fundamental_observation_set,
            derived_cash_generation_observation_set,
            setup_detection,
        )

    fundamental_by_category = _fundamental_observations_by_category(
        fundamental_observation_set
    )
    derived_by_category = _derived_observations_by_category(
        derived_cash_generation_observation_set
    )

    review_items = [
        _source_availability_review(
            fundamental_observation_set=fundamental_observation_set,
            fundamental_by_category=fundamental_by_category,
        ),
        _fundamental_observation_completeness_review(
            fundamental_observation_set=fundamental_observation_set,
            fundamental_by_category=fundamental_by_category,
        ),
        _cash_generation_review(
            derived_cash_generation_observation_set=derived_cash_generation_observation_set,
            derived_by_category=derived_by_category,
        ),
        _free_cash_flow_review(
            derived_cash_generation_observation_set=derived_cash_generation_observation_set,
            derived_by_category=derived_by_category,
        ),
    ]

    data_limitation_review = _data_limitation_review(
        fundamental_observation_set=fundamental_observation_set,
        derived_cash_generation_observation_set=derived_cash_generation_observation_set,
        fundamental_by_category=fundamental_by_category,
        derived_by_category=derived_by_category,
    )
    if data_limitation_review is not None:
        review_items.append(data_limitation_review)

    human_review_requirement = _human_review_requirement(
        fundamental_observation_set=fundamental_observation_set,
        derived_cash_generation_observation_set=derived_cash_generation_observation_set,
        fundamental_by_category=fundamental_by_category,
        derived_by_category=derived_by_category,
    )
    if human_review_requirement is not None:
        review_items.append(human_review_requirement)

    if setup_detection is not None:
        review_items.extend(_setup_detection_review_items(setup_detection))

    return SecCompanyFactsAnalysisReview(
        ticker=fundamental_observation_set.ticker,
        cik=fundamental_observation_set.cik,
        provider_name=fundamental_observation_set.provider_name,
        analysis_review_format_version=SEC_COMPANYFACTS_ANALYSIS_REVIEW_FORMAT_VERSION,
        fundamental_observation_format_version=(
            fundamental_observation_set.observation_format_version
        ),
        derived_observation_format_version=(
            derived_cash_generation_observation_set.derived_observation_format_version
        ),
        source_context_format_version=fundamental_observation_set.source_context_format_version,
        source_context_state=fundamental_observation_set.source_context_state,
        source_refresh_snapshot_id=fundamental_observation_set.source_refresh_snapshot_id,
        source_refresh_fetched_at=fundamental_observation_set.source_refresh_fetched_at,
        source_refresh_payload_format_version=(
            fundamental_observation_set.source_refresh_payload_format_version
        ),
        review_items=tuple(review_items),
        setup_detection_format_version=(
            setup_detection.setup_detection_format_version
            if setup_detection is not None
            else None
        ),
        setup_detection_run_id=(
            setup_detection.setup_detection_run_id
            if setup_detection is not None
            else None
        ),
        setup_detection_non_actionable_boundary=(
            setup_detection.non_actionable_boundary
            if setup_detection is not None
            else None
        ),
    )


def persist_sec_companyfacts_analysis_review(
    analysis_review: SecCompanyFactsAnalysisReview,
    *,
    run_id: str,
    root_dir: Path | None = None,
) -> Path:
    root = root_dir or SEC_COMPANYFACTS_ANALYSIS_REVIEW_ROOT
    analysis_review_dir = root / run_id / analysis_review.ticker
    analysis_review_dir.mkdir(parents=True, exist_ok=True)
    analysis_review_path = analysis_review_dir / "analysis_review.json"

    if analysis_review_path.exists():
        raise FileExistsError(
            "refusing to overwrite existing SEC CompanyFacts Analysis Review: "
            f"{analysis_review_path}"
        )

    analysis_review_path.write_text(
        json.dumps(_to_jsonable(analysis_review), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return analysis_review_path


def _validate_observation_set_alignment(
    fundamental_observation_set: SecCompanyFactsFundamentalObservationSet,
    derived_cash_generation_observation_set: SecCompanyFactsDerivedCashGenerationObservationSet,
) -> None:
    mismatches = []

    if fundamental_observation_set.ticker != derived_cash_generation_observation_set.ticker:
        mismatches.append("ticker")

    if fundamental_observation_set.cik != derived_cash_generation_observation_set.cik:
        mismatches.append("cik")

    if (
        fundamental_observation_set.provider_name
        != derived_cash_generation_observation_set.provider_name
    ):
        mismatches.append("provider_name")

    if (
        fundamental_observation_set.source_context_format_version
        != derived_cash_generation_observation_set.source_context_format_version
    ):
        mismatches.append("source_context_format_version")

    if (
        fundamental_observation_set.source_context_state
        != derived_cash_generation_observation_set.source_context_state
    ):
        mismatches.append("source_context_state")

    if (
        fundamental_observation_set.source_refresh_snapshot_id
        != derived_cash_generation_observation_set.source_refresh_snapshot_id
    ):
        mismatches.append("source_refresh_snapshot_id")

    if mismatches:
        raise ValueError(
            "fundamental and derived observation sets do not align on: "
            + ", ".join(mismatches)
        )


def _validate_setup_detection_alignment(
    fundamental_observation_set: SecCompanyFactsFundamentalObservationSet,
    derived_cash_generation_observation_set: SecCompanyFactsDerivedCashGenerationObservationSet,
    setup_detection: SecCompanyFactsSetupDetection,
) -> None:
    if (
        setup_detection.setup_detection_format_version
        != SEC_COMPANYFACTS_SETUP_DETECTION_FORMAT_VERSION
    ):
        raise ValueError(
            "unsupported SEC CompanyFacts Setup Detection contract: "
            f"{setup_detection.setup_detection_format_version}"
        )

    mismatches = []
    for field_name in (
        "ticker",
        "cik",
        "provider_name",
        "source_context_format_version",
        "source_context_state",
        "source_refresh_snapshot_id",
        "source_refresh_fetched_at",
        "source_refresh_payload_format_version",
    ):
        if getattr(fundamental_observation_set, field_name) != getattr(
            setup_detection,
            field_name,
        ):
            mismatches.append(field_name)

    if (
        fundamental_observation_set.observation_format_version
        != setup_detection.fundamental_observation_format_version
    ):
        mismatches.append("fundamental_observation_format_version")

    if (
        derived_cash_generation_observation_set.derived_observation_format_version
        != setup_detection.derived_observation_format_version
    ):
        mismatches.append("derived_observation_format_version")

    if mismatches:
        raise ValueError(
            "analysis review and setup detection inputs do not align on: "
            + ", ".join(mismatches)
        )


def _source_availability_review(
    *,
    fundamental_observation_set: SecCompanyFactsFundamentalObservationSet,
    fundamental_by_category: dict[
        SecCompanyFactsFundamentalObservationCategory,
        SecCompanyFactsFundamentalObservation,
    ],
) -> SecCompanyFactsAnalysisReviewItem:
    required_observations = ("SOURCE_CONTEXT_AVAILABILITY",)
    source_availability = fundamental_by_category.get(
        SecCompanyFactsFundamentalObservationCategory.SOURCE_CONTEXT_AVAILABILITY
    )

    if source_availability is None:
        return SecCompanyFactsAnalysisReviewItem(
            category=SecCompanyFactsAnalysisReviewCategory.SOURCE_AVAILABILITY_REVIEW,
            state=SecCompanyFactsAnalysisReviewState.NOT_ASSESSED,
            message=(
                "Source availability review is not assessed because the required observation is missing."
            ),
            input_observation_families=("ME-FO",),
            required_observations=required_observations,
            missing_observations=required_observations,
            source_observation_references={},
            derived_observation_references={},
        )

    if (
        source_availability.state
        == SecCompanyFactsFundamentalObservationState.PRESENT
        and fundamental_observation_set.source_context_state == "AVAILABLE"
    ):
        state = SecCompanyFactsAnalysisReviewState.SOURCE_HEALTHY
        message = "Source observations are available for the reviewed ticker."
    else:
        state = SecCompanyFactsAnalysisReviewState.SOURCE_LIMITED
        message = "Source availability review is limited by upstream source context state."

    return SecCompanyFactsAnalysisReviewItem(
        category=SecCompanyFactsAnalysisReviewCategory.SOURCE_AVAILABILITY_REVIEW,
        state=state,
        message=message,
        input_observation_families=("ME-FO",),
        required_observations=required_observations,
        missing_observations=(),
        source_observation_references={
            "SOURCE_CONTEXT_AVAILABILITY": _fundamental_observation_reference(
                source_availability
            )
        },
        derived_observation_references={},
    )


def _fundamental_observation_completeness_review(
    *,
    fundamental_observation_set: SecCompanyFactsFundamentalObservationSet,
    fundamental_by_category: dict[
        SecCompanyFactsFundamentalObservationCategory,
        SecCompanyFactsFundamentalObservation,
    ],
) -> SecCompanyFactsAnalysisReviewItem:
    required_observations = (
        "SOURCE_CONTEXT_AVAILABILITY",
        "REVENUE_SOURCE_PRESENCE",
        "NET_INCOME_SOURCE_VALUE",
        "OPERATING_CASH_FLOW_SOURCE_VALUE",
        "CAPEX_SOURCE_PRESENCE",
        "CASH_GENERATION_SOURCE_COMPLETENESS",
    )
    missing_observations = tuple(
        observation_category
        for observation_category in required_observations
        if observation_category not in {category.value for category in fundamental_by_category}
    )

    referenced_observations = {
        category.value: _fundamental_observation_reference(observation)
        for category, observation in fundamental_by_category.items()
        if category.value in required_observations
    }

    if missing_observations:
        return SecCompanyFactsAnalysisReviewItem(
            category=(
                SecCompanyFactsAnalysisReviewCategory.FUNDAMENTAL_OBSERVATION_COMPLETENESS_REVIEW
            ),
            state=SecCompanyFactsAnalysisReviewState.OBSERVATIONS_LIMITED,
            message=(
                "Fundamental observation completeness review is limited because required observations are missing."
            ),
            input_observation_families=("ME-FO",),
            required_observations=required_observations,
            missing_observations=missing_observations,
            source_observation_references=referenced_observations,
            derived_observation_references={},
        )

    if _has_limited_fundamental_observations(fundamental_by_category):
        state = SecCompanyFactsAnalysisReviewState.OBSERVATIONS_LIMITED
        message = "Fundamental observation completeness review is limited by upstream observation states."
    else:
        state = SecCompanyFactsAnalysisReviewState.OBSERVATIONS_COMPLETE
        message = "Fundamental observations are complete for the reviewed categories."

    return SecCompanyFactsAnalysisReviewItem(
        category=(
            SecCompanyFactsAnalysisReviewCategory.FUNDAMENTAL_OBSERVATION_COMPLETENESS_REVIEW
        ),
        state=state,
        message=message,
        input_observation_families=("ME-FO",),
        required_observations=required_observations,
        missing_observations=(),
        source_observation_references=referenced_observations,
        derived_observation_references={},
    )


def _cash_generation_review(
    *,
    derived_cash_generation_observation_set: SecCompanyFactsDerivedCashGenerationObservationSet,
    derived_by_category: dict[
        SecCompanyFactsDerivedCashGenerationCategory,
        SecCompanyFactsDerivedCashGenerationObservation,
    ],
) -> SecCompanyFactsAnalysisReviewItem:
    required_observations = ("FREE_CASH_FLOW_DERIVATION",)
    free_cash_flow = derived_by_category.get(
        SecCompanyFactsDerivedCashGenerationCategory.FREE_CASH_FLOW_DERIVATION
    )

    if free_cash_flow is None:
        return SecCompanyFactsAnalysisReviewItem(
            category=SecCompanyFactsAnalysisReviewCategory.CASH_GENERATION_REVIEW,
            state=SecCompanyFactsAnalysisReviewState.NOT_ASSESSED,
            message=(
                "Cash-generation review is not assessed because the required derived observation is missing."
            ),
            input_observation_families=("ME-DO",),
            required_observations=required_observations,
            missing_observations=required_observations,
            source_observation_references={},
            derived_observation_references={},
        )

    state, message = _cash_generation_state_and_message(free_cash_flow)

    return SecCompanyFactsAnalysisReviewItem(
        category=SecCompanyFactsAnalysisReviewCategory.CASH_GENERATION_REVIEW,
        state=state,
        message=message,
        input_observation_families=("ME-DO",),
        required_observations=required_observations,
        missing_observations=(),
        source_observation_references={},
        derived_observation_references={
            "FREE_CASH_FLOW_DERIVATION": _derived_observation_reference(free_cash_flow)
        },
    )


def _free_cash_flow_review(
    *,
    derived_cash_generation_observation_set: SecCompanyFactsDerivedCashGenerationObservationSet,
    derived_by_category: dict[
        SecCompanyFactsDerivedCashGenerationCategory,
        SecCompanyFactsDerivedCashGenerationObservation,
    ],
) -> SecCompanyFactsAnalysisReviewItem:
    required_observations = ("FREE_CASH_FLOW_DERIVATION",)
    free_cash_flow = derived_by_category.get(
        SecCompanyFactsDerivedCashGenerationCategory.FREE_CASH_FLOW_DERIVATION
    )

    if free_cash_flow is None:
        return SecCompanyFactsAnalysisReviewItem(
            category=SecCompanyFactsAnalysisReviewCategory.FREE_CASH_FLOW_REVIEW,
            state=SecCompanyFactsAnalysisReviewState.NOT_ASSESSED,
            message=(
                "Free cash flow review is not assessed because the required derived observation is missing."
            ),
            input_observation_families=("ME-DO",),
            required_observations=required_observations,
            missing_observations=required_observations,
            source_observation_references={},
            derived_observation_references={},
        )

    state, message = _free_cash_flow_state_and_message(free_cash_flow)

    return SecCompanyFactsAnalysisReviewItem(
        category=SecCompanyFactsAnalysisReviewCategory.FREE_CASH_FLOW_REVIEW,
        state=state,
        message=message,
        input_observation_families=("ME-DO",),
        required_observations=required_observations,
        missing_observations=(),
        source_observation_references={},
        derived_observation_references={
            "FREE_CASH_FLOW_DERIVATION": _derived_observation_reference(free_cash_flow)
        },
    )


def _data_limitation_review(
    *,
    fundamental_observation_set: SecCompanyFactsFundamentalObservationSet,
    derived_cash_generation_observation_set: SecCompanyFactsDerivedCashGenerationObservationSet,
    fundamental_by_category: dict[
        SecCompanyFactsFundamentalObservationCategory,
        SecCompanyFactsFundamentalObservation,
    ],
    derived_by_category: dict[
        SecCompanyFactsDerivedCashGenerationCategory,
        SecCompanyFactsDerivedCashGenerationObservation,
    ],
) -> SecCompanyFactsAnalysisReviewItem | None:
    limited_fundamental_observations = _limited_fundamental_observation_names(
        fundamental_by_category
    )
    limited_derived_observations = _limited_derived_observation_names(derived_by_category)

    if (
        fundamental_observation_set.source_context_state == "AVAILABLE"
        and not limited_fundamental_observations
        and not limited_derived_observations
    ):
        return None

    return SecCompanyFactsAnalysisReviewItem(
        category=SecCompanyFactsAnalysisReviewCategory.DATA_LIMITATION_REVIEW,
        state=SecCompanyFactsAnalysisReviewState.DATA_LIMITED,
        message=(
            "Analysis review is limited because one or more upstream observations are limited."
        ),
        input_observation_families=("ME-FO", "ME-DO"),
        required_observations=(
            "SOURCE_CONTEXT_AVAILABILITY",
            "FREE_CASH_FLOW_DERIVATION",
        ),
        missing_observations=(
            limited_fundamental_observations + limited_derived_observations
        ),
        source_observation_references={
            category.value: _fundamental_observation_reference(observation)
            for category, observation in fundamental_by_category.items()
            if category.value in limited_fundamental_observations
        },
        derived_observation_references={
            category.value: _derived_observation_reference(observation)
            for category, observation in derived_by_category.items()
            if category.value in limited_derived_observations
        },
    )


def _human_review_requirement(
    *,
    fundamental_observation_set: SecCompanyFactsFundamentalObservationSet,
    derived_cash_generation_observation_set: SecCompanyFactsDerivedCashGenerationObservationSet,
    fundamental_by_category: dict[
        SecCompanyFactsFundamentalObservationCategory,
        SecCompanyFactsFundamentalObservation,
    ],
    derived_by_category: dict[
        SecCompanyFactsDerivedCashGenerationCategory,
        SecCompanyFactsDerivedCashGenerationObservation,
    ],
) -> SecCompanyFactsAnalysisReviewItem | None:
    limited_fundamental_observations = _limited_fundamental_observation_names(
        fundamental_by_category
    )
    limited_derived_observations = _limited_derived_observation_names(derived_by_category)

    if (
        fundamental_observation_set.source_context_state == "AVAILABLE"
        and not limited_fundamental_observations
        and not limited_derived_observations
    ):
        return None

    return SecCompanyFactsAnalysisReviewItem(
        category=SecCompanyFactsAnalysisReviewCategory.HUMAN_REVIEW_REQUIREMENT,
        state=SecCompanyFactsAnalysisReviewState.REQUIRES_HUMAN_REVIEW,
        message=(
            "Human review is required because upstream observations are incomplete or limited."
        ),
        input_observation_families=("ME-FO", "ME-DO"),
        required_observations=(
            "SOURCE_CONTEXT_AVAILABILITY",
            "FREE_CASH_FLOW_DERIVATION",
        ),
        missing_observations=(
            limited_fundamental_observations + limited_derived_observations
        ),
        source_observation_references={
            category.value: _fundamental_observation_reference(observation)
            for category, observation in fundamental_by_category.items()
            if category.value in limited_fundamental_observations
        },
        derived_observation_references={
            category.value: _derived_observation_reference(observation)
            for category, observation in derived_by_category.items()
            if category.value in limited_derived_observations
        },
    )


def _setup_detection_review_items(
    setup_detection: SecCompanyFactsSetupDetection,
) -> list[SecCompanyFactsAnalysisReviewItem]:
    review_items = [
        _setup_detection_review_item(setup_item)
        for setup_item in setup_detection.setup_items
    ]

    human_review_setup_items = tuple(
        setup_item
        for setup_item in setup_detection.setup_items
        if setup_item.state
        in {
            SecCompanyFactsSetupState.SETUP_PARTIALLY_DETECTED,
            SecCompanyFactsSetupState.SETUP_CONFLICTED,
            SecCompanyFactsSetupState.SETUP_BLOCKED_BY_MISSING_DATA,
            SecCompanyFactsSetupState.SETUP_NOT_ASSESSED,
        }
    )
    if human_review_setup_items:
        review_items.append(_setup_human_review_requirement(human_review_setup_items))

    return review_items


def _setup_detection_review_item(
    setup_item: SecCompanyFactsSetupDetectionItem,
) -> SecCompanyFactsAnalysisReviewItem:
    setup_reference_key = setup_item.category.value
    return SecCompanyFactsAnalysisReviewItem(
        category=_analysis_category_for_setup_category(setup_item.category),
        state=_analysis_state_for_setup_state(setup_item.state),
        message=_setup_review_message(setup_item),
        input_observation_families=("ME-SD",),
        required_observations=setup_item.required_observations,
        missing_observations=setup_item.missing_observations,
        source_observation_references=setup_item.source_observation_references,
        derived_observation_references=setup_item.derived_observation_references,
        setup_detection_references={
            setup_reference_key: _setup_detection_reference(setup_item)
        },
        setup_categories=(setup_item.category.value,),
        setup_states=(setup_item.state.value,),
        setup_evidence=setup_item.setup_evidence,
        setup_limitations=setup_item.setup_limitations,
    )


def _setup_human_review_requirement(
    setup_items: tuple[SecCompanyFactsSetupDetectionItem, ...],
) -> SecCompanyFactsAnalysisReviewItem:
    setup_references = {
        setup_item.category.value: _setup_detection_reference(setup_item)
        for setup_item in setup_items
    }
    return SecCompanyFactsAnalysisReviewItem(
        category=SecCompanyFactsAnalysisReviewCategory.SETUP_HUMAN_REVIEW_REQUIREMENT,
        state=SecCompanyFactsAnalysisReviewState.SETUP_REQUIRES_HUMAN_REVIEW,
        message=(
            "Setup Detection review requires human review because setup evidence is partial, conflicted, blocked, or not assessed."
        ),
        input_observation_families=("ME-SD",),
        required_observations=tuple(
            dict.fromkeys(
                observation
                for setup_item in setup_items
                for observation in setup_item.required_observations
            )
        ),
        missing_observations=tuple(
            dict.fromkeys(
                observation
                for setup_item in setup_items
                for observation in setup_item.missing_observations
            )
        ),
        source_observation_references={
            key: value
            for setup_item in setup_items
            for key, value in setup_item.source_observation_references.items()
        },
        derived_observation_references={
            key: value
            for setup_item in setup_items
            for key, value in setup_item.derived_observation_references.items()
        },
        setup_detection_references=setup_references,
        setup_categories=tuple(
            dict.fromkeys(setup_item.category.value for setup_item in setup_items)
        ),
        setup_states=tuple(
            dict.fromkeys(setup_item.state.value for setup_item in setup_items)
        ),
        setup_evidence={
            setup_item.category.value: setup_item.setup_evidence
            for setup_item in setup_items
        },
        setup_limitations=tuple(
            dict.fromkeys(
                limitation
                for setup_item in setup_items
                for limitation in setup_item.setup_limitations
            )
        ),
    )


def _analysis_category_for_setup_category(
    setup_category: SecCompanyFactsSetupCategory,
) -> SecCompanyFactsAnalysisReviewCategory:
    mapping = {
        SecCompanyFactsSetupCategory.CASH_GENERATION_SETUP: (
            SecCompanyFactsAnalysisReviewCategory.SETUP_DETECTION_REVIEW
        ),
        SecCompanyFactsSetupCategory.FUNDAMENTAL_AVAILABILITY_SETUP: (
            SecCompanyFactsAnalysisReviewCategory.SETUP_EVIDENCE_COMPLETENESS_REVIEW
        ),
        SecCompanyFactsSetupCategory.PROFITABILITY_EVIDENCE_SETUP: (
            SecCompanyFactsAnalysisReviewCategory.SETUP_DETECTION_REVIEW
        ),
        SecCompanyFactsSetupCategory.REVENUE_EVIDENCE_SETUP: (
            SecCompanyFactsAnalysisReviewCategory.SETUP_DETECTION_REVIEW
        ),
        SecCompanyFactsSetupCategory.BALANCE_SHEET_EVIDENCE_SETUP: (
            SecCompanyFactsAnalysisReviewCategory.SETUP_DETECTION_REVIEW
        ),
        SecCompanyFactsSetupCategory.DATA_LIMITATION_SETUP: (
            SecCompanyFactsAnalysisReviewCategory.SETUP_LIMITATION_REVIEW
        ),
        SecCompanyFactsSetupCategory.NOT_ASSESSED_SETUP: (
            SecCompanyFactsAnalysisReviewCategory.SETUP_HUMAN_REVIEW_REQUIREMENT
        ),
    }
    return mapping[setup_category]


def _analysis_state_for_setup_state(
    setup_state: SecCompanyFactsSetupState,
) -> SecCompanyFactsAnalysisReviewState:
    mapping = {
        SecCompanyFactsSetupState.SETUP_DETECTED: (
            SecCompanyFactsAnalysisReviewState.SETUP_DETECTED
        ),
        SecCompanyFactsSetupState.SETUP_PARTIALLY_DETECTED: (
            SecCompanyFactsAnalysisReviewState.SETUP_PARTIALLY_DETECTED
        ),
        SecCompanyFactsSetupState.SETUP_NOT_DETECTED: (
            SecCompanyFactsAnalysisReviewState.SETUP_NOT_DETECTED
        ),
        SecCompanyFactsSetupState.SETUP_CONFLICTED: (
            SecCompanyFactsAnalysisReviewState.SETUP_CONFLICTED
        ),
        SecCompanyFactsSetupState.SETUP_BLOCKED_BY_MISSING_DATA: (
            SecCompanyFactsAnalysisReviewState.SETUP_BLOCKED_BY_MISSING_DATA
        ),
        SecCompanyFactsSetupState.SETUP_NOT_ASSESSED: (
            SecCompanyFactsAnalysisReviewState.SETUP_NOT_ASSESSED
        ),
    }
    return mapping[setup_state]


def _setup_review_message(setup_item: SecCompanyFactsSetupDetectionItem) -> str:
    if setup_item.state == SecCompanyFactsSetupState.SETUP_DETECTED:
        return (
            "Setup Detection reports detected setup evidence from approved upstream observations."
        )
    if setup_item.state == SecCompanyFactsSetupState.SETUP_PARTIALLY_DETECTED:
        return (
            "Setup Detection reports partial setup evidence and preserves incomplete upstream observations."
        )
    if setup_item.state == SecCompanyFactsSetupState.SETUP_CONFLICTED:
        return (
            "Setup Detection reports conflicted setup evidence and requires human review."
        )
    if setup_item.state == SecCompanyFactsSetupState.SETUP_BLOCKED_BY_MISSING_DATA:
        return (
            "Setup Detection review is blocked because required upstream observations are missing."
        )
    if setup_item.state == SecCompanyFactsSetupState.SETUP_NOT_ASSESSED:
        return "Setup Detection could not assess this setup from the approved inputs."
    return "Setup Detection did not detect this setup from the approved inputs."


def _setup_detection_reference(
    setup_item: SecCompanyFactsSetupDetectionItem,
) -> dict[str, Any]:
    return {
        "setup_category": setup_item.category.value,
        "setup_state": setup_item.state.value,
        "setup_message": setup_item.message,
        "setup_evidence": setup_item.setup_evidence,
        "setup_limitations": setup_item.setup_limitations,
        "missing_observations": setup_item.missing_observations,
        "source_observation_references": setup_item.source_observation_references,
        "derived_observation_references": setup_item.derived_observation_references,
        "non_actionable_boundary": setup_item.non_actionable_boundary,
    }


def _cash_generation_state_and_message(
    free_cash_flow: SecCompanyFactsDerivedCashGenerationObservation,
) -> tuple[SecCompanyFactsAnalysisReviewState, str]:
    if (
        free_cash_flow.state
        == SecCompanyFactsDerivedCashGenerationState.DERIVED_POSITIVE_SOURCE_VALUE
    ):
        return (
            SecCompanyFactsAnalysisReviewState.CASH_GENERATION_POSITIVE,
            "Cash-generation review observes a positive free cash flow derived value.",
        )

    if (
        free_cash_flow.state
        == SecCompanyFactsDerivedCashGenerationState.DERIVED_NEGATIVE_SOURCE_VALUE
    ):
        return (
            SecCompanyFactsAnalysisReviewState.CASH_GENERATION_NEGATIVE,
            "Cash-generation review observes a negative free cash flow derived value.",
        )

    if (
        free_cash_flow.state
        == SecCompanyFactsDerivedCashGenerationState.DERIVED_ZERO_SOURCE_VALUE
    ):
        return (
            SecCompanyFactsAnalysisReviewState.CASH_GENERATION_NEUTRAL,
            "Cash-generation review observes a zero free cash flow derived value.",
        )

    if free_cash_flow.state in {
        SecCompanyFactsDerivedCashGenerationState.MISSING_SOURCE_DATA,
        SecCompanyFactsDerivedCashGenerationState.SOURCE_LIMITED,
    }:
        return (
            SecCompanyFactsAnalysisReviewState.DATA_LIMITED,
            "Cash-generation review is limited because required source data is missing.",
        )

    return (
        SecCompanyFactsAnalysisReviewState.NOT_ASSESSED,
        "Cash-generation review is not assessed from the available derived observation.",
    )


def _free_cash_flow_state_and_message(
    free_cash_flow: SecCompanyFactsDerivedCashGenerationObservation,
) -> tuple[SecCompanyFactsAnalysisReviewState, str]:
    if (
        free_cash_flow.state
        == SecCompanyFactsDerivedCashGenerationState.DERIVED_POSITIVE_SOURCE_VALUE
    ):
        return (
            SecCompanyFactsAnalysisReviewState.CASH_GENERATION_POSITIVE,
            "Free cash flow derived source value is positive.",
        )

    if (
        free_cash_flow.state
        == SecCompanyFactsDerivedCashGenerationState.DERIVED_NEGATIVE_SOURCE_VALUE
    ):
        return (
            SecCompanyFactsAnalysisReviewState.CASH_GENERATION_NEGATIVE,
            "Free cash flow derived source value is negative.",
        )

    if (
        free_cash_flow.state
        == SecCompanyFactsDerivedCashGenerationState.DERIVED_ZERO_SOURCE_VALUE
    ):
        return (
            SecCompanyFactsAnalysisReviewState.CASH_GENERATION_NEUTRAL,
            "Free cash flow derived source value is zero.",
        )

    if free_cash_flow.state in {
        SecCompanyFactsDerivedCashGenerationState.MISSING_SOURCE_DATA,
        SecCompanyFactsDerivedCashGenerationState.SOURCE_LIMITED,
    }:
        return (
            SecCompanyFactsAnalysisReviewState.DATA_LIMITED,
            "Free cash flow review is limited because required source data is missing.",
        )

    return (
        SecCompanyFactsAnalysisReviewState.NOT_ASSESSED,
        "Free cash flow review is not assessed from the available derived observation.",
    )


def _has_limited_fundamental_observations(
    fundamental_by_category: dict[
        SecCompanyFactsFundamentalObservationCategory,
        SecCompanyFactsFundamentalObservation,
    ],
) -> bool:
    return bool(_limited_fundamental_observation_names(fundamental_by_category))


def _limited_fundamental_observation_names(
    fundamental_by_category: dict[
        SecCompanyFactsFundamentalObservationCategory,
        SecCompanyFactsFundamentalObservation,
    ],
) -> tuple[str, ...]:
    limited_states = {
        SecCompanyFactsFundamentalObservationState.MISSING_DATA,
        SecCompanyFactsFundamentalObservationState.NOT_ASSESSED,
        SecCompanyFactsFundamentalObservationState.SOURCE_LIMITED,
    }

    return tuple(
        category.value
        for category, observation in fundamental_by_category.items()
        if observation.state in limited_states
    )


def _limited_derived_observation_names(
    derived_by_category: dict[
        SecCompanyFactsDerivedCashGenerationCategory,
        SecCompanyFactsDerivedCashGenerationObservation,
    ],
) -> tuple[str, ...]:
    limited_states = {
        SecCompanyFactsDerivedCashGenerationState.MISSING_SOURCE_DATA,
        SecCompanyFactsDerivedCashGenerationState.NOT_ASSESSED,
        SecCompanyFactsDerivedCashGenerationState.SOURCE_LIMITED,
    }

    return tuple(
        category.value
        for category, observation in derived_by_category.items()
        if observation.state in limited_states
    )


def _fundamental_observations_by_category(
    fundamental_observation_set: SecCompanyFactsFundamentalObservationSet,
) -> dict[
    SecCompanyFactsFundamentalObservationCategory,
    SecCompanyFactsFundamentalObservation,
]:
    return {
        observation.category: observation
        for observation in fundamental_observation_set.observations
    }


def _derived_observations_by_category(
    derived_cash_generation_observation_set: SecCompanyFactsDerivedCashGenerationObservationSet,
) -> dict[
    SecCompanyFactsDerivedCashGenerationCategory,
    SecCompanyFactsDerivedCashGenerationObservation,
]:
    return {
        observation.category: observation
        for observation in derived_cash_generation_observation_set.observations
    }


def _fundamental_observation_reference(
    observation: SecCompanyFactsFundamentalObservation,
) -> dict[str, Any]:
    return {
        "category": observation.category.value,
        "state": observation.state.value,
        "message": observation.message,
        "canonical_fields": observation.canonical_fields,
        "source_values": observation.source_values,
        "source_references": observation.source_references,
        "missing_source_fields": observation.missing_source_fields,
    }


def _derived_observation_reference(
    observation: SecCompanyFactsDerivedCashGenerationObservation,
) -> dict[str, Any]:
    return {
        "category": observation.category.value,
        "state": observation.state.value,
        "message": observation.message,
        "formula": observation.formula,
        "derived_values": observation.derived_values,
        "required_source_fields": observation.required_source_fields,
        "missing_source_fields": observation.missing_source_fields,
        "source_observation_references": {
            source_field: asdict(reference)
            for source_field, reference in observation.source_observation_references.items()
        },
    }


def _to_jsonable(analysis_review: SecCompanyFactsAnalysisReview) -> dict[str, Any]:
    payload = asdict(analysis_review)
    payload["review_items"] = [
        {
            **asdict(review_item),
            "category": review_item.category.value,
            "state": review_item.state.value,
        }
        for review_item in analysis_review.review_items
    ]
    return payload
