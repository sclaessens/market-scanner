from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from market_engine.analysis_review.sec_companyfacts_analysis_review import (
    SEC_COMPANYFACTS_ANALYSIS_REVIEW_FORMAT_VERSION,
    NON_RECOMMENDATION_ANALYSIS_REVIEW_BOUNDARY,
    SecCompanyFactsAnalysisReview,
    SecCompanyFactsAnalysisReviewCategory,
    SecCompanyFactsAnalysisReviewItem,
    SecCompanyFactsAnalysisReviewState,
)
from market_engine.recommendation_review.sec_companyfacts_recommendation_review import (
    FORBIDDEN_RECOMMENDATION_REVIEW_ACTIONS,
    NON_ACTIONABLE_RECOMMENDATION_REVIEW_BOUNDARY,
    SEC_COMPANYFACTS_RECOMMENDATION_REVIEW_FORMAT_VERSION,
    SecCompanyFactsRecommendationReviewCategory,
    SecCompanyFactsRecommendationReviewState,
    build_sec_companyfacts_recommendation_review,
    persist_sec_companyfacts_recommendation_review,
)


def test_supportive_analysis_review_produces_non_actionable_human_review_candidate() -> None:
    analysis_review = _analysis_review(
        review_items=(
            _analysis_review_item(
                category=SecCompanyFactsAnalysisReviewCategory.SOURCE_AVAILABILITY_REVIEW,
                state=SecCompanyFactsAnalysisReviewState.SOURCE_HEALTHY,
                message="Source context is available for analysis review.",
            ),
            _analysis_review_item(
                category=SecCompanyFactsAnalysisReviewCategory.FUNDAMENTAL_OBSERVATION_COMPLETENESS_REVIEW,
                state=SecCompanyFactsAnalysisReviewState.OBSERVATIONS_COMPLETE,
                message="Fundamental observations are complete.",
            ),
            _analysis_review_item(
                category=SecCompanyFactsAnalysisReviewCategory.CASH_GENERATION_REVIEW,
                state=SecCompanyFactsAnalysisReviewState.CASH_GENERATION_POSITIVE,
                message="Cash generation evidence is positive.",
            ),
        )
    )

    recommendation_review = build_sec_companyfacts_recommendation_review(
        analysis_review,
        recommendation_review_run_id="rr-run-001",
    )

    assert (
        recommendation_review.recommendation_review_format_version
        == SEC_COMPANYFACTS_RECOMMENDATION_REVIEW_FORMAT_VERSION
    )
    assert (
        recommendation_review.analysis_review_format_version
        == SEC_COMPANYFACTS_ANALYSIS_REVIEW_FORMAT_VERSION
    )
    assert recommendation_review.input_contract == SEC_COMPANYFACTS_ANALYSIS_REVIEW_FORMAT_VERSION
    assert recommendation_review.recommendation_review_run_id == "rr-run-001"
    assert (
        recommendation_review.review_state
        == SecCompanyFactsRecommendationReviewState.HUMAN_REVIEW_REQUIRED
    )
    assert (
        recommendation_review.review_category
        == SecCompanyFactsRecommendationReviewCategory.ANALYSIS_SUPPORTIVE_BUT_NOT_ACTIONABLE
    )
    assert recommendation_review.non_actionable_boundary == NON_ACTIONABLE_RECOMMENDATION_REVIEW_BOUNDARY
    assert recommendation_review.forbidden_actions == FORBIDDEN_RECOMMENDATION_REVIEW_ACTIONS

    review_item = recommendation_review.review_items[0]
    assert (
        review_item.state
        == SecCompanyFactsRecommendationReviewState.HUMAN_REVIEW_REQUIRED
    )
    assert (
        review_item.category
        == SecCompanyFactsRecommendationReviewCategory.ANALYSIS_SUPPORTIVE_BUT_NOT_ACTIONABLE
    )
    assert review_item.missing_data == ()
    assert review_item.blocking_analysis_review_categories == ()
    assert "Portfolio Review" in review_item.message
    assert "Decision Engine" in review_item.message
    assert review_item.non_actionable_boundary == NON_ACTIONABLE_RECOMMENDATION_REVIEW_BOUNDARY
    assert review_item.forbidden_actions == FORBIDDEN_RECOMMENDATION_REVIEW_ACTIONS


def test_limited_analysis_review_blocks_recommendation_review_with_missing_data() -> None:
    analysis_review = _analysis_review(
        review_items=(
            _analysis_review_item(
                category=SecCompanyFactsAnalysisReviewCategory.DATA_LIMITATION_REVIEW,
                state=SecCompanyFactsAnalysisReviewState.DATA_LIMITED,
                message="Analysis is limited by missing observations.",
                missing_observations=("capital_expenditures",),
            ),
            _analysis_review_item(
                category=SecCompanyFactsAnalysisReviewCategory.HUMAN_REVIEW_REQUIREMENT,
                state=SecCompanyFactsAnalysisReviewState.REQUIRES_HUMAN_REVIEW,
                message="Human review is required because upstream data is limited.",
                missing_observations=("operating_cash_flow",),
            ),
        )
    )

    recommendation_review = build_sec_companyfacts_recommendation_review(
        analysis_review,
        recommendation_review_run_id="rr-run-002",
    )

    assert (
        recommendation_review.review_state
        == SecCompanyFactsRecommendationReviewState.BLOCKED_BY_MISSING_DATA
    )
    assert (
        recommendation_review.review_category
        == SecCompanyFactsRecommendationReviewCategory.ANALYSIS_BLOCKED_BY_MISSING_DATA
    )

    review_item = recommendation_review.review_items[0]
    assert (
        review_item.state
        == SecCompanyFactsRecommendationReviewState.BLOCKED_BY_MISSING_DATA
    )
    assert (
        review_item.category
        == SecCompanyFactsRecommendationReviewCategory.ANALYSIS_BLOCKED_BY_MISSING_DATA
    )
    assert review_item.missing_data == (
        "capital_expenditures",
        "operating_cash_flow",
    )
    assert set(review_item.blocking_analysis_review_categories) == {
        "DATA_LIMITATION_REVIEW",
        "HUMAN_REVIEW_REQUIREMENT",
    }


def test_negative_cash_generation_remains_non_actionable_human_review_candidate() -> None:
    analysis_review = _analysis_review(
        review_items=(
            _analysis_review_item(
                category=SecCompanyFactsAnalysisReviewCategory.SOURCE_AVAILABILITY_REVIEW,
                state=SecCompanyFactsAnalysisReviewState.SOURCE_HEALTHY,
                message="Source context is available.",
            ),
            _analysis_review_item(
                category=SecCompanyFactsAnalysisReviewCategory.CASH_GENERATION_REVIEW,
                state=SecCompanyFactsAnalysisReviewState.CASH_GENERATION_NEGATIVE,
                message="Cash generation evidence is negative.",
            ),
        )
    )

    recommendation_review = build_sec_companyfacts_recommendation_review(
        analysis_review,
        recommendation_review_run_id="rr-run-003",
    )

    assert (
        recommendation_review.review_state
        == SecCompanyFactsRecommendationReviewState.HUMAN_REVIEW_REQUIRED
    )
    assert (
        recommendation_review.review_category
        == SecCompanyFactsRecommendationReviewCategory.ANALYSIS_SUPPORTIVE_BUT_NOT_ACTIONABLE
    )
    assert recommendation_review.review_items[0].missing_data == ()
    assert recommendation_review.forbidden_actions == FORBIDDEN_RECOMMENDATION_REVIEW_ACTIONS


def test_invalid_analysis_review_contract_fails_closed() -> None:
    analysis_review = replace(
        _analysis_review(review_items=()),
        analysis_review_format_version="unsupported-analysis-review-v0",
    )

    with pytest.raises(ValueError, match="unsupported SEC CompanyFacts Analysis Review contract"):
        build_sec_companyfacts_recommendation_review(
            analysis_review,
            recommendation_review_run_id="rr-run-invalid",
        )


def test_recommendation_review_persistence_writes_json_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    analysis_review = _analysis_review(
        review_items=(
            _analysis_review_item(
                category=SecCompanyFactsAnalysisReviewCategory.SOURCE_AVAILABILITY_REVIEW,
                state=SecCompanyFactsAnalysisReviewState.SOURCE_HEALTHY,
                message="Source context is available.",
            ),
        )
    )
    recommendation_review = build_sec_companyfacts_recommendation_review(
        analysis_review,
        recommendation_review_run_id="rr-run-persist",
    )

    output_path = persist_sec_companyfacts_recommendation_review(
        recommendation_review,
        run_id="rr-run-persist",
        root_dir=tmp_path,
    )

    assert output_path == tmp_path / "rr-run-persist" / "NVDA" / "recommendation_review.json"

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert (
        payload["recommendation_review_format_version"]
        == SEC_COMPANYFACTS_RECOMMENDATION_REVIEW_FORMAT_VERSION
    )
    assert payload["input_contract"] == SEC_COMPANYFACTS_ANALYSIS_REVIEW_FORMAT_VERSION
    assert payload["review_state"] == "human_review_required"
    assert payload["review_category"] == "analysis_supportive_but_not_actionable"
    assert payload["input_provenance"]["ticker"] == "NVDA"
    assert payload["forbidden_actions"] == list(FORBIDDEN_RECOMMENDATION_REVIEW_ACTIONS)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        persist_sec_companyfacts_recommendation_review(
            recommendation_review,
            run_id="rr-run-persist",
            root_dir=tmp_path,
        )


def test_detected_setup_aware_analysis_review_creates_human_review_candidate() -> None:
    analysis_review = _setup_aware_analysis_review(
        _setup_analysis_review_item(
            state=SecCompanyFactsAnalysisReviewState.SETUP_DETECTED,
            setup_state="setup_detected",
            setup_evidence={"free_cash_flow": 0},
        )
    )

    recommendation_review = build_sec_companyfacts_recommendation_review(
        analysis_review,
        recommendation_review_run_id="rr-run-setup-detected",
    )

    assert (
        recommendation_review.review_state
        == SecCompanyFactsRecommendationReviewState.HUMAN_REVIEW_REQUIRED
    )
    assert (
        recommendation_review.review_category
        == SecCompanyFactsRecommendationReviewCategory.ANALYSIS_SUPPORTIVE_BUT_NOT_ACTIONABLE
    )
    assert recommendation_review.setup_detection_format_version == (
        "sec-companyfacts-setup-detection-v1"
    )
    assert recommendation_review.setup_detection_run_id == "setup-run-001"

    review_item = recommendation_review.review_items[0]
    assert review_item.setup_categories == ("cash_generation_setup",)
    assert review_item.setup_states == ("setup_detected",)
    assert review_item.setup_evidence["cash_generation_setup"]["free_cash_flow"] == 0
    assert review_item.setup_aware_analysis_review_references


def test_partial_setup_aware_analysis_review_preserves_uncertainty() -> None:
    analysis_review = _setup_aware_analysis_review(
        _setup_analysis_review_item(
            state=SecCompanyFactsAnalysisReviewState.SETUP_PARTIALLY_DETECTED,
            setup_state="setup_partially_detected",
            setup_limitations=("capital_expenditures",),
            missing_observations=("capital_expenditures",),
        )
    )

    recommendation_review = build_sec_companyfacts_recommendation_review(
        analysis_review,
        recommendation_review_run_id="rr-run-setup-partial",
    )

    assert (
        recommendation_review.review_state
        == SecCompanyFactsRecommendationReviewState.HUMAN_REVIEW_REQUIRED
    )
    assert (
        recommendation_review.review_category
        == SecCompanyFactsRecommendationReviewCategory.ANALYSIS_MIXED_OR_CONFLICTED
    )
    review_item = recommendation_review.review_items[0]
    assert review_item.missing_setup_observations == ("capital_expenditures",)
    assert review_item.setup_limitations == ("capital_expenditures",)


def test_blocked_setup_aware_analysis_review_preserves_missing_data() -> None:
    analysis_review = _setup_aware_analysis_review(
        _setup_analysis_review_item(
            state=SecCompanyFactsAnalysisReviewState.SETUP_BLOCKED_BY_MISSING_DATA,
            setup_state="setup_blocked_by_missing_data",
            setup_category="data_limitation_setup",
            category=SecCompanyFactsAnalysisReviewCategory.SETUP_LIMITATION_REVIEW,
            setup_evidence={},
            setup_limitations=("operating_cash_flow",),
            missing_observations=("operating_cash_flow",),
        )
    )

    recommendation_review = build_sec_companyfacts_recommendation_review(
        analysis_review,
        recommendation_review_run_id="rr-run-setup-blocked",
    )

    assert (
        recommendation_review.review_state
        == SecCompanyFactsRecommendationReviewState.BLOCKED_BY_MISSING_DATA
    )
    assert (
        recommendation_review.review_category
        == SecCompanyFactsRecommendationReviewCategory.ANALYSIS_BLOCKED_BY_MISSING_DATA
    )
    assert recommendation_review.review_items[0].missing_data == ("operating_cash_flow",)


def test_conflicted_setup_aware_analysis_review_routes_to_human_review() -> None:
    analysis_review = _setup_aware_analysis_review(
        _setup_analysis_review_item(
            state=SecCompanyFactsAnalysisReviewState.SETUP_CONFLICTED,
            setup_state="setup_conflicted",
            setup_evidence={"conflicted_source_observations": ("net_income",)},
            setup_limitations=("net_income",),
        )
    )

    recommendation_review = build_sec_companyfacts_recommendation_review(
        analysis_review,
        recommendation_review_run_id="rr-run-setup-conflicted",
    )

    assert (
        recommendation_review.review_state
        == SecCompanyFactsRecommendationReviewState.HUMAN_REVIEW_REQUIRED
    )
    assert (
        recommendation_review.review_category
        == SecCompanyFactsRecommendationReviewCategory.ANALYSIS_MIXED_OR_CONFLICTED
    )
    assert recommendation_review.review_items[0].setup_states == ("setup_conflicted",)


def test_not_assessed_and_not_detected_setup_do_not_become_negative_recommendations() -> None:
    for analysis_state, setup_state in (
        (
            SecCompanyFactsAnalysisReviewState.SETUP_NOT_ASSESSED,
            "setup_not_assessed",
        ),
        (
            SecCompanyFactsAnalysisReviewState.SETUP_NOT_DETECTED,
            "setup_not_detected",
        ),
    ):
        analysis_review = _setup_aware_analysis_review(
            _setup_analysis_review_item(
                state=analysis_state,
                setup_state=setup_state,
                setup_evidence={},
            )
        )

        recommendation_review = build_sec_companyfacts_recommendation_review(
            analysis_review,
            recommendation_review_run_id=f"rr-run-{setup_state}",
        )

        assert (
            recommendation_review.review_state
            == SecCompanyFactsRecommendationReviewState.INSUFFICIENT_EVIDENCE
        )
        assert (
            recommendation_review.review_category
            == SecCompanyFactsRecommendationReviewCategory.ANALYSIS_NOT_SUPPORTED
        )
        assert recommendation_review.review_items[0].setup_states == (setup_state,)


def test_missing_or_empty_setup_aware_fields_are_tolerated() -> None:
    analysis_review = _analysis_review(
        review_items=(
            _analysis_review_item(
                category=SecCompanyFactsAnalysisReviewCategory.SETUP_DETECTION_REVIEW,
                state=SecCompanyFactsAnalysisReviewState.SETUP_DETECTED,
                message="Setup-aware Analysis Review supports human review.",
            ),
        )
    )

    recommendation_review = build_sec_companyfacts_recommendation_review(
        analysis_review,
        recommendation_review_run_id="rr-run-empty-setup-fields",
    )

    assert (
        recommendation_review.review_state
        == SecCompanyFactsRecommendationReviewState.HUMAN_REVIEW_REQUIRED
    )
    assert recommendation_review.review_items[0].setup_categories == ()
    assert recommendation_review.review_items[0].setup_evidence == {}


def test_setup_aware_persistence_preserves_provenance(tmp_path: Path) -> None:
    analysis_review = _setup_aware_analysis_review(
        _setup_analysis_review_item(
            state=SecCompanyFactsAnalysisReviewState.SETUP_DETECTED,
            setup_state="setup_detected",
            setup_evidence={"free_cash_flow": 0},
        )
    )
    recommendation_review = build_sec_companyfacts_recommendation_review(
        analysis_review,
        recommendation_review_run_id="rr-run-setup-persist",
    )

    output_path = persist_sec_companyfacts_recommendation_review(
        recommendation_review,
        run_id="rr-run-setup-persist",
        root_dir=tmp_path,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["setup_detection_format_version"] == (
        "sec-companyfacts-setup-detection-v1"
    )
    assert payload["setup_detection_run_id"] == "setup-run-001"
    assert payload["review_items"][0]["setup_categories"] == ["cash_generation_setup"]
    assert payload["review_items"][0]["setup_evidence"]["cash_generation_setup"][
        "free_cash_flow"
    ] == 0


def test_recommendation_review_output_does_not_emit_action_authority_terms() -> None:
    analysis_review = _analysis_review(
        review_items=(
            _analysis_review_item(
                category=SecCompanyFactsAnalysisReviewCategory.SOURCE_AVAILABILITY_REVIEW,
                state=SecCompanyFactsAnalysisReviewState.SOURCE_HEALTHY,
                message="Source context is available.",
            ),
        )
    )

    recommendation_review = build_sec_companyfacts_recommendation_review(
        analysis_review,
        recommendation_review_run_id="rr-run-authority",
    )

    emitted_text = json.dumps(
        {
            "review_state": recommendation_review.review_state.value,
            "review_category": recommendation_review.review_category.value,
            "review_items": [
                {
                    "state": review_item.state.value,
                    "category": review_item.category.value,
                    "message": review_item.message,
                    "boundary_notes": review_item.boundary_notes,
                }
                for review_item in recommendation_review.review_items
            ],
        }
    ).lower()

    forbidden_action_terms = (
        "buy",
        "sell",
        "hold",
        "strong buy",
        "strong sell",
        "accumulate",
        "trim",
        "exit",
        "enter position",
        "increase position",
        "reduce position",
        "take profit",
        "stop loss",
        "price target",
        "target allocation",
        "position size",
        "portfolio weight",
        "conviction score",
        "urgency score",
        "tradeability score",
        "ranking",
        "top pick",
        "best candidate",
        "execute",
        "order",
        "rebalance",
        "send alert",
        "send telegram",
        "publish report",
    )

    assert not any(term in emitted_text for term in forbidden_action_terms)


def test_recommendation_review_does_not_import_legacy_runtime_modules() -> None:
    module_path = Path(
        "src/market_engine/recommendation_review/sec_companyfacts_recommendation_review.py"
    )
    module_source = module_path.read_text(encoding="utf-8")

    assert "from scripts" not in module_source
    assert "import scripts" not in module_source
    assert "from market_scanner" not in module_source
    assert "import market_scanner" not in module_source


def _analysis_review(
    *,
    review_items: tuple[SecCompanyFactsAnalysisReviewItem, ...],
    setup_detection_format_version: str | None = None,
    setup_detection_run_id: str | None = None,
    setup_detection_non_actionable_boundary: str | None = None,
) -> SecCompanyFactsAnalysisReview:
    return SecCompanyFactsAnalysisReview(
        ticker="NVDA",
        cik="0001045810",
        provider_name="sec_companyfacts",
        analysis_review_format_version=SEC_COMPANYFACTS_ANALYSIS_REVIEW_FORMAT_VERSION,
        source_context_format_version="sec-companyfacts-source-context-v1",
        fundamental_observation_format_version="sec-companyfacts-fundamental-observations-v1",
        derived_observation_format_version="sec-companyfacts-derived-cash-generation-observations-v1",
        source_context_state="AVAILABLE",
        source_refresh_snapshot_id="source-run-001",
        source_refresh_fetched_at="2026-01-01T00:00:00Z",
        source_refresh_payload_format_version="sec-companyfacts-raw-snapshot-v1",
        review_items=review_items,
        setup_detection_format_version=setup_detection_format_version,
        setup_detection_run_id=setup_detection_run_id,
        setup_detection_non_actionable_boundary=setup_detection_non_actionable_boundary,
        warnings=(),
    )


def _setup_aware_analysis_review(
    *review_items: SecCompanyFactsAnalysisReviewItem,
) -> SecCompanyFactsAnalysisReview:
    return _analysis_review(
        review_items=review_items,
        setup_detection_format_version="sec-companyfacts-setup-detection-v1",
        setup_detection_run_id="setup-run-001",
        setup_detection_non_actionable_boundary=(
            "Setup Detection describes evidence patterns only and is non-actionable."
        ),
    )


def _analysis_review_item(
    *,
    category: SecCompanyFactsAnalysisReviewCategory,
    state: SecCompanyFactsAnalysisReviewState,
    message: str,
    missing_observations: tuple[str, ...] = (),
    setup_detection_references: dict[str, dict[str, object]] | None = None,
    setup_categories: tuple[str, ...] = (),
    setup_states: tuple[str, ...] = (),
    setup_evidence: dict[str, object] | None = None,
    setup_limitations: tuple[str, ...] = (),
) -> SecCompanyFactsAnalysisReviewItem:
    return SecCompanyFactsAnalysisReviewItem(
        category=category,
        state=state,
        message=message,
        input_observation_families=("fundamental_observations", "derived_observations"),
        required_observations=("revenue", "net_income", "operating_cash_flow"),
        missing_observations=missing_observations,
        source_observation_references={
            "revenue": {
                "field": "revenue",
                "state": "PRESENT",
                "value": 100,
            }
        },
        derived_observation_references={
            "free_cash_flow": {
                "field": "free_cash_flow",
                "state": "POSITIVE_DERIVED_VALUE",
                "value": 50,
            }
        },
        setup_detection_references=setup_detection_references or {},
        setup_categories=setup_categories,
        setup_states=setup_states,
        setup_evidence=setup_evidence or {},
        setup_limitations=setup_limitations,
        non_recommendation_boundary=NON_RECOMMENDATION_ANALYSIS_REVIEW_BOUNDARY,
    )


def _setup_analysis_review_item(
    *,
    state: SecCompanyFactsAnalysisReviewState,
    setup_state: str,
    setup_category: str = "cash_generation_setup",
    category: SecCompanyFactsAnalysisReviewCategory = (
        SecCompanyFactsAnalysisReviewCategory.SETUP_DETECTION_REVIEW
    ),
    setup_evidence: dict[str, object] | None = None,
    setup_limitations: tuple[str, ...] = (),
    missing_observations: tuple[str, ...] = (),
) -> SecCompanyFactsAnalysisReviewItem:
    evidence = setup_evidence if setup_evidence is not None else {"free_cash_flow": 50}
    return _analysis_review_item(
        category=category,
        state=state,
        message="Setup-aware Analysis Review supports human review.",
        missing_observations=missing_observations,
        setup_detection_references={
            setup_category: {
                "setup_category": setup_category,
                "setup_state": setup_state,
                "setup_message": "Synthetic setup evidence for Recommendation Review.",
                "setup_evidence": evidence,
                "setup_limitations": setup_limitations,
                "missing_observations": missing_observations,
                "source_observation_references": {
                    "revenue": {"state": "PRESENT", "value": 100}
                },
                "derived_observation_references": {
                    "free_cash_flow": {"state": "DERIVED_VALUE", "value": 0}
                },
                "non_actionable_boundary": (
                    "Setup Detection describes evidence patterns only and is non-actionable."
                ),
            }
        },
        setup_categories=(setup_category,),
        setup_states=(setup_state,),
        setup_evidence=evidence,
        setup_limitations=setup_limitations,
    )
