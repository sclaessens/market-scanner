from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from market_engine.portfolio_review.sec_companyfacts_portfolio_review import (
    MARKET_ENGINE_PORTFOLIO_CONTEXT_FORMAT_VERSION,
    NON_ACTIONABLE_PORTFOLIO_REVIEW_BOUNDARY,
    SEC_COMPANYFACTS_PORTFOLIO_REVIEW_FORMAT_VERSION,
    MarketEnginePortfolioContext,
    MarketEnginePortfolioPositionState,
    SecCompanyFactsPortfolioReviewCategory,
    SecCompanyFactsPortfolioReviewState,
    build_sec_companyfacts_portfolio_review,
    persist_sec_companyfacts_portfolio_review,
)
from market_engine.recommendation_review.sec_companyfacts_recommendation_review import (
    NON_ACTIONABLE_RECOMMENDATION_REVIEW_BOUNDARY,
    SEC_COMPANYFACTS_RECOMMENDATION_REVIEW_FORMAT_VERSION,
    SecCompanyFactsRecommendationReview,
    SecCompanyFactsRecommendationReviewCategory,
    SecCompanyFactsRecommendationReviewItem,
    SecCompanyFactsRecommendationReviewState,
)


def test_valid_recommendation_review_and_portfolio_context_builds_review() -> None:
    recommendation_review = _recommendation_review()
    portfolio_context = _portfolio_context()

    portfolio_review = build_sec_companyfacts_portfolio_review(
        recommendation_review,
        portfolio_context,
        portfolio_review_run_id="portfolio-review-run",
        created_at="2026-06-16T12:00:00Z",
    )

    assert (
        portfolio_review.portfolio_review_format_version
        == SEC_COMPANYFACTS_PORTFOLIO_REVIEW_FORMAT_VERSION
    )
    assert portfolio_review.ticker == "NVDA"
    assert portfolio_review.recommendation_review_run_id == "rr-run-001"
    assert portfolio_review.portfolio_context_run_id == "portfolio-context-run"
    assert (
        portfolio_review.review_state
        == SecCompanyFactsPortfolioReviewState.PORTFOLIO_REVIEW_REQUIRED
    )
    assert (
        portfolio_review.review_category
        == SecCompanyFactsPortfolioReviewCategory.PORTFOLIO_FIT_CONTEXT_REVIEW
    )
    assert portfolio_review.non_actionable_boundary == NON_ACTIONABLE_PORTFOLIO_REVIEW_BOUNDARY
    assert portfolio_review.recommendation_review_provenance["ticker"] == "NVDA"
    assert portfolio_review.portfolio_context_provenance["current_quantity"] == 10


def test_missing_recommendation_review_returns_invalid_input_review() -> None:
    portfolio_review = build_sec_companyfacts_portfolio_review(
        None,
        _portfolio_context(),
        portfolio_review_run_id="portfolio-review-run",
    )

    assert (
        portfolio_review.review_state
        == SecCompanyFactsPortfolioReviewState.BLOCKED_BY_INVALID_INPUT
    )
    assert (
        portfolio_review.review_category
        == SecCompanyFactsPortfolioReviewCategory.INPUT_CONTRACT_INVALID
    )
    assert portfolio_review.portfolio_review_items[0].message == (
        "Recommendation Review input is missing."
    )


def test_wrong_recommendation_review_contract_fails_closed() -> None:
    recommendation_review = replace(
        _recommendation_review(),
        recommendation_review_format_version="unsupported-recommendation-review-v0",
    )

    with pytest.raises(
        ValueError,
        match="unsupported SEC CompanyFacts Recommendation Review contract",
    ):
        build_sec_companyfacts_portfolio_review(
            recommendation_review,
            _portfolio_context(),
            portfolio_review_run_id="portfolio-review-run",
        )


def test_not_reviewable_recommendation_review_blocks_portfolio_review() -> None:
    recommendation_review = replace(
        _recommendation_review(),
        review_state=SecCompanyFactsRecommendationReviewState.NOT_APPLICABLE,
        review_category=SecCompanyFactsRecommendationReviewCategory.INPUT_CONTRACT_INVALID,
    )

    portfolio_review = build_sec_companyfacts_portfolio_review(
        recommendation_review,
        _portfolio_context(),
        portfolio_review_run_id="portfolio-review-run",
    )

    assert (
        portfolio_review.review_state
        == SecCompanyFactsPortfolioReviewState.BLOCKED_BY_INVALID_INPUT
    )
    assert portfolio_review.portfolio_review_items[0].message == (
        "Recommendation Review input is not reviewable for Portfolio Review."
    )


def test_missing_portfolio_context_blocks_without_crash() -> None:
    recommendation_review = _recommendation_review()

    portfolio_review = build_sec_companyfacts_portfolio_review(
        recommendation_review,
        None,
        portfolio_review_run_id="portfolio-review-run",
    )

    assert (
        portfolio_review.review_state
        == SecCompanyFactsPortfolioReviewState.BLOCKED_BY_MISSING_PORTFOLIO_CONTEXT
    )
    assert portfolio_review.missing_portfolio_context_fields == ("portfolio_context",)
    assert portfolio_review.setup_aware_provenance


def test_stale_portfolio_context_is_marked_stale() -> None:
    portfolio_context = replace(
        _portfolio_context(),
        position_state=MarketEnginePortfolioPositionState.STALE,
        stale_portfolio_context_fields=("portfolio_snapshot_timestamp",),
    )

    portfolio_review = build_sec_companyfacts_portfolio_review(
        _recommendation_review(),
        portfolio_context,
        portfolio_review_run_id="portfolio-review-run",
    )

    assert (
        portfolio_review.review_state
        == SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_STALE
    )
    assert portfolio_review.stale_portfolio_context_fields == (
        "portfolio_snapshot_timestamp",
    )


def test_partial_portfolio_context_is_marked_partial() -> None:
    portfolio_context = replace(
        _portfolio_context(),
        position_state=MarketEnginePortfolioPositionState.PARTIALLY_KNOWN,
        current_ticker_exposure_pct=None,
        missing_portfolio_context_fields=("current_ticker_exposure_pct",),
    )

    portfolio_review = build_sec_companyfacts_portfolio_review(
        _recommendation_review(),
        portfolio_context,
        portfolio_review_run_id="portfolio-review-run",
    )

    assert (
        portfolio_review.review_state
        == SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_PARTIAL
    )
    assert portfolio_review.missing_portfolio_context_fields == (
        "current_ticker_exposure_pct",
    )
    exposure_item = _portfolio_item_by_category(portfolio_review)[
        SecCompanyFactsPortfolioReviewCategory.EXPOSURE_CONTEXT_REVIEW
    ]
    assert exposure_item.state == SecCompanyFactsPortfolioReviewState.EXPOSURE_MISSING


def test_ticker_held_is_identified_without_action_advice() -> None:
    portfolio_review = build_sec_companyfacts_portfolio_review(
        _recommendation_review(),
        _portfolio_context(position_state=MarketEnginePortfolioPositionState.HELD),
        portfolio_review_run_id="portfolio-review-run",
    )

    position_item = _portfolio_item_by_category(portfolio_review)[
        SecCompanyFactsPortfolioReviewCategory.POSITION_CONTEXT_REVIEW
    ]
    assert position_item.state == SecCompanyFactsPortfolioReviewState.POSITION_ALREADY_HELD


def test_ticker_not_held_is_identified_without_action_advice() -> None:
    portfolio_review = build_sec_companyfacts_portfolio_review(
        _recommendation_review(),
        _portfolio_context(
            position_state=MarketEnginePortfolioPositionState.NOT_HELD,
            current_quantity=0,
            current_market_value=0,
            current_ticker_exposure_pct=0,
        ),
        portfolio_review_run_id="portfolio-review-run",
    )

    position_item = _portfolio_item_by_category(portfolio_review)[
        SecCompanyFactsPortfolioReviewCategory.POSITION_CONTEXT_REVIEW
    ]
    assert position_item.state == SecCompanyFactsPortfolioReviewState.POSITION_NOT_HELD


def test_unknown_holding_state_is_identified() -> None:
    portfolio_review = build_sec_companyfacts_portfolio_review(
        _recommendation_review(),
        _portfolio_context(position_state=MarketEnginePortfolioPositionState.UNKNOWN),
        portfolio_review_run_id="portfolio-review-run",
    )

    position_item = _portfolio_item_by_category(portfolio_review)[
        SecCompanyFactsPortfolioReviewCategory.POSITION_CONTEXT_REVIEW
    ]
    assert position_item.state == SecCompanyFactsPortfolioReviewState.POSITION_UNKNOWN


def test_numeric_zero_fields_are_preserved_and_not_missing() -> None:
    portfolio_review = build_sec_companyfacts_portfolio_review(
        _recommendation_review(),
        _portfolio_context(
            position_state=MarketEnginePortfolioPositionState.NOT_HELD,
            current_quantity=0,
            current_market_value=0.0,
            portfolio_total_value=0,
            current_ticker_exposure_pct=0,
            concentration_thresholds={"max_ticker_exposure_pct": 0},
        ),
        portfolio_review_run_id="portfolio-review-run",
    )

    assert portfolio_review.portfolio_context_provenance["current_quantity"] == 0
    assert portfolio_review.portfolio_context_provenance["current_market_value"] == 0.0
    assert portfolio_review.portfolio_context_provenance["portfolio_total_value"] == 0
    assert portfolio_review.portfolio_context_provenance[
        "current_ticker_exposure_pct"
    ] == 0
    exposure_item = _portfolio_item_by_category(portfolio_review)[
        SecCompanyFactsPortfolioReviewCategory.EXPOSURE_CONTEXT_REVIEW
    ]
    assert exposure_item.state == SecCompanyFactsPortfolioReviewState.EXPOSURE_KNOWN


def test_recommendation_and_setup_provenance_are_preserved() -> None:
    recommendation_review = _recommendation_review()

    portfolio_review = build_sec_companyfacts_portfolio_review(
        recommendation_review,
        _portfolio_context(),
        portfolio_review_run_id="portfolio-review-run",
    )

    assert portfolio_review.recommendation_review_provenance[
        "recommendation_review_run_id"
    ] == "rr-run-001"
    assert portfolio_review.setup_aware_provenance["setup_detection_run_id"] == (
        "setup-run-001"
    )
    assert portfolio_review.setup_aware_provenance["review_items"][0][
        "setup_evidence"
    ] == {"cash_generation_setup": {"free_cash_flow": 0}}


def test_unsupported_portfolio_context_contract_returns_invalid_review() -> None:
    portfolio_context = replace(
        _portfolio_context(),
        portfolio_context_format_version="unsupported-portfolio-context-v0",
    )

    portfolio_review = build_sec_companyfacts_portfolio_review(
        _recommendation_review(),
        portfolio_context,
        portfolio_review_run_id="portfolio-review-run",
    )

    assert (
        portfolio_review.review_state
        == SecCompanyFactsPortfolioReviewState.BLOCKED_BY_INVALID_INPUT
    )
    assert (
        portfolio_review.review_category
        == SecCompanyFactsPortfolioReviewCategory.INPUT_CONTRACT_INVALID
    )


def test_persistence_writes_json_and_refuses_overwrite(tmp_path: Path) -> None:
    portfolio_review = build_sec_companyfacts_portfolio_review(
        _recommendation_review(),
        _portfolio_context(),
        portfolio_review_run_id="portfolio-review-run",
    )

    output_path = persist_sec_companyfacts_portfolio_review(
        portfolio_review,
        run_id="portfolio-review-run",
        root_dir=tmp_path,
    )

    assert (
        output_path
        == tmp_path / "portfolio-review-run" / "NVDA" / "portfolio_review.json"
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["portfolio_review_format_version"] == (
        SEC_COMPANYFACTS_PORTFOLIO_REVIEW_FORMAT_VERSION
    )
    assert payload["setup_aware_provenance"]["setup_detection_run_id"] == (
        "setup-run-001"
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        persist_sec_companyfacts_portfolio_review(
            portfolio_review,
            run_id="portfolio-review-run",
            root_dir=tmp_path,
        )


def test_portfolio_review_output_does_not_emit_action_authority_terms() -> None:
    portfolio_review = build_sec_companyfacts_portfolio_review(
        _recommendation_review(),
        _portfolio_context(),
        portfolio_review_run_id="portfolio-review-run",
    )
    emitted_text = json.dumps(
        {
            "review_state": portfolio_review.review_state.value,
            "review_category": portfolio_review.review_category.value,
            "portfolio_review_items": [
                {
                    "state": item.state.value,
                    "category": item.category.value,
                    "message": item.message,
                    "boundary_notes": item.boundary_notes,
                }
                for item in portfolio_review.portfolio_review_items
            ],
        }
    ).lower()

    forbidden_guidance_terms = (
        "buy",
        "sell",
        "strong buy",
        "strong sell",
        "target weight",
        "target price",
        "position size",
        "conviction",
        "urgency",
        "tradeability",
        "ranking",
        "score",
        "execute",
        "order",
        "rebalance",
        "send telegram",
        "publish report",
    )

    assert not any(term in emitted_text for term in forbidden_guidance_terms)


def test_portfolio_review_does_not_import_legacy_runtime_modules() -> None:
    module_path = Path(
        "src/market_engine/portfolio_review/sec_companyfacts_portfolio_review.py"
    )
    module_source = module_path.read_text(encoding="utf-8")

    assert "from scripts" not in module_source
    assert "import scripts" not in module_source
    assert "from market_scanner" not in module_source
    assert "import market_scanner" not in module_source


def _recommendation_review() -> SecCompanyFactsRecommendationReview:
    return SecCompanyFactsRecommendationReview(
        ticker="NVDA",
        cik="0001045810",
        provider_name="sec_companyfacts",
        recommendation_review_format_version=(
            SEC_COMPANYFACTS_RECOMMENDATION_REVIEW_FORMAT_VERSION
        ),
        analysis_review_format_version="sec-companyfacts-analysis-review-v1",
        source_context_format_version="sec-companyfacts-source-context-v1",
        source_context_state="AVAILABLE",
        source_refresh_snapshot_id="source-run-001",
        source_refresh_fetched_at="2026-01-01T00:00:00Z",
        source_refresh_payload_format_version="sec-companyfacts-raw-snapshot-v1",
        recommendation_review_run_id="rr-run-001",
        input_contract="sec-companyfacts-analysis-review-v1",
        setup_detection_format_version="sec-companyfacts-setup-detection-v1",
        setup_detection_run_id="setup-run-001",
        review_state=SecCompanyFactsRecommendationReviewState.HUMAN_REVIEW_REQUIRED,
        review_category=(
            SecCompanyFactsRecommendationReviewCategory.ANALYSIS_SUPPORTIVE_BUT_NOT_ACTIONABLE
        ),
        review_items=(_recommendation_review_item(),),
        input_provenance={
            "ticker": "NVDA",
            "setup_detection_format_version": "sec-companyfacts-setup-detection-v1",
            "setup_detection_run_id": "setup-run-001",
        },
        non_actionable_boundary=NON_ACTIONABLE_RECOMMENDATION_REVIEW_BOUNDARY,
    )


def _recommendation_review_item() -> SecCompanyFactsRecommendationReviewItem:
    return SecCompanyFactsRecommendationReviewItem(
        category=(
            SecCompanyFactsRecommendationReviewCategory.ANALYSIS_SUPPORTIVE_BUT_NOT_ACTIONABLE
        ),
        state=SecCompanyFactsRecommendationReviewState.HUMAN_REVIEW_REQUIRED,
        message="Setup-aware Analysis Review supports human review and remains non-actionable.",
        supporting_analysis_review_categories=("SETUP_DETECTION_REVIEW",),
        blocking_analysis_review_categories=(),
        missing_data=(),
        analysis_review_references={
            "SETUP_DETECTION_REVIEW": {
                "category": "SETUP_DETECTION_REVIEW",
                "state": "SETUP_DETECTED",
            }
        },
        setup_aware_analysis_review_references={
            "SETUP_DETECTION_REVIEW": {
                "setup_detection_references": {
                    "cash_generation_setup": {
                        "setup_state": "setup_detected",
                        "setup_evidence": {"free_cash_flow": 0},
                    }
                }
            }
        },
        setup_categories=("cash_generation_setup",),
        setup_states=("setup_detected",),
        setup_evidence={"cash_generation_setup": {"free_cash_flow": 0}},
        setup_limitations=(),
        missing_setup_observations=(),
        boundary_notes=("Recommendation Review is non-actionable.",),
        non_actionable_boundary=NON_ACTIONABLE_RECOMMENDATION_REVIEW_BOUNDARY,
    )


def _portfolio_context(
    *,
    position_state: MarketEnginePortfolioPositionState = (
        MarketEnginePortfolioPositionState.HELD
    ),
    current_quantity: float | int | None = 10,
    current_market_value: float | int | None = 1000,
    portfolio_total_value: float | int | None = 10000,
    current_ticker_exposure_pct: float | int | None = 10,
    concentration_thresholds: dict[str, object] | None = None,
) -> MarketEnginePortfolioContext:
    return MarketEnginePortfolioContext(
        portfolio_context_format_version=MARKET_ENGINE_PORTFOLIO_CONTEXT_FORMAT_VERSION,
        portfolio_context_run_id="portfolio-context-run",
        portfolio_snapshot_timestamp="2026-06-16T12:00:00Z",
        portfolio_base_currency="USD",
        ticker="NVDA",
        position_state=position_state,
        current_quantity=current_quantity,
        current_market_value=current_market_value,
        portfolio_total_value=portfolio_total_value,
        current_ticker_exposure_pct=current_ticker_exposure_pct,
        exposure_buckets={"sector": "Technology"},
        concentration_thresholds=concentration_thresholds
        if concentration_thresholds is not None
        else {"max_ticker_exposure_pct": 15},
        policy_constraints={"review_only": True},
        missing_portfolio_context_fields=(),
        stale_portfolio_context_fields=(),
        context_provenance={"source": "synthetic-test-context"},
    )


def _portfolio_item_by_category(portfolio_review):
    return {
        item.category: item
        for item in portfolio_review.portfolio_review_items
    }
