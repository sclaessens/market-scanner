from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from market_engine.decision_engine_handoff.sec_companyfacts_handoff import (
    MARKET_ENGINE_DECISION_ENGINE_HANDOFF_BOUNDARY,
    MARKET_ENGINE_DECISION_ENGINE_HANDOFF_FORMAT_VERSION,
    MarketEngineDecisionEngineHandoffReadinessState,
    build_market_engine_decision_engine_handoff,
)
from market_engine.portfolio_review.sec_companyfacts_portfolio_review import (
    MARKET_ENGINE_PORTFOLIO_CONTEXT_FORMAT_VERSION,
    SEC_COMPANYFACTS_PORTFOLIO_REVIEW_FORMAT_VERSION,
    MarketEnginePortfolioContext,
    MarketEnginePortfolioPositionState,
    SecCompanyFactsPortfolioReview,
    SecCompanyFactsPortfolioReviewCategory,
    SecCompanyFactsPortfolioReviewItem,
    SecCompanyFactsPortfolioReviewState,
    build_sec_companyfacts_portfolio_review,
)
from market_engine.recommendation_review.sec_companyfacts_recommendation_review import (
    NON_ACTIONABLE_RECOMMENDATION_REVIEW_BOUNDARY,
    SEC_COMPANYFACTS_RECOMMENDATION_REVIEW_FORMAT_VERSION,
    SecCompanyFactsRecommendationReview,
    SecCompanyFactsRecommendationReviewCategory,
    SecCompanyFactsRecommendationReviewItem,
    SecCompanyFactsRecommendationReviewState,
)


def test_eligible_portfolio_review_builds_ready_handoff() -> None:
    handoff = build_market_engine_decision_engine_handoff(
        _portfolio_review(),
        handoff_run_id="handoff-run-001",
        created_at="2026-06-17T12:00:00Z",
    )

    assert (
        handoff.handoff_format_version
        == MARKET_ENGINE_DECISION_ENGINE_HANDOFF_FORMAT_VERSION
    )
    assert handoff.ticker == "NVDA"
    assert handoff.portfolio_review_format_version == (
        SEC_COMPANYFACTS_PORTFOLIO_REVIEW_FORMAT_VERSION
    )
    assert handoff.handoff_readiness_state == (
        MarketEngineDecisionEngineHandoffReadinessState.READY_FOR_DECISION_ENGINE_REVIEW
    )
    assert handoff.blocked_reasons == ()
    assert handoff.authority_boundary == MARKET_ENGINE_DECISION_ENGINE_HANDOFF_BOUNDARY


def test_missing_portfolio_review_builds_blocked_handoff() -> None:
    handoff = build_market_engine_decision_engine_handoff(
        None,
        handoff_run_id="handoff-run-001",
    )

    assert handoff.handoff_readiness_state == (
        MarketEngineDecisionEngineHandoffReadinessState.BLOCKED_MISSING_PORTFOLIO_REVIEW
    )
    assert handoff.blocked_reasons == ("Portfolio Review input is missing.",)
    assert handoff.ticker == ""


def test_wrong_portfolio_review_contract_blocks_handoff() -> None:
    portfolio_review = replace(
        _portfolio_review(),
        portfolio_review_format_version="unsupported-portfolio-review-v0",
    )

    handoff = build_market_engine_decision_engine_handoff(
        portfolio_review,
        handoff_run_id="handoff-run-001",
    )

    assert handoff.handoff_readiness_state == (
        MarketEngineDecisionEngineHandoffReadinessState.BLOCKED_INVALID_PORTFOLIO_REVIEW_CONTRACT
    )
    assert "Portfolio Review contract is unsupported." in handoff.blocked_reasons


def test_blocked_portfolio_review_state_blocks_handoff() -> None:
    portfolio_review = replace(
        _portfolio_review(),
        review_state=SecCompanyFactsPortfolioReviewState.BLOCKED_BY_INVALID_INPUT,
        review_category=SecCompanyFactsPortfolioReviewCategory.INPUT_CONTRACT_INVALID,
    )

    handoff = build_market_engine_decision_engine_handoff(
        portfolio_review,
        handoff_run_id="handoff-run-001",
    )

    assert handoff.handoff_readiness_state == (
        MarketEngineDecisionEngineHandoffReadinessState.BLOCKED_UNAPPROVED_PORTFOLIO_REVIEW
    )
    assert "Portfolio Review input is invalid or not approved." in (
        handoff.blocked_reasons
    )


def test_missing_portfolio_context_blocks_handoff() -> None:
    portfolio_review = replace(
        _portfolio_review(),
        portfolio_context_format_version=None,
        portfolio_context_run_id=None,
        portfolio_context_provenance={},
        missing_portfolio_context_fields=("portfolio_context",),
    )

    handoff = build_market_engine_decision_engine_handoff(
        portfolio_review,
        handoff_run_id="handoff-run-001",
    )

    assert handoff.handoff_readiness_state == (
        MarketEngineDecisionEngineHandoffReadinessState.BLOCKED_MISSING_PORTFOLIO_CONTEXT
    )
    assert "Portfolio context contract is missing or unsupported." in (
        handoff.blocked_reasons
    )
    assert "portfolio_context" in handoff.missing_data_markers


def test_stale_portfolio_context_blocks_handoff() -> None:
    portfolio_review = replace(
        _portfolio_review(),
        review_state=SecCompanyFactsPortfolioReviewState.PORTFOLIO_CONTEXT_STALE,
        stale_portfolio_context_fields=("portfolio_snapshot_timestamp",),
    )

    handoff = build_market_engine_decision_engine_handoff(
        portfolio_review,
        handoff_run_id="handoff-run-001",
    )

    assert handoff.handoff_readiness_state == (
        MarketEngineDecisionEngineHandoffReadinessState.BLOCKED_STALE_PORTFOLIO_CONTEXT
    )
    assert handoff.stale_data_markers == ("portfolio_snapshot_timestamp",)


def test_missing_handoff_readiness_item_blocks_as_insufficient_evidence() -> None:
    portfolio_review = replace(
        _portfolio_review(),
        portfolio_review_items=tuple(
            item
            for item in _portfolio_review().portfolio_review_items
            if item.category
            != SecCompanyFactsPortfolioReviewCategory.DOWNSTREAM_HANDOFF_READINESS_REVIEW
        ),
    )

    handoff = build_market_engine_decision_engine_handoff(
        portfolio_review,
        handoff_run_id="handoff-run-001",
    )

    assert handoff.handoff_readiness_state == (
        MarketEngineDecisionEngineHandoffReadinessState.BLOCKED_INSUFFICIENT_EVIDENCE
    )
    assert "Portfolio Review handoff-readiness evidence is missing." in (
        handoff.blocked_reasons
    )


def test_invalid_review_value_blocks_deterministically() -> None:
    portfolio_review = replace(
        _portfolio_review(),
        review_state="not-a-portfolio-review-state",
    )

    handoff = build_market_engine_decision_engine_handoff(
        portfolio_review,
        handoff_run_id="handoff-run-001",
    )

    assert handoff.handoff_readiness_state == (
        MarketEngineDecisionEngineHandoffReadinessState.BLOCKED_UNAPPROVED_PORTFOLIO_REVIEW
    )
    assert "Portfolio Review state or category is unsupported." in (
        handoff.blocked_reasons
    )


def test_ticker_mismatch_blocks_handoff() -> None:
    portfolio_review = replace(
        _portfolio_review(),
        portfolio_context_provenance={
            **_portfolio_review().portfolio_context_provenance,
            "ticker": "AMD",
        },
    )

    handoff = build_market_engine_decision_engine_handoff(
        portfolio_review,
        handoff_run_id="handoff-run-001",
    )

    assert handoff.handoff_readiness_state == (
        MarketEngineDecisionEngineHandoffReadinessState.BLOCKED_TICKER_MISMATCH
    )
    assert "Portfolio context ticker does not match Portfolio Review ticker." in (
        handoff.blocked_reasons
    )


def test_numeric_zero_values_are_preserved_and_not_missing() -> None:
    portfolio_review = _portfolio_review(
        portfolio_context=_portfolio_context(
            position_state=MarketEnginePortfolioPositionState.NOT_HELD,
            current_quantity=0,
            current_market_value=0.0,
            portfolio_total_value=0,
            current_ticker_exposure_pct=0,
            concentration_thresholds={"max_ticker_exposure_pct": 0},
        )
    )

    handoff = build_market_engine_decision_engine_handoff(
        portfolio_review,
        handoff_run_id="handoff-run-001",
    )

    assert handoff.portfolio_context_reference["current_quantity"] == 0
    assert handoff.portfolio_context_reference["current_market_value"] == 0.0
    assert handoff.portfolio_context_reference["portfolio_total_value"] == 0
    assert handoff.portfolio_context_reference["current_ticker_exposure_pct"] == 0
    assert handoff.missing_data_markers == ()
    assert handoff.handoff_readiness_state == (
        MarketEngineDecisionEngineHandoffReadinessState.READY_FOR_DECISION_ENGINE_REVIEW
    )


def test_provenance_and_lineage_are_preserved_without_invention() -> None:
    portfolio_review = _portfolio_review()

    handoff = build_market_engine_decision_engine_handoff(
        portfolio_review,
        handoff_run_id="handoff-run-001",
    )

    assert handoff.recommendation_review_reference == (
        portfolio_review.recommendation_review_provenance
    )
    assert handoff.setup_detection_reference == portfolio_review.setup_aware_provenance
    assert handoff.portfolio_context_reference == (
        portfolio_review.portfolio_context_provenance
    )
    assert handoff.audit_provenance["portfolio_review"]["portfolio_review_run_id"] == (
        "portfolio-review-run"
    )
    assert handoff.analysis_review_reference["review_items"][0][
        "SETUP_DETECTION_REVIEW"
    ]["state"] == "SETUP_DETECTED"


def test_payload_contract_name_and_blocked_reason_determinism() -> None:
    portfolio_review = replace(
        _portfolio_review(),
        portfolio_context_format_version=None,
        portfolio_context_provenance={},
        recommendation_review_provenance={},
        missing_portfolio_context_fields=("portfolio_context",),
    )

    handoff = build_market_engine_decision_engine_handoff(
        portfolio_review,
        handoff_run_id="handoff-run-001",
    )

    assert handoff.to_payload()["handoff_format_version"] == (
        MARKET_ENGINE_DECISION_ENGINE_HANDOFF_FORMAT_VERSION
    )
    assert handoff.blocked_reasons == (
        "Portfolio context contract is missing or unsupported.",
        "Portfolio context provenance is missing.",
        "Recommendation Review provenance is missing.",
        "Portfolio context is missing or incomplete.",
    )


def test_handoff_output_does_not_emit_action_authority_terms() -> None:
    handoff = build_market_engine_decision_engine_handoff(
        _portfolio_review(),
        handoff_run_id="handoff-run-001",
    )
    emitted_text = json.dumps(
        {
            "handoff_readiness_state": handoff.handoff_readiness_state.value,
            "authority_boundary": handoff.authority_boundary,
            "blocked_reasons": handoff.blocked_reasons,
            "warnings": handoff.warnings,
        }
    ).lower()

    forbidden_guidance_terms = (
        "buy",
        "sell",
        "hold",
        "target weight",
        "target price",
        "position size",
        "conviction",
        "urgency",
        "tradeability",
        "ranking",
        "score",
        "rebalance",
        "send telegram",
        "publish report",
    )

    assert not any(term in emitted_text for term in forbidden_guidance_terms)


def test_handoff_does_not_import_legacy_runtime_modules() -> None:
    module_path = Path(
        "src/market_engine/decision_engine_handoff/sec_companyfacts_handoff.py"
    )
    module_source = module_path.read_text(encoding="utf-8")

    assert "from scripts" not in module_source
    assert "import scripts" not in module_source
    assert "from market_scanner" not in module_source
    assert "import market_scanner" not in module_source


def _portfolio_review(
    *,
    portfolio_context: MarketEnginePortfolioContext | None = None,
) -> SecCompanyFactsPortfolioReview:
    return build_sec_companyfacts_portfolio_review(
        _recommendation_review(),
        portfolio_context if portfolio_context is not None else _portfolio_context(),
        portfolio_review_run_id="portfolio-review-run",
        created_at="2026-06-17T12:00:00Z",
    )


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
            "source_context_format_version": "sec-companyfacts-source-context-v1",
            "source_context_state": "AVAILABLE",
            "source_refresh_snapshot_id": "source-run-001",
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
        portfolio_snapshot_timestamp="2026-06-17T12:00:00Z",
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
