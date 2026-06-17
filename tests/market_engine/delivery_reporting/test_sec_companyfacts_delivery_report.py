from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from market_engine.decision_engine_handoff.sec_companyfacts_handoff import (
    MARKET_ENGINE_DECISION_ENGINE_HANDOFF_FORMAT_VERSION,
    MarketEngineDecisionEngineHandoff,
    MarketEngineDecisionEngineHandoffReadinessState,
    build_market_engine_decision_engine_handoff,
)
from market_engine.delivery_reporting.sec_companyfacts_delivery_report import (
    MARKET_ENGINE_DELIVERY_REPORT_BOUNDARY,
    MARKET_ENGINE_DELIVERY_REPORT_FORMAT_VERSION,
    MarketEngineDeliveryReportDisplaySection,
    MarketEngineDeliveryReportState,
    build_market_engine_delivery_report,
)
from market_engine.portfolio_review.sec_companyfacts_portfolio_review import (
    MARKET_ENGINE_PORTFOLIO_CONTEXT_FORMAT_VERSION,
    MarketEnginePortfolioContext,
    MarketEnginePortfolioPositionState,
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


def test_valid_handoff_builds_delivery_report_payload() -> None:
    report = build_market_engine_delivery_report(
        _handoff(),
        report_id="delivery-report-001",
        generated_at="2026-06-17T12:00:00Z",
    )

    assert report.report_format_version == MARKET_ENGINE_DELIVERY_REPORT_FORMAT_VERSION
    assert report.source_handoff_format_version == (
        MARKET_ENGINE_DECISION_ENGINE_HANDOFF_FORMAT_VERSION
    )
    assert report.ticker == "NVDA"
    assert report.delivery_state == MarketEngineDeliveryReportState.READY_FOR_USER_REVIEW
    assert report.blocked_unavailable_reasons == ()
    assert report.non_execution_boundary == MARKET_ENGINE_DELIVERY_REPORT_BOUNDARY


def test_blocked_upstream_handoff_remains_blocked() -> None:
    handoff = replace(
        _handoff(),
        handoff_readiness_state=(
            MarketEngineDecisionEngineHandoffReadinessState.BLOCKED_UNAPPROVED_PORTFOLIO_REVIEW
        ),
        blocked_reasons=("Portfolio Review input is invalid or not approved.",),
    )

    report = build_market_engine_delivery_report(
        handoff,
        report_id="delivery-report-001",
    )

    assert report.delivery_state == MarketEngineDeliveryReportState.BLOCKED_UPSTREAM
    assert report.blocked_unavailable_reasons == (
        "Portfolio Review input is invalid or not approved.",
    )


def test_missing_or_insufficient_data_remains_visible() -> None:
    handoff = replace(
        _handoff(),
        handoff_readiness_state=(
            MarketEngineDecisionEngineHandoffReadinessState.BLOCKED_INSUFFICIENT_EVIDENCE
        ),
        missing_data_markers=("portfolio_context",),
        blocked_reasons=("Portfolio Review handoff-readiness evidence is missing.",),
    )

    report = build_market_engine_delivery_report(
        handoff,
        report_id="delivery-report-001",
    )

    assert report.delivery_state == MarketEngineDeliveryReportState.INSUFFICIENT_DATA
    assert report.missing_data_summary == ("portfolio_context",)
    assert "Portfolio Review handoff-readiness evidence is missing." in (
        report.blocked_unavailable_reasons
    )


def test_stale_data_remains_visible() -> None:
    handoff = replace(
        _handoff(),
        handoff_readiness_state=(
            MarketEngineDecisionEngineHandoffReadinessState.BLOCKED_STALE_PORTFOLIO_CONTEXT
        ),
        stale_data_markers=("portfolio_snapshot_timestamp",),
        blocked_reasons=("Portfolio context is stale.",),
    )

    report = build_market_engine_delivery_report(
        handoff,
        report_id="delivery-report-001",
    )

    assert report.delivery_state == MarketEngineDeliveryReportState.STALE_DATA
    assert report.stale_data_summary == ("portfolio_snapshot_timestamp",)


def test_unsupported_input_format_becomes_unsupported_input() -> None:
    payload = _handoff().to_payload()
    payload["handoff_format_version"] = "unsupported-handoff-v0"

    report = build_market_engine_delivery_report(
        payload,
        report_id="delivery-report-001",
    )

    assert report.delivery_state == MarketEngineDeliveryReportState.UNSUPPORTED_INPUT
    assert report.blocked_unavailable_reasons == (
        "Decision Engine handoff format is unsupported.",
    )


def test_malformed_input_becomes_contract_violation() -> None:
    report = build_market_engine_delivery_report(
        object(),
        report_id="delivery-report-001",
    )

    assert report.delivery_state == MarketEngineDeliveryReportState.CONTRACT_VIOLATION
    assert report.blocked_unavailable_reasons == (
        "Decision Engine handoff input must be a handoff object or payload.",
    )


def test_missing_handoff_payload_becomes_contract_violation() -> None:
    report = build_market_engine_delivery_report(
        None,
        report_id="delivery-report-001",
    )

    assert report.delivery_state == MarketEngineDeliveryReportState.CONTRACT_VIOLATION
    assert report.blocked_unavailable_reasons == (
        "Decision Engine handoff input is missing.",
    )


def test_numeric_zero_values_are_preserved_and_not_missing() -> None:
    report = build_market_engine_delivery_report(
        _handoff(
            portfolio_context=_portfolio_context(
                position_state=MarketEnginePortfolioPositionState.NOT_HELD,
                current_quantity=0,
                current_market_value=0.0,
                portfolio_total_value=0,
                current_ticker_exposure_pct=0,
                concentration_thresholds={"max_ticker_exposure_pct": 0},
            )
        ),
        report_id="delivery-report-001",
    )

    assert report.delivery_state == MarketEngineDeliveryReportState.READY_FOR_USER_REVIEW
    assert report.missing_data_summary == ()
    assert report.numeric_zero_evidence[
        "portfolio_context.current_quantity"
    ] == 0
    assert report.numeric_zero_evidence[
        "portfolio_context.current_market_value"
    ] == 0.0
    assert report.numeric_zero_evidence[
        "portfolio_context.current_ticker_exposure_pct"
    ] == 0


def test_provenance_and_lineage_are_preserved() -> None:
    handoff = _handoff()

    report = build_market_engine_delivery_report(
        handoff,
        report_id="delivery-report-001",
    )

    assert report.upstream_provenance_summary["portfolio_review"] == (
        handoff.portfolio_review_reference
    )
    assert report.upstream_provenance_summary["recommendation_review"] == (
        handoff.recommendation_review_reference
    )
    assert report.upstream_provenance_summary["analysis_review"] == (
        handoff.analysis_review_reference
    )
    assert report.upstream_provenance_summary["setup_detection"] == (
        handoff.setup_detection_reference
    )
    assert report.audit_metadata["source_handoff_run_id"] == "handoff-run-001"


def test_payload_dict_input_is_supported() -> None:
    report = build_market_engine_delivery_report(
        _handoff().to_payload(),
        report_id="delivery-report-001",
    )

    assert report.report_format_version == MARKET_ENGINE_DELIVERY_REPORT_FORMAT_VERSION
    assert report.delivery_state == MarketEngineDeliveryReportState.READY_FOR_USER_REVIEW


def test_forbidden_terms_are_not_emitted_as_user_facing_language() -> None:
    report = build_market_engine_delivery_report(
        _handoff(),
        report_id="delivery-report-001",
    )

    emitted_text = json.dumps(
        [
            {
                "title": section.title,
                "body": section.body,
                "category": section.category.value,
            }
            for section in report.display_sections
        ]
    ).lower()
    forbidden_terms = (
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

    assert not any(term in emitted_text for term in forbidden_terms)


def test_forbidden_user_facing_language_is_rejected() -> None:
    sections = [
        MarketEngineDeliveryReportDisplaySection(
            category="factual_summary",
            title="Factual summary",
            body="This body contains a forbidden target price phrase.",
        )
    ]

    with pytest.raises(ValueError, match="forbidden language"):
        from market_engine.delivery_reporting.sec_companyfacts_delivery_report import (
            _validate_display_language,
        )

        _validate_display_language(sections)


def test_delivery_reporting_does_not_import_legacy_or_channel_modules() -> None:
    module_path = Path(
        "src/market_engine/delivery_reporting/sec_companyfacts_delivery_report.py"
    )
    module_source = module_path.read_text(encoding="utf-8")

    forbidden_imports = (
        "from scripts",
        "import scripts",
        "from market_scanner",
        "import market_scanner",
        "telegram",
        "smtplib",
        "yfinance",
        "urllib",
        "requests",
    )

    assert not any(term in module_source for term in forbidden_imports)


def _handoff(
    *,
    portfolio_context: MarketEnginePortfolioContext | None = None,
) -> MarketEngineDecisionEngineHandoff:
    portfolio_review = build_sec_companyfacts_portfolio_review(
        _recommendation_review(),
        portfolio_context if portfolio_context is not None else _portfolio_context(),
        portfolio_review_run_id="portfolio-review-run",
        created_at="2026-06-17T12:00:00Z",
    )
    return build_market_engine_decision_engine_handoff(
        portfolio_review,
        handoff_run_id="handoff-run-001",
        created_at="2026-06-17T12:30:00Z",
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
