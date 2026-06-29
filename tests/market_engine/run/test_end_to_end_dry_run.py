from __future__ import annotations

from pathlib import Path

from market_engine.run.end_to_end_dry_run import (
    MARKET_ENGINE_END_TO_END_DRY_RUN_BOUNDARY,
    MARKET_ENGINE_END_TO_END_DRY_RUN_FORMAT_VERSION,
    REQUIRED_DRY_RUN_STAGE_NAMES,
    MarketEngineEndToEndDryRunStageStatus,
    MarketEngineEndToEndDryRunState,
    build_market_engine_end_to_end_dry_run,
)


def test_complete_synthetic_fixture_builds_dry_run_payload() -> None:
    dry_run = build_market_engine_end_to_end_dry_run(
        _stage_payloads(),
        dry_run_id="dry-run-001",
        input_mode="synthetic_contract_fixture",
        generated_at="2026-06-17T13:00:00Z",
    )

    assert dry_run.dry_run_format_version == MARKET_ENGINE_END_TO_END_DRY_RUN_FORMAT_VERSION
    assert dry_run.dry_run_id == "dry-run-001"
    assert dry_run.input_mode == "synthetic_contract_fixture"
    assert dry_run.ticker == "NVDA"
    assert dry_run.cik == "0001045810"
    assert dry_run.provider_name == "sec_companyfacts"
    assert dry_run.run_state == MarketEngineEndToEndDryRunState.DRY_RUN_COMPLETED
    assert dry_run.blocked_stage is None
    assert dry_run.blocked_reasons == ()
    assert tuple(stage.stage_name for stage in dry_run.stage_results) == REQUIRED_DRY_RUN_STAGE_NAMES
    assert all(
        stage.status == MarketEngineEndToEndDryRunStageStatus.COMPLETED
        for stage in dry_run.stage_results
    )
    assert dry_run.non_execution_boundary == MARKET_ENGINE_END_TO_END_DRY_RUN_BOUNDARY


def test_completed_with_limitations_preserves_missing_and_stale_markers() -> None:
    payloads = _stage_payloads()
    payloads["analysis_review"] = {
        **payloads["analysis_review"],
        "missing_data_markers": ("analysis_review.free_cash_flow_component",),
    }
    payloads["portfolio_review"] = {
        **payloads["portfolio_review"],
        "stale_data_markers": ("portfolio_context.snapshot_timestamp",),
    }

    dry_run = build_market_engine_end_to_end_dry_run(
        payloads,
        dry_run_id="dry-run-001",
        input_mode="synthetic_contract_fixture",
    )

    assert dry_run.run_state == (
        MarketEngineEndToEndDryRunState.DRY_RUN_COMPLETED_WITH_LIMITATIONS
    )
    assert "analysis_review.free_cash_flow_component" in dry_run.missing_data_summary
    assert "portfolio_context.snapshot_timestamp" in dry_run.stale_data_summary
    assert _stage(dry_run, "analysis_review").status == (
        MarketEngineEndToEndDryRunStageStatus.COMPLETED_WITH_LIMITATIONS
    )
    assert _stage(dry_run, "portfolio_review").status == (
        MarketEngineEndToEndDryRunStageStatus.COMPLETED_WITH_LIMITATIONS
    )


def test_blocked_upstream_state_remains_blocked_and_downstream_is_not_started() -> None:
    payloads = _stage_payloads()
    payloads["portfolio_review"] = {
        **payloads["portfolio_review"],
        "review_state": "blocked_missing_portfolio_context",
        "blocked_reasons": ("Portfolio context is missing.",),
    }

    dry_run = build_market_engine_end_to_end_dry_run(
        payloads,
        dry_run_id="dry-run-001",
        input_mode="synthetic_contract_fixture",
    )

    assert dry_run.run_state == MarketEngineEndToEndDryRunState.DRY_RUN_BLOCKED
    assert dry_run.blocked_stage == "portfolio_review"
    assert dry_run.blocked_reasons == ("Portfolio context is missing.",)
    assert _stage(dry_run, "portfolio_review").status == (
        MarketEngineEndToEndDryRunStageStatus.BLOCKED
    )
    assert _stage(dry_run, "decision_engine_handoff").status == (
        MarketEngineEndToEndDryRunStageStatus.NOT_STARTED
    )
    assert _stage(dry_run, "delivery_reporting").status == (
        MarketEngineEndToEndDryRunStageStatus.NOT_STARTED
    )


def test_unsupported_contract_version_blocks_the_run() -> None:
    payloads = _stage_payloads()
    payloads["setup_detection"] = {
        **payloads["setup_detection"],
        "setup_detection_format_version": "sec-companyfacts-setup-detection-v0",
    }

    dry_run = build_market_engine_end_to_end_dry_run(
        payloads,
        dry_run_id="dry-run-001",
        input_mode="synthetic_contract_fixture",
    )

    assert dry_run.run_state == (
        MarketEngineEndToEndDryRunState.DRY_RUN_UNSUPPORTED_INPUT
    )
    assert dry_run.blocked_stage == "setup_detection"
    assert dry_run.blocked_reasons == ("Setup Detection contract version is unsupported.",)
    assert _stage(dry_run, "setup_detection").status == (
        MarketEngineEndToEndDryRunStageStatus.UNSUPPORTED_INPUT
    )


def test_malformed_stage_payload_becomes_contract_violation() -> None:
    payloads = _stage_payloads()
    payloads["source_context"] = object()

    dry_run = build_market_engine_end_to_end_dry_run(
        payloads,
        dry_run_id="dry-run-001",
        input_mode="synthetic_contract_fixture",
    )

    assert dry_run.run_state == (
        MarketEngineEndToEndDryRunState.DRY_RUN_CONTRACT_VIOLATION
    )
    assert dry_run.blocked_stage == "source_context"
    assert dry_run.blocked_reasons == ("Source Context payload must be a mapping.",)


def test_missing_required_stage_blocks_the_run() -> None:
    payloads = _stage_payloads()
    del payloads["fundamental_observations"]

    dry_run = build_market_engine_end_to_end_dry_run(
        payloads,
        dry_run_id="dry-run-001",
        input_mode="synthetic_contract_fixture",
    )

    assert dry_run.run_state == MarketEngineEndToEndDryRunState.DRY_RUN_BLOCKED
    assert dry_run.blocked_stage == "fundamental_observations"
    assert dry_run.blocked_reasons == (
        "Fundamental Observations payload is missing.",
    )


def test_unsupported_input_mode_is_fail_closed() -> None:
    dry_run = build_market_engine_end_to_end_dry_run(
        _stage_payloads(),
        dry_run_id="dry-run-001",
        input_mode="live_provider_fetch",
    )

    assert dry_run.run_state == (
        MarketEngineEndToEndDryRunState.DRY_RUN_UNSUPPORTED_INPUT
    )
    assert dry_run.blocked_stage == "input_mode"
    assert dry_run.blocked_reasons == ("Dry-run input mode is unsupported.",)
    assert all(
        stage.status == MarketEngineEndToEndDryRunStageStatus.NOT_STARTED
        for stage in dry_run.stage_results
    )


def test_numeric_zero_values_are_preserved_and_not_missing() -> None:
    payloads = _stage_payloads()
    payloads["portfolio_review"] = {
        **payloads["portfolio_review"],
        "portfolio_context_reference": {
            "current_quantity": 0,
            "current_market_value": 0.0,
            "current_ticker_exposure_pct": 0,
        },
    }

    dry_run = build_market_engine_end_to_end_dry_run(
        payloads,
        dry_run_id="dry-run-001",
        input_mode="synthetic_contract_fixture",
    )

    assert dry_run.run_state == MarketEngineEndToEndDryRunState.DRY_RUN_COMPLETED
    assert dry_run.missing_data_summary == ()
    assert dry_run.numeric_zero_evidence_summary[
        "portfolio_review.portfolio_context_reference.current_quantity"
    ] == 0
    assert dry_run.numeric_zero_evidence_summary[
        "portfolio_review.portfolio_context_reference.current_market_value"
    ] == 0.0
    assert dry_run.numeric_zero_evidence_summary[
        "portfolio_review.portfolio_context_reference.current_ticker_exposure_pct"
    ] == 0


def test_provenance_and_delivery_report_reference_are_preserved() -> None:
    dry_run = build_market_engine_end_to_end_dry_run(
        _stage_payloads(),
        dry_run_id="dry-run-001",
        input_mode="synthetic_contract_fixture",
    )

    assert dry_run.provenance_summary["source_context"]["source_refresh_snapshot_id"] == (
        "source-run-001"
    )
    assert dry_run.provenance_summary["recommendation_review"]["input_provenance"] == {
        "analysis_review_run_id": "analysis-review-run-001",
        "setup_detection_run_id": "setup-run-001",
    }
    assert dry_run.delivery_report_reference["source_handoff_run_id"] == (
        "handoff-run-001"
    )


def test_forbidden_semantic_fields_are_rejected() -> None:
    payloads = _stage_payloads()
    payloads["delivery_reporting"] = {
        **payloads["delivery_reporting"],
        "target_price": 100,
    }

    dry_run = build_market_engine_end_to_end_dry_run(
        payloads,
        dry_run_id="dry-run-001",
        input_mode="synthetic_contract_fixture",
    )

    assert dry_run.run_state == (
        MarketEngineEndToEndDryRunState.DRY_RUN_CONTRACT_VIOLATION
    )
    assert dry_run.blocked_stage == "delivery_reporting"
    assert dry_run.blocked_reasons == (
        "Delivery / Reporting payload contains prohibited dry-run semantics.",
    )


def test_payload_output_contains_serialized_enum_values() -> None:
    payload = build_market_engine_end_to_end_dry_run(
        _stage_payloads(),
        dry_run_id="dry-run-001",
        input_mode="synthetic_contract_fixture",
    ).to_payload()

    assert payload["dry_run_format_version"] == MARKET_ENGINE_END_TO_END_DRY_RUN_FORMAT_VERSION
    assert payload["run_state"] == "dry_run_completed"
    assert payload["stage_results"][0]["status"] == "completed"
    assert payload["analysis_context_readiness"]["readiness_level"] == (
        "recommendation_eligible"
    )
    assert payload["analysis_context_readiness"][
        "recommendation_review_eligible"
    ] is True
    assert payload["analysis_context_readiness"]["actionable_review_allowed"] is False
    assert payload["analysis_context_readiness"]["decision_engine_ready"] is False


def test_dry_run_harness_does_not_import_legacy_or_side_effect_modules() -> None:
    module_path = Path("src/market_engine/run/end_to_end_dry_run.py")
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
        "socket",
        "subprocess",
    )

    assert not any(term in module_source for term in forbidden_imports)


def _stage(dry_run, stage_name: str):
    return next(stage for stage in dry_run.stage_results if stage.stage_name == stage_name)


def _stage_payloads() -> dict[str, dict[str, object]]:
    return {
        "source_context": {
            "source_context_format_version": "sec-companyfacts-source-context-v1",
            "ticker": "NVDA",
            "cik": "0001045810",
            "provider_name": "sec_companyfacts",
            "source_context_state": "AVAILABLE",
            "source_refresh_snapshot_id": "source-run-001",
            "fixture_backed": True,
        },
        "fundamental_observations": {
            "fundamental_observations_format_version": (
                "sec-companyfacts-fundamental-observations-v1"
            ),
            "ticker": "NVDA",
            "cik": "0001045810",
            "provider_name": "sec_companyfacts",
            "observation_run_id": "fundamental-run-001",
            "source_context_reference": {"source_refresh_snapshot_id": "source-run-001"},
        },
        "derived_observations": {
            "derived_observations_format_version": (
                "sec-companyfacts-derived-cash-generation-observations-v1"
            ),
            "ticker": "NVDA",
            "cik": "0001045810",
            "provider_name": "sec_companyfacts",
            "derived_observation_run_id": "derived-run-001",
            "fundamental_observations_reference": {
                "observation_run_id": "fundamental-run-001"
            },
        },
        "setup_detection": {
            "setup_detection_format_version": "sec-companyfacts-setup-detection-v1",
            "ticker": "NVDA",
            "cik": "0001045810",
            "provider_name": "sec_companyfacts",
            "setup_detection_run_id": "setup-run-001",
            "derived_observations_reference": {
                "derived_observation_run_id": "derived-run-001"
            },
        },
        "analysis_review": {
            "analysis_review_format_version": "sec-companyfacts-analysis-review-v1",
            "ticker": "NVDA",
            "cik": "0001045810",
            "provider_name": "sec_companyfacts",
            "analysis_review_run_id": "analysis-review-run-001",
            "setup_detection_reference": {"setup_detection_run_id": "setup-run-001"},
        },
        "recommendation_review": {
            "recommendation_review_format_version": (
                "sec-companyfacts-recommendation-review-v1"
            ),
            "ticker": "NVDA",
            "cik": "0001045810",
            "provider_name": "sec_companyfacts",
            "recommendation_review_run_id": "rr-run-001",
            "input_provenance": {
                "analysis_review_run_id": "analysis-review-run-001",
                "setup_detection_run_id": "setup-run-001",
            },
        },
        "portfolio_review": {
            "portfolio_review_format_version": "sec-companyfacts-portfolio-review-v1",
            "ticker": "NVDA",
            "cik": "0001045810",
            "provider_name": "sec_companyfacts",
            "portfolio_review_run_id": "portfolio-review-run-001",
            "portfolio_context_reference": {
                "portfolio_context_format_version": "market-engine-portfolio-context-v1",
                "portfolio_context_run_id": "portfolio-context-run-001",
            },
            "recommendation_review_reference": {
                "recommendation_review_run_id": "rr-run-001"
            },
        },
        "decision_engine_handoff": {
            "handoff_format_version": "market-engine-decision-engine-handoff-v1",
            "ticker": "NVDA",
            "cik": "0001045810",
            "provider_name": "sec_companyfacts",
            "handoff_run_id": "handoff-run-001",
            "portfolio_review_reference": {
                "portfolio_review_run_id": "portfolio-review-run-001"
            },
            "portfolio_context_reference": {
                "portfolio_context_run_id": "portfolio-context-run-001"
            },
            "handoff_readiness_state": "ready_for_decision_engine_review",
            "audit_provenance": {"portfolio_review_run_id": "portfolio-review-run-001"},
        },
        "delivery_reporting": {
            "report_format_version": "market-engine-delivery-report-v1",
            "ticker": "NVDA",
            "cik": "0001045810",
            "provider_name": "sec_companyfacts",
            "report_id": "delivery-report-001",
            "source_handoff_run_id": "handoff-run-001",
            "delivery_state": "ready_for_user_review",
            "upstream_provenance_summary": {
                "decision_engine_handoff": {"handoff_run_id": "handoff-run-001"}
            },
            "forbidden_language_guardrails": ("buy", "sell", "hold"),
        },
    }
