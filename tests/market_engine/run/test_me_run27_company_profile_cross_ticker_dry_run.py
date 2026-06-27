from __future__ import annotations

from pathlib import Path

from scripts.market_engine.me_run27_company_profile_cross_ticker_dry_run import (
    BOUNDED_VALIDATION_TICKERS,
    run_company_profile_cross_ticker_dry_run,
)


def test_cross_ticker_runner_records_controlled_profile_outcomes(tmp_path) -> None:
    artifact_root = tmp_path / "me-run27"

    summary = run_company_profile_cross_ticker_dry_run(
        run_id="me-run27-test",
        generated_at="2026-06-27T15:00:00Z",
        artifact_root=artifact_root,
    )

    assert BOUNDED_VALIDATION_TICKERS == ("NVDA", "AMD", "ASML")
    assert summary["overall_result"] == "completed_with_controlled_stop"
    assert summary["ticker_agnostic_execution"] is True
    assert summary["acquisition_safety"] == {
        "provider_calls_performed": False,
        "network_used": False,
        "telegram_sent": False,
        "portfolio_written": False,
        "watchlist_written": False,
        "broker_action_performed": False,
        "production_write_performed": False,
    }
    assert [result["ticker"] for result in summary["ticker_results"]] == [
        "NVDA",
        "AMD",
        "ASML",
    ]
    for result in summary["ticker_results"]:
        assert result["acquisition_state"] == "completed"
        assert result["staging_validation_state"] == "accepted"
        assert result["compatibility_gate_state"] == "allowed"
        assert result["source_context_state"] == "consumed"
        assert result["fundamental_observations_state"] == "completed"
        assert result["analysis_review_state"] == "completed"
        assert result["analysis_context_available"] is True
        assert result["completed_stages"] == (
            "source_context",
            "fundamental_observations",
            "derived_observations",
            "setup_detection",
            "analysis_review",
        )
        assert result["stop_stage"] == "recommendation_review"
        assert result["blocker_reasons"] == (
            "company_profile_descriptive_analysis_context_has_no_recommendation_input",
        )
        assert result["company_profile_observations_produced"] is True
        assert result["company_profile_observation_codes"]
    assert (artifact_root / "me_run27_summary.json").exists()
    assert (artifact_root / "me_run27_summary.md").exists()


def test_cross_ticker_script_has_no_runtime_side_effect_dependencies() -> None:
    script_path = Path(
        "scripts/market_engine/me_run27_company_profile_cross_ticker_dry_run.py"
    )
    script_source = script_path.read_text(encoding="utf-8").lower()

    forbidden_dependencies = (
        "import requests",
        "import urllib",
        "import yfinance",
        "import telegram",
        "import smtplib",
        "import socket",
        "import subprocess",
    )
    assert not any(term in script_source for term in forbidden_dependencies)
