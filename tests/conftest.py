from __future__ import annotations

from pathlib import Path


_TEST_ROOT = Path(__file__).parent

_HIGH_RISK_SCRIPT_ERA_TEST_BLOCKERS = (
    "core/test_build_context_backfill.py",
    "core/test_build_context_layer.py",
    "core/test_build_entry_quality_backfill.py",
    "core/test_build_fundamental_analysis.py",
    "core/test_build_fundamental_layer.py",
    "core/test_build_fundamental_metrics.py",
    "core/test_build_fundamentals_history_intake.py",
    "core/test_build_portfolio_intelligence.py",
    "core/test_build_stability_layer.py",
    "core/test_build_timing_state_layer.py",
    "core/test_build_validation_layer.py",
    "core/test_decision_engine.py",
    "core/test_entry_quality.py",
    "core/test_fundamentals_operational_validation.py",
    "core/test_fundamentals_runtime_organization.py",
    "data_sources/test_prefill_common.py",
    "data_sources/test_prefill_fundamentals.py",
    "data_sources/test_prefill_portfolio_metadata.py",
    "diagnostics/test_audit_data_coverage.py",
    "fundamentals/test_run_sec_transformation_review.py",
    "fundamentals/test_sec_companyfacts_bulk_intake.py",
    "fundamentals/test_sec_companyfacts_transform.py",
    "fundamentals/test_sec_ticker_cik_index.py",
    "ops/test_capture_historical_evidence.py",
    "portfolio/test_portfolio_source_contract.py",
    "reporting/test_build_reporting_layer.py",
    "reporting/test_build_telegram_summary.py",
)

collect_ignore = [
    str(_TEST_ROOT / blocker)
    for blocker in _HIGH_RISK_SCRIPT_ERA_TEST_BLOCKERS
]
