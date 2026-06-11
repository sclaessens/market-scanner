from __future__ import annotations

from pathlib import Path


_TEST_ROOT = Path(__file__).parent

_HIGH_RISK_SCRIPT_ERA_TEST_BLOCKERS = (
    "core/test_build_context_backfill.py",
    "core/test_build_context_layer.py",
    "core/test_build_entry_quality_backfill.py",
    "core/test_build_portfolio_intelligence.py",
    "core/test_build_stability_layer.py",
    "core/test_build_timing_state_layer.py",
    "core/test_build_validation_layer.py",
    "core/test_decision_engine.py",
    "core/test_entry_quality.py",
    "diagnostics/test_audit_data_coverage.py",
    "ops/test_capture_historical_evidence.py",
    "portfolio/test_portfolio_source_contract.py",
)

collect_ignore = [
    str(_TEST_ROOT / blocker)
    for blocker in _HIGH_RISK_SCRIPT_ERA_TEST_BLOCKERS
]
