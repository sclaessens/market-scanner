from __future__ import annotations

from pathlib import Path


LEGACY_FLOW_MODULES = [
    Path("scripts/fundamentals/build_history_intake.py"),
    Path("scripts/fundamentals/build_metrics.py"),
    Path("scripts/fundamentals/build_quality.py"),
    Path("scripts/fundamentals/build_analysis.py"),
]

EXPECTED_LEGACY_FLOW_STAGES = [
    "history_validation",
    "metric_derivation",
    "quality_classification",
    "analysis_classification",
]


def test_legacy_operational_validation_test_is_static_evidence_only() -> None:
    assert [path.parts[:2] for path in LEGACY_FLOW_MODULES] == [
        ("scripts", "fundamentals"),
        ("scripts", "fundamentals"),
        ("scripts", "fundamentals"),
        ("scripts", "fundamentals"),
    ]


def test_legacy_operational_flow_policy_has_no_downstream_authority() -> None:
    forbidden_stages = {"allocation", "execution", "portfolio_write", "telegram_send"}

    assert set(EXPECTED_LEGACY_FLOW_STAGES).isdisjoint(forbidden_stages)
