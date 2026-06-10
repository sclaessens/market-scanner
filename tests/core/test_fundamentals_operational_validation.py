from __future__ import annotations

from pathlib import Path

CANONICAL_CONTRACT_MODULES = [
    Path("src/market_scanner/fundamentals/fundamental_contracts.py"),
    Path("src/market_scanner/fundamentals/fundamentals_metrics_contracts.py"),
]

REMAINING_LEGACY_FLOW_MODULES = [
    Path("scripts/fundamentals/build_quality.py"),
    Path("scripts/fundamentals/build_analysis.py"),
]

EXPECTED_CANONICAL_FLOW_STAGES = [
    "history_validation_contract",
    "metric_derivation_contract",
    "quality_classification_review",
    "analysis_classification_review",
]


def test_operational_validation_no_longer_depends_on_history_or_metrics_script_paths() -> None:
    assert [path.parts[:3] for path in CANONICAL_CONTRACT_MODULES] == [
        ("src", "market_scanner", "fundamentals"),
        ("src", "market_scanner", "fundamentals"),
    ]

    assert [path.parts[:2] for path in REMAINING_LEGACY_FLOW_MODULES] == [
        ("scripts", "fundamentals"),
        ("scripts", "fundamentals"),
    ]


def test_legacy_operational_flow_policy_has_no_downstream_authority() -> None:
    forbidden_stages = {"allocation", "execution", "portfolio_write", "telegram_send"}

    assert set(EXPECTED_CANONICAL_FLOW_STAGES).isdisjoint(forbidden_stages)