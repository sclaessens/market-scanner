from __future__ import annotations

from pathlib import Path

CANONICAL_CONTRACT_MODULES = [
    Path("src/market_scanner/fundamentals/fundamental_contracts.py"),
    Path("src/market_scanner/fundamentals/fundamentals_metrics_contracts.py"),
    Path("src/market_scanner/analysis/analysis_boundary.py"),
    Path("src/market_scanner/analysis/analysis_contracts.py"),
]

EXPECTED_CANONICAL_FLOW_STAGES = [
    "history_validation_contract",
    "metric_derivation_contract",
    "analysis_boundary_contract",
    "quality_evidence_review_contract",
]


def test_operational_validation_uses_canonical_contract_paths_only() -> None:
    for path in CANONICAL_CONTRACT_MODULES:
        assert path.parts[0] == "src"
        assert path.parts[1] == "market_scanner"
        assert "scripts" not in path.parts


def test_legacy_operational_flow_policy_has_no_downstream_authority() -> None:
    forbidden_stages = {"allocation", "execution", "portfolio_write", "telegram_send"}

    assert set(EXPECTED_CANONICAL_FLOW_STAGES).isdisjoint(forbidden_stages)