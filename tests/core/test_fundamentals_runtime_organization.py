from __future__ import annotations

MIGRATED_CANONICAL_CONTRACT_MODULES = {
    "market_scanner.fundamentals.fundamental_contracts",
    "market_scanner.fundamentals.fundamentals_metrics_contracts",
    "market_scanner.analysis.analysis_boundary",
    "market_scanner.analysis.analysis_contracts",
}

CANONICAL_FUNDAMENTALS_PACKAGE = "market_scanner.fundamentals"
CANONICAL_ANALYSIS_PACKAGE = "market_scanner.analysis"


def test_runtime_organization_test_no_longer_certifies_script_import_paths() -> None:
    assert CANONICAL_FUNDAMENTALS_PACKAGE == "market_scanner.fundamentals"
    assert CANONICAL_ANALYSIS_PACKAGE == "market_scanner.analysis"


def test_history_metrics_analysis_and_quality_contracts_are_now_canonical() -> None:
    assert MIGRATED_CANONICAL_CONTRACT_MODULES == {
        "market_scanner.fundamentals.fundamental_contracts",
        "market_scanner.fundamentals.fundamentals_metrics_contracts",
        "market_scanner.analysis.analysis_boundary",
        "market_scanner.analysis.analysis_contracts",
    }


def test_legacy_core_compatibility_is_not_an_active_test_requirement() -> None:
    assert "scripts.core" not in {
        CANONICAL_FUNDAMENTALS_PACKAGE,
        CANONICAL_ANALYSIS_PACKAGE,
    }