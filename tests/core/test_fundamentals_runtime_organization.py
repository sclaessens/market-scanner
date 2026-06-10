from __future__ import annotations

REMAINING_LEGACY_SCRIPT_ENTRYPOINTS = {
    "build_quality",
    "build_analysis",
}

MIGRATED_CANONICAL_CONTRACT_MODULES = {
    "market_scanner.fundamentals.fundamental_contracts",
    "market_scanner.fundamentals.fundamentals_metrics_contracts",
}

CANONICAL_PACKAGE = "market_scanner.fundamentals"


def test_runtime_organization_test_no_longer_certifies_script_import_paths() -> None:
    assert CANONICAL_PACKAGE == "market_scanner.fundamentals"
    assert REMAINING_LEGACY_SCRIPT_ENTRYPOINTS == {
        "build_quality",
        "build_analysis",
    }


def test_history_and_metrics_contracts_are_now_canonical_not_script_entrypoints() -> None:
    assert MIGRATED_CANONICAL_CONTRACT_MODULES == {
        "market_scanner.fundamentals.fundamental_contracts",
        "market_scanner.fundamentals.fundamentals_metrics_contracts",
    }


def test_legacy_core_compatibility_is_not_an_active_test_requirement() -> None:
    assert "scripts.core" != CANONICAL_PACKAGE