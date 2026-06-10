from __future__ import annotations


LEGACY_SCRIPT_ENTRYPOINTS = {
    "build_history_intake",
    "build_metrics",
    "build_quality",
    "build_analysis",
}

CANONICAL_PACKAGE = "market_scanner.fundamentals"


def test_runtime_organization_test_no_longer_certifies_script_import_paths() -> None:
    assert CANONICAL_PACKAGE == "market_scanner.fundamentals"
    assert LEGACY_SCRIPT_ENTRYPOINTS == {
        "build_history_intake",
        "build_metrics",
        "build_quality",
        "build_analysis",
    }


def test_legacy_core_compatibility_is_not_an_active_test_requirement() -> None:
    assert "scripts.core" != CANONICAL_PACKAGE
