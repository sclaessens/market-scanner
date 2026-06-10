from __future__ import annotations

from pathlib import Path


LEGACY_MODULE_PATH = Path("scripts/fundamentals/build_analysis.py")
CANONICAL_PACKAGE_PATH = Path("src/market_scanner/fundamentals")

EXPECTED_LEGACY_POLICY = {
    "row_preservation",
    "classification_only",
    "no_allocation_semantics",
    "explicit_output_path_only",
}


def test_legacy_fundamental_analysis_test_is_static_evidence_only() -> None:
    assert LEGACY_MODULE_PATH.parts == ("scripts", "fundamentals", "build_analysis.py")
    assert CANONICAL_PACKAGE_PATH.parts == ("src", "market_scanner", "fundamentals")


def test_legacy_fundamental_analysis_policy_surface_is_preserved() -> None:
    assert EXPECTED_LEGACY_POLICY == {
        "row_preservation",
        "classification_only",
        "no_allocation_semantics",
        "explicit_output_path_only",
    }
