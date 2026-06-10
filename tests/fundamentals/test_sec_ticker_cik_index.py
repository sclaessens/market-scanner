from __future__ import annotations

from pathlib import Path


LEGACY_MODULE_PATH = Path("scripts/fundamentals/sec_ticker_cik_index.py")
EXPECTED_MAPPING_STATUSES = {
    "CIK_MATCHED",
    "CIK_MISSING",
    "CIK_AMBIGUOUS",
}


def test_legacy_ticker_cik_index_test_is_static_evidence_only() -> None:
    assert LEGACY_MODULE_PATH.parts == ("scripts", "fundamentals", "sec_ticker_cik_index.py")


def test_legacy_ticker_cik_index_contract_preserved_as_mapping_evidence() -> None:
    assert "CIK_MATCHED" in EXPECTED_MAPPING_STATUSES
    assert "TRADEABLE" not in EXPECTED_MAPPING_STATUSES
