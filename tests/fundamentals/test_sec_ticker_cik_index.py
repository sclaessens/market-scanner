from __future__ import annotations

from market_scanner.fundamentals.sec_companyfacts_smoke_boundary import (
    SEC_COMPANYFACTS_SOURCE_FAMILY,
)

CANONICAL_CIK_MAPPING_STATUSES = {
    "CIK_MATCHED",
    "CIK_MISSING",
    "CIK_AMBIGUOUS",
}


def test_sec_ticker_cik_mapping_policy_is_canonical_source_mapping_only() -> None:
    assert SEC_COMPANYFACTS_SOURCE_FAMILY == "SEC EDGAR / SEC CompanyFacts"
    assert "CIK_MATCHED" in CANONICAL_CIK_MAPPING_STATUSES
    assert "CIK_MISSING" in CANONICAL_CIK_MAPPING_STATUSES
    assert "CIK_AMBIGUOUS" in CANONICAL_CIK_MAPPING_STATUSES


def test_sec_ticker_cik_mapping_policy_has_no_investment_authority() -> None:
    rendered = " ".join(CANONICAL_CIK_MAPPING_STATUSES).lower()

    assert "tradeable" not in rendered
    assert "allocation" not in rendered
    assert "conviction" not in rendered