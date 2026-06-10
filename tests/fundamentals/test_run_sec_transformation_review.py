from __future__ import annotations

from market_scanner.fundamentals.sec_companyfacts_smoke_boundary import (
    SEC_COMPANYFACTS_SOURCE_FAMILY,
)

CANONICAL_REVIEW_STATUSES = {
    "TRANSFORMED",
    "CIK_REVIEW_REQUIRED",
    "COMPANYFACTS_MISSING",
}


def test_sec_transformation_review_policy_is_canonical_source_review_only() -> None:
    assert SEC_COMPANYFACTS_SOURCE_FAMILY == "SEC EDGAR / SEC CompanyFacts"
    assert CANONICAL_REVIEW_STATUSES == {
        "TRANSFORMED",
        "CIK_REVIEW_REQUIRED",
        "COMPANYFACTS_MISSING",
    }


def test_sec_transformation_review_policy_has_no_investment_authority() -> None:
    forbidden = {"allocation", "tradeability", "urgency", "conviction", "buy", "sell"}
    rendered = " ".join(CANONICAL_REVIEW_STATUSES).lower()

    assert all(term not in rendered for term in forbidden)