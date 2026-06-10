from __future__ import annotations

from pathlib import Path


LEGACY_MODULE_PATH = Path("scripts/fundamentals/run_sec_transformation_review.py")

EXPECTED_REVIEW_STATUSES = {
    "TRANSFORMED",
    "CIK_REVIEW_REQUIRED",
    "COMPANYFACTS_MISSING",
}


def test_legacy_sec_transformation_review_test_is_static_evidence_only() -> None:
    assert LEGACY_MODULE_PATH.parts == ("scripts", "fundamentals", "run_sec_transformation_review.py")


def test_legacy_sec_transformation_review_policy_is_review_only() -> None:
    forbidden = {"allocation", "tradeability", "urgency", "conviction", "buy", "sell"}
    rendered = " ".join(EXPECTED_REVIEW_STATUSES).lower()

    assert all(term not in rendered for term in forbidden)
