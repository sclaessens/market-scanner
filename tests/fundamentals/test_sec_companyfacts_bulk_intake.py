from __future__ import annotations

from pathlib import Path


LEGACY_MODULE_PATH = Path("scripts/fundamentals/sec_companyfacts_bulk_intake.py")
SEC_COMPANYFACTS_BULK_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"


def test_legacy_companyfacts_bulk_intake_test_is_static_evidence_only() -> None:
    assert LEGACY_MODULE_PATH.parts == ("scripts", "fundamentals", "sec_companyfacts_bulk_intake.py")


def test_legacy_companyfacts_bulk_intake_policy_requires_explicit_operator_action() -> None:
    assert SEC_COMPANYFACTS_BULK_URL.startswith("https://www.sec.gov/")
    assert SEC_COMPANYFACTS_BULK_URL.endswith("/companyfacts.zip")
