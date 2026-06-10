from __future__ import annotations

from pathlib import Path


LEGACY_MODULE_PATH = Path("scripts/fundamentals/sec_companyfacts_transform.py")

EXPECTED_TRANSFORM_OUTPUT_COLUMNS = [
    "ticker",
    "fiscal_year",
    "fiscal_period",
    "period_end_date",
    "report_date",
    "currency",
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "diluted_eps",
    "total_debt",
    "total_equity",
    "free_cash_flow",
    "source_name",
    "source_reference",
    "source_freshness_date",
    "extraction_date",
    "notes",
]


def test_legacy_companyfacts_transform_test_is_static_evidence_only() -> None:
    assert LEGACY_MODULE_PATH.parts == ("scripts", "fundamentals", "sec_companyfacts_transform.py")


def test_legacy_companyfacts_transform_contract_is_source_evidence_only() -> None:
    assert EXPECTED_TRANSFORM_OUTPUT_COLUMNS[:3] == ["ticker", "fiscal_year", "fiscal_period"]
    assert "notes" in EXPECTED_TRANSFORM_OUTPUT_COLUMNS
    assert "allocation" not in EXPECTED_TRANSFORM_OUTPUT_COLUMNS
    assert "tradeability" not in EXPECTED_TRANSFORM_OUTPUT_COLUMNS
