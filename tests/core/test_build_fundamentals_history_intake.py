from __future__ import annotations

from pathlib import Path


LEGACY_MODULE_PATH = Path("scripts/fundamentals/build_history_intake.py")

REQUIRED_COLUMNS = [
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


def test_legacy_history_intake_test_is_static_evidence_only() -> None:
    assert LEGACY_MODULE_PATH.parts == ("scripts", "fundamentals", "build_history_intake.py")


def test_legacy_history_intake_contract_stays_upstream_source_evidence() -> None:
    assert REQUIRED_COLUMNS[:3] == ["ticker", "fiscal_year", "fiscal_period"]
    assert "source_reference" in REQUIRED_COLUMNS
    assert "tradeability" not in REQUIRED_COLUMNS
    assert "conviction" not in REQUIRED_COLUMNS
