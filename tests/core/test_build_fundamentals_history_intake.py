from __future__ import annotations

from market_scanner.fundamentals.fundamental_contracts import (
    required_fundamental_history_fields,
)


def test_fundamentals_history_intake_contract_is_now_canonical_evidence_only() -> None:
    assert required_fundamental_history_fields() == (
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
    )


def test_fundamentals_history_contract_stays_upstream_source_evidence() -> None:
    fields = required_fundamental_history_fields()

    assert fields[:3] == ("ticker", "fiscal_year", "fiscal_period")
    assert "source_reference" in fields
    assert "tradeability" not in fields
    assert "conviction" not in fields