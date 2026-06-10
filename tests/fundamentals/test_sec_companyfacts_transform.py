from __future__ import annotations

from market_scanner.fundamentals.fundamental_contracts import (
    required_fundamental_history_fields,
)


def test_companyfacts_transform_contract_is_now_canonical_history_evidence() -> None:
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


def test_companyfacts_transform_contract_remains_source_evidence_only() -> None:
    fields = required_fundamental_history_fields()

    assert fields[:3] == ("ticker", "fiscal_year", "fiscal_period")
    assert "source_reference" in fields
    assert "notes" in fields
    assert "allocation" not in fields
    assert "tradeability" not in fields
    assert "final_action" not in fields