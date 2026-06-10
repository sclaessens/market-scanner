from __future__ import annotations

from market_scanner.fundamentals.fundamentals_metrics_contracts import (
    fundamental_derived_metric_fields,
    fundamental_metrics_helper_fields,
    fundamental_metrics_identity_fields,
)


def test_fundamental_metrics_contract_is_now_canonical_evidence_only() -> None:
    assert fundamental_metrics_identity_fields() == (
        "ticker",
        "fiscal_year",
        "fiscal_period",
        "period_end_date",
        "report_date",
        "currency",
        "source_name",
        "source_reference",
        "source_freshness_date",
        "extraction_date",
    )


def test_fundamental_metrics_contract_kept_as_non_allocation_evidence() -> None:
    columns = (
        fundamental_metrics_identity_fields()
        + fundamental_derived_metric_fields()
        + fundamental_metrics_helper_fields()
    )

    assert "ticker" in columns
    assert "free_cash_flow_yoy_growth" in columns
    assert "decision" not in columns
    assert "allocation" not in columns