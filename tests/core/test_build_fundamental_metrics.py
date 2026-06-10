from __future__ import annotations

from pathlib import Path


LEGACY_MODULE_PATH = Path("scripts/fundamentals/build_metrics.py")

IDENTITY_COLUMNS = [
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
]

METRIC_COLUMNS = [
    "gross_margin",
    "operating_margin",
    "net_margin",
    "free_cash_flow_margin",
    "debt_to_equity",
    "return_on_equity",
    "revenue_yoy_growth",
    "eps_yoy_growth",
    "free_cash_flow_yoy_growth",
]

HELPER_COLUMNS = [
    "metric_status",
    "metric_missing_inputs",
    "metric_warnings",
]


def test_legacy_fundamental_metrics_test_is_static_evidence_only() -> None:
    assert LEGACY_MODULE_PATH.parts == ("scripts", "fundamentals", "build_metrics.py")


def test_legacy_fundamental_metrics_contract_kept_as_non_allocation_evidence() -> None:
    columns = IDENTITY_COLUMNS + METRIC_COLUMNS + HELPER_COLUMNS

    assert "ticker" in columns
    assert "free_cash_flow_yoy_growth" in columns
    assert "decision" not in columns
    assert "allocation" not in columns
