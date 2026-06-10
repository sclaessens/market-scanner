from importlib import reload
from pathlib import Path

import pytest

from market_scanner.fundamentals import fundamentals_metrics_contracts
from market_scanner.fundamentals.fundamentals_metrics_contracts import (
    build_fundamental_metrics_contract_records,
    forbidden_fundamental_metrics_authority_fields,
    fundamental_derived_metric_fields,
    fundamental_metrics_helper_fields,
    fundamental_metrics_identity_fields,
    fundamental_metrics_input_fields,
)


def _history_record(**overrides):
    record = {
        "ticker": "ASML",
        "fiscal_year": "2025",
        "fiscal_period": "FY",
        "period_end_date": "2025-12-31",
        "report_date": "2026-02-15",
        "currency": "EUR",
        "revenue": "1000",
        "gross_profit": "600",
        "operating_income": "300",
        "net_income": "200",
        "diluted_eps": "5",
        "total_debt": "150",
        "total_equity": "500",
        "free_cash_flow": "250",
        "source_name": "synthetic filing",
        "source_reference": "synthetic FY2025 filing",
        "source_freshness_date": "2026-05-08",
        "extraction_date": "2026-05-08",
        "notes": "",
    }
    record.update(overrides)
    return record


def test_metrics_contract_exposes_bl81_identity_input_metric_and_helper_fields():
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

    assert fundamental_metrics_input_fields() == (
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "diluted_eps",
        "total_debt",
        "total_equity",
        "free_cash_flow",
    )

    assert fundamental_derived_metric_fields() == (
        "gross_margin",
        "operating_margin",
        "net_margin",
        "free_cash_flow_margin",
        "debt_to_equity",
        "return_on_equity",
        "revenue_yoy_growth",
        "eps_yoy_growth",
        "free_cash_flow_yoy_growth",
    )

    assert fundamental_metrics_helper_fields() == (
        "metric_status",
        "metric_missing_inputs",
        "metric_warnings",
    )


def test_metrics_contract_calculates_bl81_ratio_formulas():
    result = build_fundamental_metrics_contract_records((_history_record(),))
    record = result[0]

    assert record["gross_margin"] == pytest.approx(0.6)
    assert record["operating_margin"] == pytest.approx(0.3)
    assert record["net_margin"] == pytest.approx(0.2)
    assert record["free_cash_flow_margin"] == pytest.approx(0.25)
    assert record["debt_to_equity"] == pytest.approx(0.3)
    assert record["return_on_equity"] == pytest.approx(0.4)


def test_metrics_contract_calculates_bl81_yoy_growth_formulas():
    prior = _history_record(
        fiscal_year="2024",
        revenue="800",
        diluted_eps="4",
        free_cash_flow="200",
    )
    current = _history_record(
        fiscal_year="2025",
        revenue="1000",
        diluted_eps="5",
        free_cash_flow="250",
    )

    result = build_fundamental_metrics_contract_records((prior, current))
    current_metrics = result[1]

    assert current_metrics["revenue_yoy_growth"] == pytest.approx(0.25)
    assert current_metrics["eps_yoy_growth"] == pytest.approx(0.25)
    assert current_metrics["free_cash_flow_yoy_growth"] == pytest.approx(0.25)
    assert current_metrics["metric_status"] == "complete"
    assert current_metrics["metric_missing_inputs"] == ""
    assert current_metrics["metric_warnings"] == ""


def test_metrics_contract_preserves_identity_and_source_provenance():
    result = build_fundamental_metrics_contract_records(
        (
            _history_record(
                ticker="NVDA",
                source_name="SEC CompanyFacts",
                source_reference="CIK0001045810/FY2025",
            ),
        )
    )

    record = result[0]

    assert record["ticker"] == "NVDA"
    assert record["fiscal_year"] == "2025"
    assert record["fiscal_period"] == "FY"
    assert record["currency"] == "EUR"
    assert record["source_name"] == "SEC CompanyFacts"
    assert record["source_reference"] == "CIK0001045810/FY2025"


def test_metrics_contract_reports_missing_ratio_inputs_without_converting_to_zero():
    result = build_fundamental_metrics_contract_records(
        (
            _history_record(
                gross_profit="",
                total_equity="",
            ),
        )
    )

    record = result[0]

    assert record["gross_margin"] is None
    assert record["debt_to_equity"] is None
    assert record["return_on_equity"] is None
    assert "gross_margin:missing:gross_profit" in record["metric_missing_inputs"]
    assert "debt_to_equity:missing:total_equity" in record["metric_missing_inputs"]
    assert "return_on_equity:missing:total_equity" in record["metric_missing_inputs"]
    assert record["metric_status"] == "partial"


def test_metrics_contract_reports_zero_denominators_without_runtime_errors():
    result = build_fundamental_metrics_contract_records(
        (
            _history_record(
                revenue="0",
                total_equity="0",
            ),
        )
    )

    record = result[0]

    assert record["gross_margin"] is None
    assert record["operating_margin"] is None
    assert record["net_margin"] is None
    assert record["free_cash_flow_margin"] is None
    assert record["debt_to_equity"] is None
    assert record["return_on_equity"] is None
    assert "gross_margin:zero_denominator:revenue" in record["metric_missing_inputs"]
    assert "return_on_equity:zero_denominator:total_equity" in record["metric_missing_inputs"]
    assert record["metric_status"] == "partial"


def test_metrics_contract_reports_missing_prior_year_for_yoy_growth():
    result = build_fundamental_metrics_contract_records((_history_record(),))
    record = result[0]

    assert record["revenue_yoy_growth"] is None
    assert record["eps_yoy_growth"] is None
    assert record["free_cash_flow_yoy_growth"] is None
    assert record["metric_warnings"] == "yoy_growth:missing_prior_year"
    assert record["metric_status"] == "partial"


def test_metrics_contract_uses_absolute_prior_year_denominator_for_yoy_growth():
    prior = _history_record(
        fiscal_year="2024",
        revenue="-800",
        diluted_eps="-4",
        free_cash_flow="-200",
    )
    current = _history_record(
        fiscal_year="2025",
        revenue="1000",
        diluted_eps="5",
        free_cash_flow="250",
    )

    result = build_fundamental_metrics_contract_records((prior, current))
    current_metrics = result[1]

    assert current_metrics["revenue_yoy_growth"] == pytest.approx(2.25)
    assert current_metrics["eps_yoy_growth"] == pytest.approx(2.25)
    assert current_metrics["free_cash_flow_yoy_growth"] == pytest.approx(2.25)


def test_metrics_contract_reports_zero_prior_yoy_denominators():
    prior = _history_record(
        fiscal_year="2024",
        revenue="0",
        diluted_eps="0",
        free_cash_flow="0",
    )
    current = _history_record(
        fiscal_year="2025",
        revenue="1000",
        diluted_eps="5",
        free_cash_flow="250",
    )

    result = build_fundamental_metrics_contract_records((prior, current))
    current_metrics = result[1]

    assert current_metrics["revenue_yoy_growth"] is None
    assert current_metrics["eps_yoy_growth"] is None
    assert current_metrics["free_cash_flow_yoy_growth"] is None
    assert "revenue_yoy_growth:zero_denominator:prior_revenue" in current_metrics["metric_warnings"]
    assert "eps_yoy_growth:zero_denominator:prior_diluted_eps" in current_metrics["metric_warnings"]
    assert (
        "free_cash_flow_yoy_growth:zero_denominator:prior_free_cash_flow"
        in current_metrics["metric_warnings"]
    )
    assert current_metrics["metric_status"] == "partial"


def test_metrics_contract_does_not_expose_investment_authority_fields():
    output_fields = (
        set(fundamental_metrics_identity_fields())
        | set(fundamental_metrics_input_fields())
        | set(fundamental_derived_metric_fields())
        | set(fundamental_metrics_helper_fields())
    )

    for field_name in forbidden_fundamental_metrics_authority_fields():
        assert field_name not in output_fields


def test_metrics_contract_source_does_not_import_legacy_network_or_provider_modules():
    source = Path(fundamentals_metrics_contracts.__file__).read_text(encoding="utf-8")

    for forbidden in (
        "scripts",
        "requests",
        "urllib",
        "httpx",
        "aiohttp",
        "yfinance",
        "EDGAR",
    ):
        assert forbidden not in source


def test_metrics_contract_helpers_create_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    reload(fundamentals_metrics_contracts)
    build_fundamental_metrics_contract_records((_history_record(),))

    assert list(tmp_path.iterdir()) == []
    assert not Path("data/raw").exists()
    assert not Path("data/processed").exists()
    assert not Path("data/local").exists()
    assert not Path("reports").exists()