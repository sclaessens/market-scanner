from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.fundamentals.build_metrics import (
    HELPER_COLUMNS,
    IDENTITY_COLUMNS,
    METRIC_COLUMNS,
    build_fundamental_metrics,
)

FORBIDDEN_FIELDS = {
    "buy",
    "sell",
    "action",
    "final_action",
    "decision",
    "allocation",
    "position_size",
    "urgency",
    "conviction",
    "tradeability",
    "eligible",
    "eligibility",
    "ranking",
    "score",
    "priority",
    "entry",
    "stop",
    "target",
}


def _raw_row(ticker: str = "MSFT", fiscal_year: str = "2025", fiscal_period: str = "FY", **overrides: str) -> dict[str, str]:
    row = {
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "fiscal_period": fiscal_period,
        "period_end_date": f"{fiscal_year}-06-30",
        "report_date": f"{fiscal_year}-07-30",
        "currency": "USD",
        "revenue": "1000",
        "gross_profit": "600",
        "operating_income": "300",
        "net_income": "200",
        "diluted_eps": "5",
        "total_debt": "150",
        "total_equity": "500",
        "free_cash_flow": "250",
        "source_name": "company filing",
        "source_reference": f"{ticker} {fiscal_year} filing",
        "source_freshness_date": "2026-05-28",
        "extraction_date": "2026-05-28",
        "notes": "",
    }
    row.update(overrides)
    return row


def _write_history(path: Path, rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_valid_fixture_computes_margin_metrics(tmp_path: Path) -> None:
    input_path = tmp_path / "fundamentals_history.csv"
    _write_history(input_path, [_raw_row()])

    metrics = build_fundamental_metrics(input_path)

    row = metrics.iloc[0]
    assert row["gross_margin"] == pytest.approx(0.6)
    assert row["operating_margin"] == pytest.approx(0.3)
    assert row["net_margin"] == pytest.approx(0.2)
    assert row["free_cash_flow_margin"] == pytest.approx(0.25)


def test_valid_fixture_computes_leverage_and_roe(tmp_path: Path) -> None:
    input_path = tmp_path / "fundamentals_history.csv"
    _write_history(input_path, [_raw_row(total_debt="125", total_equity="500", net_income="75")])

    metrics = build_fundamental_metrics(input_path)

    row = metrics.iloc[0]
    assert row["debt_to_equity"] == pytest.approx(0.25)
    assert row["return_on_equity"] == pytest.approx(0.15)


def test_yoy_growth_matches_same_ticker_same_period_prior_year(tmp_path: Path) -> None:
    input_path = tmp_path / "fundamentals_history.csv"
    _write_history(
        input_path,
        [
            _raw_row(fiscal_year="2024", revenue="800", diluted_eps="4", free_cash_flow="200"),
            _raw_row(fiscal_year="2025", revenue="1000", diluted_eps="5", free_cash_flow="250"),
        ],
    )

    metrics = build_fundamental_metrics(input_path)

    current = metrics.loc[metrics["fiscal_year"] == "2025"].iloc[0]
    assert current["revenue_yoy_growth"] == pytest.approx(0.25)
    assert current["eps_yoy_growth"] == pytest.approx(0.25)
    assert current["free_cash_flow_yoy_growth"] == pytest.approx(0.25)
    assert current["metric_status"] == "complete"


def test_yoy_does_not_compare_different_fiscal_periods(tmp_path: Path) -> None:
    input_path = tmp_path / "fundamentals_history.csv"
    _write_history(
        input_path,
        [
            _raw_row(fiscal_year="2024", fiscal_period="Q1", revenue="800"),
            _raw_row(fiscal_year="2025", fiscal_period="FY", revenue="1000"),
        ],
    )

    metrics = build_fundamental_metrics(input_path)

    current = metrics.loc[metrics["fiscal_year"] == "2025"].iloc[0]
    assert pd.isna(current["revenue_yoy_growth"])
    assert "yoy_growth:missing_prior_year" in current["metric_warnings"]


def test_missing_prior_year_produces_blank_yoy_metrics(tmp_path: Path) -> None:
    input_path = tmp_path / "fundamentals_history.csv"
    _write_history(input_path, [_raw_row(fiscal_year="2025")])

    metrics = build_fundamental_metrics(input_path)

    row = metrics.iloc[0]
    assert pd.isna(row["revenue_yoy_growth"])
    assert pd.isna(row["eps_yoy_growth"])
    assert pd.isna(row["free_cash_flow_yoy_growth"])
    assert row["metric_status"] == "partial"


def test_zero_denominators_produce_blank_metric_values(tmp_path: Path) -> None:
    input_path = tmp_path / "fundamentals_history.csv"
    _write_history(input_path, [_raw_row(revenue="0", total_equity="0")])

    metrics = build_fundamental_metrics(input_path)

    row = metrics.iloc[0]
    assert pd.isna(row["gross_margin"])
    assert pd.isna(row["operating_margin"])
    assert pd.isna(row["net_margin"])
    assert pd.isna(row["free_cash_flow_margin"])
    assert pd.isna(row["debt_to_equity"])
    assert pd.isna(row["return_on_equity"])
    assert "gross_margin:zero_denominator:revenue" in row["metric_missing_inputs"]
    assert "debt_to_equity:zero_denominator:total_equity" in row["metric_missing_inputs"]


def test_missing_numeric_inputs_produce_blank_metric_values(tmp_path: Path) -> None:
    input_path = tmp_path / "fundamentals_history.csv"
    _write_history(input_path, [_raw_row(gross_profit="", revenue="", net_income="")])

    metrics = build_fundamental_metrics(input_path)

    row = metrics.iloc[0]
    assert pd.isna(row["gross_margin"])
    assert pd.isna(row["net_margin"])
    assert pd.isna(row["return_on_equity"])
    assert "gross_margin:missing:gross_profit|revenue" in row["metric_missing_inputs"]


def test_negative_numerator_values_are_accepted(tmp_path: Path) -> None:
    input_path = tmp_path / "fundamentals_history.csv"
    _write_history(input_path, [_raw_row(net_income="-100", free_cash_flow="-50")])

    metrics = build_fundamental_metrics(input_path)

    row = metrics.iloc[0]
    assert row["net_margin"] == pytest.approx(-0.1)
    assert row["return_on_equity"] == pytest.approx(-0.2)
    assert row["free_cash_flow_margin"] == pytest.approx(-0.05)


def test_negative_prior_year_values_use_absolute_denominator_for_yoy(tmp_path: Path) -> None:
    input_path = tmp_path / "fundamentals_history.csv"
    _write_history(
        input_path,
        [
            _raw_row(fiscal_year="2024", revenue="-800", diluted_eps="-4", free_cash_flow="-200"),
            _raw_row(fiscal_year="2025", revenue="1000", diluted_eps="5", free_cash_flow="250"),
        ],
    )

    metrics = build_fundamental_metrics(input_path)

    current = metrics.loc[metrics["fiscal_year"] == "2025"].iloc[0]
    assert current["revenue_yoy_growth"] == pytest.approx(2.25)
    assert current["eps_yoy_growth"] == pytest.approx(2.25)
    assert current["free_cash_flow_yoy_growth"] == pytest.approx(2.25)


def test_invalid_raw_history_input_fails_before_metrics_are_computed(tmp_path: Path) -> None:
    input_path = tmp_path / "fundamentals_history.csv"
    row = _raw_row()
    row.pop("source_reference")
    _write_history(input_path, [row])

    with pytest.raises(ValueError, match="fundamentals history validation failed"):
        build_fundamental_metrics(input_path)


def test_output_is_row_preserving_in_input_order(tmp_path: Path) -> None:
    input_path = tmp_path / "fundamentals_history.csv"
    rows = [
        _raw_row(ticker="MSFT", fiscal_year="2025"),
        _raw_row(ticker="AAPL", fiscal_year="2025"),
        _raw_row(ticker="NVDA", fiscal_year="2025"),
    ]
    _write_history(input_path, rows)

    metrics = build_fundamental_metrics(input_path)

    assert len(metrics) == len(rows)
    assert list(metrics["ticker"]) == ["MSFT", "AAPL", "NVDA"]


def test_output_contains_no_forbidden_semantic_fields(tmp_path: Path) -> None:
    input_path = tmp_path / "fundamentals_history.csv"
    _write_history(input_path, [_raw_row()])

    metrics = build_fundamental_metrics(input_path)

    output_fields = {column.lower() for column in metrics.columns}
    assert output_fields.isdisjoint(FORBIDDEN_FIELDS)
    assert list(metrics.columns) == IDENTITY_COLUMNS + METRIC_COLUMNS + HELPER_COLUMNS


def test_output_path_writes_only_when_explicitly_supplied(tmp_path: Path) -> None:
    input_path = tmp_path / "fundamentals_history.csv"
    output_path = tmp_path / "nested" / "fundamental_metrics.csv"
    implicit_output_path = tmp_path / "fundamental_metrics.csv"
    _write_history(input_path, [_raw_row()])

    build_fundamental_metrics(input_path)
    assert not implicit_output_path.exists()

    metrics = build_fundamental_metrics(input_path, output_path)

    assert output_path.exists()
    written = pd.read_csv(output_path)
    assert len(written) == len(metrics)
    assert list(written.columns) == IDENTITY_COLUMNS + METRIC_COLUMNS + HELPER_COLUMNS
