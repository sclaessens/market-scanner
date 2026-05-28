from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.fundamentals.build_analysis import ANALYSIS_COLUMNS, build_fundamental_analysis

FORBIDDEN_TERMS = {
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


def _quality_row(ticker: str = "MSFT", date: str = "2026-05-28", **overrides: str) -> dict[str, str]:
    row = {
        "ticker": ticker,
        "date": date,
        "quality_state": "SUFFICIENT_DATA",
        "quality_reason": "fundamental metrics available",
        "profitability_profile": "OBSERVED",
        "balance_sheet_profile": "OBSERVED",
        "earnings_quality_profile": "OBSERVED",
        "capital_efficiency_profile": "OBSERVED",
        "cashflow_profile": "OBSERVED",
        "stability_profile": "OBSERVED",
        "quality_metadata_status": "complete",
        "source_data_status": "source_available",
        "source_timestamp": "2026-05-28",
        "source_name": "company filing",
        "source_last_updated": "2026-05-28",
        "source_freshness_days": "0",
        "missing_required_fields": "",
        "partial_data_reason": "",
        "stale_data_reason": "",
        "invalid_data_reason": "",
        "generated_at": "2026-05-28 12:00:00",
    }
    row.update(overrides)
    return row


def _metrics_row(ticker: str = "MSFT", fiscal_year: str = "2025", fiscal_period: str = "FY", **overrides: str) -> dict[str, str]:
    row = {
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "fiscal_period": fiscal_period,
        "period_end_date": "2025-12-31",
        "report_date": "2026-02-15",
        "currency": "USD",
        "source_name": "company filing",
        "source_reference": "FY2025 filing",
        "source_freshness_date": "2026-05-28",
        "extraction_date": "2026-05-28",
        "gross_margin": "0.60",
        "operating_margin": "0.30",
        "net_margin": "0.20",
        "free_cash_flow_margin": "0.25",
        "debt_to_equity": "0.30",
        "return_on_equity": "0.40",
        "revenue_yoy_growth": "0.10",
        "eps_yoy_growth": "0.20",
        "free_cash_flow_yoy_growth": "0.30",
        "metric_status": "complete",
        "metric_missing_inputs": "",
        "metric_warnings": "",
    }
    row.update(overrides)
    return row


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _build(
    tmp_path: Path,
    quality_rows: list[dict[str, str]],
    metric_rows: list[dict[str, str]] | None = None,
) -> pd.DataFrame:
    quality_path = tmp_path / "fundamental_quality.csv"
    metrics_path = tmp_path / "fundamental_metrics.csv"
    _write_csv(quality_path, quality_rows)
    if metric_rows is None:
        return build_fundamental_analysis(quality_path)
    _write_csv(metrics_path, metric_rows)
    return build_fundamental_analysis(quality_path, metrics_path)


def test_valid_quality_and_metrics_fixtures_produce_row_preserving_output(tmp_path: Path) -> None:
    rows = [_quality_row("MSFT"), _quality_row("AAPL")]
    analysis = _build(tmp_path, rows, [_metrics_row("MSFT"), _metrics_row("AAPL")])

    assert list(analysis.columns) == ANALYSIS_COLUMNS
    assert len(analysis) == len(rows)
    assert list(analysis["ticker"]) == ["MSFT", "AAPL"]
    assert list(analysis["date"]) == ["2026-05-28", "2026-05-28"]


def test_output_preserves_ticker_date_identity_and_input_order(tmp_path: Path) -> None:
    rows = [
        _quality_row("MSFT", date="2026-05-01"),
        _quality_row("NVDA", date="2026-05-02"),
        _quality_row("AAPL", date="2026-05-03"),
    ]
    metrics = [_metrics_row("AAPL"), _metrics_row("MSFT"), _metrics_row("NVDA")]

    analysis = _build(tmp_path, rows, metrics)

    assert list(zip(analysis["ticker"], analysis["date"], strict=True)) == [
        ("MSFT", "2026-05-01"),
        ("NVDA", "2026-05-02"),
        ("AAPL", "2026-05-03"),
    ]


def test_insufficient_quality_data_produces_insufficient_data(tmp_path: Path) -> None:
    analysis = _build(
        tmp_path,
        [
            _quality_row(
                quality_state="INSUFFICIENT_DATA",
                quality_metadata_status="row_missing",
                source_data_status="row_missing",
            )
        ],
        [_metrics_row()],
    )

    row = analysis.iloc[0]
    assert row["fundamental_analysis_state"] == "INSUFFICIENT_DATA"
    assert row["fundamental_review_flag"] == "REVIEW_DATA_LIMITATION"
    assert row["analysis_input_coverage"] == "quality_limited"


def test_missing_metrics_produce_limited_analysis(tmp_path: Path) -> None:
    analysis = _build(tmp_path, [_quality_row()])

    row = analysis.iloc[0]
    assert row["fundamental_analysis_state"] == "LIMITED_ANALYSIS"
    assert row["fundamental_profile_state"] == "UNKNOWN_PROFILE"
    assert row["analysis_data_status"] == "metrics_missing"


def test_sufficient_quality_and_complete_metrics_produce_analysis_ready(tmp_path: Path) -> None:
    analysis = _build(tmp_path, [_quality_row()], [_metrics_row()])

    row = analysis.iloc[0]
    assert row["fundamental_analysis_state"] == "ANALYSIS_READY"
    assert row["analysis_input_coverage"] == "quality_and_metrics"
    assert row["fundamental_review_flag"] == "NO_REVIEW_FLAG"


def test_positive_margin_metrics_produce_descriptive_margin_state(tmp_path: Path) -> None:
    analysis = _build(tmp_path, [_quality_row()], [_metrics_row(operating_margin="0.12", net_margin="0.08")])

    assert analysis.iloc[0]["margin_profile_state"] == "MARGIN_STABLE"


def test_negative_margin_metrics_produce_negative_margin_state(tmp_path: Path) -> None:
    analysis = _build(tmp_path, [_quality_row()], [_metrics_row(operating_margin="-0.02", net_margin="0.03")])

    row = analysis.iloc[0]
    assert row["margin_profile_state"] == "MARGIN_NEGATIVE"
    assert row["fundamental_profile_state"] == "DETERIORATING_PROFILE"


def test_positive_and_negative_growth_metrics_map_to_growth_states(tmp_path: Path) -> None:
    positive = _build(
        tmp_path / "positive",
        [_quality_row()],
        [_metrics_row(revenue_yoy_growth="0.12", eps_yoy_growth="0.04")],
    )
    negative = _build(
        tmp_path / "negative",
        [_quality_row()],
        [_metrics_row(revenue_yoy_growth="-0.12", eps_yoy_growth="-0.04")],
    )

    assert positive.iloc[0]["growth_profile_state"] == "GROWTH_POSITIVE"
    assert negative.iloc[0]["growth_profile_state"] == "GROWTH_NEGATIVE"


def test_mixed_growth_signals_create_descriptive_review_context(tmp_path: Path) -> None:
    analysis = _build(tmp_path, [_quality_row()], [_metrics_row(revenue_yoy_growth="0.12", eps_yoy_growth="-0.04")])

    row = analysis.iloc[0]
    assert row["growth_profile_state"] == "GROWTH_MIXED"
    assert row["fundamental_profile_state"] == "MIXED_PROFILE"
    assert row["fundamental_review_flag"] == "REVIEW_METRIC_CONFLICT"


def test_missing_leverage_inputs_produce_unknown_leverage_state(tmp_path: Path) -> None:
    analysis = _build(tmp_path, [_quality_row()], [_metrics_row(debt_to_equity="", metric_status="partial")])

    row = analysis.iloc[0]
    assert row["leverage_profile_state"] == "LEVERAGE_UNKNOWN"
    assert row["fundamental_analysis_state"] == "LIMITED_ANALYSIS"


def test_positive_and_negative_free_cash_flow_margin_map_to_cash_flow_states(tmp_path: Path) -> None:
    positive = _build(tmp_path / "positive", [_quality_row()], [_metrics_row(free_cash_flow_margin="0.15")])
    negative = _build(tmp_path / "negative", [_quality_row()], [_metrics_row(free_cash_flow_margin="-0.15")])

    assert positive.iloc[0]["cash_flow_profile_state"] == "CASH_FLOW_POSITIVE"
    assert negative.iloc[0]["cash_flow_profile_state"] == "CASH_FLOW_NEGATIVE"


def test_stale_quality_metadata_creates_descriptive_review_flag_only(tmp_path: Path) -> None:
    analysis = _build(
        tmp_path,
        [_quality_row(quality_state="STALE_DATA", quality_metadata_status="stale", source_data_status="stale_data")],
        [_metrics_row()],
    )

    row = analysis.iloc[0]
    assert row["fundamental_analysis_state"] == "LIMITED_ANALYSIS"
    assert row["fundamental_review_flag"] == "REVIEW_STALE_SOURCE"
    assert row["analysis_warnings"] == "stale_source"


def test_forbidden_semantic_fields_and_values_are_not_emitted(tmp_path: Path) -> None:
    analysis = _build(tmp_path, [_quality_row()], [_metrics_row()])

    assert {column.lower() for column in analysis.columns}.isdisjoint(FORBIDDEN_TERMS)
    for value in analysis.astype(str).to_numpy().ravel():
        assert value.strip().lower() not in FORBIDDEN_TERMS


def test_rows_are_not_filtered_when_fundamentals_are_missing(tmp_path: Path) -> None:
    rows = [
        _quality_row("MSFT"),
        _quality_row("MISSING", quality_state="INSUFFICIENT_DATA", quality_metadata_status="row_missing"),
    ]

    analysis = _build(tmp_path, rows, [_metrics_row("MSFT")])

    assert len(analysis) == 2
    missing = analysis.loc[analysis["ticker"] == "MISSING"].iloc[0]
    assert missing["fundamental_analysis_state"] == "INSUFFICIENT_DATA"


def test_metrics_rows_not_present_in_quality_input_do_not_create_extra_rows(tmp_path: Path) -> None:
    analysis = _build(tmp_path, [_quality_row("MSFT")], [_metrics_row("MSFT"), _metrics_row("EXTRA")])

    assert len(analysis) == 1
    assert list(analysis["ticker"]) == ["MSFT"]


def test_no_ticker_category_input_is_required(tmp_path: Path) -> None:
    analysis = _build(tmp_path, [_quality_row()], [_metrics_row()])

    assert "ticker_category" not in analysis.columns
    assert analysis.iloc[0]["fundamental_analysis_state"] == "ANALYSIS_READY"


def test_output_path_writes_only_when_explicitly_supplied(tmp_path: Path) -> None:
    quality_path = tmp_path / "fundamental_quality.csv"
    metrics_path = tmp_path / "fundamental_metrics.csv"
    output_path = tmp_path / "nested" / "fundamental_analysis.csv"
    implicit_output_path = tmp_path / "fundamental_analysis.csv"
    _write_csv(quality_path, [_quality_row()])
    _write_csv(metrics_path, [_metrics_row()])

    build_fundamental_analysis(quality_path, metrics_path)
    assert not implicit_output_path.exists()

    analysis = build_fundamental_analysis(quality_path, metrics_path, output_path)

    assert output_path.exists()
    written = pd.read_csv(output_path)
    assert len(written) == len(analysis)
    assert list(written.columns) == ANALYSIS_COLUMNS


def test_invalid_quality_input_fails_deterministically(tmp_path: Path) -> None:
    quality_path = tmp_path / "fundamental_quality.csv"
    _write_csv(quality_path, [{"ticker": "MSFT", "date": "2026-05-28"}])

    with pytest.raises(ValueError, match="fundamental_quality.csv is missing required columns"):
        build_fundamental_analysis(quality_path)
