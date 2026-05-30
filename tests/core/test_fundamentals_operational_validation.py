from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.fundamentals import build_quality as quality_module
from scripts.fundamentals.build_analysis import ANALYSIS_COLUMNS, build_fundamental_analysis
from scripts.fundamentals.build_history_intake import validate_fundamentals_history
from scripts.fundamentals.build_metrics import (
    HELPER_COLUMNS,
    METRIC_COLUMNS,
    build_fundamental_metrics,
)

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


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _context_row(ticker: str) -> dict[str, str]:
    return {
        "ticker": ticker,
        "date": "2026-05-30",
        "rs_score": "1.0",
        "rs_percentile": "50.0",
        "rs_rank": "1",
        "rs_vs_market": "1.0",
        "rs_vs_sector": "",
        "context_strength": "NEUTRAL",
        "context_reason": "synthetic operational validation",
        "leadership_state": "NEUTRAL",
    }


def _history_row(ticker: str, fiscal_year: str, **overrides: str) -> dict[str, str]:
    row = {
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "fiscal_period": "FY",
        "period_end_date": f"{fiscal_year}-12-31",
        "report_date": f"{int(fiscal_year) + 1}-02-15",
        "currency": "USD",
        "revenue": "1200" if fiscal_year == "2025" else "1000",
        "gross_profit": "720" if fiscal_year == "2025" else "600",
        "operating_income": "360" if fiscal_year == "2025" else "300",
        "net_income": "240" if fiscal_year == "2025" else "200",
        "diluted_eps": "6" if fiscal_year == "2025" else "5",
        "total_debt": "150" if fiscal_year == "2025" else "120",
        "total_equity": "600" if fiscal_year == "2025" else "550",
        "free_cash_flow": "260" if fiscal_year == "2025" else "200",
        "source_name": "synthetic filing",
        "source_reference": f"{ticker} synthetic FY{fiscal_year}",
        "source_freshness_date": "2026-05-30",
        "extraction_date": "2026-05-30",
        "notes": "synthetic E8 validation fixture",
    }
    row.update(overrides)
    return row


def _synthetic_history_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for ticker in ["AAPL_SAMPLE", "NEG_MARGIN_SAMPLE", "PARTIAL_SAMPLE", "STALE_OR_LIMITED_SAMPLE"]:
        rows.append(_history_row(ticker, "2024"))
        rows.append(_history_row(ticker, "2025"))

    for row in rows:
        if row["ticker"] == "NEG_MARGIN_SAMPLE" and row["fiscal_year"] == "2025":
            row["operating_income"] = "-50"
            row["net_income"] = "-80"
        if row["ticker"] == "PARTIAL_SAMPLE" and row["fiscal_year"] == "2025":
            row["total_equity"] = ""
        if row["ticker"] == "STALE_OR_LIMITED_SAMPLE":
            row["source_freshness_date"] = "2025-12-01"
    return rows


def _run_operational_flow(tmp_path: Path, monkeypatch) -> dict[str, pd.DataFrame | Path | dict]:
    context_path = tmp_path / "context_strength.csv"
    history_path = tmp_path / "fundamentals_history.csv"
    metrics_path = tmp_path / "fundamental_metrics.csv"
    quality_path = tmp_path / "fundamental_quality.csv"
    analysis_path = tmp_path / "fundamental_analysis.csv"
    log_path = tmp_path / "fundamental_layer_log.csv"

    _write_csv(
        context_path,
        [
            _context_row("AAPL_SAMPLE"),
            _context_row("NEG_MARGIN_SAMPLE"),
            _context_row("PARTIAL_SAMPLE"),
            _context_row("STALE_OR_LIMITED_SAMPLE"),
            _context_row("MISSING_SAMPLE"),
        ],
    )
    _write_csv(history_path, _synthetic_history_rows())

    monkeypatch.setattr(quality_module, "CONTEXT_PATH", context_path)
    monkeypatch.setattr(quality_module, "OUTPUT_PATH", quality_path)
    monkeypatch.setattr(quality_module, "LOG_PATH", log_path)

    validation_result = validate_fundamentals_history(history_path)
    metrics_df = build_fundamental_metrics(history_path, metrics_path)
    second_metrics_df = build_fundamental_metrics(history_path)
    pd.testing.assert_frame_equal(metrics_df, second_metrics_df)

    quality_df = quality_module.build_fundamental_layer(
        generated_at="2026-05-30 12:00:00",
        fundamentals_history_path=history_path,
        fundamental_metrics_path=metrics_path,
    )
    analysis_df = build_fundamental_analysis(quality_path, metrics_path, analysis_path)

    return {
        "validation_result": validation_result,
        "metrics_df": metrics_df,
        "quality_df": quality_df,
        "analysis_df": analysis_df,
        "history_path": history_path,
        "metrics_path": metrics_path,
        "quality_path": quality_path,
        "analysis_path": analysis_path,
        "log_path": log_path,
    }


def test_operational_fundamentals_flow_produces_reviewable_outputs(tmp_path: Path, monkeypatch) -> None:
    result = _run_operational_flow(tmp_path, monkeypatch)
    metrics_df = result["metrics_df"]
    quality_df = result["quality_df"]
    analysis_df = result["analysis_df"]

    assert result["validation_result"]["status"] == "VALID"
    assert result["validation_result"]["row_count"] == 8

    expected_metric_columns = set(METRIC_COLUMNS + HELPER_COLUMNS)
    assert expected_metric_columns.issubset(metrics_df.columns)
    aapl_metrics = metrics_df[
        (metrics_df["ticker"] == "AAPL_SAMPLE")
        & (metrics_df["fiscal_year"].astype(str) == "2025")
    ].iloc[0]
    assert aapl_metrics["gross_margin"] == 0.6
    assert aapl_metrics["net_margin"] == 0.2
    assert aapl_metrics["revenue_yoy_growth"] == 0.2
    assert aapl_metrics["eps_yoy_growth"] == 0.2
    assert aapl_metrics["free_cash_flow_yoy_growth"] == 0.3

    assert list(quality_df.columns) == quality_module.OUTPUT_COLUMNS
    assert len(quality_df) == 5
    assert list(quality_df["ticker"]) == [
        "AAPL_SAMPLE",
        "NEG_MARGIN_SAMPLE",
        "PARTIAL_SAMPLE",
        "STALE_OR_LIMITED_SAMPLE",
        "MISSING_SAMPLE",
    ]
    assert quality_df.loc[quality_df["ticker"] == "AAPL_SAMPLE", "quality_state"].iloc[0] == "SUFFICIENT_DATA"
    assert quality_df.loc[quality_df["ticker"] == "PARTIAL_SAMPLE", "quality_state"].iloc[0] == "PARTIAL_DATA"
    assert quality_df.loc[quality_df["ticker"] == "STALE_OR_LIMITED_SAMPLE", "quality_state"].iloc[0] == "STALE_DATA"
    missing_quality = quality_df.loc[quality_df["ticker"] == "MISSING_SAMPLE"].iloc[0]
    assert missing_quality["quality_state"] == "INSUFFICIENT_DATA"
    assert missing_quality["quality_metadata_status"] == "row_missing"

    assert list(analysis_df.columns) == ANALYSIS_COLUMNS
    assert len(analysis_df) == 5
    assert list(analysis_df["ticker"]) == list(quality_df["ticker"])

    aapl_analysis = analysis_df.loc[analysis_df["ticker"] == "AAPL_SAMPLE"].iloc[0]
    assert aapl_analysis["fundamental_analysis_state"] == "ANALYSIS_READY"
    assert aapl_analysis["margin_profile_state"] == "MARGIN_STABLE"
    assert aapl_analysis["growth_profile_state"] == "GROWTH_POSITIVE"
    assert aapl_analysis["cash_flow_profile_state"] == "CASH_FLOW_POSITIVE"

    negative_analysis = analysis_df.loc[analysis_df["ticker"] == "NEG_MARGIN_SAMPLE"].iloc[0]
    assert negative_analysis["margin_profile_state"] == "MARGIN_NEGATIVE"
    assert negative_analysis["fundamental_profile_state"] == "DETERIORATING_PROFILE"

    partial_analysis = analysis_df.loc[analysis_df["ticker"] == "PARTIAL_SAMPLE"].iloc[0]
    assert partial_analysis["fundamental_analysis_state"] == "LIMITED_ANALYSIS"
    assert partial_analysis["analysis_data_status"] == "metrics_partial"

    stale_analysis = analysis_df.loc[analysis_df["ticker"] == "STALE_OR_LIMITED_SAMPLE"].iloc[0]
    assert stale_analysis["fundamental_review_flag"] == "REVIEW_STALE_SOURCE"
    assert stale_analysis["analysis_data_status"] == "stale_source"

    missing_analysis = analysis_df.loc[analysis_df["ticker"] == "MISSING_SAMPLE"].iloc[0]
    assert missing_analysis["fundamental_analysis_state"] == "INSUFFICIENT_DATA"
    assert missing_analysis["fundamental_review_flag"] == "REVIEW_DATA_LIMITATION"

    assert result["metrics_path"].exists()
    assert result["quality_path"].exists()
    assert result["analysis_path"].exists()
    assert result["log_path"].exists()


def test_invalid_raw_history_fails_before_metrics_output(tmp_path: Path) -> None:
    history_path = tmp_path / "invalid_fundamentals_history.csv"
    metrics_path = tmp_path / "fundamental_metrics.csv"
    duplicate_rows = [
        _history_row("DUPLICATE_SAMPLE", "2025"),
        _history_row("DUPLICATE_SAMPLE", "2025"),
    ]
    _write_csv(history_path, duplicate_rows)

    validation_result = validate_fundamentals_history(history_path)

    assert validation_result["status"] == "INVALID"
    assert validation_result["duplicate_key_count"] == 2
    assert not metrics_path.exists()


def test_operational_analysis_output_contains_no_forbidden_semantics(tmp_path: Path, monkeypatch) -> None:
    result = _run_operational_flow(tmp_path, monkeypatch)
    analysis_df = result["analysis_df"]

    forbidden_columns = [
        column for column in analysis_df.columns if any(term in column.lower() for term in FORBIDDEN_TERMS)
    ]
    assert forbidden_columns == []

    forbidden_values = []
    for column in analysis_df.columns:
        for value in analysis_df[column].astype(str):
            normalized_value = value.strip().lower()
            if any(term in normalized_value for term in FORBIDDEN_TERMS):
                forbidden_values.append(f"{column}={value}")
    assert forbidden_values == []
