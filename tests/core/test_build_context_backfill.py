from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.core import build_context_backfill as b

FORBIDDEN_CONTEXT_FIELDS = {
    "context_tradeable",
    "tradeability",
    "conviction",
    "allocation_priority",
    "final_action",
    "urgency",
    "actionable",
    "BUY",
    "SELL",
    "HOLD",
    "TRIM",
    "REMOVE",
}


def _prices(start="2026-01-01", periods=40, base=100.0, step=1.0):
    idx = pd.bdate_range(start, periods=periods)
    return pd.DataFrame({"close": [base + i * step for i in range(periods)]}, index=idx)


def test_missing_required_column_fails():
    df = pd.DataFrame({"ticker": ["AAA"], "scan_date": ["2026-02-01"]})
    with pytest.raises(ValueError, match="Missing required scan columns"):
        b.validate_scans_input(df)


def test_duplicate_ticker_date_fails():
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "scan_date": ["2026-02-01", "2026-02-01"],
            "primary_setup": ["BREAKOUT", "BREAKOUT"],
        }
    )
    with pytest.raises(ValueError, match="Duplicate primary key"):
        b.validate_scans_input(df)


def test_output_row_count_validation():
    out = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "date": "2026-02-01",
                "rs_score": 1.0,
                "rs_rank": 1,
                "rs_percentile": 100.0,
                "rs_20d": 1.0,
                "benchmark_return_20d": 0.5,
                "rs_vs_market": 1.0,
                "rs_vs_sector": np.nan,
                "context_strength": "STRONG",
                "context_reason": "upper_quartile_leadership",
                "leadership_state": "STRONG",
            }
        ]
    )
    b.validate_output(out, expected_rows=1)
    with pytest.raises(ValueError, match="row count mismatch"):
        b.validate_output(out, expected_rows=2)


def test_backfill_schema_is_exact_and_governance_clean():
    assert b.OUTPUT_COLUMNS == [
        "ticker",
        "date",
        "rs_score",
        "rs_rank",
        "rs_percentile",
        "rs_20d",
        "benchmark_return_20d",
        "rs_vs_market",
        "rs_vs_sector",
        "context_strength",
        "context_reason",
        "leadership_state",
    ]
    assert set(b.OUTPUT_COLUMNS).isdisjoint(FORBIDDEN_CONTEXT_FIELDS)


def test_no_future_candle_usage():
    prices = _prices(periods=30, base=100, step=1)
    scan_date = pd.Timestamp("2026-02-18")  # after available data; aligns to last <= scan_date
    result = b.calculate_20d_return(prices, scan_date)
    assert result.aligned_date == prices.index.max()
    assert result.aligned_date <= scan_date


def test_strong_classification_correct():
    assert b.classify_context(0.26, np.nan) == ("STRONG", "upper_quartile_leadership")


def test_weak_classification_correct():
    assert b.classify_context(-0.26, np.nan) == ("WEAK", "lower_distribution")


def test_neutral_classification_correct():
    assert b.classify_context(0.25, np.nan) == ("NEUTRAL", "middle_distribution")
    assert b.classify_context(-0.25, np.nan) == ("NEUTRAL", "middle_distribution")


def test_positive_relative_strength_does_not_require_sector_data():
    strength, reason = b.classify_context(5.0, np.nan)
    assert strength == "STRONG"
    assert reason == "upper_quartile_leadership"


def test_cross_sectional_percentile_classification():
    assert b.classify_percentile(95.0) == ("LEADING", "top_decile_leadership")


def test_unknown_missing_price_data_policy(tmp_path: Path, monkeypatch):
    scans_path = tmp_path / "scans.csv"
    out_path = tmp_path / "context.csv"
    log_path = tmp_path / "log.csv"

    pd.DataFrame(
        {
            "ticker": ["AAA"],
            "scan_date": ["2026-02-01"],
            "primary_setup": ["BREAKOUT"],
        }
    ).to_csv(scans_path, index=False)

    def fake_download(symbols, start, end):
        return {symbol: pd.DataFrame(columns=["close"]) for symbol in symbols}, {"AAA": "empty_download"}

    monkeypatch.setattr(b, "_download_ohlcv", fake_download)

    output = b.build_context_backfill(scans_path, out_path, log_path)
    row = output.iloc[0]
    assert row["context_strength"] == "UNKNOWN"
    assert row["context_reason"] == "missing_price_data"
    assert pd.isna(row["rs_score"])
    assert pd.isna(row["rs_20d"])
    assert row["benchmark_return_20d"] == 0.0
    assert pd.isna(row["rs_vs_market"])
    assert pd.isna(row["rs_vs_sector"])
    assert set(output.columns).isdisjoint(FORBIDDEN_CONTEXT_FIELDS)


def test_allowed_enum_validation_fails():
    out = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "date": "2026-02-01",
                "rs_score": 1.0,
                "rs_rank": 1,
                "rs_percentile": 100.0,
                "rs_20d": 1.0,
                "benchmark_return_20d": 0.5,
                "rs_vs_market": 1.0,
                "rs_vs_sector": np.nan,
                "context_strength": "INVALID",
                "context_reason": "upper_quartile_leadership",
                "leadership_state": "INVALID",
            }
        ]
    )
    with pytest.raises(ValueError, match="Invalid context_strength"):
        b.validate_output(out, expected_rows=1)


def test_build_context_backfill_preserves_rows_and_writes_clean_schema(tmp_path: Path, monkeypatch):
    scans_path = tmp_path / "scans.csv"
    out_path = tmp_path / "context_strength_historical.csv"
    log_path = tmp_path / "context_backfill_log.csv"

    pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "scan_date": ["2026-02-25", "2026-02-25", "2026-02-25"],
            "primary_setup": ["BREAKOUT", "PULLBACK", "BASE"],
        }
    ).to_csv(scans_path, index=False)

    def fake_download(symbols, start, end):
        prices = {
            "AAA": _prices(periods=45, base=100, step=2),
            "BBB": _prices(periods=45, base=100, step=1),
            "CCC": _prices(periods=45, base=100, step=-1),
            "SPY": _prices(periods=45, base=100, step=0.5),
        }
        return {symbol: prices[symbol] for symbol in symbols}, {}

    monkeypatch.setattr(b, "_download_ohlcv", fake_download)

    output = b.build_context_backfill(scans_path, out_path, log_path)
    written_df = pd.read_csv(out_path)

    assert len(output) == 3
    assert len(written_df) == 3
    assert list(output.columns) == b.OUTPUT_COLUMNS
    assert list(written_df.columns) == b.OUTPUT_COLUMNS
    assert set(output.columns).isdisjoint(FORBIDDEN_CONTEXT_FIELDS)
    assert set(written_df.columns).isdisjoint(FORBIDDEN_CONTEXT_FIELDS)
    assert output["rs_vs_sector"].isna().all()
    assert set(output["leadership_state"]) == set(output["context_strength"])
