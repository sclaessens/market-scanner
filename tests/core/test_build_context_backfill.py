from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.core import build_context_backfill as b


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
                "rs_20d": 1.0,
                "benchmark_return_20d": 0.5,
                "rs_vs_market": 1.0,
                "rs_vs_sector": np.nan,
                "context_strength": "STRONG",
                "context_reason": "market_outperformance",
            }
        ]
    )
    b.validate_output(out, expected_rows=1)
    with pytest.raises(ValueError, match="row count mismatch"):
        b.validate_output(out, expected_rows=2)


def test_no_future_candle_usage():
    prices = _prices(periods=30, base=100, step=1)
    scan_date = pd.Timestamp("2026-02-18")  # after available data; aligns to last <= scan_date
    result = b.calculate_20d_return(prices, scan_date)
    assert result.aligned_date == prices.index.max()
    assert result.aligned_date <= scan_date


def test_strong_classification_correct():
    assert b.classify_context(0.26, np.nan) == ("STRONG", "market_outperformance")


def test_weak_classification_correct():
    assert b.classify_context(-0.26, np.nan) == ("WEAK", "negative_rs")


def test_neutral_classification_correct():
    assert b.classify_context(0.25, np.nan) == ("NEUTRAL", "neutral_rs")
    assert b.classify_context(-0.25, np.nan) == ("NEUTRAL", "neutral_rs")


def test_leading_impossible_without_sector_data():
    strength, reason = b.classify_context(5.0, np.nan)
    assert strength == "STRONG"
    assert reason == "market_outperformance"


def test_leading_when_sector_data_available():
    assert b.classify_context(5.0, 1.0) == ("LEADING", "market_and_sector_outperformance")


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
    assert row["rs_20d"] == 0.0
    assert row["benchmark_return_20d"] == 0.0
    assert row["rs_vs_market"] == 0.0
    assert pd.isna(row["rs_vs_sector"])


def test_allowed_enum_validation_fails():
    out = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "date": "2026-02-01",
                "rs_20d": 1.0,
                "benchmark_return_20d": 0.5,
                "rs_vs_market": 1.0,
                "rs_vs_sector": np.nan,
                "context_strength": "INVALID",
                "context_reason": "market_outperformance",
            }
        ]
    )
    with pytest.raises(ValueError, match="Invalid context_strength"):
        b.validate_output(out, expected_rows=1)
