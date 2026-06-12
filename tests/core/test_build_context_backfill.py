from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


DEFAULT_BENCHMARK = "SPY"
NEUTRAL_BAND = 0.25
MAX_ALIGNMENT_TOLERANCE_DAYS = 7
LOOKBACK_BUFFER_DAYS = 90

REQUIRED_SCAN_COLUMNS = {"ticker", "scan_date", "primary_setup"}

OUTPUT_COLUMNS = [
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

ALLOWED_CONTEXT_STRENGTH = {"WEAK", "NEUTRAL", "STRONG", "LEADING", "UNKNOWN"}
ALLOWED_CONTEXT_REASON = {
    "missing_rs_20d",
    "top_decile_leadership",
    "upper_quartile_leadership",
    "middle_distribution",
    "lower_distribution",
    "missing_price_data",
    "fallback",
}

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


@dataclass(frozen=True)
class PriceResult:
    return_20d: float | None
    aligned_date: pd.Timestamp | None
    reason: str | None = None


def _prices(start="2026-01-01", periods=40, base=100.0, step=1.0):
    idx = pd.bdate_range(start, periods=periods)
    return pd.DataFrame({"close": [base + i * step for i in range(periods)]}, index=idx)


def _normalise_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _validate_scans_input_contract(scans: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_SCAN_COLUMNS - set(scans.columns)
    if missing:
        raise ValueError(f"Missing required scan columns: {sorted(missing)}")
    if scans.empty:
        raise ValueError("Input scans file is empty")

    scans = scans.copy()
    scans["ticker"] = scans["ticker"].astype(str).str.strip().str.upper()
    scans["scan_date"] = _normalise_date(scans["scan_date"])

    duplicate_mask = scans.duplicated(["ticker", "scan_date"], keep=False)
    if duplicate_mask.any():
        examples = scans.loc[duplicate_mask, ["ticker", "scan_date"]].head(10).to_dict("records")
        raise ValueError(f"Duplicate primary key found in scans input: {examples}")
    return scans


def _aligned_position_contract(price_df: pd.DataFrame, scan_date: pd.Timestamp) -> int | None:
    if price_df.empty:
        return None
    eligible_dates = price_df.index[price_df.index <= scan_date]
    if len(eligible_dates) == 0:
        return None
    aligned_date = eligible_dates.max()
    if (scan_date - aligned_date).days > MAX_ALIGNMENT_TOLERANCE_DAYS:
        return None
    return int(price_df.index.get_loc(aligned_date))


def _calculate_20d_return_contract(price_df: pd.DataFrame, scan_date: pd.Timestamp) -> PriceResult:
    pos = _aligned_position_contract(price_df, scan_date)
    if pos is None:
        return PriceResult(None, None, "missing_price_data")
    if pos < 20:
        return PriceResult(None, price_df.index[pos], "missing_price_data")

    end_close = float(price_df.iloc[pos]["close"])
    start_close = float(price_df.iloc[pos - 20]["close"])
    if not np.isfinite(end_close) or not np.isfinite(start_close) or start_close <= 0:
        return PriceResult(None, price_df.index[pos], "missing_price_data")

    return_pct = (end_close / start_close - 1.0) * 100.0
    return PriceResult(float(return_pct), price_df.index[pos], None)


def _classify_context_contract(rs_20d: float | None, rs_vs_sector: float | None) -> tuple[str, str]:
    del rs_vs_sector
    if rs_20d is None or pd.isna(rs_20d):
        return "UNKNOWN", "missing_rs_20d"
    if abs(rs_20d) <= NEUTRAL_BAND:
        return "NEUTRAL", "middle_distribution"
    if rs_20d < -NEUTRAL_BAND:
        return "WEAK", "lower_distribution"
    if rs_20d > NEUTRAL_BAND:
        return "STRONG", "upper_quartile_leadership"
    return "UNKNOWN", "fallback"


def _classify_percentile_contract(percentile: float | None) -> tuple[str, str]:
    if percentile is None or pd.isna(percentile):
        return "UNKNOWN", "missing_rs_20d"
    if percentile >= 90:
        return "LEADING", "top_decile_leadership"
    if percentile >= 75:
        return "STRONG", "upper_quartile_leadership"
    if percentile >= 40:
        return "NEUTRAL", "middle_distribution"
    return "WEAK", "lower_distribution"


def _validate_output_contract(output: pd.DataFrame, expected_rows: int) -> None:
    missing_cols = set(OUTPUT_COLUMNS) - set(output.columns)
    if missing_cols:
        raise ValueError(f"Output missing columns: {sorted(missing_cols)}")
    if len(output) != expected_rows:
        raise ValueError(f"Output row count mismatch: output={len(output)} input={expected_rows}")
    invalid_strength = set(output["context_strength"].dropna()) - ALLOWED_CONTEXT_STRENGTH
    if invalid_strength:
        raise ValueError(f"Invalid context_strength values: {sorted(invalid_strength)}")
    invalid_reason = set(output["context_reason"].dropna()) - ALLOWED_CONTEXT_REASON
    if invalid_reason:
        raise ValueError(f"Invalid context_reason values: {sorted(invalid_reason)}")


def _build_context_backfill_contract(
    scans_path: Path,
    output_path: Path,
    log_path: Path,
    prices: dict[str, pd.DataFrame],
    fetch_errors: dict[str, str] | None = None,
    benchmark: str = DEFAULT_BENCHMARK,
) -> pd.DataFrame:
    scans = _validate_scans_input_contract(pd.read_csv(scans_path))
    min_date = scans["scan_date"].min() - pd.Timedelta(days=LOOKBACK_BUFFER_DAYS)
    max_date = scans["scan_date"].max() + pd.Timedelta(days=1)
    assert min_date < max_date

    benchmark_prices = prices.get(benchmark.upper(), pd.DataFrame(columns=["close"]))
    rows: list[dict[str, Any]] = []
    for record in scans.to_dict("records"):
        ticker = record["ticker"]
        scan_date = pd.Timestamp(record["scan_date"]).normalize()
        ticker_result = _calculate_20d_return_contract(prices.get(ticker, pd.DataFrame(columns=["close"])), scan_date)
        benchmark_result = _calculate_20d_return_contract(benchmark_prices, scan_date)

        if ticker_result.return_20d is None or benchmark_result.return_20d is None:
            context_strength = "UNKNOWN"
            context_reason = "missing_price_data"
            rs_20d = np.nan
            benchmark_return_20d = 0.0
            rs_vs_market = np.nan
            rs_vs_sector = np.nan
        else:
            benchmark_return_20d = benchmark_result.return_20d
            rs_vs_market = ticker_result.return_20d - benchmark_return_20d
            rs_20d = rs_vs_market
            rs_vs_sector = np.nan
            context_strength, context_reason = _classify_context_contract(rs_20d, rs_vs_sector)

        rows.append(
            {
                "ticker": ticker,
                "date": scan_date.strftime("%Y-%m-%d"),
                "rs_score": rs_20d,
                "rs_rank": pd.NA,
                "rs_percentile": np.nan,
                "rs_20d": rs_20d,
                "benchmark_return_20d": round(float(benchmark_return_20d), 4),
                "rs_vs_market": rs_vs_market,
                "rs_vs_sector": rs_vs_sector,
                "context_strength": context_strength,
                "context_reason": context_reason,
                "leadership_state": context_strength,
            }
        )

    output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    ready = output["rs_score"].notna()
    if ready.any():
        output.loc[ready, "rs_rank"] = (
            output.loc[ready].groupby("date")["rs_score"].rank(method="first", ascending=False).astype("Int64")
        )
        output.loc[ready, "rs_percentile"] = (
            output.loc[ready].groupby("date")["rs_score"].rank(method="average", pct=True) * 100
        )
        classifications = output.loc[ready, "rs_percentile"].apply(_classify_percentile_contract)
        output.loc[ready, "context_strength"] = classifications.apply(lambda item: item[0])
        output.loc[ready, "context_reason"] = classifications.apply(lambda item: item[1])
        output.loc[ready, "leadership_state"] = output.loc[ready, "context_strength"]

    for column in ["rs_score", "rs_percentile", "rs_20d", "rs_vs_market", "rs_vs_sector"]:
        output[column] = pd.to_numeric(output[column], errors="coerce").round(4)

    _validate_output_contract(output, expected_rows=len(scans))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    pd.DataFrame(
        [
            {
                "total_rows": len(output),
                "benchmark": benchmark.upper(),
                "fetch_errors": str(fetch_errors or {}),
            }
        ]
    ).to_csv(log_path, index=False)
    return output


def test_missing_required_column_fails():
    df = pd.DataFrame({"ticker": ["AAA"], "scan_date": ["2026-02-01"]})
    with pytest.raises(ValueError, match="Missing required scan columns"):
        _validate_scans_input_contract(df)


def test_duplicate_ticker_date_fails():
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "scan_date": ["2026-02-01", "2026-02-01"],
            "primary_setup": ["BREAKOUT", "BREAKOUT"],
        }
    )
    with pytest.raises(ValueError, match="Duplicate primary key"):
        _validate_scans_input_contract(df)


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
    _validate_output_contract(out, expected_rows=1)
    with pytest.raises(ValueError, match="row count mismatch"):
        _validate_output_contract(out, expected_rows=2)


def test_backfill_schema_is_exact_and_governance_clean():
    assert OUTPUT_COLUMNS == [
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
    assert set(OUTPUT_COLUMNS).isdisjoint(FORBIDDEN_CONTEXT_FIELDS)


def test_no_future_candle_usage():
    prices = _prices(periods=30, base=100, step=1)
    scan_date = pd.Timestamp("2026-02-18")
    result = _calculate_20d_return_contract(prices, scan_date)
    assert result.aligned_date == prices.index.max()
    assert result.aligned_date <= scan_date


def test_strong_classification_correct():
    assert _classify_context_contract(0.26, np.nan) == ("STRONG", "upper_quartile_leadership")


def test_weak_classification_correct():
    assert _classify_context_contract(-0.26, np.nan) == ("WEAK", "lower_distribution")


def test_neutral_classification_correct():
    assert _classify_context_contract(0.25, np.nan) == ("NEUTRAL", "middle_distribution")
    assert _classify_context_contract(-0.25, np.nan) == ("NEUTRAL", "middle_distribution")


def test_positive_relative_strength_does_not_require_sector_data():
    strength, reason = _classify_context_contract(5.0, np.nan)
    assert strength == "STRONG"
    assert reason == "upper_quartile_leadership"


def test_cross_sectional_percentile_classification():
    assert _classify_percentile_contract(95.0) == ("LEADING", "top_decile_leadership")


def test_unknown_missing_price_data_policy(tmp_path: Path):
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

    output = _build_context_backfill_contract(
        scans_path,
        out_path,
        log_path,
        prices={"AAA": pd.DataFrame(columns=["close"])},
        fetch_errors={"AAA": "empty_download"},
    )
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
        _validate_output_contract(out, expected_rows=1)


def test_build_context_backfill_preserves_rows_and_writes_clean_schema(tmp_path: Path):
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

    prices = {
        "AAA": _prices(periods=45, base=100, step=2),
        "BBB": _prices(periods=45, base=100, step=1),
        "CCC": _prices(periods=45, base=100, step=-1),
        "SPY": _prices(periods=45, base=100, step=0.5),
    }
    output = _build_context_backfill_contract(scans_path, out_path, log_path, prices=prices)
    written_df = pd.read_csv(out_path)

    assert len(output) == 3
    assert len(written_df) == 3
    assert list(output.columns) == OUTPUT_COLUMNS
    assert list(written_df.columns) == OUTPUT_COLUMNS
    assert set(output.columns).isdisjoint(FORBIDDEN_CONTEXT_FIELDS)
    assert set(written_df.columns).isdisjoint(FORBIDDEN_CONTEXT_FIELDS)
    assert output["rs_vs_sector"].isna().all()
    assert set(output["leadership_state"]) == set(output["context_strength"])
