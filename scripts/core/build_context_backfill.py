"""
Historical Context Strength Backfill

Additive, point-in-time reconstruction of historical context_strength records.

Output:
    data/processed/context_strength_historical.csv
    data/logs/context_backfill_log.csv

This script does NOT implement decision logic.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


DEFAULT_BENCHMARK = "SPY"
NEUTRAL_BAND = 0.25  # percentage points
MAX_ALIGNMENT_TOLERANCE_DAYS = 7
LOOKBACK_BUFFER_DAYS = 90

REQUIRED_SCAN_COLUMNS = {"ticker", "scan_date", "primary_setup"}
OPTIONAL_SCAN_COLUMNS = {"sector", "regime"}

OUTPUT_COLUMNS = [
    "ticker",
    "date",
    "rs_20d",
    "benchmark_return_20d",
    "rs_vs_market",
    "rs_vs_sector",
    "context_strength",
    "context_reason",
]

ALLOWED_CONTEXT_STRENGTH = {"WEAK", "NEUTRAL", "STRONG", "LEADING", "UNKNOWN"}
ALLOWED_CONTEXT_REASON = {
    "missing_rs_20d",
    "neutral_rs",
    "negative_rs",
    "market_outperformance",
    "market_and_sector_outperformance",
    "missing_price_data",
    "fallback",
}


@dataclass(frozen=True)
class PriceResult:
    return_20d: float | None
    aligned_date: pd.Timestamp | None
    reason: str | None = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalise_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def validate_scans_input(scans: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_SCAN_COLUMNS - set(scans.columns)
    if missing:
        raise ValueError(f"Missing required scan columns: {sorted(missing)}")

    if scans.empty:
        raise ValueError("Input scans file is empty")

    scans = scans.copy()
    scans["ticker"] = scans["ticker"].astype(str).str.strip().str.upper()
    scans["scan_date"] = _normalise_date(scans["scan_date"])

    if scans["ticker"].eq("").any():
        raise ValueError("Empty ticker found in scans input")
    if scans["scan_date"].isna().any():
        raise ValueError("Invalid scan_date found in scans input")

    duplicate_mask = scans.duplicated(["ticker", "scan_date"], keep=False)
    if duplicate_mask.any():
        examples = scans.loc[duplicate_mask, ["ticker", "scan_date"]].head(10).to_dict("records")
        raise ValueError(f"Duplicate primary key found in scans input: {examples}")

    return scans


def _download_ohlcv(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    if yf is None:
        raise ImportError("yfinance is required. Install with: pip install yfinance")

    data: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}

    for symbol in sorted(set(symbols)):
        try:
            raw = yf.download(
                symbol,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=False,
            )
            if raw is None or raw.empty:
                errors[symbol] = "empty_download"
                data[symbol] = pd.DataFrame(columns=["close"])
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                # yfinance can return multi-index columns even for one ticker.
                close = raw.xs("Close", axis=1, level=0, drop_level=False)
                close_series = close.iloc[:, 0]
            else:
                close_series = raw["Close"]

            df = pd.DataFrame({"close": pd.to_numeric(close_series, errors="coerce")})
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            df = df.dropna(subset=["close"]).sort_index()
            data[symbol] = df
        except Exception as exc:  # noqa: BLE001 - explicit logging requirement
            errors[symbol] = repr(exc)
            data[symbol] = pd.DataFrame(columns=["close"])

    return data, errors


def _aligned_position(price_df: pd.DataFrame, scan_date: pd.Timestamp) -> int | None:
    if price_df.empty:
        return None
    eligible_dates = price_df.index[price_df.index <= scan_date]
    if len(eligible_dates) == 0:
        return None
    aligned_date = eligible_dates.max()
    if (scan_date - aligned_date).days > MAX_ALIGNMENT_TOLERANCE_DAYS:
        return None
    return int(price_df.index.get_loc(aligned_date))


def calculate_20d_return(price_df: pd.DataFrame, scan_date: pd.Timestamp) -> PriceResult:
    pos = _aligned_position(price_df, scan_date)
    if pos is None:
        return PriceResult(None, None, "missing_price_data")
    if pos < 20:
        return PriceResult(None, price_df.index[pos], "missing_price_data")

    end_close = float(price_df.iloc[pos]["close"])
    start_close = float(price_df.iloc[pos - 20]["close"])

    if not np.isfinite(end_close) or not np.isfinite(start_close) or start_close <= 0:
        return PriceResult(None, price_df.index[pos], "missing_price_data")

    # Return in percentage points, matching NEUTRAL_BAND = 0.25 percentage points.
    return_pct = (end_close / start_close - 1.0) * 100.0
    return PriceResult(float(return_pct), price_df.index[pos], None)


def classify_context(rs_20d: float | None, rs_vs_sector: float | None) -> tuple[str, str]:
    if rs_20d is None or pd.isna(rs_20d):
        return "UNKNOWN", "missing_rs_20d"

    if abs(rs_20d) <= NEUTRAL_BAND:
        return "NEUTRAL", "neutral_rs"

    if rs_20d < -NEUTRAL_BAND:
        return "WEAK", "negative_rs"

    if rs_20d > NEUTRAL_BAND:
        if rs_vs_sector is not None and not pd.isna(rs_vs_sector) and rs_vs_sector > NEUTRAL_BAND:
            return "LEADING", "market_and_sector_outperformance"
        return "STRONG", "market_outperformance"

    return "UNKNOWN", "fallback"


def build_context_backfill(
    scans_path: Path,
    output_path: Path,
    log_path: Path,
    benchmark: str = DEFAULT_BENCHMARK,
) -> pd.DataFrame:
    scans = validate_scans_input(pd.read_csv(scans_path))

    min_date = scans["scan_date"].min() - pd.Timedelta(days=LOOKBACK_BUFFER_DAYS)
    max_date = scans["scan_date"].max() + pd.Timedelta(days=1)

    tickers = scans["ticker"].dropna().unique().tolist()
    symbols = sorted(set(tickers + [benchmark.upper()]))
    prices, fetch_errors = _download_ohlcv(symbols, min_date, max_date)
    benchmark_prices = prices.get(benchmark.upper(), pd.DataFrame(columns=["close"]))

    rows: list[dict[str, Any]] = []

    for record in scans.to_dict("records"):
        ticker = record["ticker"]
        scan_date = pd.Timestamp(record["scan_date"]).normalize()

        ticker_result = calculate_20d_return(prices.get(ticker, pd.DataFrame(columns=["close"])), scan_date)
        benchmark_result = calculate_20d_return(benchmark_prices, scan_date)

        if ticker_result.return_20d is None or benchmark_result.return_20d is None:
            context_strength = "UNKNOWN"
            context_reason = "missing_price_data"
            rs_20d = 0.0
            benchmark_return_20d = 0.0
            rs_vs_market = 0.0
            rs_vs_sector = np.nan
        else:
            benchmark_return_20d = benchmark_result.return_20d
            rs_vs_market = ticker_result.return_20d - benchmark_return_20d
            rs_20d = rs_vs_market
            rs_vs_sector = np.nan  # Phase 1: sector ETF mapping intentionally not implemented.
            context_strength, context_reason = classify_context(rs_20d, rs_vs_sector)

        rows.append(
            {
                "ticker": ticker,
                "date": scan_date.strftime("%Y-%m-%d"),
                "rs_20d": round(float(rs_20d), 4),
                "benchmark_return_20d": round(float(benchmark_return_20d), 4),
                "rs_vs_market": round(float(rs_vs_market), 4),
                "rs_vs_sector": rs_vs_sector,
                "context_strength": context_strength,
                "context_reason": context_reason,
            }
        )

    output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    validate_output(output, expected_rows=len(scans))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    write_log(output, log_path, benchmark, fetch_errors)

    return output


def validate_output(output: pd.DataFrame, expected_rows: int) -> None:
    missing_cols = set(OUTPUT_COLUMNS) - set(output.columns)
    if missing_cols:
        raise ValueError(f"Output missing columns: {sorted(missing_cols)}")

    if len(output) != expected_rows:
        raise ValueError(f"Output row count mismatch: output={len(output)} input={expected_rows}")

    duplicate_mask = output.duplicated(["ticker", "date"], keep=False)
    if duplicate_mask.any():
        examples = output.loc[duplicate_mask, ["ticker", "date"]].head(10).to_dict("records")
        raise ValueError(f"Duplicate primary key found in output: {examples}")

    invalid_strength = set(output["context_strength"].dropna()) - ALLOWED_CONTEXT_STRENGTH
    if invalid_strength:
        raise ValueError(f"Invalid context_strength values: {sorted(invalid_strength)}")

    invalid_reason = set(output["context_reason"].dropna()) - ALLOWED_CONTEXT_REASON
    if invalid_reason:
        raise ValueError(f"Invalid context_reason values: {sorted(invalid_reason)}")


def write_log(output: pd.DataFrame, log_path: Path, benchmark: str, fetch_errors: dict[str, str]) -> None:
    counts = output["context_strength"].value_counts().to_dict()
    log_row = {
        "run_date": datetime.now(timezone.utc).isoformat(),
        "total_rows": int(len(output)),
        "weak_count": int(counts.get("WEAK", 0)),
        "neutral_count": int(counts.get("NEUTRAL", 0)),
        "strong_count": int(counts.get("STRONG", 0)),
        "leading_count": int(counts.get("LEADING", 0)),
        "unknown_count": int(counts.get("UNKNOWN", 0)),
        "missing_price_data_count": int((output["context_reason"] == "missing_price_data").sum()),
        "benchmark": benchmark.upper(),
        "fetch_errors": json.dumps(fetch_errors, sort_keys=True),
    }

    log_df = pd.DataFrame([log_row])
    if log_path.exists():
        existing = pd.read_csv(log_path)
        log_df = pd.concat([existing, log_df], ignore_index=True)
    log_df.to_csv(log_path, index=False)


def parse_args() -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Build historical context strength backfill")
    parser.add_argument("--scans", type=Path, default=root / "data/logs/scans_validation_dedup.csv")
    parser.add_argument("--output", type=Path, default=root / "data/processed/context_strength_historical.csv")
    parser.add_argument("--log", type=Path, default=root / "data/logs/context_backfill_log.csv")
    parser.add_argument("--benchmark", type=str, default=DEFAULT_BENCHMARK)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_context_backfill(
        scans_path=args.scans,
        output_path=args.output,
        log_path=args.log,
        benchmark=args.benchmark,
    )
    print(f"Wrote {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()
