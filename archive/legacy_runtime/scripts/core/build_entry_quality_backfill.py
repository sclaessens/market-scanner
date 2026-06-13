#!/usr/bin/env python3
"""
Historical Entry Quality Backfill

Additive data-reconstruction layer for the market-scanner project.

Outputs:
- data/processed/entry_quality_metrics_historical.csv
- data/logs/entry_quality_backfill_log.csv

Hard guarantees:
- Point-in-time rolling indicators only use candles up to and including entry date.
- No decision logic, no context logic, no fundamentals.
- One output row per scans_validation input row.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    yf = None


REQUIRED_INPUT_COLUMNS = {"ticker", "entry", "rr"}
DATE_CANDIDATES = ("entry_date", "date", "scan_date")

OUTPUT_COLUMNS = [
    "ticker",
    "date",
    "distance_to_breakout_pct",
    "breakout_extension_atr",
    "extension_atr",
    "distance_ma20_pct",
    "volume_ratio",
    "range_atr",
    "entry_quality_state",
    "entry_quality_reason",
]

ALLOWED_STATES = {
    "BALANCED",
    "EXTENDED",
    "VOLUME_DIVERGENCE",
    "WIDE_RANGE",
    "STRUCTURE_GAP",
    "DATA_GAP",
}

ALLOWED_REASONS = {
    "balanced_structure",
    "too_far_from_breakout",
    "overextended_atr",
    "overextended_ma20",
    "weak_volume",
    "excessive_volume",
    "wide_recent_range",
    "invalid_structure",
    "missing_data",
}

FAIL_CLOSED_MESSAGE = (
    "FAIL_CLOSED: This legacy historical backfill module is fail-closed and "
    "must not be executed manually. It is retained only for historical review "
    "pending controlled archive governance."
)


@dataclass(frozen=True)
class EntryQualityConfig:
    max_distance_breakout_pct: float = 3.0
    max_breakout_extension_atr: float = 2.0
    max_extension_atr: float = 2.5
    min_volume_ratio: float = 0.10
    max_volume_ratio: float = 4.0
    max_range_atr: float = 4.0

    def validate(self) -> None:
        if not (0 < self.min_volume_ratio < self.max_volume_ratio):
            raise ValueError("Invalid config: expected 0 < min_volume_ratio < max_volume_ratio")
        if self.max_distance_breakout_pct <= 0:
            raise ValueError("Invalid config: max_distance_breakout_pct must be > 0")
        if self.max_breakout_extension_atr <= 0:
            raise ValueError("Invalid config: max_breakout_extension_atr must be > 0")
        if self.max_extension_atr <= 0:
            raise ValueError("Invalid config: max_extension_atr must be > 0")
        if self.max_range_atr <= 0:
            raise ValueError("Invalid config: max_range_atr must be > 0")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_date_column(df: pd.DataFrame) -> str:
    for col in DATE_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"Missing date column. Expected one of: {', '.join(DATE_CANDIDATES)}")


def load_scans(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing scans validation file: {path}")

    df = normalize_columns(pd.read_csv(path))
    if df.empty:
        raise ValueError("scans_validation.csv is empty")

    date_col = find_date_column(df)
    missing = sorted((REQUIRED_INPUT_COLUMNS - {"entry"}) - set(df.columns))
    if "entry" not in df.columns and "entry_price" not in df.columns:
        missing.append("entry or entry_price")
    if missing:
        raise ValueError(f"Missing required scans_validation columns: {missing}")

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None).dt.normalize()
    if df["date"].isna().any():
        bad = df[df["date"].isna()].head(10).to_dict("records")
        raise ValueError(f"Invalid dates in scans_validation: {bad}")

    if "entry_price" not in df.columns:
        df["entry_price"] = pd.to_numeric(df["entry"], errors="coerce")
    else:
        df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")

    if df["entry_price"].isna().any():
        raise ValueError("entry_price/entry contains NaN or non-numeric values")

    duplicated = df.duplicated(["ticker", "date"], keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, ["ticker", "date"]].head(20).to_dict("records")
        raise ValueError(
            "Duplicate primary key in scans_validation.csv. "
            "Historical backfill requires UNIQUE (ticker, date). "
            f"Examples: {examples}"
        )

    return df.reset_index(drop=True)


def load_optional_scanner(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    df = normalize_columns(pd.read_csv(path))
    if df.empty or "ticker" not in df.columns:
        return None
    date_col = find_date_column(df)
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["date"])
    if df.duplicated(["ticker", "date"]).any():
        df = df.drop_duplicates(["ticker", "date"], keep="last")
    return df


def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def fetch_ohlcv(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed. Install with: pip install yfinance")

    fetch_start = (start - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    # yfinance end is exclusive; add a small buffer for non-trading-day alignment.
    fetch_end = (end + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    raw = yf.download(
        ticker,
        start=fetch_start,
        end=fetch_end,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise ValueError(f"No OHLCV data returned for {ticker}")

    raw = flatten_yfinance_columns(raw).reset_index()
    raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
    rename = {
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj_close": "adj_close",
        "volume": "volume",
    }
    raw = raw.rename(columns=rename)
    required = ["date", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"OHLCV for {ticker} missing columns: {missing}")

    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None).dt.normalize()
    raw["ticker"] = ticker
    for col in ["high", "low", "close", "volume"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["date", "high", "low", "close", "volume"])
    raw = raw.sort_values("date").drop_duplicates(["ticker", "date"], keep="last")
    return raw[["ticker", "date", "high", "low", "close", "volume"]]


def compute_point_in_time_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    required = ["ticker", "date", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in ohlcv.columns]
    if missing:
        raise ValueError(f"OHLCV dataset missing required columns before indicator calculation: {missing}")
    if ohlcv.empty:
        raise ValueError("OHLCV dataset is empty before indicator calculation")

    df = ohlcv[required].sort_values(["ticker", "date"]).copy()

    frames: list[pd.DataFrame] = []
    for _, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        prev_close = g["close"].shift(1)
        true_range = pd.concat(
            [
                g["high"] - g["low"],
                (g["high"] - prev_close).abs(),
                (g["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        g["ma20"] = g["close"].rolling(window=20, min_periods=20).mean()
        g["atr14"] = true_range.rolling(window=14, min_periods=14).mean()
        g["high_20d"] = g["high"].rolling(window=20, min_periods=20).max()
        g["avg_vol_20"] = g["volume"].rolling(window=20, min_periods=20).mean()
        frames.append(g)

    if not frames:
        raise ValueError("No ticker groups available for indicator calculation")
    return pd.concat(frames, ignore_index=True)


def build_ohlcv_for_scans(scans: pd.DataFrame, scanner_df: pd.DataFrame | None) -> pd.DataFrame:
    ticker_frames: list[pd.DataFrame] = []

    # Use optional scanner rows as a cache only when they match the historical key.
    if scanner_df is not None:
        cols = {"ticker", "date", "high", "low", "close", "volume"}
        if cols.issubset(scanner_df.columns):
            cached = scanner_df[list(cols)].copy()
            for col in ["high", "low", "close", "volume"]:
                cached[col] = pd.to_numeric(cached[col], errors="coerce")
            cached = cached.dropna(subset=list(cols))
            if not cached.empty:
                ticker_frames.append(cached)

    needed = scans.groupby("ticker")["date"].agg(["min", "max"]).reset_index()
    fetched_frames: list[pd.DataFrame] = []
    fetch_errors: dict[str, str] = {}
    for row in needed.itertuples(index=False):
        ticker = str(row.ticker)
        try:
            fetched_frames.append(fetch_ohlcv(ticker, pd.Timestamp(row.min), pd.Timestamp(row.max)))
        except Exception as exc:
            fetch_errors[ticker] = str(exc)

    if fetched_frames:
        ticker_frames.extend(fetched_frames)

    if not ticker_frames:
        raise RuntimeError(
            "No OHLCV data available. Provide scanner data with OHLCV columns or enable yfinance access. "
            f"Fetch errors: {fetch_errors}"
        )

    ohlcv = pd.concat(ticker_frames, ignore_index=True)
    required = ["ticker", "date", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in ohlcv.columns]
    if missing:
        raise RuntimeError(f"OHLCV reconstruction returned an invalid schema. Missing columns: {missing}")

    ohlcv = ohlcv[required].copy()
    for col in ["high", "low", "close", "volume"]:
        ohlcv[col] = pd.to_numeric(ohlcv[col], errors="coerce")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    ohlcv["ticker"] = ohlcv["ticker"].astype(str).str.strip().str.upper()
    ohlcv = ohlcv.dropna(subset=required)

    if ohlcv.empty:
        raise RuntimeError(
            "OHLCV reconstruction returned zero usable rows after cleaning. "
            f"Fetch errors: {fetch_errors}"
        )

    ohlcv = ohlcv.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    ohlcv.attrs["fetch_errors"] = fetch_errors
    return ohlcv


def align_trade_dates(scans: pd.DataFrame, indicators: pd.DataFrame) -> pd.DataFrame:
    required = ["ticker", "date", "high", "low", "close", "volume", "ma20", "atr14", "high_20d", "avg_vol_20"]
    missing = [c for c in required if c not in indicators.columns]
    if missing:
        raise ValueError(f"Indicator dataset missing required columns before date alignment: {missing}")
    if indicators.empty:
        raise ValueError("Indicator dataset is empty before date alignment")

    left = scans[["ticker", "date"]].copy()
    right = indicators.copy()

    # pandas merge_asof requires exactly matching datetime resolutions.
    # Some CSV/yfinance paths produce datetime64[us] while others produce datetime64[s].
    # Normalize both sides explicitly to timezone-naive datetime64[ns] to prevent
    # dtype-dependent failures without changing the calendar date semantics.
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
    left["ticker"] = left["ticker"].astype(str).str.strip().str.upper()
    right["ticker"] = right["ticker"].astype(str).str.strip().str.upper()

    if left["date"].isna().any() or right["date"].isna().any():
        raise ValueError("Date alignment failed because scans or indicators contain invalid dates after normalization.")

    left = left.sort_values(["ticker", "date"])
    right = right.sort_values(["ticker", "date"])

    aligned_parts = []
    missing_tickers = []
    for ticker, g_left in left.groupby("ticker"):
        g_right = right[right["ticker"] == ticker]
        if g_right.empty:
            missing_tickers.append(ticker)
            continue
        aligned = pd.merge_asof(
            g_left.sort_values("date"),
            g_right.sort_values("date"),
            on="date",
            by="ticker",
            direction="backward",
            tolerance=pd.Timedelta(days=7),
        )
        aligned_parts.append(aligned)

    if not aligned_parts:
        raise ValueError(f"No trade dates could be aligned. Missing tickers: {missing_tickers}")
    aligned = pd.concat(aligned_parts, ignore_index=True)
    return scans.merge(aligned, on=["ticker", "date"], how="left")


def calculate_entry_quality(df: pd.DataFrame, config: EntryQualityConfig) -> pd.DataFrame:
    out = df.copy()

    required = ["close", "high", "low", "volume", "high_20d", "ma20", "atr14", "avg_vol_20"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Internal error: missing enriched columns: {missing}")

    out["distance_to_breakout_pct"] = (out["close"] - out["high_20d"]) / out["high_20d"] * 100
    out["breakout_extension_atr"] = (out["close"] - out["high_20d"]) / out["atr14"]
    out["extension_atr"] = (out["close"] - out["ma20"]) / out["atr14"]
    out["distance_ma20_pct"] = (out["close"] - out["ma20"]) / out["ma20"] * 100
    out["volume_ratio"] = out["volume"] / out["avg_vol_20"]
    out["range_atr"] = (out["high"] - out["low"]) / out["atr14"]

    metrics = [
        "distance_to_breakout_pct",
        "breakout_extension_atr",
        "extension_atr",
        "distance_ma20_pct",
        "volume_ratio",
        "range_atr",
    ]

    invalid_structure = (
        (out["high_20d"] <= 0)
        | (out["ma20"] <= 0)
        | (out["atr14"] <= 0)
        | (out["avg_vol_20"] <= 0)
        | (out["volume"] < 0)
        | (out["high"] < out["low"])
    )
    missing_data = out[required].isna().any(axis=1) | out[metrics].replace([np.inf, -np.inf], np.nan).isna().any(axis=1)

    out["entry_quality_state"] = "BALANCED"
    out["entry_quality_reason"] = "balanced_structure"

    out.loc[missing_data, ["entry_quality_state", "entry_quality_reason"]] = ["DATA_GAP", "missing_data"]
    out.loc[invalid_structure & ~missing_data, ["entry_quality_state", "entry_quality_reason"]] = ["STRUCTURE_GAP", "invalid_structure"]

    classifiable = ~(missing_data | invalid_structure)
    rules = [
        (out["distance_to_breakout_pct"] > config.max_distance_breakout_pct, "EXTENDED", "too_far_from_breakout"),
        (out["breakout_extension_atr"] > config.max_breakout_extension_atr, "EXTENDED", "overextended_atr"),
        (out["extension_atr"] > config.max_extension_atr, "EXTENDED", "overextended_ma20"),
        (out["volume_ratio"] < config.min_volume_ratio, "VOLUME_DIVERGENCE", "weak_volume"),
        (out["volume_ratio"] > config.max_volume_ratio, "VOLUME_DIVERGENCE", "excessive_volume"),
        (out["range_atr"] > config.max_range_atr, "WIDE_RANGE", "wide_recent_range"),
    ]
    unclassified = classifiable.copy()
    for mask, state, reason in rules:
        hit = unclassified & mask
        out.loc[hit, ["entry_quality_state", "entry_quality_reason"]] = [state, reason]
        unclassified = unclassified & ~mask

    if not set(out["entry_quality_state"].dropna().unique()).issubset(ALLOWED_STATES):
        raise AssertionError("Unexpected entry_quality_state detected")
    if not set(out["entry_quality_reason"].dropna().unique()).issubset(ALLOWED_REASONS):
        raise AssertionError("Unexpected entry_quality_reason detected")

    for col in metrics:
        out[col] = out[col].astype("float64").round(4)
    return out[OUTPUT_COLUMNS]


def validate_output(scans: pd.DataFrame, output: pd.DataFrame) -> None:
    missing_cols = [c for c in OUTPUT_COLUMNS if c not in output.columns]
    if missing_cols:
        raise AssertionError(f"Output missing columns: {missing_cols}")

    if output.duplicated(["ticker", "date"]).any():
        dupes = output[output.duplicated(["ticker", "date"], keep=False)][["ticker", "date"]]
        raise AssertionError(f"Duplicate output primary keys: {dupes.head(20).to_dict('records')}")

    if len(output) != len(scans):
        raise AssertionError(f"Row count mismatch: output={len(output)} input={len(scans)}")

    if output.isna().any().any():
        cols = output.columns[output.isna().any()].tolist()
        raise AssertionError(f"NaN values in output columns: {cols}")

    expected_keys = set(map(tuple, scans[["ticker", "date"]].to_numpy()))
    actual_keys = set(map(tuple, output[["ticker", "date"]].to_numpy()))
    if expected_keys != actual_keys:
        missing = list(expected_keys - actual_keys)[:10]
        extra = list(actual_keys - expected_keys)[:10]
        raise AssertionError(f"Output keys do not match input. Missing={missing}, extra={extra}")


def write_log(output: pd.DataFrame, log_path: Path, fetch_errors: dict[str, str] | None = None) -> None:
    state_distribution = output["entry_quality_state"].value_counts().to_dict()
    reason_distribution = output["entry_quality_reason"].value_counts().to_dict()
    row = {
        "run_date": pd.Timestamp.utcnow().isoformat(),
        "total_rows": int(len(output)),
        "state_distribution": json.dumps(state_distribution, sort_keys=True),
        "reason_distribution": json.dumps(reason_distribution, sort_keys=True),
        "avg_extension_atr": float(output["extension_atr"].mean()),
        "avg_volume_ratio": float(output["volume_ratio"].mean()),
        "fetch_errors": json.dumps(fetch_errors or {}, sort_keys=True),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(log_path, index=False)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Build historical Entry Quality metrics backfill")
    parser.add_argument("--scans", default=str(root / "data/logs/scans_validation.csv"))
    parser.add_argument("--scanner", default=str(root / "data/processed/scanner_ranked.csv"))
    parser.add_argument("--output", default=str(root / "data/processed/entry_quality_metrics_historical.csv"))
    parser.add_argument("--log", default=str(root / "data/logs/entry_quality_backfill_log.csv"))
    parser.add_argument("--max-distance-breakout-pct", type=float, default=EntryQualityConfig.max_distance_breakout_pct)
    parser.add_argument("--max-breakout-extension-atr", type=float, default=EntryQualityConfig.max_breakout_extension_atr)
    parser.add_argument("--max-extension-atr", type=float, default=EntryQualityConfig.max_extension_atr)
    parser.add_argument("--min-volume-ratio", type=float, default=EntryQualityConfig.min_volume_ratio)
    parser.add_argument("--max-volume-ratio", type=float, default=EntryQualityConfig.max_volume_ratio)
    parser.add_argument("--max-range-atr", type=float, default=EntryQualityConfig.max_range_atr)
    return parser.parse_args(list(argv))


def _legacy_main_impl(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    config = EntryQualityConfig(
        max_distance_breakout_pct=args.max_distance_breakout_pct,
        max_breakout_extension_atr=args.max_breakout_extension_atr,
        max_extension_atr=args.max_extension_atr,
        min_volume_ratio=args.min_volume_ratio,
        max_volume_ratio=args.max_volume_ratio,
        max_range_atr=args.max_range_atr,
    )
    config.validate()

    scans = load_scans(Path(args.scans))
    scanner_df = load_optional_scanner(Path(args.scanner)) if args.scanner else None
    ohlcv = build_ohlcv_for_scans(scans, scanner_df)
    fetch_errors = ohlcv.attrs.get("fetch_errors", {})
    indicators = compute_point_in_time_indicators(ohlcv)
    enriched = align_trade_dates(scans, indicators)
    output = calculate_entry_quality(enriched, config)
    validate_output(scans, output)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    write_log(output, Path(args.log), fetch_errors=fetch_errors)

    print(f"Wrote {out_path} ({len(output)} rows)")
    print(f"Wrote {args.log}")
    print(output["entry_quality_reason"].value_counts().to_string())
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    raise SystemExit(FAIL_CLOSED_MESSAGE)


if __name__ == "__main__":
    raise SystemExit(FAIL_CLOSED_MESSAGE)
