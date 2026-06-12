from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


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


@dataclass(frozen=True)
class EntryQualityConfig:
    max_distance_breakout_pct: float = 3.0
    max_breakout_extension_atr: float = 2.0
    max_extension_atr: float = 2.5
    min_volume_ratio: float = 0.10
    max_volume_ratio: float = 4.0
    max_range_atr: float = 4.0


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_date_column(df: pd.DataFrame) -> str:
    for column in DATE_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(f"Missing date column. Expected one of: {', '.join(DATE_CANDIDATES)}")


def _load_scans_contract(path: Path) -> pd.DataFrame:
    df = _normalize_columns(pd.read_csv(path))
    date_column = _find_date_column(df)
    missing = sorted((REQUIRED_INPUT_COLUMNS - {"entry"}) - set(df.columns))
    if "entry" not in df.columns and "entry_price" not in df.columns:
        missing.append("entry or entry_price")
    if missing:
        raise ValueError(f"Missing required scans_validation columns: {missing}")

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["date"] = pd.to_datetime(df[date_column], errors="coerce").dt.tz_localize(None).dt.normalize()
    if "entry_price" not in df.columns:
        df["entry_price"] = pd.to_numeric(df["entry"], errors="coerce")
    else:
        df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")

    duplicated = df.duplicated(["ticker", "date"], keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, ["ticker", "date"]].head(20).to_dict("records")
        raise ValueError(
            "Duplicate primary key in scans_validation.csv. "
            "Historical backfill requires UNIQUE (ticker, date). "
            f"Examples: {examples}"
        )
    return df.reset_index(drop=True)


def _compute_point_in_time_indicators_contract(ohlcv: pd.DataFrame) -> pd.DataFrame:
    required = ["ticker", "date", "high", "low", "close", "volume"]
    df = ohlcv[required].sort_values(["ticker", "date"]).copy()

    frames: list[pd.DataFrame] = []
    for _, group in df.groupby("ticker", sort=False):
        group = group.sort_values("date").copy()
        prev_close = group["close"].shift(1)
        true_range = pd.concat(
            [
                group["high"] - group["low"],
                (group["high"] - prev_close).abs(),
                (group["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        group["ma20"] = group["close"].rolling(window=20, min_periods=20).mean()
        group["atr14"] = true_range.rolling(window=14, min_periods=14).mean()
        group["high_20d"] = group["high"].rolling(window=20, min_periods=20).max()
        group["avg_vol_20"] = group["volume"].rolling(window=20, min_periods=20).mean()
        frames.append(group)

    return pd.concat(frames, ignore_index=True)


def _calculate_entry_quality_contract(df: pd.DataFrame, config: EntryQualityConfig) -> pd.DataFrame:
    out = df.copy()
    required = ["close", "high", "low", "volume", "high_20d", "ma20", "atr14", "avg_vol_20"]

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
    out.loc[invalid_structure & ~missing_data, ["entry_quality_state", "entry_quality_reason"]] = [
        "STRUCTURE_GAP",
        "invalid_structure",
    ]

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

    assert set(out["entry_quality_state"].dropna()).issubset(ALLOWED_STATES)
    assert set(out["entry_quality_reason"].dropna()).issubset(ALLOWED_REASONS)

    for column in metrics:
        out[column] = out[column].astype("float64").round(4)
    return out[OUTPUT_COLUMNS]


def _validate_output_contract(scans: pd.DataFrame, output: pd.DataFrame) -> None:
    missing_columns = [column for column in OUTPUT_COLUMNS if column not in output.columns]
    if missing_columns:
        raise AssertionError(f"Output missing columns: {missing_columns}")

    if output.duplicated(["ticker", "date"]).any():
        dupes = output[output.duplicated(["ticker", "date"], keep=False)][["ticker", "date"]]
        raise AssertionError(f"Duplicate output primary keys: {dupes.head(20).to_dict('records')}")

    if len(output) != len(scans):
        raise AssertionError(f"Row count mismatch: output={len(output)} input={len(scans)}")

    if output.isna().any().any():
        columns = output.columns[output.isna().any()].tolist()
        raise AssertionError(f"NaN values in output columns: {columns}")

    expected_keys = set(map(tuple, scans[["ticker", "date"]].to_numpy()))
    actual_keys = set(map(tuple, output[["ticker", "date"]].to_numpy()))
    if expected_keys != actual_keys:
        missing = list(expected_keys - actual_keys)[:10]
        extra = list(actual_keys - expected_keys)[:10]
        raise AssertionError(f"Output keys do not match input. Missing={missing}, extra={extra}")


def test_duplicate_ticker_date_fails(tmp_path: Path):
    path = tmp_path / "scans_validation.csv"
    pd.DataFrame(
        [
            {"scan_date": "2026-01-01", "ticker": "ABC", "entry": 10.0, "rr": 2.0},
            {"scan_date": "2026-01-01", "ticker": "ABC", "entry": 10.1, "rr": 2.0},
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Duplicate primary key"):
        _load_scans_contract(path)


def test_point_in_time_indicators_use_current_and_past_only():
    dates = pd.date_range("2026-01-01", periods=25, freq="D")
    ohlcv = pd.DataFrame(
        {
            "ticker": "ABC",
            "date": dates,
            "high": range(11, 36),
            "low": range(9, 34),
            "close": range(10, 35),
            "volume": [100] * 25,
        }
    )
    indicators = _compute_point_in_time_indicators_contract(ohlcv)
    row = indicators[indicators["date"] == dates[19]].iloc[0]

    assert row["ma20"] == pytest.approx(sum(range(10, 30)) / 20)
    assert row["high_20d"] == 30
    assert row["avg_vol_20"] == 100


def test_entry_quality_reason_order_and_schema():
    df = pd.DataFrame(
        [
            {
                "ticker": "ABC",
                "date": pd.Timestamp("2026-01-30"),
                "close": 110.0,
                "high": 111.0,
                "low": 109.0,
                "volume": 100.0,
                "high_20d": 100.0,
                "ma20": 100.0,
                "atr14": 2.0,
                "avg_vol_20": 100.0,
            }
        ]
    )
    output = _calculate_entry_quality_contract(df, EntryQualityConfig())

    assert list(output.columns) == OUTPUT_COLUMNS
    assert output.loc[0, "entry_quality_state"] == "EXTENDED"
    assert output.loc[0, "entry_quality_reason"] == "too_far_from_breakout"


def test_validate_output_fails_on_nan():
    scans = pd.DataFrame({"ticker": ["ABC"], "date": [pd.Timestamp("2026-01-01")]})
    output = pd.DataFrame(
        {
            "ticker": ["ABC"],
            "date": [pd.Timestamp("2026-01-01")],
            "distance_to_breakout_pct": [float("nan")],
            "breakout_extension_atr": [0.0],
            "extension_atr": [0.0],
            "distance_ma20_pct": [0.0],
            "volume_ratio": [1.0],
            "range_atr": [1.0],
            "entry_quality_state": ["BALANCED"],
            "entry_quality_reason": ["balanced_structure"],
        }
    )
    with pytest.raises(AssertionError, match="NaN"):
        _validate_output_contract(scans, output)
