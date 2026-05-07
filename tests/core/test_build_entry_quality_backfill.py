from pathlib import Path
import pandas as pd
import pytest

from scripts.core.build_entry_quality_backfill import (
    EntryQualityConfig,
    calculate_entry_quality,
    compute_point_in_time_indicators,
    load_scans,
    validate_output,
)


def test_duplicate_ticker_date_fails(tmp_path: Path):
    p = tmp_path / "scans_validation.csv"
    pd.DataFrame(
        [
            {"scan_date": "2026-01-01", "ticker": "ABC", "entry": 10.0, "rr": 2.0},
            {"scan_date": "2026-01-01", "ticker": "ABC", "entry": 10.1, "rr": 2.0},
        ]
    ).to_csv(p, index=False)

    with pytest.raises(ValueError, match="Duplicate primary key"):
        load_scans(p)


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
    ind = compute_point_in_time_indicators(ohlcv)
    row = ind[ind["date"] == dates[19]].iloc[0]

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
    out = calculate_entry_quality(df, EntryQualityConfig())

    assert list(out.columns) == [
        "ticker",
        "date",
        "distance_to_breakout_pct",
        "breakout_extension_atr",
        "extension_atr",
        "distance_ma20_pct",
        "volume_ratio",
        "range_atr",
        "entry_quality_flag",
        "entry_quality_reason",
    ]
    assert out.loc[0, "entry_quality_flag"] is False or out.loc[0, "entry_quality_flag"] == False
    assert out.loc[0, "entry_quality_reason"] == "too_far_from_breakout"


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
            "entry_quality_flag": [True],
            "entry_quality_reason": ["ok"],
        }
    )
    with pytest.raises(AssertionError, match="NaN"):
        validate_output(scans, output)
