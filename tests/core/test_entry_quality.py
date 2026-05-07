from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.core import build_validation_layer as validation_module


def _base_row(**overrides) -> dict:
    row = {
        "ticker": "AAPL",
        "date": "2026-05-06",
        "primary_setup": "BREAKOUT",
        "rr": 2.5,
        "close": 102.0,
        "ma20": 100.0,
        "ma50": 90.0,
        "ma200": 80.0,
        "high_20d": 100.0,
        "low_20d": 95.0,
        "atr14": 2.0,
        "volume_ratio": 1.5,
        "extension_atr": 1.0,
    }
    row.update(overrides)
    return row


@pytest.fixture()
def patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    processed_dir = tmp_path / "data" / "processed"
    logs_dir = tmp_path / "data" / "logs"

    processed_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    scanner_path = processed_dir / "scanner_ranked.csv"
    validation_path = processed_dir / "validation_layer.csv"
    metrics_path = processed_dir / "entry_quality_metrics.csv"
    log_path = logs_dir / "validation_layer_log.csv"

    monkeypatch.setattr(validation_module, "INPUT_PATH", scanner_path)
    monkeypatch.setattr(validation_module, "OUTPUT_PATH", validation_path)
    monkeypatch.setattr(validation_module, "ENTRY_QUALITY_OUTPUT_PATH", metrics_path)
    monkeypatch.setattr(validation_module, "LOG_PATH", log_path)

    return scanner_path, validation_path, metrics_path, log_path


def _run_with_rows(patch_paths, rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scanner_path, validation_path, metrics_path, _ = patch_paths
    pd.DataFrame(rows).to_csv(scanner_path, index=False)

    validation_df = validation_module.build_validation_layer()
    metrics_df = pd.read_csv(metrics_path)

    return validation_df, metrics_df


def test_clean_row_entry_quality_balanced(patch_paths):
    _, metrics_df = _run_with_rows(patch_paths, [_base_row()])

    assert metrics_df.loc[0, "entry_quality_state"] == "BALANCED"
    assert metrics_df.loc[0, "entry_quality_reason"] == "balanced_structure"


def test_distance_to_breakout_metric_is_descriptive(patch_paths):
    _, metrics_df = _run_with_rows(
        patch_paths,
        [_base_row(close=104.0, high_20d=100.0, atr14=10.0, ma20=100.0)],
    )

    assert float(metrics_df.loc[0, "distance_to_breakout_pct"]) == pytest.approx(4.0)
    assert metrics_df.loc[0, "entry_quality_state"] == "BALANCED"


def test_breakout_extension_atr_metric_is_descriptive(patch_paths):
    _, metrics_df = _run_with_rows(
        patch_paths,
        [_base_row(close=102.5, high_20d=100.0, atr14=1.0, ma20=101.5)],
    )

    assert float(metrics_df.loc[0, "breakout_extension_atr"]) == pytest.approx(2.5)
    assert metrics_df.loc[0, "entry_quality_state"] == "BALANCED"


def test_extension_atr_classified_extended(patch_paths):
    _, metrics_df = _run_with_rows(
        patch_paths,
        [_base_row(close=102.0, high_20d=100.0, atr14=1.0, ma20=99.0)],
    )

    assert metrics_df.loc[0, "entry_quality_state"] == "EXTENDED"
    assert metrics_df.loc[0, "entry_quality_reason"] == "extended_vs_ma20"


def test_volume_ratio_below_min_is_metric_only(patch_paths):
    _, metrics_df = _run_with_rows(
        patch_paths,
        [_base_row(volume_ratio=1.0)],
    )

    assert float(metrics_df.loc[0, "volume_ratio"]) == pytest.approx(1.0)
    assert metrics_df.loc[0, "entry_quality_state"] == "BALANCED"


def test_volume_ratio_above_max_is_metric_only(patch_paths):
    _, metrics_df = _run_with_rows(
        patch_paths,
        [_base_row(volume_ratio=4.5)],
    )

    assert float(metrics_df.loc[0, "volume_ratio"]) == pytest.approx(4.5)
    assert metrics_df.loc[0, "entry_quality_state"] == "BALANCED"


def test_range_atr_classified_wide_range(patch_paths):
    _, metrics_df = _run_with_rows(
        patch_paths,
        [_base_row(high_20d=100.0, low_20d=86.0, atr14=2.0)],
    )

    assert metrics_df.loc[0, "entry_quality_state"] == "WIDE_RANGE"
    assert metrics_df.loc[0, "entry_quality_reason"] == "wide_recent_range"


def test_atr14_zero_hard_fail(patch_paths):
    scanner_path, *_ = patch_paths
    pd.DataFrame([_base_row(atr14=0)]).to_csv(scanner_path, index=False)

    with pytest.raises(ValueError, match="atr14 must be > 0"):
        validation_module.build_validation_layer()


def test_duplicate_ticker_date_hard_fail(patch_paths):
    scanner_path, *_ = patch_paths
    pd.DataFrame([_base_row(), _base_row()]).to_csv(scanner_path, index=False)

    with pytest.raises(ValueError, match="duplicate ticker/date"):
        validation_module.build_validation_layer()


def test_missing_required_column_hard_fail(patch_paths):
    scanner_path, *_ = patch_paths
    row = _base_row()
    row.pop("atr14")
    pd.DataFrame([row]).to_csv(scanner_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        validation_module.build_validation_layer()


def test_validation_layer_schema_unchanged(patch_paths):
    validation_df, _ = _run_with_rows(patch_paths, [_base_row()])

    assert list(validation_df.columns) == [
        "ticker",
        "date",
        "structure_state",
        "structure_reason",
        "setup_type",
        "valid_setup",
        "validation_reason",
    ]


def test_entry_quality_metrics_schema_exact(patch_paths):
    _, metrics_df = _run_with_rows(patch_paths, [_base_row()])

    assert list(metrics_df.columns) == [
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


def test_high_equals_low_sets_range_atr_zero(patch_paths):
    _, metrics_df = _run_with_rows(
        patch_paths,
        [_base_row(high_20d=100.0, low_20d=100.0)],
    )

    assert float(metrics_df.loc[0, "range_atr"]) == 0.0


def test_entry_quality_does_not_change_structure_contract(patch_paths):
    validation_df, metrics_df = _run_with_rows(
        patch_paths,
        [_base_row(close=104.0, high_20d=100.0, atr14=10.0, ma20=100.0)],
    )

    assert metrics_df.loc[0, "entry_quality_state"] == "BALANCED"
    assert "structure_state" in validation_df.columns
    assert "valid_setup" in validation_df.columns
