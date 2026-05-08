from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core import build_validation_layer as validation_module


BASE_COLUMNS = [
    "ticker",
    "date",
    "primary_setup",
    "rr",
    "close",
    "ma20",
    "ma50",
    "ma200",
    "high_20d",
    "low_20d",
    "atr14",
    "volume_ratio",
    "extension_atr",
]

OUTPUT_COLUMNS = [
    "ticker",
    "date",
    "structure_state",
    "structure_reason",
    "setup_type",
    "valid_setup",
    "validation_reason",
]

FORBIDDEN_VALIDATION_FIELDS = [
    "tradeable_setup",
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
]


@pytest.fixture()
def isolated_paths(tmp_path, monkeypatch):
    scanner_path = tmp_path / "data" / "processed" / "scanner_ranked.csv"
    output_path = tmp_path / "data" / "processed" / "validation_layer.csv"
    metrics_path = tmp_path / "data" / "processed" / "entry_quality_metrics.csv"
    log_path = tmp_path / "data" / "logs" / "validation_layer_log.csv"

    scanner_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(validation_module, "INPUT_PATH", scanner_path)
    monkeypatch.setattr(validation_module, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(validation_module, "ENTRY_QUALITY_OUTPUT_PATH", metrics_path)
    monkeypatch.setattr(validation_module, "LOG_PATH", log_path)

    return scanner_path, output_path, log_path


def _base_row(**overrides):
    row = {
        "ticker": "TEST",
        "date": "2026-05-05",
        "primary_setup": "BREAKOUT",
        "rr": 2.0,
        "close": 100.0,
        "ma20": 95.0,
        "ma50": 90.0,
        "ma200": 80.0,
        "high_20d": 102.0,
        "low_20d": 95.0,
        "atr14": 2.0,
        "volume_ratio": 1.4,
        "extension_atr": 1.5,
    }
    row.update(overrides)
    return row


def _write_scanner(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows, columns=BASE_COLUMNS).to_csv(path, index=False)


def _run(path: Path, rows: list[dict]) -> pd.DataFrame:
    _write_scanner(path, rows)
    return validation_module.build_validation_layer()


def test_valid_breakout_correct(isolated_paths):
    scanner_path, _, _ = isolated_paths

    result = _run(scanner_path, [_base_row(primary_setup="BREAKOUT")])

    assert result.loc[0, "structure_state"] == "COHERENT"
    assert result.loc[0, "structure_reason"] == "coherent_breakout"
    assert result.loc[0, "valid_setup"] is True or result.loc[0, "valid_setup"] == True
    assert result.loc[0, "validation_reason"] == "coherent_breakout"


def test_valid_pullback_correct(isolated_paths):
    scanner_path, _, _ = isolated_paths

    result = _run(
        scanner_path,
        [
            _base_row(
                primary_setup="PULLBACK",
                close=100.0,
                ma20=99.0,
                ma50=90.0,
                high_20d=110.0,
                volume_ratio=0.8,
            )
        ],
    )

    assert result.loc[0, "structure_state"] == "COHERENT"
    assert result.loc[0, "structure_reason"] == "coherent_pullback"
    assert result.loc[0, "valid_setup"] is True or result.loc[0, "valid_setup"] == True
    assert result.loc[0, "validation_reason"] == "coherent_pullback"


def test_valid_vcp_correct(isolated_paths):
    scanner_path, _, _ = isolated_paths

    result = _run(
        scanner_path,
        [
            _base_row(
                primary_setup="VCP",
                close=100.0,
                ma20=95.0,
                ma50=90.0,
                high_20d=110.0,
                volume_ratio=0.9,
            )
        ],
    )

    assert result.loc[0, "structure_state"] == "COHERENT"
    assert result.loc[0, "structure_reason"] == "coherent_vcp"
    assert result.loc[0, "valid_setup"] is True or result.loc[0, "valid_setup"] == True
    assert result.loc[0, "validation_reason"] == "coherent_vcp"


def test_rr_is_metadata_not_structure_gate(isolated_paths):
    scanner_path, _, _ = isolated_paths

    result = _run(scanner_path, [_base_row(rr=1.79)])

    assert result.loc[0, "structure_state"] == "COHERENT"
    assert result.loc[0, "valid_setup"] is True or result.loc[0, "valid_setup"] == True
    assert result.loc[0, "validation_reason"] == "coherent_breakout"


def test_broken_price_structure_correct(isolated_paths):
    scanner_path, _, _ = isolated_paths

    result = _run(scanner_path, [_base_row(close=89.0, ma50=90.0)])

    assert result.loc[0, "structure_state"] == "BROKEN"
    assert result.loc[0, "valid_setup"] is False or result.loc[0, "valid_setup"] == False
    assert result.loc[0, "validation_reason"] == "structure_broken"


def test_missing_data_correct(isolated_paths):
    scanner_path, _, _ = isolated_paths

    result = _run(scanner_path, [_base_row(close=None)])

    assert result.loc[0, "structure_state"] == "INCOMPLETE"
    assert result.loc[0, "valid_setup"] is False or result.loc[0, "valid_setup"] == False
    assert result.loc[0, "validation_reason"] == "missing_data"


def test_no_setup_correct(isolated_paths):
    scanner_path, _, _ = isolated_paths

    result = _run(scanner_path, [_base_row(primary_setup="")])

    assert result.loc[0, "structure_state"] == "INCOMPLETE"
    assert result.loc[0, "valid_setup"] is False or result.loc[0, "valid_setup"] == False
    assert result.loc[0, "validation_reason"] == "no_setup"


def test_broken_structure_correct(isolated_paths):
    scanner_path, _, _ = isolated_paths

    result = _run(
        scanner_path,
        [
            _base_row(
                primary_setup="BREAKOUT",
                close=100.0,
                ma20=105.0,
                ma50=90.0,
                high_20d=130.0,
                volume_ratio=0.5,
            )
        ],
    )

    assert result.loc[0, "structure_state"] == "BROKEN"
    assert result.loc[0, "valid_setup"] is False or result.loc[0, "valid_setup"] == False
    assert result.loc[0, "validation_reason"] == "structure_broken"


def test_legacy_tradeability_alias_not_emitted(isolated_paths):
    scanner_path, _, _ = isolated_paths

    result = _run(
        scanner_path,
        [
            _base_row(ticker="AAA", primary_setup="BREAKOUT"),
            _base_row(ticker="BBB", rr=1.2),
        ],
    )

    assert "trad" + "eable_setup" not in result.columns


def test_forbidden_validation_fields_not_emitted(isolated_paths):
    scanner_path, output_path, _ = isolated_paths

    result = _run(scanner_path, [_base_row()])
    written = pd.read_csv(output_path)

    for field in FORBIDDEN_VALIDATION_FIELDS:
        assert field not in result.columns
        assert field not in written.columns


def test_compatibility_aliases_mirror_primary_contract(isolated_paths):
    scanner_path, _, _ = isolated_paths

    result = _run(
        scanner_path,
        [
            _base_row(ticker="AAA", primary_setup="BREAKOUT"),
            _base_row(ticker="BBB", close=80.0, ma20=95.0),
            _base_row(ticker="CCC", primary_setup=""),
        ],
    )

    assert result.loc[result["structure_state"] == "COHERENT", "valid_setup"].all()
    assert not result.loc[result["structure_state"] != "COHERENT", "valid_setup"].any()
    assert result["validation_reason"].equals(result["structure_reason"])


def test_duplicate_ticker_date_fails(isolated_paths):
    scanner_path, _, _ = isolated_paths

    _write_scanner(
        scanner_path,
        [
            _base_row(ticker="AAA", date="2026-05-05"),
            _base_row(ticker="AAA", date="2026-05-05"),
        ],
    )

    with pytest.raises(ValueError, match="duplicate ticker/date"):
        validation_module.build_validation_layer()


def test_missing_column_fails(isolated_paths):
    scanner_path, _, _ = isolated_paths

    df = pd.DataFrame([_base_row()])
    df = df.drop(columns=["volume_ratio"])
    df.to_csv(scanner_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        validation_module.build_validation_layer()


def test_empty_scanner_fails(isolated_paths):
    scanner_path, _, _ = isolated_paths

    scanner_path.write_text("")

    with pytest.raises(ValueError, match="empty"):
        validation_module.build_validation_layer()


def test_output_schema_exact_correct(isolated_paths):
    scanner_path, output_path, _ = isolated_paths

    result = _run(scanner_path, [_base_row()])

    written = pd.read_csv(output_path)

    assert list(result.columns) == OUTPUT_COLUMNS
    assert list(written.columns) == OUTPUT_COLUMNS


def test_scanner_ranked_csv_is_not_modified(isolated_paths):
    scanner_path, _, _ = isolated_paths

    _write_scanner(scanner_path, [_base_row()])
    before = scanner_path.read_bytes()

    validation_module.build_validation_layer()

    after = scanner_path.read_bytes()

    assert after == before


def test_validation_layer_log_csv_is_written(isolated_paths):
    scanner_path, _, log_path = isolated_paths

    _run(
        scanner_path,
        [
            _base_row(ticker="AAA", primary_setup="BREAKOUT"),
            _base_row(ticker="BBB", primary_setup="PULLBACK", close=100, ma20=99),
            _base_row(ticker="CCC", rr=1.0),
        ],
    )

    assert log_path.exists()

    log_df = pd.read_csv(log_path)

    expected_log_columns = [
        "run_date",
        "total_rows",
        "coherent_count",
        "broken_count",
        "incomplete_count",
        "avg_extension_atr",
        "avg_volume_ratio",
        "median_range_atr",
    ]

    assert list(log_df.columns) == expected_log_columns
    assert int(log_df.iloc[-1]["total_rows"]) == 3
    assert int(log_df.iloc[-1]["coherent_count"]) == 3

def test_breakout_extension_is_metadata_not_structure_gate(isolated_paths):
    scanner_path, _, _ = isolated_paths

    result = _run(
        scanner_path,
        [
            _base_row(
                primary_setup="BREAKOUT",
                close=100.0,
                ma20=95.0,
                ma50=90.0,
                high_20d=102.0,
                volume_ratio=1.5,
                extension_atr=2.51,
            )
        ],
    )

    assert result.loc[0, "structure_state"] == "COHERENT"
    assert result.loc[0, "valid_setup"] == True
    assert result.loc[0, "validation_reason"] == "coherent_breakout"


def test_breakout_distance_is_metadata_not_structure_gate(isolated_paths):
    scanner_path, _, _ = isolated_paths

    result = _run(
        scanner_path,
        [
            _base_row(
                primary_setup="BREAKOUT",
                close=100.0,
                ma20=95.0,
                ma50=90.0,
                high_20d=104.0,
                volume_ratio=1.5,
                extension_atr=1.5,
            )
        ],
    )

    assert result.loc[0, "structure_state"] == "COHERENT"
    assert result.loc[0, "valid_setup"] == True
    assert result.loc[0, "validation_reason"] == "coherent_breakout"


def test_breakout_volume_is_metadata_not_structure_gate(isolated_paths):
    scanner_path, _, _ = isolated_paths

    result = _run(
        scanner_path,
        [
            _base_row(
                primary_setup="BREAKOUT",
                close=100.0,
                ma20=95.0,
                ma50=90.0,
                high_20d=102.0,
                volume_ratio=1.29,
                extension_atr=1.5,
            )
        ],
    )

    assert result.loc[0, "structure_state"] == "COHERENT"
    assert result.loc[0, "valid_setup"] == True
    assert result.loc[0, "validation_reason"] == "coherent_breakout"
