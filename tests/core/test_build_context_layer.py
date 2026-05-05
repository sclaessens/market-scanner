from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from scripts.core import build_context_layer as context_module


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_base_files(
    tmp_path: Path,
    scanner_rows: list[dict],
    validation_rows: list[dict],
    sector_rows: list[dict] | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    processed_dir = tmp_path / "data" / "processed"
    logs_dir = tmp_path / "data" / "logs"

    processed_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    scanner_path = processed_dir / "scanner_ranked.csv"
    validation_path = processed_dir / "validation_layer.csv"
    sector_path = processed_dir / "sector_relative_strength.csv"
    output_path = processed_dir / "context_strength.csv"
    log_path = logs_dir / "context_layer_log.csv"

    pd.DataFrame(scanner_rows).to_csv(scanner_path, index=False)
    pd.DataFrame(validation_rows).to_csv(validation_path, index=False)

    if sector_rows is not None:
        pd.DataFrame(sector_rows).to_csv(sector_path, index=False)

    return scanner_path, validation_path, sector_path, output_path, log_path


@pytest.fixture()
def patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def _patch(
        scanner_rows: list[dict],
        validation_rows: list[dict],
        sector_rows: list[dict] | None = None,
    ):
        scanner_path, validation_path, sector_path, output_path, log_path = (
            _write_base_files(tmp_path, scanner_rows, validation_rows, sector_rows)
        )

        monkeypatch.setattr(context_module, "SCANNER_PATH", scanner_path)
        monkeypatch.setattr(context_module, "VALIDATION_PATH", validation_path)
        monkeypatch.setattr(context_module, "SECTOR_RS_PATH", sector_path)
        monkeypatch.setattr(context_module, "OUTPUT_PATH", output_path)
        monkeypatch.setattr(context_module, "LOG_PATH", log_path)

        return scanner_path, validation_path, sector_path, output_path, log_path

    return _patch


def _scanner_row(
    ticker: str = "AAPL",
    date: str = "2026-05-05",
    rs_20d_pct: float | None = 1.0,
    sector: str | None = "Technology",
) -> dict:
    return {
        "ticker": ticker,
        "date": date,
        "rs_20d_pct": rs_20d_pct,
        "sector": sector,
    }


def _validation_row(
    ticker: str = "AAPL",
    date: str = "2026-05-05",
    valid_setup: bool = True,
    tradeable_setup: bool = True,
) -> dict:
    return {
        "ticker": ticker,
        "date": date,
        "valid_setup": valid_setup,
        "tradeable_setup": tradeable_setup,
        "validation_reason": "valid_breakout",
    }


def _sector_row(
    sector: str = "TECHNOLOGY",
    date: str = "2026-05-05",
    sector_rs_20d_pct: float = 0.5,
) -> dict:
    return {
        "sector": sector,
        "date": date,
        "sector_rs_20d_pct": sector_rs_20d_pct,
    }


def test_leading_correct(patch_paths):
    patch_paths(
        scanner_rows=[_scanner_row(rs_20d_pct=1.0, sector=" Technology ")],
        validation_rows=[_validation_row(valid_setup=True)],
        sector_rows=[_sector_row(sector="TECHNOLOGY", sector_rs_20d_pct=0.5)],
    )

    df = context_module.build_context_layer()

    assert df.loc[0, "context_strength"] == "LEADING"
    assert df.loc[0, "context_reason"] == "market_and_sector_outperformance"


def test_strong_correct(patch_paths):
    patch_paths(
        scanner_rows=[_scanner_row(rs_20d_pct=1.0)],
        validation_rows=[_validation_row(valid_setup=True)],
        sector_rows=[_sector_row(sector_rs_20d_pct=0.9)],
    )

    df = context_module.build_context_layer()

    assert df.loc[0, "context_strength"] == "STRONG"
    assert df.loc[0, "context_reason"] == "market_outperformance"


def test_weak_correct(patch_paths):
    patch_paths(
        scanner_rows=[_scanner_row(rs_20d_pct=-0.5)],
        validation_rows=[_validation_row(valid_setup=True)],
        sector_rows=[_sector_row(sector_rs_20d_pct=0.0)],
    )

    df = context_module.build_context_layer()

    assert df.loc[0, "context_strength"] == "WEAK"
    assert df.loc[0, "context_reason"] == "negative_rs"


def test_neutral_correct(patch_paths):
    patch_paths(
        scanner_rows=[_scanner_row(rs_20d_pct=0.25)],
        validation_rows=[_validation_row(valid_setup=True)],
        sector_rows=[_sector_row(sector_rs_20d_pct=0.0)],
    )

    df = context_module.build_context_layer()

    assert df.loc[0, "context_strength"] == "NEUTRAL"
    assert df.loc[0, "context_reason"] == "neutral_rs"


def test_unknown_correct(patch_paths):
    patch_paths(
        scanner_rows=[_scanner_row(rs_20d_pct=None)],
        validation_rows=[_validation_row(valid_setup=True)],
        sector_rows=[_sector_row(sector_rs_20d_pct=0.0)],
    )

    df = context_module.build_context_layer()

    assert df.loc[0, "context_strength"] == "UNKNOWN"
    assert df.loc[0, "context_reason"] == "missing_rs_20d"


def test_context_tradeable_correct(patch_paths):
    patch_paths(
        scanner_rows=[
            _scanner_row(ticker="AAA", rs_20d_pct=1.0, sector="Technology"),
            _scanner_row(ticker="BBB", rs_20d_pct=1.0, sector="Technology"),
            _scanner_row(ticker="CCC", rs_20d_pct=-1.0, sector="Technology"),
        ],
        validation_rows=[
            _validation_row(ticker="AAA", valid_setup=True),
            _validation_row(ticker="BBB", valid_setup=False),
            _validation_row(ticker="CCC", valid_setup=True),
        ],
        sector_rows=[_sector_row(sector="TECHNOLOGY", sector_rs_20d_pct=0.9)],
    )

    df = context_module.build_context_layer().set_index("ticker")

    assert bool(df.loc["AAA", "context_tradeable"]) is True
    assert bool(df.loc["BBB", "context_tradeable"]) is False
    assert bool(df.loc["CCC", "context_tradeable"]) is False


def test_duplicate_ticker_date_fails(patch_paths):
    patch_paths(
        scanner_rows=[
            _scanner_row(ticker="AAPL"),
            _scanner_row(ticker="AAPL"),
        ],
        validation_rows=[_validation_row(ticker="AAPL")],
        sector_rows=[_sector_row()],
    )

    with pytest.raises(ValueError, match="duplicate"):
        context_module.build_context_layer()


def test_missing_column_fails(patch_paths):
    scanner = _scanner_row()
    scanner.pop("rs_20d_pct")

    patch_paths(
        scanner_rows=[scanner],
        validation_rows=[_validation_row()],
        sector_rows=[_sector_row()],
    )

    with pytest.raises(ValueError, match="missing required columns"):
        context_module.build_context_layer()


def test_validation_layer_not_modified(patch_paths):
    scanner_path, validation_path, *_ = patch_paths(
        scanner_rows=[_scanner_row(rs_20d_pct=1.0)],
        validation_rows=[_validation_row(valid_setup=True)],
        sector_rows=[_sector_row(sector_rs_20d_pct=0.5)],
    )

    before_hash = _hash_file(validation_path)

    context_module.build_context_layer()

    after_hash = _hash_file(validation_path)

    assert before_hash == after_hash


def test_scanner_ranked_not_modified(patch_paths):
    scanner_path, validation_path, *_ = patch_paths(
        scanner_rows=[_scanner_row(rs_20d_pct=1.0)],
        validation_rows=[_validation_row(valid_setup=True)],
        sector_rows=[_sector_row(sector_rs_20d_pct=0.5)],
    )

    before_hash = _hash_file(scanner_path)

    context_module.build_context_layer()

    after_hash = _hash_file(scanner_path)

    assert before_hash == after_hash


def test_empty_scanner_fails(patch_paths):
    patch_paths(
        scanner_rows=[],
        validation_rows=[_validation_row()],
        sector_rows=[_sector_row()],
    )

    with pytest.raises(ValueError, match="scanner_ranked.csv is empty"):
        context_module.build_context_layer()


def test_empty_validation_fails(patch_paths):
    patch_paths(
        scanner_rows=[_scanner_row()],
        validation_rows=[],
        sector_rows=[_sector_row()],
    )

    with pytest.raises(ValueError, match="validation_layer.csv is empty"):
        context_module.build_context_layer()


def test_missing_sector_data_no_crash_but_never_leading(patch_paths):
    patch_paths(
        scanner_rows=[_scanner_row(rs_20d_pct=1.0, sector="Technology")],
        validation_rows=[_validation_row(valid_setup=True)],
        sector_rows=None,
    )

    df = context_module.build_context_layer()

    assert df.loc[0, "context_strength"] == "STRONG"
    assert df.loc[0, "context_strength"] != "LEADING"
    assert bool(df.loc[0, "context_tradeable"]) is True