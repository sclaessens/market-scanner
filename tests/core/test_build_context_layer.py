from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from scripts.core import build_context_layer as context_module

EXPECTED_CONTEXT_COLUMNS = [
    "ticker",
    "date",
    "rs_score",
    "rs_percentile",
    "rs_rank",
    "rs_vs_market",
    "rs_vs_sector",
    "context_strength",
    "context_reason",
    "leadership_state",
]

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


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_base_files(
    tmp_path: Path,
    scanner_rows: list[dict],
    validation_rows: list[dict] | None = None,
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
    pd.DataFrame(validation_rows or []).to_csv(validation_path, index=False)

    if sector_rows is not None:
        pd.DataFrame(sector_rows).to_csv(sector_path, index=False)

    return scanner_path, validation_path, sector_path, output_path, log_path


@pytest.fixture()
def patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def _patch(
        scanner_rows: list[dict],
        validation_rows: list[dict] | None = None,
        sector_rows: list[dict] | None = None,
    ):
        scanner_path, validation_path, sector_path, output_path, log_path = (
            _write_base_files(tmp_path, scanner_rows, validation_rows, sector_rows)
        )

        monkeypatch.setattr(context_module, "SCANNER_PATH", scanner_path)
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
) -> dict:
    return {
        "ticker": ticker,
        "date": date,
        "structure_state": "COHERENT" if valid_setup else "BROKEN",
        "structure_reason": "coherent_breakout" if valid_setup else "structure_broken",
        "setup_type": "BREAKOUT",
        "valid_setup": valid_setup,
        "validation_reason": "coherent_breakout" if valid_setup else "structure_broken",
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
        sector_rows=[_sector_row(sector="TECHNOLOGY", sector_rs_20d_pct=0.5)],
    )

    df = context_module.build_context_layer()

    assert df.loc[0, "context_strength"] == "LEADING"
    assert df.loc[0, "context_reason"] == "top_decile_leadership"


def test_strong_correct(patch_paths):
    patch_paths(
        scanner_rows=[
            _scanner_row(ticker="AAA", rs_20d_pct=1.0),
            _scanner_row(ticker="BBB", rs_20d_pct=2.0),
            _scanner_row(ticker="CCC", rs_20d_pct=0.2),
            _scanner_row(ticker="DDD", rs_20d_pct=-0.1),
        ],
        sector_rows=[_sector_row(sector_rs_20d_pct=0.9)],
    )

    df = context_module.build_context_layer().set_index("ticker")

    assert df.loc["AAA", "context_strength"] == "STRONG"
    assert df.loc["AAA", "context_reason"] == "upper_quartile_leadership"


def test_weak_correct(patch_paths):
    patch_paths(
        scanner_rows=[
            _scanner_row(ticker="AAA", rs_20d_pct=-0.5),
            _scanner_row(ticker="BBB", rs_20d_pct=2.0),
            _scanner_row(ticker="CCC", rs_20d_pct=1.0),
            _scanner_row(ticker="DDD", rs_20d_pct=0.0),
        ],
        sector_rows=[_sector_row(sector_rs_20d_pct=0.0)],
    )

    df = context_module.build_context_layer().set_index("ticker")

    assert df.loc["AAA", "context_strength"] == "WEAK"
    assert df.loc["AAA", "context_reason"] == "lower_distribution"


def test_neutral_correct(patch_paths):
    patch_paths(
        scanner_rows=[
            _scanner_row(ticker="AAA", rs_20d_pct=0.25),
            _scanner_row(ticker="BBB", rs_20d_pct=2.0),
            _scanner_row(ticker="CCC", rs_20d_pct=1.0),
            _scanner_row(ticker="DDD", rs_20d_pct=-0.5),
        ],
        sector_rows=[_sector_row(sector_rs_20d_pct=0.0)],
    )

    df = context_module.build_context_layer().set_index("ticker")

    assert df.loc["AAA", "context_strength"] == "NEUTRAL"
    assert df.loc["AAA", "context_reason"] == "middle_distribution"


def test_unknown_correct(patch_paths):
    strength, reason = context_module._classify_from_percentile(float("nan"))

    assert strength == "UNKNOWN"
    assert reason == "missing_percentile"


def test_context_outputs_classification_columns_only(patch_paths):
    patch_paths(
        scanner_rows=[
            _scanner_row(ticker="AAA", rs_20d_pct=1.0, sector="Technology"),
            _scanner_row(ticker="BBB", rs_20d_pct=1.0, sector="Technology"),
            _scanner_row(ticker="CCC", rs_20d_pct=-1.0, sector="Technology"),
        ],
        sector_rows=[_sector_row(sector="TECHNOLOGY", sector_rs_20d_pct=0.9)],
    )

    df = context_module.build_context_layer()

    assert list(df.columns) == EXPECTED_CONTEXT_COLUMNS
    assert set(df.columns).isdisjoint(FORBIDDEN_CONTEXT_FIELDS)


def test_context_output_file_schema_is_exact_and_governance_clean(patch_paths):
    *_, output_path, _ = patch_paths(
        scanner_rows=[
            _scanner_row(ticker="AAA", rs_20d_pct=3.0, sector="Technology"),
            _scanner_row(ticker="BBB", rs_20d_pct=2.0, sector="Technology"),
            _scanner_row(ticker="CCC", rs_20d_pct=1.0, sector="Technology"),
            _scanner_row(ticker="DDD", rs_20d_pct=-1.0, sector="Technology"),
        ],
        sector_rows=[_sector_row(sector="TECHNOLOGY", sector_rs_20d_pct=0.5)],
    )

    context_module.build_context_layer()

    written_df = pd.read_csv(output_path)

    assert list(written_df.columns) == EXPECTED_CONTEXT_COLUMNS
    assert set(written_df.columns).isdisjoint(FORBIDDEN_CONTEXT_FIELDS)


def test_duplicate_ticker_date_fails(patch_paths):
    patch_paths(
        scanner_rows=[
            _scanner_row(ticker="AAPL"),
            _scanner_row(ticker="AAPL"),
        ],
        sector_rows=[_sector_row()],
    )

    with pytest.raises(ValueError, match="duplicate"):
        context_module.build_context_layer()


def test_missing_column_fails(patch_paths):
    scanner = _scanner_row()
    scanner.pop("rs_20d_pct")

    patch_paths(
        scanner_rows=[scanner],
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
        sector_rows=[_sector_row(sector_rs_20d_pct=0.5)],
    )

    before_hash = _hash_file(scanner_path)

    context_module.build_context_layer()

    after_hash = _hash_file(scanner_path)

    assert before_hash == after_hash


def test_empty_scanner_fails(patch_paths):
    patch_paths(
        scanner_rows=[],
        sector_rows=[_sector_row()],
    )

    with pytest.raises(ValueError, match="scanner_ranked.csv is empty"):
        context_module.build_context_layer()


def test_context_layer_does_not_require_validation_layer(patch_paths):
    patch_paths(
        scanner_rows=[_scanner_row()],
        validation_rows=[],
        sector_rows=[_sector_row()],
    )

    df = context_module.build_context_layer()

    assert len(df) == 1
    assert set(df.columns).isdisjoint(FORBIDDEN_CONTEXT_FIELDS)


def test_missing_sector_data_no_crash_and_remains_distribution_only(patch_paths):
    patch_paths(
        scanner_rows=[_scanner_row(rs_20d_pct=1.0, sector="Technology")],
        sector_rows=None,
    )

    df = context_module.build_context_layer()

    assert df.loc[0, "context_strength"] == "LEADING"
    assert df.loc[0, "context_reason"] == "top_decile_leadership"
    assert pd.isna(df.loc[0, "rs_vs_sector"])


def test_missing_sector_data_preserves_all_rows(patch_paths):
    scanner_rows = [
        _scanner_row(ticker="AAA", rs_20d_pct=3.0, sector="Technology"),
        _scanner_row(ticker="BBB", rs_20d_pct=2.0, sector="Healthcare"),
        _scanner_row(ticker="CCC", rs_20d_pct=1.0, sector=None),
        _scanner_row(ticker="DDD", rs_20d_pct=-1.0, sector="Financials"),
    ]
    patch_paths(scanner_rows=scanner_rows, sector_rows=None)

    df = context_module.build_context_layer()

    assert len(df) == len(scanner_rows)
    assert set(df["ticker"]) == {"AAA", "BBB", "CCC", "DDD"}
    assert df["rs_vs_sector"].isna().all()
    assert set(df.columns).isdisjoint(FORBIDDEN_CONTEXT_FIELDS)


def test_leadership_states_are_classification_only_and_preserve_distribution(patch_paths):
    scanner_rows = [
        _scanner_row(ticker="LEAD", rs_20d_pct=4.0),
        _scanner_row(ticker="STRONG", rs_20d_pct=3.0),
        _scanner_row(ticker="NEUTRAL", rs_20d_pct=2.0),
        _scanner_row(ticker="WEAK", rs_20d_pct=-1.0),
    ]
    patch_paths(
        scanner_rows=scanner_rows,
        sector_rows=[_sector_row(sector_rs_20d_pct=0.0)],
    )

    df = context_module.build_context_layer().set_index("ticker")

    assert len(df) == len(scanner_rows)
    assert df.loc["LEAD", "context_strength"] == "LEADING"
    assert df.loc["STRONG", "context_strength"] == "STRONG"
    assert df.loc["NEUTRAL", "context_strength"] == "NEUTRAL"
    assert df.loc["WEAK", "context_strength"] == "WEAK"
    assert df["leadership_state"].equals(df["context_strength"])
    assert set(df.columns).isdisjoint(FORBIDDEN_CONTEXT_FIELDS)


def test_context_output_is_deterministic(patch_paths):
    *_, output_path, _ = patch_paths(
        scanner_rows=[
            _scanner_row(ticker="AAA", rs_20d_pct=3.0),
            _scanner_row(ticker="BBB", rs_20d_pct=2.0),
            _scanner_row(ticker="CCC", rs_20d_pct=1.0),
            _scanner_row(ticker="DDD", rs_20d_pct=-1.0),
        ],
        sector_rows=[_sector_row(sector_rs_20d_pct=0.0)],
    )

    first = context_module.build_context_layer()
    first_hash = _hash_file(output_path)
    second = context_module.build_context_layer()
    second_hash = _hash_file(output_path)

    pd.testing.assert_frame_equal(first, second)
    assert first_hash == second_hash
