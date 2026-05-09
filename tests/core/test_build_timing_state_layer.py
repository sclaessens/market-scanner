from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.core import build_timing_state_layer as timing_module

UPSTREAM_COLUMNS = [
    "ticker",
    "date",
    "quality_state",
    "quality_reason",
    "profitability_profile",
    "balance_sheet_profile",
    "earnings_quality_profile",
    "capital_efficiency_profile",
    "cashflow_profile",
    "stability_profile",
    "quality_metadata_status",
    "source_data_status",
    "source_timestamp",
    "generated_at",
]

TIMING_COLUMNS = [
    "timing_state",
    "timing_reason",
    "breakout_state",
    "pullback_state",
    "compression_state",
    "extension_state",
    "participation_state",
    "timing_environment",
    "timing_pattern_state",
    "trend_participation_state",
    "timing_structure_state",
    "timing_metadata_status",
    "timing_source_data_status",
    "timing_source_timestamp",
    "timing_generated_at",
]

EXPECTED_OUTPUT_COLUMNS = UPSTREAM_COLUMNS + TIMING_COLUMNS

EXPECTED_LOG_COLUMNS = [
    "generated_at",
    "input_row_count",
    "output_row_count",
    "unique_ticker_date_count",
    "duplicate_ticker_date_count",
    "missing_auxiliary_source_count",
    "timing_state_distribution",
    "extension_state_distribution",
    "compression_state_distribution",
    "pullback_state_distribution",
    "breakout_state_distribution",
    "timing_metadata_status_distribution",
    "source_data_status_distribution",
]

FORBIDDEN_FIELDS = {
    "tradeable",
    "approved",
    "rejected",
    "high_conviction",
    "conviction",
    "conviction_score",
    "priority",
    "rank",
    "ranking",
    "score",
    "scoring",
    "actionable",
    "buy_candidate",
    "sell_candidate",
    "execution_ready",
    "readiness",
    "best_opportunity",
    "allocation",
    "allocation_weight",
    "expected_return",
    "alpha_score",
    "opportunity_rank",
    "preferred_setup",
    "timing_grade",
    "timing_signal",
    "final_action",
    "final_score",
}

FORBIDDEN_VALUES = {
    "BUY",
    "SELL",
    "REMOVE",
    "APPROVED",
    "REJECTED",
    "TRADEABLE",
    "HIGH_CONVICTION",
    "PRIORITY",
    "ACTIONABLE",
    "EXECUTION_READY",
    "READY",
    "FAILED",
}


@pytest.fixture()
def patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    processed_dir = tmp_path / "data" / "processed"
    logs_dir = tmp_path / "data" / "logs"
    processed_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    input_path = processed_dir / "fundamental_quality.csv"
    auxiliary_path = processed_dir / "entry_quality_metrics.csv"
    output_path = processed_dir / "timing_state_layer.csv"
    log_path = logs_dir / "timing_state_layer_log.csv"

    monkeypatch.setattr(timing_module, "INPUT_PATH", input_path)
    monkeypatch.setattr(timing_module, "AUXILIARY_PATH", auxiliary_path)
    monkeypatch.setattr(timing_module, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(timing_module, "LOG_PATH", log_path)

    return input_path, auxiliary_path, output_path, log_path


def _fundamental_row(ticker: str, date: str = "2026-05-09") -> dict:
    return {
        "ticker": ticker,
        "date": date,
        "quality_state": "INSUFFICIENT_DATA",
        "quality_reason": "fundamental data unavailable",
        "profitability_profile": "UNAVAILABLE",
        "balance_sheet_profile": "UNAVAILABLE",
        "earnings_quality_profile": "UNAVAILABLE",
        "capital_efficiency_profile": "UNAVAILABLE",
        "cashflow_profile": "UNAVAILABLE",
        "stability_profile": "UNAVAILABLE",
        "quality_metadata_status": "source_missing",
        "source_data_status": "source_missing",
        "source_timestamp": "",
        "generated_at": "2026-05-09 11:00:00",
    }


def _write_fundamental(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_auxiliary(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_with_rows(patch_paths, rows: list[dict]) -> tuple[pd.DataFrame, Path, Path]:
    input_path, _, output_path, log_path = patch_paths
    _write_fundamental(input_path, rows)
    df = timing_module.build_timing_state_layer(generated_at="2026-05-09 12:00:00")
    return df, output_path, log_path


def test_successful_build_preserves_upstream_columns_and_appends_metadata(patch_paths):
    rows = [_fundamental_row("AAA")]
    df, output_path, _ = _build_with_rows(patch_paths, rows)
    written_df = pd.read_csv(output_path)

    assert list(df.columns) == EXPECTED_OUTPUT_COLUMNS
    assert list(written_df.columns) == EXPECTED_OUTPUT_COLUMNS
    assert df.loc[0, "ticker"] == "AAA"
    assert df.loc[0, "quality_state"] == "INSUFFICIENT_DATA"
    assert df.loc[0, "timing_state"] == "UNCLASSIFIED"


def test_output_preserves_row_count_ticker_universe_and_ordering(patch_paths):
    rows = [
        _fundamental_row("CCC"),
        _fundamental_row("AAA"),
        _fundamental_row("BBB"),
    ]

    df, _, _ = _build_with_rows(patch_paths, rows)

    assert len(df) == len(rows)
    assert df["ticker"].tolist() == ["CCC", "AAA", "BBB"]
    assert set(df["ticker"]) == {"AAA", "BBB", "CCC"}


def test_non_mutating_enrichment_preserves_upstream_values(patch_paths):
    rows = [
        _fundamental_row("AAA"),
        _fundamental_row("BBB"),
    ]

    df, _, _ = _build_with_rows(patch_paths, rows)
    upstream_df = pd.DataFrame(rows)

    for column in UPSTREAM_COLUMNS:
        assert df[column].astype("string").fillna("").tolist() == upstream_df[column].astype("string").fillna("").tolist()


def test_missing_auxiliary_source_preserves_rows_with_descriptive_metadata(patch_paths):
    rows = [
        _fundamental_row("AAA"),
        _fundamental_row("BBB"),
    ]

    df, _, _ = _build_with_rows(patch_paths, rows)

    assert len(df) == 2
    assert set(df["timing_metadata_status"]) == {"SOURCE_MISSING"}
    assert set(df["timing_source_data_status"]) == {"SOURCE_MISSING"}


def test_auxiliary_observations_append_descriptive_metadata_without_filtering(patch_paths):
    input_path, auxiliary_path, _, _ = patch_paths
    rows = [
        _fundamental_row("AAA"),
        _fundamental_row("BBB"),
        _fundamental_row("CCC"),
    ]
    _write_fundamental(input_path, rows)
    _write_auxiliary(
        auxiliary_path,
        [
            {"ticker": "AAA", "date": "2026-05-09", "entry_quality_state": "EXTENDED"},
            {"ticker": "BBB", "date": "2026-05-09", "entry_quality_state": "PULLBACK"},
        ],
    )

    df = timing_module.build_timing_state_layer(generated_at="2026-05-09 12:00:00")

    assert df["ticker"].tolist() == ["AAA", "BBB", "CCC"]
    assert df.loc[0, "extension_state"] == "EXTENDED"
    assert df.loc[1, "pullback_state"] == "PULLBACK_OBSERVED"
    assert df.loc[2, "timing_metadata_status"] == "SOURCE_MISSING"


def test_missing_input_column_fails_fast(patch_paths):
    input_path, _, _, _ = patch_paths
    pd.DataFrame([{"ticker": "AAA"}]).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        timing_module.build_timing_state_layer(generated_at="2026-05-09 12:00:00")


def test_missing_ticker_fails_fast(patch_paths):
    input_path, _, _, _ = patch_paths
    _write_fundamental(input_path, [_fundamental_row("")])

    with pytest.raises(ValueError, match="missing ticker"):
        timing_module.build_timing_state_layer(generated_at="2026-05-09 12:00:00")


def test_invalid_date_fails_fast(patch_paths):
    input_path, _, _, _ = patch_paths
    _write_fundamental(input_path, [_fundamental_row("AAA", date="not-a-date")])

    with pytest.raises(ValueError, match="invalid date"):
        timing_module.build_timing_state_layer(generated_at="2026-05-09 12:00:00")


def test_duplicate_ticker_date_fails_fast(patch_paths):
    input_path, _, _, _ = patch_paths
    _write_fundamental(
        input_path,
        [
            _fundamental_row("AAA"),
            _fundamental_row("AAA"),
        ],
    )

    with pytest.raises(ValueError, match="duplicate ticker/date"):
        timing_module.build_timing_state_layer(generated_at="2026-05-09 12:00:00")


def test_reserved_timing_columns_in_input_fail_fast(patch_paths):
    input_path, _, _, _ = patch_paths
    row = _fundamental_row("AAA")
    row["timing_state"] = "UNCLASSIFIED"
    _write_fundamental(input_path, [row])

    with pytest.raises(ValueError, match="reserved timing columns"):
        timing_module.build_timing_state_layer(generated_at="2026-05-09 12:00:00")


def test_forbidden_columns_are_absent_from_output(patch_paths):
    df, _, _ = _build_with_rows(patch_paths, [_fundamental_row("AAA")])
    normalized_columns = {column.lower() for column in df.columns}

    assert normalized_columns.isdisjoint(FORBIDDEN_FIELDS)


def test_forbidden_semantic_values_are_absent_from_output(patch_paths):
    df, _, _ = _build_with_rows(patch_paths, [_fundamental_row("AAA")])
    values = {
        str(value).upper()
        for value in df.astype("string").fillna("").to_numpy().ravel()
        if str(value)
    }

    assert values.isdisjoint(FORBIDDEN_VALUES)


def test_deterministic_output_repeated_runs_match(patch_paths):
    rows = [
        _fundamental_row("CCC"),
        _fundamental_row("AAA"),
        _fundamental_row("BBB"),
    ]

    first_df, _, _ = _build_with_rows(patch_paths, rows)
    second_df = timing_module.build_timing_state_layer(generated_at="2026-05-09 12:00:00")

    pd.testing.assert_frame_equal(first_df, second_df)


def test_log_creation_and_schema(patch_paths):
    rows = [_fundamental_row("AAA"), _fundamental_row("BBB")]
    _, _, log_path = _build_with_rows(patch_paths, rows)

    log_df = pd.read_csv(log_path)

    assert list(log_df.columns) == EXPECTED_LOG_COLUMNS
    assert int(log_df.loc[0, "input_row_count"]) == 2
    assert int(log_df.loc[0, "output_row_count"]) == 2


def test_no_legacy_watchlist_states_are_emitted(patch_paths):
    df, _, _ = _build_with_rows(patch_paths, [_fundamental_row("AAA")])

    values = {
        str(value).upper()
        for value in df.astype("string").fillna("").to_numpy().ravel()
        if str(value)
    }

    assert "READY" not in values
    assert "FAILED" not in values
