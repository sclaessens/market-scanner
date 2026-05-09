from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.core import build_fundamental_layer as fundamental_module

EXPECTED_OUTPUT_COLUMNS = [
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

EXPECTED_LOG_COLUMNS = [
    "generated_at",
    "input_row_count",
    "output_row_count",
    "unique_ticker_date_count",
    "duplicate_ticker_date_count",
    "missing_fundamentals_count",
    "partial_data_count",
    "stale_data_count",
    "quality_state_distribution",
    "quality_metadata_status_distribution",
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
    "best_opportunity",
    "allocation",
    "allocation_weight",
    "urgency",
    "final_action",
    "final_score",
    "decision",
    "signal_strength",
    "gate",
    "pass_fail",
}

FORBIDDEN_VALUES = {
    "BUY",
    "SELL",
    "HOLD",
    "WAIT",
    "REVIEW",
    "APPROVED",
    "REJECTED",
    "TRADEABLE",
    "NOT_TRADEABLE",
    "HIGH_CONVICTION",
    "LOW_CONVICTION",
    "PRIORITY",
    "ACTIONABLE",
    "EXECUTION_READY",
}


@pytest.fixture()
def patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    processed_dir = tmp_path / "data" / "processed"
    logs_dir = tmp_path / "data" / "logs"
    processed_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    context_path = processed_dir / "context_strength.csv"
    output_path = processed_dir / "fundamental_quality.csv"
    log_path = logs_dir / "fundamental_layer_log.csv"

    monkeypatch.setattr(fundamental_module, "CONTEXT_PATH", context_path)
    monkeypatch.setattr(fundamental_module, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(fundamental_module, "LOG_PATH", log_path)

    return context_path, output_path, log_path


def _context_row(ticker: str, date: str = "2026-05-07", strength: str = "NEUTRAL") -> dict:
    return {
        "ticker": ticker,
        "date": date,
        "rs_score": 1.0,
        "rs_percentile": 50.0,
        "rs_rank": 1,
        "rs_vs_market": 1.0,
        "rs_vs_sector": None,
        "context_strength": strength,
        "context_reason": "middle_distribution",
        "leadership_state": strength,
    }


def _write_context(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_with_rows(patch_paths, rows: list[dict]) -> tuple[pd.DataFrame, Path, Path]:
    context_path, output_path, log_path = patch_paths
    _write_context(context_path, rows)
    df = fundamental_module.build_fundamental_layer(generated_at="2026-05-09 12:00:00")
    return df, output_path, log_path


def test_output_schema_contains_required_columns_in_order(patch_paths):
    df, output_path, _ = _build_with_rows(patch_paths, [_context_row("AAA")])

    written_df = pd.read_csv(output_path)

    assert list(df.columns) == EXPECTED_OUTPUT_COLUMNS
    assert list(written_df.columns) == EXPECTED_OUTPUT_COLUMNS


def test_output_preserves_all_input_rows_and_keys(patch_paths):
    rows = [
        _context_row("AAA", strength="WEAK"),
        _context_row("BBB", strength="NEUTRAL"),
        _context_row("CCC", strength="STRONG"),
    ]

    df, _, _ = _build_with_rows(patch_paths, rows)

    input_keys = {(row["ticker"], row["date"]) for row in rows}
    output_keys = set(zip(df["ticker"], df["date"], strict=True))

    assert len(df) == len(rows)
    assert output_keys == input_keys


def test_missing_fundamentals_preserve_rows_with_descriptive_states(patch_paths):
    df, _, _ = _build_with_rows(
        patch_paths,
        [
            _context_row("AAA", strength="WEAK"),
            _context_row("BBB", strength="LEADING"),
        ],
    )

    assert set(df["quality_state"]) == {"INSUFFICIENT_DATA"}
    assert set(df["quality_reason"]) == {"fundamental data unavailable"}
    assert set(df["quality_metadata_status"]) == {"source_missing"}
    assert set(df["source_data_status"]) == {"source_missing"}
    assert len(df) == 2


def test_duplicate_ticker_date_fails_fast(patch_paths):
    context_path, _, _ = patch_paths
    _write_context(
        context_path,
        [
            _context_row("AAA"),
            _context_row("AAA"),
        ],
    )

    with pytest.raises(ValueError, match="duplicate ticker/date"):
        fundamental_module.build_fundamental_layer(generated_at="2026-05-09 12:00:00")


def test_missing_ticker_fails_fast(patch_paths):
    context_path, _, _ = patch_paths
    _write_context(context_path, [_context_row("")])

    with pytest.raises(ValueError, match="missing ticker"):
        fundamental_module.build_fundamental_layer(generated_at="2026-05-09 12:00:00")


def test_missing_date_fails_fast(patch_paths):
    context_path, _, _ = patch_paths
    _write_context(context_path, [_context_row("AAA", date="")])

    with pytest.raises(ValueError, match="missing date"):
        fundamental_module.build_fundamental_layer(generated_at="2026-05-09 12:00:00")


def test_forbidden_fields_are_absent_from_output(patch_paths):
    df, _, _ = _build_with_rows(patch_paths, [_context_row("AAA")])

    normalized_columns = {column.lower() for column in df.columns}

    assert normalized_columns.isdisjoint(FORBIDDEN_FIELDS)


def test_forbidden_semantic_values_are_absent_from_output(patch_paths):
    df, _, _ = _build_with_rows(patch_paths, [_context_row("AAA")])

    values = {
        str(value)
        for value in df.astype("string").fillna("").to_numpy().ravel()
        if str(value)
    }

    assert values.isdisjoint(FORBIDDEN_VALUES)


def test_deterministic_ordering_is_reproducible_and_not_quality_based(patch_paths):
    rows = [
        _context_row("CCC", strength="WEAK"),
        _context_row("AAA", strength="LEADING"),
        _context_row("BBB", strength="NEUTRAL"),
    ]

    first_df, _, _ = _build_with_rows(patch_paths, rows)
    second_df = fundamental_module.build_fundamental_layer(generated_at="2026-05-09 12:00:00")

    assert first_df["ticker"].tolist() == ["CCC", "AAA", "BBB"]
    pd.testing.assert_frame_equal(first_df, second_df)


def test_no_actionability_tradeability_ranking_or_scoring_fields_are_produced(patch_paths):
    df, _, _ = _build_with_rows(patch_paths, [_context_row("AAA")])
    columns_text = " ".join(df.columns).lower()

    for term in FORBIDDEN_FIELDS:
        assert term not in columns_text


def test_log_output_contains_required_audit_metrics(patch_paths):
    df, _, log_path = _build_with_rows(
        patch_paths,
        [
            _context_row("AAA", strength="WEAK"),
            _context_row("BBB", strength="NEUTRAL"),
        ],
    )

    log_df = pd.read_csv(log_path)

    assert list(log_df.columns) == EXPECTED_LOG_COLUMNS
    assert log_df.loc[0, "input_row_count"] == len(df)
    assert log_df.loc[0, "output_row_count"] == len(df)
    assert log_df.loc[0, "unique_ticker_date_count"] == len(df)
    assert log_df.loc[0, "duplicate_ticker_date_count"] == 0
    assert log_df.loc[0, "missing_fundamentals_count"] == len(df)
    assert log_df.loc[0, "partial_data_count"] == 0
    assert log_df.loc[0, "stale_data_count"] == 0
    assert log_df.loc[0, "quality_state_distribution"] == '{"INSUFFICIENT_DATA":2}'
    assert log_df.loc[0, "quality_metadata_status_distribution"] == '{"source_missing":2}'
    assert log_df.loc[0, "source_data_status_distribution"] == '{"source_missing":2}'


def test_one_row_per_ticker_date_is_enforced(patch_paths):
    rows = [
        _context_row("AAA", date="2026-05-07"),
        _context_row("AAA", date="2026-05-08"),
    ]

    df, _, _ = _build_with_rows(patch_paths, rows)

    assert len(df) == 2
    assert not df.duplicated(subset=["ticker", "date"]).any()
