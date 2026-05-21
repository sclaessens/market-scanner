from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from scripts.core import build_portfolio_intelligence as portfolio_module

UPSTREAM_COLUMNS = [
    "ticker",
    "date",
    "quality_state",
    "quality_reason",
    "timing_state",
    "timing_reason",
]

PORTFOLIO_COLUMNS = [
    "in_portfolio",
    "portfolio_position_state",
    "exposure_state",
    "diversification_state",
    "concentration_state",
    "overlap_state",
    "sector_exposure_state",
    "position_context_state",
    "portfolio_environment",
    "portfolio_metadata_status",
    "portfolio_metadata_reason",
    "portfolio_source_provenance",
    "portfolio_classification_rationale",
]

EXPECTED_OUTPUT_COLUMNS = UPSTREAM_COLUMNS + PORTFOLIO_COLUMNS

EXPECTED_LOG_COLUMNS = [
    "ticker",
    "date",
    "input_row_index",
    "output_row_index",
    "row_identity_preserved",
    "ticker_preserved",
    "date_preserved",
    "ordering_preserved",
    "upstream_columns_preserved",
    "upstream_values_preserved",
    "portfolio_source_status",
    "portfolio_source_provenance",
    "portfolio_classification_rationale",
    "portfolio_metadata_status",
    "portfolio_metadata_reason",
    "forbidden_semantics_absent",
]

FORBIDDEN_FIELDS = {
    "allocation_weight",
    "recommended_weight",
    "ideal_position_size",
    "high_conviction",
    "conviction_score",
    "portfolio_priority",
    "actionable",
    "execution_ready",
    "best_opportunity",
    "buy_candidate",
    "sell_candidate",
    "ranking_score",
    "portfolio_score",
    "final_score",
    "allocation_signal",
    "recommended_trade",
    "preferred_position",
    "preferred_opportunity",
    "execution_signal",
    "urgency",
    "priority",
    "recommendation",
    "suitability",
    "attractiveness",
    "optimal_weight",
    "target_weight",
    "rebalance_action",
    "portfolio_fit",
    "portfolio_capacity",
    "exposure_allowance",
    "tradeable",
    "tradeability",
    "conviction",
    "score",
    "rank",
    "weight",
    "signal",
}

FORBIDDEN_VALUES = {
    "BUY",
    "SELL",
    "REMOVE",
    "ACTIONABLE",
    "EXECUTION_READY",
    "TRADEABLE",
    "URGENT",
    "HIGH_CONVICTION",
    "RECOMMENDED",
    "PREFERRED",
}


@pytest.fixture()
def patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    processed_dir = tmp_path / "data" / "processed"
    portfolio_dir = tmp_path / "data" / "portfolio"
    logs_dir = tmp_path / "data" / "logs"
    processed_dir.mkdir(parents=True, exist_ok=True)
    portfolio_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    input_path = processed_dir / "timing_state_layer.csv"
    portfolio_path = portfolio_dir / "portfolio_positions.csv"
    metadata_path = portfolio_dir / "portfolio_metadata.csv"
    output_path = processed_dir / "portfolio_intelligence.csv"
    log_path = logs_dir / "portfolio_intelligence_log.csv"

    monkeypatch.setattr(portfolio_module, "INPUT_PATH", input_path)
    monkeypatch.setattr(portfolio_module, "PORTFOLIO_PATH", portfolio_path)
    monkeypatch.setattr(portfolio_module, "PORTFOLIO_METADATA_PATH", metadata_path)
    monkeypatch.setattr(portfolio_module, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(portfolio_module, "LOG_PATH", log_path)

    return input_path, portfolio_path, output_path, log_path


def _timing_row(ticker: str, date: str = "2026-05-09", sector: str | None = None) -> dict:
    row = {
        "ticker": ticker,
        "date": date,
        "quality_state": "INSUFFICIENT_DATA",
        "quality_reason": "fundamental data unavailable",
        "timing_state": "UNCLASSIFIED",
        "timing_reason": "auxiliary timing source unavailable",
    }
    if sector is not None:
        row["sector"] = sector
    return row


def _write_timing(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_portfolio(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _metadata_row(ticker: str, **overrides) -> dict:
    row = {
        "ticker": ticker,
        "sector": "Technology",
        "industry": "Software",
        "asset_class": "Equity",
        "currency": "USD",
        "metadata_source": "manual",
        "metadata_last_updated": "2026-05-01",
    }
    row.update(overrides)
    return row


def _write_metadata(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_with_rows(patch_paths, rows: list[dict]) -> tuple[pd.DataFrame, Path, Path]:
    input_path, _, output_path, log_path = patch_paths
    _write_timing(input_path, rows)
    df = portfolio_module.build_portfolio_intelligence()
    return df, output_path, log_path


def test_successful_build_preserves_upstream_columns_and_appends_metadata(patch_paths):
    input_path, portfolio_path, output_path, _ = patch_paths
    rows = [_timing_row("AAA"), _timing_row("BBB")]
    _write_timing(input_path, rows)
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 2, "status": "OPEN"}])

    df = portfolio_module.build_portfolio_intelligence()
    written_df = pd.read_csv(output_path)

    assert list(df.columns) == EXPECTED_OUTPUT_COLUMNS
    assert list(written_df.columns) == EXPECTED_OUTPUT_COLUMNS
    assert df["ticker"].tolist() == ["AAA", "BBB"]
    assert df.loc[0, "in_portfolio"] == "PRESENT"
    assert df.loc[1, "in_portfolio"] == "ABSENT"


def test_output_preserves_row_count_ticker_universe_and_ordering(patch_paths):
    rows = [_timing_row("CCC"), _timing_row("AAA"), _timing_row("BBB")]
    df, _, _ = _build_with_rows(patch_paths, rows)

    assert len(df) == len(rows)
    assert df["ticker"].tolist() == ["CCC", "AAA", "BBB"]
    assert set(df["ticker"]) == {"AAA", "BBB", "CCC"}


def test_non_mutating_enrichment_preserves_upstream_values(patch_paths):
    rows = [_timing_row("AAA"), _timing_row("BBB")]
    df, _, _ = _build_with_rows(patch_paths, rows)
    upstream_df = pd.DataFrame(rows)

    for column in UPSTREAM_COLUMNS:
        assert df[column].astype("string").fillna("").tolist() == upstream_df[column].astype("string").fillna("").tolist()


def test_missing_portfolio_source_preserves_rows_with_neutral_metadata(patch_paths):
    rows = [_timing_row("AAA"), _timing_row("BBB")]
    df, _, _ = _build_with_rows(patch_paths, rows)

    assert len(df) == 2
    assert set(df["in_portfolio"]) == {"UNKNOWN"}
    assert set(df["portfolio_metadata_status"]) == {"MISSING"}
    assert set(df["portfolio_environment"]) == {"SOURCE_MISSING"}


def test_empty_portfolio_source_preserves_rows_with_empty_environment(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    rows = [_timing_row("AAA"), _timing_row("BBB")]
    _write_timing(input_path, rows)
    pd.DataFrame(columns=["ticker", "quantity", "status"]).to_csv(portfolio_path, index=False)

    df = portfolio_module.build_portfolio_intelligence()

    assert df["ticker"].tolist() == ["AAA", "BBB"]
    assert set(df["in_portfolio"]) == {"ABSENT"}
    assert set(df["portfolio_environment"]) == {"EMPTY_PORTFOLIO"}
    assert set(df["exposure_state"]) == {"NONE"}


def test_closed_portfolio_rows_do_not_create_presence(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    rows = [_timing_row("AAA"), _timing_row("BBB")]
    _write_timing(input_path, rows)
    _write_portfolio(
        portfolio_path,
        [
            {"ticker": "AAA", "quantity": 0, "status": "CLOSED"},
            {"ticker": "BBB", "quantity": 0, "status": "CLOSED"},
        ],
    )

    df = portfolio_module.build_portfolio_intelligence()

    assert df["in_portfolio"].tolist() == ["ABSENT", "ABSENT"]
    assert set(df["portfolio_environment"]) == {"EMPTY_PORTFOLIO"}


def test_partial_portfolio_source_preserves_rows_with_partial_metadata(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    _write_timing(input_path, [_timing_row("AAA")])
    pd.DataFrame([{"quantity": 1, "status": "OPEN"}]).to_csv(portfolio_path, index=False)

    df = portfolio_module.build_portfolio_intelligence()

    assert df.loc[0, "in_portfolio"] == "UNKNOWN"
    assert df.loc[0, "portfolio_metadata_status"] == "PARTIAL"
    assert df.loc[0, "portfolio_environment"] == "SOURCE_PARTIAL"


def test_duplicate_portfolio_source_tickers_are_grouped_without_reordering(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    rows = [_timing_row("BBB"), _timing_row("AAA"), _timing_row("CCC")]
    _write_timing(input_path, rows)
    _write_portfolio(
        portfolio_path,
        [
            {"ticker": "AAA", "quantity": 1, "status": "OPEN"},
            {"ticker": "AAA", "quantity": 2, "status": "OPEN"},
            {"ticker": "CCC", "quantity": 1, "status": "OPEN"},
        ],
    )

    df = portfolio_module.build_portfolio_intelligence()

    assert df["ticker"].tolist() == ["BBB", "AAA", "CCC"]
    assert df["in_portfolio"].tolist() == ["ABSENT", "PRESENT", "PRESENT"]


def test_sector_metadata_is_descriptive_when_sources_exist(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    rows = [
        _timing_row("AAA", sector="Technology"),
        _timing_row("BBB", sector="Healthcare"),
    ]
    _write_timing(input_path, rows)
    _write_portfolio(
        portfolio_path,
        [
            {"ticker": "AAA", "quantity": 1, "status": "OPEN", "sector": "Technology"},
            {"ticker": "CCC", "quantity": 1, "status": "OPEN", "sector": "Technology"},
        ],
    )

    df = portfolio_module.build_portfolio_intelligence()

    assert df.loc[0, "sector_exposure_state"] == "MODERATE"
    assert df.loc[1, "sector_exposure_state"] == "NONE"


def test_missing_portfolio_metadata_artifact_preserves_existing_partial_behavior(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    _write_timing(input_path, [_timing_row("AAA")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])

    df = portfolio_module.build_portfolio_intelligence()

    assert df.loc[0, "portfolio_metadata_status"] == "PARTIAL"
    assert df.loc[0, "portfolio_metadata_reason"] == "portfolio source available with partial sector metadata"


def test_complete_portfolio_metadata_can_produce_complete_status(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    _write_metadata(metadata_path, [_metadata_row(" aaa ")])

    df = portfolio_module.build_portfolio_intelligence()

    assert df.loc[0, "portfolio_metadata_status"] == "COMPLETE"
    assert df.loc[0, "portfolio_metadata_reason"] == "portfolio metadata complete"
    assert df.loc[0, "sector_exposure_state"] == "LOW"
    assert "portfolio_metadata.csv" in df.loc[0, "portfolio_source_provenance"]


def test_metadata_only_tickers_do_not_create_output_rows(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    _write_metadata(
        metadata_path,
        [
            _metadata_row("AAA"),
            _metadata_row("RAWONLY", sector="Healthcare"),
        ],
    )

    df = portfolio_module.build_portfolio_intelligence()

    assert len(df) == 1
    assert df["ticker"].tolist() == ["AAA"]


def test_missing_metadata_row_keeps_metadata_incomplete(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    _write_metadata(metadata_path, [_metadata_row("BBB")])

    df = portfolio_module.build_portfolio_intelligence()

    assert df.loc[0, "portfolio_metadata_status"] == "MISSING"
    assert df.loc[0, "portfolio_metadata_reason"] == "portfolio metadata row missing"
    assert df.loc[0, "sector_exposure_state"] == "SOURCE_PARTIAL"


def test_missing_required_metadata_values_produce_partial_status(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    _write_metadata(metadata_path, [_metadata_row("AAA", industry="")])

    df = portfolio_module.build_portfolio_intelligence()

    assert df.loc[0, "portfolio_metadata_status"] == "PARTIAL"
    assert df.loc[0, "portfolio_metadata_reason"] == "portfolio metadata partial: missing fields industry"


def test_stale_portfolio_metadata_remains_incomplete(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    _write_metadata(metadata_path, [_metadata_row("AAA", metadata_last_updated="2025-01-01")])

    df = portfolio_module.build_portfolio_intelligence()

    assert df.loc[0, "portfolio_metadata_status"] == "PARTIAL"
    assert df.loc[0, "portfolio_metadata_reason"] == "portfolio metadata stale: metadata_last_updated older than 365 days"


def test_empty_sector_metadata_value_is_partial_and_deterministic(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    _write_metadata(metadata_path, [_metadata_row("AAA", sector="")])

    df = portfolio_module.build_portfolio_intelligence()

    assert df.loc[0, "portfolio_metadata_status"] == "PARTIAL"
    assert df.loc[0, "portfolio_metadata_reason"] == "portfolio metadata partial: missing fields sector"


def test_invalid_asset_class_metadata_value_is_invalid_and_incomplete(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    _write_metadata(metadata_path, [_metadata_row("AAA", asset_class="Crypto")])

    df = portfolio_module.build_portfolio_intelligence()

    assert df.loc[0, "portfolio_metadata_status"] == "PARTIAL"
    assert df.loc[0, "portfolio_metadata_reason"] == "portfolio metadata invalid: asset_class"


def test_invalid_metadata_date_is_invalid_and_incomplete(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    _write_metadata(metadata_path, [_metadata_row("AAA", metadata_last_updated="2026/05/01")])

    df = portfolio_module.build_portfolio_intelligence()

    assert df.loc[0, "portfolio_metadata_status"] == "PARTIAL"
    assert df.loc[0, "portfolio_metadata_reason"] == "portfolio metadata invalid: metadata_last_updated"


def test_metadata_freshness_date_alias_is_backward_compatible(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    metadata = _metadata_row("AAA")
    metadata["metadata_freshness_date"] = metadata.pop("metadata_last_updated")
    _write_metadata(metadata_path, [metadata])

    df = portfolio_module.build_portfolio_intelligence()

    assert df.loc[0, "portfolio_metadata_status"] == "COMPLETE"
    assert df.loc[0, "portfolio_metadata_reason"] == "portfolio metadata complete"


def test_future_metadata_date_is_invalid_and_incomplete(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    _write_metadata(metadata_path, [_metadata_row("AAA", metadata_last_updated="2026-05-10")])

    df = portfolio_module.build_portfolio_intelligence()

    assert df.loc[0, "portfolio_metadata_status"] == "PARTIAL"
    assert df.loc[0, "portfolio_metadata_reason"] == "portfolio metadata invalid: metadata_last_updated after opportunity date"


def test_metadata_source_secret_marker_is_invalid_and_incomplete(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    _write_metadata(metadata_path, [_metadata_row("AAA", metadata_source="token:local")])

    df = portfolio_module.build_portfolio_intelligence()

    assert df.loc[0, "portfolio_metadata_status"] == "PARTIAL"
    assert df.loc[0, "portfolio_metadata_reason"] == "portfolio metadata invalid: metadata_source"


def test_duplicate_portfolio_metadata_rows_fail_fast_before_output_generation(patch_paths):
    input_path, portfolio_path, output_path, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    _write_metadata(metadata_path, [_metadata_row("AAA"), _metadata_row(" aaa ")])

    with pytest.raises(ValueError, match="duplicate normalized ticker rows"):
        portfolio_module.build_portfolio_intelligence()

    assert not output_path.exists()


def test_missing_required_metadata_schema_fields_fail_fast(patch_paths):
    input_path, portfolio_path, output_path, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    _write_timing(input_path, [_timing_row("AAA", date="2026-05-09")])
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    metadata = _metadata_row("AAA")
    metadata.pop("asset_class")
    _write_metadata(metadata_path, [metadata])

    with pytest.raises(ValueError, match="missing required columns"):
        portfolio_module.build_portfolio_intelligence()

    assert not output_path.exists()


def test_complete_metadata_preserves_upstream_row_count_identity_and_order(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    metadata_path = portfolio_path.with_name("portfolio_metadata.csv")
    rows = [
        _timing_row("CCC", date="2026-05-09"),
        _timing_row("AAA", date="2026-05-09"),
        _timing_row("BBB", date="2026-05-09"),
    ]
    _write_timing(input_path, rows)
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 1, "status": "OPEN"}])
    _write_metadata(
        metadata_path,
        [
            _metadata_row("AAA"),
            _metadata_row("BBB", sector="Healthcare"),
            _metadata_row("CCC", sector="Industrials"),
        ],
    )

    df = portfolio_module.build_portfolio_intelligence()

    assert len(df) == len(rows)
    assert list(zip(df["ticker"], df["date"], strict=True)) == [
        (row["ticker"], row["date"]) for row in rows
    ]
    assert set(df["portfolio_metadata_status"]) == {"COMPLETE"}


def test_missing_input_file_fails_fast(patch_paths):
    with pytest.raises(FileNotFoundError, match="timing_state_layer.csv not found"):
        portfolio_module.build_portfolio_intelligence()


def test_missing_ticker_column_fails_fast(patch_paths):
    input_path, _, _, _ = patch_paths
    pd.DataFrame([{"date": "2026-05-09"}]).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        portfolio_module.build_portfolio_intelligence()


def test_reserved_portfolio_columns_in_input_fail_fast(patch_paths):
    input_path, _, _, _ = patch_paths
    row = _timing_row("AAA")
    row["in_portfolio"] = "ABSENT"
    _write_timing(input_path, [row])

    with pytest.raises(ValueError, match="reserved portfolio columns"):
        portfolio_module.build_portfolio_intelligence()


def test_forbidden_columns_are_absent_from_output_and_log(patch_paths):
    df, _, log_path = _build_with_rows(patch_paths, [_timing_row("AAA")])
    log_df = pd.read_csv(log_path)
    normalized = {column.lower() for column in df.columns} | {column.lower() for column in log_df.columns}

    assert normalized.isdisjoint(FORBIDDEN_FIELDS)


def test_forbidden_semantic_values_are_absent_from_generated_metadata_and_log(patch_paths):
    df, _, log_path = _build_with_rows(patch_paths, [_timing_row("AAA")])
    log_df = pd.read_csv(log_path)
    generated = pd.concat([df[PORTFOLIO_COLUMNS], log_df.drop(columns=["ticker", "date"])], axis=1)
    values = {
        str(value).upper()
        for value in generated.astype("string").fillna("").to_numpy().ravel()
        if str(value).strip()
    }

    assert values.isdisjoint(FORBIDDEN_VALUES)


def test_deterministic_output_repeated_runs_match(patch_paths):
    input_path, portfolio_path, _, _ = patch_paths
    rows = [_timing_row("CCC"), _timing_row("AAA"), _timing_row("BBB")]
    _write_timing(input_path, rows)
    _write_portfolio(portfolio_path, [{"ticker": "AAA", "quantity": 2, "status": "OPEN"}])

    first_df = portfolio_module.build_portfolio_intelligence()
    second_df = portfolio_module.build_portfolio_intelligence()

    pd.testing.assert_frame_equal(first_df, second_df)


def test_log_creation_and_schema(patch_paths):
    rows = [_timing_row("AAA"), _timing_row("BBB")]
    _, _, log_path = _build_with_rows(patch_paths, rows)
    log_df = pd.read_csv(log_path)

    assert list(log_df.columns) == EXPECTED_LOG_COLUMNS
    assert len(log_df) == 2
    assert log_df["input_row_index"].tolist() == [0, 1]
    assert log_df["output_row_index"].tolist() == [0, 1]
    assert set(log_df["row_identity_preserved"]) == {True}
    assert set(log_df["upstream_values_preserved"]) == {True}


def test_no_decision_engine_dependency_or_leakage():
    source = inspect.getsource(portfolio_module)

    assert "decision_engine" not in source
    assert "final_action" not in source
    assert "allocation_priority" not in source


def test_no_reporting_or_telegram_dependency_or_leakage():
    source = inspect.getsource(portfolio_module)

    assert "scripts.reporting" not in source
    assert "build_reporting_layer" not in source
    assert "build_telegram_summary" not in source
    assert "send_telegram" not in source


def test_only_approved_output_files_are_written(patch_paths, tmp_path: Path):
    rows = [_timing_row("AAA")]
    _, output_path, log_path = _build_with_rows(patch_paths, rows)
    files = {path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*.csv")}

    assert output_path.exists()
    assert log_path.exists()
    assert files == {
        "data/processed/timing_state_layer.csv",
        "data/processed/portfolio_intelligence.csv",
        "data/logs/portfolio_intelligence_log.csv",
    }
