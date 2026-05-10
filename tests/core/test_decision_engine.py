from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from scripts.core import decision_engine

REQUIRED_INPUT_COLUMNS = decision_engine.REQUIRED_INPUT_COLUMNS
REQUIRED_OUTPUT_COLUMNS = decision_engine.OUTPUT_COLUMNS
REQUIRED_LOG_COLUMNS = decision_engine.LOG_COLUMNS

FORBIDDEN_OUTPUT_COLUMNS = {
    "decision_output",
    "conviction_score",
    "ranking_score",
    "portfolio_score",
    "final_score",
    "recommended_trade",
    "recommended_weight",
    "optimal_weight",
    "target_weight",
    "allocation_queue",
    "execution_urgency",
    "urgency",
    "actionable",
    "execution_ready",
}

DEFERRED_TERMS = {
    "persistence",
    "smoothing",
    "allocation_queue",
}


@pytest.fixture()
def patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    processed_dir = tmp_path / "data" / "processed"
    logs_dir = tmp_path / "data" / "logs"
    processed_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    input_path = processed_dir / "portfolio_intelligence.csv"
    output_path = processed_dir / "final_decisions.csv"
    log_path = logs_dir / "decision_engine_log.csv"

    monkeypatch.setattr(decision_engine, "INPUT_PATH", input_path)
    monkeypatch.setattr(decision_engine, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(decision_engine, "LOG_PATH", log_path)

    return input_path, output_path, log_path


def _input_row(ticker: str, date: str = "2026-05-09", **overrides) -> dict:
    row = {
        "ticker": ticker,
        "date": date,
        "quality_state": "QUALITY_CONFIRMED",
        "timing_state": "READY",
        "in_portfolio": "ABSENT",
        "portfolio_position_state": "ABSENT",
        "exposure_state": "LOW",
        "diversification_state": "BROAD",
        "concentration_state": "BALANCED",
        "overlap_state": "UNMATCHED",
        "sector_exposure_state": "AVAILABLE",
        "position_context_state": "ABSENT",
        "portfolio_environment": "POSITIONS_PRESENT",
        "portfolio_metadata_status": "AVAILABLE",
        "portfolio_metadata_reason": "portfolio metadata available",
        "quality_reason": "quality metadata available",
        "timing_reason": "timing metadata available",
        "portfolio_source_provenance": "data/portfolio/portfolio_positions.csv",
        "portfolio_classification_rationale": "portfolio metadata observed",
    }
    row.update(overrides)
    return row


def _write_input(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_with_rows(patch_paths, rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    input_path, output_path, log_path = patch_paths
    _write_input(input_path, rows)
    df = decision_engine.build_final_decisions()
    log_df = pd.read_csv(log_path)
    written_df = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(df, written_df)
    return df, log_df, input_path


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_output_schema_contract(patch_paths):
    df, _, _ = _build_with_rows(patch_paths, [_input_row("AAA")])

    assert list(df.columns[: len(REQUIRED_OUTPUT_COLUMNS)]) == REQUIRED_OUTPUT_COLUMNS
    assert not (set(df.columns) & FORBIDDEN_OUTPUT_COLUMNS)


def test_required_input_schema_contract(patch_paths):
    input_path, _, _ = patch_paths
    row = _input_row("AAA")
    row.pop("portfolio_metadata_status")
    _write_input(input_path, [row])

    with pytest.raises(ValueError, match="missing required columns"):
        decision_engine.build_final_decisions()


def test_one_decision_per_ticker_date_row(patch_paths):
    rows = [_input_row("AAA"), _input_row("BBB")]
    df, _, _ = _build_with_rows(patch_paths, rows)

    assert len(df) == 2
    assert not df[["ticker", "date"]].duplicated().any()


def test_deterministic_output_under_identical_inputs(patch_paths):
    rows = [_input_row("BBB"), _input_row("AAA", in_portfolio="PRESENT")]
    first_df, _, _ = _build_with_rows(patch_paths, rows)
    second_df = decision_engine.build_final_decisions()

    pd.testing.assert_frame_equal(first_df, second_df)


def test_deterministic_tie_breaking_preserves_input_order(patch_paths):
    rows = [_input_row("CCC"), _input_row("AAA"), _input_row("BBB")]
    df, log_df, _ = _build_with_rows(patch_paths, rows)

    assert df["ticker"].tolist() == ["CCC", "AAA", "BBB"]
    assert log_df.loc[0, "input_order_preserved"] in {True, "True", "true"}


def test_no_upstream_artifact_mutation(patch_paths):
    input_path, _, _ = patch_paths
    rows = [_input_row("AAA"), _input_row("BBB")]
    _write_input(input_path, rows)
    before_hash = _file_hash(input_path)

    decision_engine.build_final_decisions()

    assert _file_hash(input_path) == before_hash


def test_no_hidden_filtering(patch_paths):
    rows = [_input_row("AAA"), _input_row("BBB", portfolio_metadata_status="PARTIAL")]
    df, log_df, _ = _build_with_rows(patch_paths, rows)

    assert len(df) == len(rows)
    assert log_df.loc[0, "hidden_filtering_detected"] in {False, "False", "false"}


def test_no_silent_opportunity_suppression(patch_paths):
    rows = [_input_row("AAA"), _input_row("BBB", quality_state="INSUFFICIENT_DATA")]
    df, log_df, _ = _build_with_rows(patch_paths, rows)

    assert set(df["ticker"]) == {"AAA", "BBB"}
    assert log_df.loc[0, "silent_suppression_detected"] in {False, "False", "false"}
    assert df["allocation_rationale"].str.len().gt(0).all()


def test_ticker_date_visibility_preservation(patch_paths):
    rows = [_input_row("AAA", "2026-05-08"), _input_row("AAA", "2026-05-09")]
    df, log_df, _ = _build_with_rows(patch_paths, rows)

    input_keys = {(row["ticker"], row["date"]) for row in rows}
    output_keys = set(map(tuple, df[["ticker", "date"]].values.tolist()))
    assert output_keys == input_keys
    assert log_df.loc[0, "ticker_date_universe_preserved"] in {True, "True", "true"}


def test_source_provenance_completeness(patch_paths):
    df, log_df, _ = _build_with_rows(patch_paths, [_input_row("AAA")])

    assert df["source_provenance"].str.contains("portfolio_intelligence.csv").all()
    assert set(log_df["source_provenance_status"]) == {"COMPLETE"}


def test_decision_rationale_completeness(patch_paths):
    df, log_df, _ = _build_with_rows(patch_paths, [_input_row("AAA")])
    rationale_columns = [
        "allocation_rationale",
        "execution_rationale",
        "arbitration_reason",
        "conflict_resolution_reason",
    ]

    assert df[rationale_columns].notna().all().all()
    assert (df[rationale_columns].astype(str).apply(lambda col: col.str.len() > 0)).all().all()
    assert set(log_df["rationale_completeness_status"]) == {"COMPLETE"}


def test_forbidden_semantics_outside_decision_engine_are_not_imported():
    source = Path(decision_engine.__file__).read_text()

    assert "watchlist_status.csv" not in source
    assert "portfolio_review.csv" not in source
    assert "action_now" not in source
    assert "entry_bias" not in source


def test_no_reporting_authority_leakage():
    source = Path(decision_engine.__file__).read_text()

    assert "scripts/reporting" not in source
    assert "reporting" not in source.lower()


def test_no_watchlist_authority_leakage():
    source = Path(decision_engine.__file__).read_text()

    assert "watchlist" not in source.lower()


def test_no_portfolio_authority_leakage():
    source = Path(decision_engine.__file__).read_text()

    assert "portfolio_review.csv" not in source
    assert "data/portfolio" not in source


def test_no_portfolio_intelligence_authority_leakage(patch_paths):
    rows = [_input_row("AAA", portfolio_metadata_status="PARTIAL")]
    df, _, _ = _build_with_rows(patch_paths, rows)

    assert df.loc[0, "final_action"] == "REVIEW"
    assert df.loc[0, "arbitration_state"] == "MISSING_METADATA"
    assert df.loc[0, "source_provenance"].startswith("data/processed/portfolio_intelligence.csv")


def test_missing_input_fail_fast(patch_paths):
    with pytest.raises(FileNotFoundError, match="Required Decision Engine input is missing"):
        decision_engine.build_final_decisions()


def test_missing_optional_metadata_behavior(patch_paths):
    input_path, _, _ = patch_paths
    row = _input_row("AAA")
    row.pop("quality_reason")
    row.pop("portfolio_source_provenance")
    _write_input(input_path, [row])

    df = decision_engine.build_final_decisions()

    assert len(df) == 1
    assert df.loc[0, "source_provenance"] == "data/processed/portfolio_intelligence.csv"


def test_log_schema_contract(patch_paths):
    _, log_df, _ = _build_with_rows(patch_paths, [_input_row("AAA")])

    assert list(log_df.columns) == REQUIRED_LOG_COLUMNS
    assert log_df.loc[0, "decision_contract_version"] == decision_engine.DECISION_CONTRACT_VERSION


def test_forbidden_output_column_absence(patch_paths):
    df, _, _ = _build_with_rows(patch_paths, [_input_row("AAA")])

    assert not (set(df.columns) & FORBIDDEN_OUTPUT_COLUMNS)


def test_deferred_concept_absence_for_persistence_and_smoothing(patch_paths):
    df, log_df, _ = _build_with_rows(patch_paths, [_input_row("AAA")])
    combined_columns = " ".join(list(df.columns) + list(log_df.columns)).lower()

    for term in DEFERRED_TERMS:
        assert term not in combined_columns


def test_duplicate_ticker_date_rows_fail_fast(patch_paths):
    input_path, _, _ = patch_paths
    _write_input(input_path, [_input_row("AAA"), _input_row("AAA")])

    with pytest.raises(ValueError, match="duplicate ticker/date"):
        decision_engine.build_final_decisions()


def test_allowed_decision_values(patch_paths):
    df, _, _ = _build_with_rows(
        patch_paths,
        [
            _input_row("AAA"),
            _input_row("BBB", in_portfolio="PRESENT"),
            _input_row("CCC", quality_state="INSUFFICIENT_DATA"),
        ],
    )

    assert set(df["final_action"]).issubset(decision_engine.FINAL_ACTIONS)
    assert set(df["allocation_decision"]).issubset(decision_engine.ALLOCATION_DECISIONS)
    assert set(df["execution_decision"]).issubset(decision_engine.EXECUTION_DECISIONS)
    assert set(df["arbitration_state"]).issubset(decision_engine.ARBITRATION_STATES)
