from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from scripts.ops import capture_historical_evidence as capture


CAPTURED_AT = "2026-05-18T12:00:00Z"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision_row(
    ticker: str = "AAA",
    date: str = "2026-05-18",
    input_row_hash: str = "hash-aaa",
) -> dict:
    return {
        "ticker": ticker,
        "date": date,
        "final_action": "REVIEW",
        "allocation_decision": "REVIEW_REQUIRED",
        "execution_decision": "REVIEW_REQUIRED",
        "portfolio_decision_state": "OBSERVED",
        "opportunity_decision_state": "OBSERVED",
        "arbitration_state": "NOT_REQUIRED",
        "allocation_rationale": "source decision rationale",
        "execution_rationale": "source execution rationale",
        "arbitration_reason": "source arbitration reason",
        "conflict_resolution_reason": "source conflict reason",
        "source_provenance": "data/processed/portfolio_intelligence.csv",
        "decision_contract_version": "TEST_DECISION_CONTRACT",
        "input_row_hash": input_row_hash,
    }


def reporting_row(
    ticker: str = "AAA",
    date: str = "2026-05-18",
    source_input_row_hash: str = "hash-aaa",
    source_row_index: str = "0",
) -> dict:
    return {
        "ticker": ticker,
        "date": date,
        "source_artifact_path": "data/processed/final_decisions.csv",
        "source_row_identity": (
            f"data/processed/final_decisions.csv#{source_row_index}#"
            f"{ticker}#{date}#{source_input_row_hash}"
        ),
        "source_row_index": source_row_index,
        "reporting_contract_version": "TEST_REPORTING_CONTRACT",
        "report_section": "DECISION_OUTPUTS",
        "display_mode": "ROW_DISPLAYED",
        "source_final_action": "REVIEW",
        "source_allocation_decision": "REVIEW_REQUIRED",
        "source_execution_decision": "REVIEW_REQUIRED",
        "source_portfolio_decision_state": "OBSERVED",
        "source_opportunity_decision_state": "OBSERVED",
        "source_arbitration_state": "NOT_REQUIRED",
        "source_input_row_hash": source_input_row_hash,
    }


def seed_source_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    decision_path = tmp_path / "data" / "processed" / "final_decisions.csv"
    reporting_path = tmp_path / "data" / "processed" / "reporting_dashboard_data.csv"
    write_csv(decision_path, [decision_row()])
    write_csv(reporting_path, [reporting_row()])
    return decision_path, reporting_path


def read_history(history_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(history_dir / "pipeline_runs.csv"),
        pd.read_csv(history_dir / "pipeline_artifacts.csv"),
        pd.read_csv(history_dir / "decision_reporting_observations.csv"),
    )


def test_historical_capture_creates_expected_artifacts(tmp_path: Path):
    seed_source_artifacts(tmp_path)
    history_dir = tmp_path / "data" / "history"

    result = capture.capture_historical_evidence(
        project_root=tmp_path,
        history_dir=history_dir,
        captured_at=CAPTURED_AT,
    )

    assert result.pipeline_runs_file == history_dir / "pipeline_runs.csv"
    assert result.pipeline_artifacts_file == history_dir / "pipeline_artifacts.csv"
    assert result.decision_reporting_observations_file == (
        history_dir / "decision_reporting_observations.csv"
    )
    assert result.pipeline_runs_file.exists()
    assert result.pipeline_artifacts_file.exists()
    assert result.decision_reporting_observations_file.exists()


def test_run_id_is_confined_to_historical_outputs(tmp_path: Path):
    decision_path, reporting_path = seed_source_artifacts(tmp_path)
    decision_before = pd.read_csv(decision_path).copy()
    reporting_before = pd.read_csv(reporting_path).copy()

    result = capture.capture_historical_evidence(
        project_root=tmp_path,
        history_dir=tmp_path / "data" / "history",
        captured_at=CAPTURED_AT,
    )

    runs, artifacts, observations = read_history(tmp_path / "data" / "history")
    assert runs.loc[0, "run_id"] == result.run_id
    assert set(artifacts["run_id"]) == {result.run_id}
    assert set(observations["run_id"]) == {result.run_id}
    assert "run_id" not in pd.read_csv(decision_path).columns
    assert "run_id" not in pd.read_csv(reporting_path).columns
    pd.testing.assert_frame_equal(pd.read_csv(decision_path), decision_before)
    pd.testing.assert_frame_equal(pd.read_csv(reporting_path), reporting_before)


def test_artifact_manifest_records_paths_and_row_counts(tmp_path: Path):
    seed_source_artifacts(tmp_path)
    capture.capture_historical_evidence(
        project_root=tmp_path,
        history_dir=tmp_path / "data" / "history",
        captured_at=CAPTURED_AT,
    )

    _, artifacts, _ = read_history(tmp_path / "data" / "history")
    decision_artifact = artifacts.loc[
        artifacts["artifact_path"] == "data/processed/final_decisions.csv"
    ].iloc[0]
    missing_artifact = artifacts.loc[
        artifacts["artifact_path"] == "data/processed/scanner_ranked.csv"
    ].iloc[0]

    assert decision_artifact["artifact_exists"] in {True, "True"}
    assert decision_artifact["row_count"] == 1
    assert int(decision_artifact["file_size_bytes"]) > 0
    assert isinstance(decision_artifact["content_hash"], str)
    assert missing_artifact["artifact_exists"] in {False, "False"}
    assert missing_artifact["diagnostic_notes"] == "ARTIFACT_MISSING"


def test_decision_reporting_observations_preserve_source_row_evidence(tmp_path: Path):
    seed_source_artifacts(tmp_path)
    capture.capture_historical_evidence(
        project_root=tmp_path,
        history_dir=tmp_path / "data" / "history",
        captured_at=CAPTURED_AT,
    )

    _, _, observations = read_history(tmp_path / "data" / "history")
    row = observations.iloc[0]

    assert row["ticker"] == "AAA"
    assert row["date"] == "2026-05-18"
    assert row["decision_row_index"] == 0
    assert row["reporting_row_index"] == 0
    assert row["decision_artifact_path"] == "data/processed/final_decisions.csv"
    assert row["reporting_artifact_path"] == "data/processed/reporting_dashboard_data.csv"
    assert row["decision_input_row_hash"] == "hash-aaa"
    assert "data/processed/final_decisions.csv#0#AAA#2026-05-18#hash-aaa" == row[
        "reporting_source_row_identity"
    ]
    assert row["reporting_represented_flag"] in {True, "True"}
    assert row["diagnostic_note"] == "LINKED"


def test_unmatched_decision_and_reporting_rows_are_diagnostic(tmp_path: Path):
    decision_path = tmp_path / "data" / "processed" / "final_decisions.csv"
    reporting_path = tmp_path / "data" / "processed" / "reporting_dashboard_data.csv"
    write_csv(decision_path, [decision_row(input_row_hash="hash-decision-only")])
    write_csv(
        reporting_path,
        [reporting_row(ticker="BBB", source_input_row_hash="hash-reporting-only")],
    )

    capture.capture_historical_evidence(
        project_root=tmp_path,
        history_dir=tmp_path / "data" / "history",
        captured_at=CAPTURED_AT,
    )

    runs, _, observations = read_history(tmp_path / "data" / "history")
    assert set(observations["diagnostic_note"]) == {
        "REPORTING_ROW_NOT_MATCHED",
        "REPORTING_ROW_WITHOUT_DECISION_MATCH",
    }
    assert runs.loc[0, "decision_reporting_linkage_status"] == "PARTIAL"


def test_missing_decision_and_reporting_sources_create_empty_observation_artifact(
    tmp_path: Path,
):
    history_dir = tmp_path / "data" / "history"
    decision_path = tmp_path / "data" / "processed" / "final_decisions.csv"
    reporting_path = tmp_path / "data" / "processed" / "reporting_dashboard_data.csv"

    capture.capture_historical_evidence(
        project_root=tmp_path,
        history_dir=history_dir,
        captured_at=CAPTURED_AT,
    )

    runs_path = history_dir / "pipeline_runs.csv"
    artifacts_path = history_dir / "pipeline_artifacts.csv"
    observations_path = history_dir / "decision_reporting_observations.csv"
    assert runs_path.exists()
    assert artifacts_path.exists()
    assert observations_path.exists()

    observations = pd.read_csv(observations_path)
    assert observations.empty
    assert list(observations.columns) == capture.DECISION_REPORTING_OBSERVATION_COLUMNS

    runs = pd.read_csv(runs_path)
    assert runs.loc[0, "decision_reporting_linkage_status"] == "NO_DECISION_OR_REPORTING_ROWS"
    assert runs.loc[0, "decision_reporting_observation_count"] == 0

    artifacts = pd.read_csv(artifacts_path)
    decision_artifact = artifacts.loc[
        artifacts["artifact_path"] == "data/processed/final_decisions.csv"
    ].iloc[0]
    reporting_artifact = artifacts.loc[
        artifacts["artifact_path"] == "data/processed/reporting_dashboard_data.csv"
    ].iloc[0]
    assert decision_artifact["artifact_exists"] in {False, "False"}
    assert reporting_artifact["artifact_exists"] in {False, "False"}
    assert decision_artifact["diagnostic_notes"] == "ARTIFACT_MISSING"
    assert reporting_artifact["diagnostic_notes"] == "ARTIFACT_MISSING"
    assert not decision_path.exists()
    assert not reporting_path.exists()


def test_duplicate_or_missing_identity_is_recorded_without_filtering(tmp_path: Path):
    decision_path = tmp_path / "data" / "processed" / "final_decisions.csv"
    reporting_path = tmp_path / "data" / "processed" / "reporting_dashboard_data.csv"
    write_csv(
        decision_path,
        [
            decision_row(ticker="AAA", input_row_hash="duplicate-hash"),
            decision_row(ticker="BBB", input_row_hash="duplicate-hash"),
            decision_row(ticker="", date="", input_row_hash=""),
        ],
    )
    write_csv(
        reporting_path,
        [
            reporting_row(ticker="AAA", source_input_row_hash="duplicate-hash"),
            {
                **reporting_row(ticker="", date="", source_input_row_hash=""),
                "source_row_identity": "",
            },
        ],
    )

    capture.capture_historical_evidence(
        project_root=tmp_path,
        history_dir=tmp_path / "data" / "history",
        captured_at=CAPTURED_AT,
    )

    _, _, observations = read_history(tmp_path / "data" / "history")
    assert len(observations) == 5
    assert "DUPLICATE_DECISION_IDENTITY" in set(observations["diagnostic_note"])
    assert "MISSING_DECISION_IDENTITY" in set(observations["diagnostic_note"])
    assert "MISSING_REPORTING_IDENTITY" in set(observations["diagnostic_note"])


def test_source_artifacts_are_not_modified(tmp_path: Path):
    decision_path, reporting_path = seed_source_artifacts(tmp_path)
    decision_hash_before = file_hash(decision_path)
    reporting_hash_before = file_hash(reporting_path)

    capture.capture_historical_evidence(
        project_root=tmp_path,
        history_dir=tmp_path / "data" / "history",
        captured_at=CAPTURED_AT,
    )

    assert file_hash(decision_path) == decision_hash_before
    assert file_hash(reporting_path) == reporting_hash_before


def test_output_ordering_is_deterministic_for_same_input(tmp_path: Path):
    decision_path = tmp_path / "data" / "processed" / "final_decisions.csv"
    reporting_path = tmp_path / "data" / "processed" / "reporting_dashboard_data.csv"
    write_csv(
        decision_path,
        [
            decision_row(ticker="BBB", input_row_hash="hash-bbb"),
            decision_row(ticker="AAA", input_row_hash="hash-aaa"),
        ],
    )
    write_csv(
        reporting_path,
        [
            reporting_row(ticker="BBB", source_input_row_hash="hash-bbb", source_row_index="0"),
            reporting_row(ticker="AAA", source_input_row_hash="hash-aaa", source_row_index="1"),
        ],
    )

    first_history = tmp_path / "first_history"
    second_history = tmp_path / "second_history"
    capture.capture_historical_evidence(
        project_root=tmp_path,
        history_dir=first_history,
        captured_at=CAPTURED_AT,
    )
    capture.capture_historical_evidence(
        project_root=tmp_path,
        history_dir=second_history,
        captured_at=CAPTURED_AT,
    )

    first_observations = pd.read_csv(first_history / "decision_reporting_observations.csv")
    second_observations = pd.read_csv(second_history / "decision_reporting_observations.csv")
    assert first_observations["ticker"].tolist() == ["BBB", "AAA"]
    pd.testing.assert_frame_equal(first_observations, second_observations)
