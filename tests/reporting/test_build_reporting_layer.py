from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.reporting import build_reporting_layer as reporting


def sample_final_decisions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "date": "2026-05-10",
                "final_action": "HOLD",
                "allocation_decision": "SOURCE_HOLD",
                "execution_decision": "SOURCE_NONE",
                "portfolio_decision_state": "SOURCE_PORTFOLIO",
                "opportunity_decision_state": "SOURCE_OPPORTUNITY",
                "arbitration_state": "SOURCE_CLEAR",
                "allocation_rationale": "source allocation rationale",
                "execution_rationale": "source execution rationale",
                "arbitration_reason": "source arbitration reason",
                "conflict_resolution_reason": "source conflict reason",
                "source_provenance": "DECISION_ENGINE",
                "decision_contract_version": "DECISION_CONTRACT_V1",
                "input_row_hash": "hash-aaa",
            },
            {
                "ticker": "BBB",
                "date": "2026-05-10",
                "final_action": "WAIT",
                "allocation_decision": "SOURCE_WAIT",
                "execution_decision": "SOURCE_NONE",
                "portfolio_decision_state": "SOURCE_PORTFOLIO",
                "opportunity_decision_state": "SOURCE_OPPORTUNITY",
                "arbitration_state": "SOURCE_CLEAR",
                "allocation_rationale": "source allocation rationale",
                "execution_rationale": "source execution rationale",
                "arbitration_reason": "source arbitration reason",
                "conflict_resolution_reason": "source conflict reason",
                "source_provenance": "DECISION_ENGINE",
                "decision_contract_version": "DECISION_CONTRACT_V1",
                "input_row_hash": "hash-bbb",
            },
            {
                "ticker": "CCC",
                "date": "2026-05-10",
                "final_action": "WAIT",
                "allocation_decision": "SOURCE_WAIT",
                "execution_decision": "SOURCE_NONE",
                "portfolio_decision_state": "SOURCE_PORTFOLIO",
                "opportunity_decision_state": "SOURCE_OPPORTUNITY",
                "arbitration_state": "SOURCE_CLEAR",
                "allocation_rationale": "source allocation rationale",
                "execution_rationale": "source execution rationale",
                "arbitration_reason": "source arbitration reason",
                "conflict_resolution_reason": "source conflict reason",
                "source_provenance": "DECISION_ENGINE",
                "decision_contract_version": "DECISION_CONTRACT_V1",
                "input_row_hash": "hash-ccc",
            },
            {
                "ticker": "DDD",
                "date": "2026-05-10",
                "final_action": "WAIT",
                "allocation_decision": "SOURCE_WAIT",
                "execution_decision": "SOURCE_NONE",
                "portfolio_decision_state": "SOURCE_PORTFOLIO",
                "opportunity_decision_state": "SOURCE_OPPORTUNITY",
                "arbitration_state": "SOURCE_CLEAR",
                "allocation_rationale": "source allocation rationale",
                "execution_rationale": "source execution rationale",
                "arbitration_reason": "source arbitration reason",
                "conflict_resolution_reason": "source conflict reason",
                "source_provenance": "DECISION_ENGINE",
                "decision_contract_version": "DECISION_CONTRACT_V1",
                "input_row_hash": "hash-ddd",
            },
        ]
    )


def sample_stability_state() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "date": "2026-05-10",
                "stability_state": "STABLE",
                "conviction_persistence": "SOURCE_UNAVAILABLE",
                "action_persistence": "PERSISTENT",
                "behavioural_stability": "STABLE",
                "transition_frequency": 0,
                "escalation_frequency": 0,
                "stability_reason": "metadata only",
                "persistence_duration": 1,
            }
        ]
    )


def configure_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(reporting, "FINAL_DECISIONS_FILE", tmp_path / "data/processed/final_decisions.csv")
    monkeypatch.setattr(reporting, "STABILITY_STATE_FILE", tmp_path / "data/processed/stability_state.csv")
    monkeypatch.setattr(
        reporting,
        "REPORTING_DASHBOARD_FILE",
        tmp_path / "data/processed/reporting_dashboard_data.csv",
    )
    monkeypatch.setattr(reporting, "REPORTING_LOG_FILE", tmp_path / "data/logs/reporting_layer_log.csv")
    monkeypatch.setattr(reporting, "TELEGRAM_MESSAGE_FILE", tmp_path / "reports/daily/telegram_message.txt")


def write_sources(tmp_path: Path, final_decisions: pd.DataFrame, stability_state: pd.DataFrame | None = None) -> None:
    final_path = tmp_path / "data/processed/final_decisions.csv"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_decisions.to_csv(final_path, index=False)
    if stability_state is not None:
        stability_path = tmp_path / "data/processed/stability_state.csv"
        stability_state.to_csv(stability_path, index=False)


def test_reporting_dashboard_and_log_schema_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    configure_paths(monkeypatch, tmp_path)
    write_sources(tmp_path, sample_final_decisions(), sample_stability_state())

    dashboard, log_row, text = reporting.build_reporting_layer()

    assert list(dashboard.columns) == reporting.DASHBOARD_COLUMNS
    assert list(log_row.keys()) == reporting.LOG_COLUMNS
    assert len(dashboard) == len(sample_final_decisions())
    assert log_row["source_row_count"] == len(sample_final_decisions())
    assert log_row["dashboard_row_count"] == len(sample_final_decisions())
    assert log_row["omitted_row_count"] == 0
    assert log_row["row_count_preserved"] == "True"
    assert log_row["ticker_date_universe_preserved"] == "True"
    assert "Source row count: 4" in text
    assert "Represented row count: 4" in text


def test_source_universe_and_source_row_identity_are_preserved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    configure_paths(monkeypatch, tmp_path)
    source = sample_final_decisions()
    write_sources(tmp_path, source)

    dashboard, _, _ = reporting.build_reporting_layer()

    assert dashboard["ticker"].tolist() == source["ticker"].tolist()
    assert dashboard["date"].tolist() == source["date"].tolist()
    assert dashboard["source_row_index"].tolist() == [0, 1, 2, 3]
    assert dashboard.loc[0, "source_row_identity"] == (
        "data/processed/final_decisions.csv#0#AAA#2026-05-10#hash-aaa"
    )
    assert dashboard["source_row_identity"].is_unique


def test_deterministic_output_generation_and_grouping(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    configure_paths(monkeypatch, tmp_path)
    write_sources(tmp_path, sample_final_decisions())

    first_dashboard, _, first_text = reporting.build_reporting_layer()
    second_dashboard, _, second_text = reporting.build_reporting_layer()

    pd.testing.assert_frame_equal(first_dashboard, second_dashboard)
    assert first_text == second_text
    assert first_text.index("Decision output: HOLD") < first_text.index("Decision output: WAIT")
    wait_rows = first_dashboard[first_dashboard["source_final_action"] == "WAIT"]
    assert wait_rows["source_row_index"].tolist() == [1, 2, 3]


def test_telegram_representation_metadata_has_no_hidden_omission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    configure_paths(monkeypatch, tmp_path)
    write_sources(tmp_path, sample_final_decisions())

    dashboard, log_row, text = reporting.build_reporting_layer()

    wait_rows = dashboard[dashboard["source_final_action"] == "WAIT"]
    assert wait_rows["display_mode"].tolist() == [
        reporting.ROW_DISPLAYED,
        reporting.ROW_DISPLAYED,
        reporting.ROW_DISPLAYED,
    ]
    assert log_row["omitted_row_count"] == 0
    assert "Low-information scanner observations omitted" not in text


def test_group_represented_rows_are_traceable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    configure_paths(monkeypatch, tmp_path)
    source = sample_final_decisions()
    extra = source.iloc[[3]].copy()
    extra.loc[:, "ticker"] = "EEE"
    extra.loc[:, "input_row_hash"] = "hash-eee"
    source = pd.concat([source, extra], ignore_index=True)
    write_sources(tmp_path, source)

    dashboard, log_row, text = reporting.build_reporting_layer()

    wait_rows = dashboard[dashboard["source_final_action"] == "WAIT"]
    assert wait_rows["display_mode"].tolist() == [
        reporting.ROW_DISPLAYED,
        reporting.ROW_DISPLAYED,
        reporting.ROW_DISPLAYED,
        reporting.GROUP_REPRESENTED,
    ]
    assert log_row["summarized_row_count"] == 1
    assert "Group represented rows: 1" in text
    assert "EEE: action=WAIT" not in text
    assert dashboard[dashboard["ticker"] == "EEE"]["source_row_identity"].notna().all()


def test_forbidden_keyword_and_english_only_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    configure_paths(monkeypatch, tmp_path)
    write_sources(tmp_path, sample_final_decisions())

    _, _, text = reporting.build_reporting_layer()

    assert all(ord(char) < 128 for char in text)
    forbidden_terms = [
        "BUY NOW",
        "urgent",
        "ranked",
        "score",
        "best",
        "top",
        "recommended",
        "priority",
        "tradeable",
        "actionable",
    ]
    for term in forbidden_terms:
        assert term.lower() not in text.lower()


def test_fail_fast_missing_required_columns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    configure_paths(monkeypatch, tmp_path)
    source = sample_final_decisions().drop(columns=["input_row_hash"])
    write_sources(tmp_path, source)

    with pytest.raises(ValueError, match="missing required columns"):
        reporting.build_reporting_layer()


def test_optional_stability_layer_missing_handling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    configure_paths(monkeypatch, tmp_path)
    write_sources(tmp_path, sample_final_decisions())

    dashboard, log_row, _ = reporting.build_reporting_layer()

    assert log_row["stability_status"] == reporting.SOURCE_UNAVAILABLE
    assert set(dashboard["stability_state"]) == {reporting.SOURCE_UNAVAILABLE}


def test_duplicate_stability_identity_fails_fast(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    configure_paths(monkeypatch, tmp_path)
    stability = pd.concat([sample_stability_state(), sample_stability_state()], ignore_index=True)
    write_sources(tmp_path, sample_final_decisions(), stability)

    with pytest.raises(ValueError, match="duplicate ticker/date"):
        reporting.build_reporting_layer()


def test_source_artifact_is_not_mutated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    configure_paths(monkeypatch, tmp_path)
    write_sources(tmp_path, sample_final_decisions())
    source_path = tmp_path / "data/processed/final_decisions.csv"
    before = source_path.read_text(encoding="utf-8")

    _, log_row, _ = reporting.build_reporting_layer()

    after = source_path.read_text(encoding="utf-8")
    assert before == after
    assert log_row["upstream_artifacts_mutated"] == "False"
