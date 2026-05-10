from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.core import build_stability_layer as stability_module

REQUIRED_OUTPUT_COLUMNS = stability_module.OUTPUT_COLUMNS
REQUIRED_LOG_COLUMNS = stability_module.LOG_COLUMNS

FORBIDDEN_FIELDS = {
    "suppression_flag",
    "hard_block",
    "cooldown_lock",
    "allocation_gate",
    "execution_gate",
    "hidden_filter",
    "action_override",
    "final_action_override",
    "remove_opportunity",
}

FORBIDDEN_SOURCE_TERMS = [
    "BUY NOW",
    "tradeable",
    "invalid",
]


@pytest.fixture()
def patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    processed_dir = tmp_path / "data" / "processed"
    logs_dir = tmp_path / "data" / "logs"
    processed_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    input_path = processed_dir / "final_decisions.csv"
    output_path = processed_dir / "stability_state.csv"
    log_path = logs_dir / "stability_layer_log.csv"

    monkeypatch.setattr(stability_module, "INPUT_PATH", input_path)
    monkeypatch.setattr(stability_module, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(stability_module, "LOG_PATH", log_path)

    return input_path, output_path, log_path


def _decision_row(ticker: str, date: str, action: str = "WAIT", **overrides) -> dict:
    row = {
        "ticker": ticker,
        "date": date,
        "final_action": action,
        "allocation_decision": "NO_ALLOCATION_ACTION",
        "execution_decision": "MONITOR",
        "decision_contract_version": "SPRINT_6_DECISION_ENGINE_CORE_V1",
        "input_row_hash": f"{ticker}-{date}",
    }
    row.update(overrides)
    return row


def _write_input(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_with_rows(patch_paths, rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_path, output_path, log_path = patch_paths
    _write_input(input_path, rows)
    df = stability_module.build_stability_layer()
    written_df = pd.read_csv(output_path)
    log_df = pd.read_csv(log_path)
    pd.testing.assert_frame_equal(df, written_df)
    return df, log_df


def test_output_schema_contract(patch_paths):
    df, log_df = _build_with_rows(
        patch_paths,
        [_decision_row("AAA", "2026-05-08")],
    )

    assert list(df.columns) == REQUIRED_OUTPUT_COLUMNS
    assert list(log_df.columns) == REQUIRED_LOG_COLUMNS
    assert not (set(df.columns) & FORBIDDEN_FIELDS)


def test_required_input_schema_contract(patch_paths):
    input_path, _, _ = patch_paths
    row = _decision_row("AAA", "2026-05-08")
    row.pop("final_action")
    _write_input(input_path, [row])

    with pytest.raises(ValueError, match="missing required columns"):
        stability_module.build_stability_layer()


def test_deterministic_output_under_identical_inputs(patch_paths):
    rows = [
        _decision_row("AAA", "2026-05-08", "WAIT"),
        _decision_row("AAA", "2026-05-09", "WAIT"),
        _decision_row("AAA", "2026-05-10", "PREPARE"),
    ]
    first_df, _ = _build_with_rows(patch_paths, rows)
    second_df = stability_module.build_stability_layer()

    pd.testing.assert_frame_equal(first_df, second_df)


def test_no_hidden_filtering_fields_or_row_loss(patch_paths):
    rows = [
        _decision_row("AAA", "2026-05-08"),
        _decision_row("BBB", "2026-05-08"),
    ]
    df, log_df = _build_with_rows(patch_paths, rows)

    assert len(df) == len(rows)
    assert set(df["ticker"]) == {"AAA", "BBB"}
    assert not (set(df.columns) & FORBIDDEN_FIELDS)
    assert log_df.loc[0, "row_count_preserved"] in {True, "True", "true"}
    assert log_df.loc[0, "ticker_date_universe_preserved"] in {True, "True", "true"}


def test_persistence_duration_correctness(patch_paths):
    rows = [
        _decision_row("AAA", "2026-05-08", "WAIT"),
        _decision_row("AAA", "2026-05-09", "WAIT"),
        _decision_row("AAA", "2026-05-10", "WAIT"),
        _decision_row("AAA", "2026-05-11", "PREPARE"),
    ]
    df, _ = _build_with_rows(patch_paths, rows)

    assert df["persistence_duration"].tolist() == [1, 2, 3, 1]
    assert df["action_persistence"].tolist() == ["UNCHANGED", "UNCHANGED", "PERSISTENT", "CHANGED"]


def test_transition_frequency_correctness(patch_paths):
    rows = [
        _decision_row("AAA", "2026-05-08", "WAIT"),
        _decision_row("AAA", "2026-05-09", "PREPARE"),
        _decision_row("AAA", "2026-05-10", "WAIT"),
        _decision_row("AAA", "2026-05-11", "REVIEW"),
    ]
    df, _ = _build_with_rows(patch_paths, rows)

    assert df["transition_frequency"].tolist() == [0, 1, 2, 3]
    assert df.loc[3, "behavioural_stability"] == "NOISY"


def test_escalation_state_behaviour(patch_paths):
    rows = [
        _decision_row("AAA", "2026-05-08", "WAIT"),
        _decision_row("AAA", "2026-05-09", "PREPARE"),
        _decision_row("AAA", "2026-05-10", "PREPARE"),
    ]
    df, _ = _build_with_rows(patch_paths, rows)

    assert df["escalation_frequency"].tolist() == [0, 1, 1]
    assert df.loc[1, "stability_state"] == "ESCALATING"


def test_missing_input_handling_writes_empty_outputs(patch_paths):
    _, output_path, log_path = patch_paths

    df = stability_module.build_stability_layer()
    written_df = pd.read_csv(output_path)
    log_df = pd.read_csv(log_path)

    assert df.empty
    assert list(written_df.columns) == REQUIRED_OUTPUT_COLUMNS
    assert log_df.loc[0, "input_status"] == "SOURCE_MISSING"
    assert log_df.loc[0, "output_row_count"] == 0


def test_empty_input_handling_writes_empty_outputs(patch_paths):
    input_path, output_path, log_path = patch_paths
    pd.DataFrame(columns=stability_module.REQUIRED_INPUT_COLUMNS).to_csv(input_path, index=False)

    df = stability_module.build_stability_layer()
    written_df = pd.read_csv(output_path)
    log_df = pd.read_csv(log_path)

    assert df.empty
    assert list(written_df.columns) == REQUIRED_OUTPUT_COLUMNS
    assert log_df.loc[0, "input_status"] == "SOURCE_EMPTY"


def test_conviction_persistence_when_source_exists(patch_paths):
    rows = [
        _decision_row("AAA", "2026-05-08", conviction="HIGH"),
        _decision_row("AAA", "2026-05-09", conviction="HIGH"),
        _decision_row("AAA", "2026-05-10", conviction="LOW"),
    ]
    df, _ = _build_with_rows(patch_paths, rows)

    assert df["conviction_persistence"].tolist() == ["UNCHANGED", "UNCHANGED", "CHANGED"]


def test_source_avoids_blocked_terms():
    source = Path(stability_module.__file__).read_text()

    for term in FORBIDDEN_SOURCE_TERMS:
        assert term not in source


def test_new_stability_artifacts_remain_ascii_text():
    artifact_paths = [
        Path(stability_module.__file__),
        Path(__file__),
    ]

    for path in artifact_paths:
        assert path.read_text().isascii()
