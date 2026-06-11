from __future__ import annotations

from pathlib import Path

LEGACY_STABILITY_LAYER_MODULE_PATH = Path("scripts/core/build_stability_layer.py")

REQUIRED_INPUT_COLUMNS = [
    "ticker",
    "date",
    "final_action",
    "allocation_decision",
    "execution_decision",
    "decision_contract_version",
    "input_row_hash",
]

STABILITY_OUTPUT_COLUMNS = [
    "ticker",
    "date",
    "final_action",
    "allocation_decision",
    "execution_decision",
    "decision_contract_version",
    "input_row_hash",
    "previous_final_action",
    "action_persistence",
    "persistence_duration",
    "transition_frequency",
    "escalation_frequency",
    "conviction_persistence",
    "behavioural_stability",
    "stability_state",
    "stability_reason",
]

STABILITY_LOG_COLUMNS = [
    "generated_at",
    "input_status",
    "input_row_count",
    "output_row_count",
    "row_count_preserved",
    "ticker_date_universe_preserved",
    "stability_state_distribution",
    "behavioural_stability_distribution",
]

FORBIDDEN_STABILITY_FIELDS = {
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


def test_stability_layer_script_remains_legacy_reference_only():
    assert LEGACY_STABILITY_LAYER_MODULE_PATH == Path("scripts/core/build_stability_layer.py")


def test_stability_contract_preserves_decision_identity_fields():
    for column in REQUIRED_INPUT_COLUMNS:
        assert column in STABILITY_OUTPUT_COLUMNS
    assert "input_row_hash" in STABILITY_OUTPUT_COLUMNS


def test_stability_contract_appends_observation_only_fields():
    expected_observation_fields = {
        "previous_final_action",
        "action_persistence",
        "persistence_duration",
        "transition_frequency",
        "escalation_frequency",
        "conviction_persistence",
        "behavioural_stability",
        "stability_state",
        "stability_reason",
    }
    assert expected_observation_fields.issubset(STABILITY_OUTPUT_COLUMNS)
    assert set(STABILITY_OUTPUT_COLUMNS).isdisjoint(FORBIDDEN_STABILITY_FIELDS)


def test_stability_log_contract_tracks_preservation_not_authority():
    assert "row_count_preserved" in STABILITY_LOG_COLUMNS
    assert "ticker_date_universe_preserved" in STABILITY_LOG_COLUMNS
    assert "stability_state_distribution" in STABILITY_LOG_COLUMNS


def test_stability_contract_has_no_hidden_execution_gate_fields():
    for field in FORBIDDEN_STABILITY_FIELDS:
        assert field not in STABILITY_OUTPUT_COLUMNS
        assert field not in STABILITY_LOG_COLUMNS


def test_active_code_no_longer_imports_stability_layer_script():
    for path in [Path("src"), Path("tests"), Path(".github")]:
        if not path.exists():
            continue
        for source_path in path.rglob("*.py"):
            if "__pycache__" in source_path.parts:
                continue
            if source_path == Path("tests/core/test_build_stability_layer.py"):
                continue
            source = source_path.read_text(encoding="utf-8")
            assert "from scripts.core import build_stability_layer" not in source
            assert "import scripts.core.build_stability_layer" not in source
