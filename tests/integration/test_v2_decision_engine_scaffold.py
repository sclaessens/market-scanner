import importlib
import sys
from pathlib import Path

from market_scanner.decisions.decision_engine import (
    REVIEW_ONLY_RATIONALE,
    run_decision_engine,
)
from market_scanner.decisions.decision_records import DecisionState
from market_scanner.orchestration.pipeline_core import (
    run_minimal_pipeline_from_fixtures,
)
from market_scanner.shared.records import PipelineRecord, PipelineResult


FORBIDDEN_UPSTREAM_FIELDS = {
    "final_action",
    "decision_state",
    "decision",
    "allocation",
    "allocation_amount",
    "position_size",
    "rank",
    "score",
    "urgency",
    "tradeability",
    "conviction",
    "execution_instruction",
}

FORBIDDEN_DECISION_FIELDS = {
    "allocation",
    "allocation_amount",
    "position_size",
    "rank",
    "score",
    "urgency",
    "tradeability",
    "conviction",
    "execution_instruction",
}


def test_decision_engine_accepts_minimal_pipeline_result():
    pipeline_result = run_minimal_pipeline_from_fixtures()

    result = run_decision_engine(pipeline_result)

    assert result.input_row_count == pipeline_result.output_row_count
    assert result.layers_consumed == pipeline_result.layers_visited
    assert result.fixture_source_names == pipeline_result.fixture_source_names


def test_decision_engine_emits_one_review_record_per_pipeline_row():
    pipeline_result = run_minimal_pipeline_from_fixtures()

    result = run_decision_engine(pipeline_result)

    assert result.output_row_count == pipeline_result.output_row_count
    assert len(result.decision_records) == pipeline_result.output_row_count
    assert {record.final_action for record in result.decision_records} == {
        DecisionState.REVIEW
    }
    assert {record.rationale for record in result.decision_records} == {
        REVIEW_ONLY_RATIONALE
    }


def test_decision_engine_preserves_row_identity():
    pipeline_result = run_minimal_pipeline_from_fixtures()

    result = run_decision_engine(pipeline_result)

    assert result.preserved_row_ids == pipeline_result.preserved_row_ids
    assert tuple(record.row_id for record in result.decision_records) == (
        pipeline_result.preserved_row_ids
    )


def test_decision_engine_repeated_runs_are_identical():
    pipeline_result = run_minimal_pipeline_from_fixtures()

    first = run_decision_engine(pipeline_result)
    second = run_decision_engine(pipeline_result)

    assert first == second


def test_no_buy_sell_hold_states_are_implemented():
    state_values = {state.value for state in DecisionState}

    assert state_values == {"REVIEW"}
    assert "BUY" not in state_values
    assert "SELL" not in state_values
    assert "HOLD" not in state_values


def test_decision_engine_output_has_no_sizing_or_execution_fields():
    pipeline_result = run_minimal_pipeline_from_fixtures()
    result = run_decision_engine(pipeline_result)
    result_fields = set(result.__dataclass_fields__)
    record_fields = {
        field
        for record in result.decision_records
        for field in record.__dataclass_fields__
    }

    assert result_fields.isdisjoint(FORBIDDEN_DECISION_FIELDS)
    assert record_fields.isdisjoint(FORBIDDEN_DECISION_FIELDS)


def test_upstream_pipeline_types_do_not_contain_decision_authority_fields():
    assert set(PipelineRecord.__dataclass_fields__).isdisjoint(
        FORBIDDEN_UPSTREAM_FIELDS
    )
    assert set(PipelineResult.__dataclass_fields__).isdisjoint(
        FORBIDDEN_UPSTREAM_FIELDS
    )


def test_reporting_package_is_not_imported_or_used_by_decision_engine():
    reporting_modules_before = {
        module_name
        for module_name in sys.modules
        if module_name == "market_scanner.reporting"
        or module_name.startswith("market_scanner.reporting.")
    }

    importlib.import_module("market_scanner.decisions.decision_engine")
    run_decision_engine(run_minimal_pipeline_from_fixtures())

    reporting_modules_after = {
        module_name
        for module_name in sys.modules
        if module_name == "market_scanner.reporting"
        or module_name.startswith("market_scanner.reporting.")
    }

    assert reporting_modules_after == reporting_modules_before


def test_decision_engine_import_and_run_has_no_filesystem_side_effects(
    tmp_path, monkeypatch
):
    before = set(Path(tmp_path).iterdir())
    monkeypatch.chdir(tmp_path)

    importlib.import_module("market_scanner.decisions.decision_engine")
    run_decision_engine(run_minimal_pipeline_from_fixtures())

    after = set(Path(tmp_path).iterdir())

    assert after == before


def test_decision_engine_does_not_import_legacy_scripts():
    scripts_modules_before = {
        module_name
        for module_name in sys.modules
        if module_name == "scripts" or module_name.startswith("scripts.")
    }

    importlib.import_module("market_scanner.decisions.decision_engine")
    run_decision_engine(run_minimal_pipeline_from_fixtures())

    scripts_modules_after = {
        module_name
        for module_name in sys.modules
        if module_name == "scripts" or module_name.startswith("scripts.")
    }

    assert scripts_modules_after == scripts_modules_before


def test_missing_source_data_values_remain_explicit_after_decision_engine():
    pipeline_result = run_minimal_pipeline_from_fixtures()
    run_decision_engine(pipeline_result)
    source_records = [
        record
        for record in pipeline_result.records
        if record.source_name == "synthetic_source_data_readiness"
    ]
    review_required_records = [
        record
        for record in source_records
        if record.values["readiness_state"] == "review_required"
    ]

    assert review_required_records
    for record in review_required_records:
        assert record.values["metric_value"] == ""
        assert record.values["missing_value_policy"] == "missing_not_zero"
