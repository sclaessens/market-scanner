import importlib
import sys
from pathlib import Path

from market_scanner.orchestration.pipeline_core import (
    MINIMAL_PIPELINE_LAYER_ORDER,
    load_fixture_records,
    run_minimal_pipeline,
    run_minimal_pipeline_from_fixtures,
)


FORBIDDEN_RESULT_FIELDS = {
    "final_action",
    "action",
    "decision",
    "tradeability",
    "conviction",
    "rank",
    "score",
    "urgency",
    "allocation",
}


def test_minimal_pipeline_loads_approved_fixture_records():
    records = load_fixture_records()

    assert len(records) == 6
    assert {record.source_name for record in records} == {
        "synthetic_universe_candidates",
        "synthetic_portfolio_transactions",
        "synthetic_source_data_readiness",
    }


def test_minimal_pipeline_preserves_row_count_and_identity():
    records = load_fixture_records()
    input_ids = tuple(record.row_id for record in records)

    result = run_minimal_pipeline(records)

    assert result.input_row_count == len(records)
    assert result.output_row_count == len(records)
    assert result.preserved_row_ids == input_ids
    assert tuple(record.row_id for record in result.records) == input_ids


def test_minimal_pipeline_layer_order_is_deterministic():
    result = run_minimal_pipeline_from_fixtures()

    assert result.layers_visited == MINIMAL_PIPELINE_LAYER_ORDER
    assert tuple(stage.stage_name for stage in result.stage_results) == (
        MINIMAL_PIPELINE_LAYER_ORDER
    )


def test_minimal_pipeline_repeated_runs_are_identical():
    first = run_minimal_pipeline_from_fixtures()
    second = run_minimal_pipeline_from_fixtures()

    assert first == second


def test_minimal_pipeline_import_and_run_has_no_filesystem_side_effects(
    tmp_path, monkeypatch
):
    before = set(Path(tmp_path).iterdir())
    monkeypatch.chdir(tmp_path)

    importlib.import_module("market_scanner.orchestration.pipeline_core")
    run_minimal_pipeline_from_fixtures()

    after = set(Path(tmp_path).iterdir())

    assert after == before


def test_minimal_pipeline_result_has_no_final_action_or_allocation_fields():
    result = run_minimal_pipeline_from_fixtures()
    result_fields = set(result.__dataclass_fields__)
    stage_fields = {
        field
        for stage in result.stage_results
        for field in stage.__dataclass_fields__
    }
    record_fields = {
        field for record in result.records for field in record.__dataclass_fields__
    }

    assert result_fields.isdisjoint(FORBIDDEN_RESULT_FIELDS)
    assert stage_fields.isdisjoint(FORBIDDEN_RESULT_FIELDS)
    assert record_fields.isdisjoint(FORBIDDEN_RESULT_FIELDS)


def test_source_data_readiness_remains_metadata_with_explicit_missing_values():
    result = run_minimal_pipeline_from_fixtures()
    source_records = [
        record
        for record in result.records
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
        assert record.values["review_required_reason"]


def test_v2_pipeline_core_does_not_import_legacy_scripts():
    scripts_modules_before = {
        module_name
        for module_name in sys.modules
        if module_name == "scripts" or module_name.startswith("scripts.")
    }

    importlib.import_module("market_scanner.orchestration.pipeline_core")

    scripts_modules_after = {
        module_name
        for module_name in sys.modules
        if module_name == "scripts" or module_name.startswith("scripts.")
    }

    assert scripts_modules_after == scripts_modules_before
