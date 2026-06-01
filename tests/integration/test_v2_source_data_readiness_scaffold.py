import importlib
import sys
from pathlib import Path

from market_scanner.decisions.decision_engine import run_decision_engine
from market_scanner.fundamentals.source_data_readiness import (
    SOURCE_DATA_READINESS_FIXTURE_NAME,
    evaluate_source_data_readiness,
    load_source_data_readiness_fixture,
)
from market_scanner.fundamentals.source_data_records import SourceDataStatus
from market_scanner.orchestration.pipeline_core import (
    run_minimal_pipeline_from_fixtures,
)
from market_scanner.reporting.reporting_engine import build_communication_report


FORBIDDEN_SOURCE_DATA_FIELDS = {
    "final_action",
    "decision_state",
    "allocation",
    "allocation_amount",
    "position_size",
    "rank",
    "quality_score",
    "score",
    "urgency",
    "tradeability",
    "conviction",
    "execution_instruction",
}

NETWORK_MODULE_NAMES = {
    "requests",
    "urllib",
    "httpx",
    "aiohttp",
    "yfinance",
}


def test_source_data_readiness_fixture_loads_successfully():
    rows = load_source_data_readiness_fixture()

    assert len(rows) == 2
    assert {row["source_record_id"] for row in rows} == {"src-001", "src-002"}


def test_source_data_readiness_emits_one_record_per_fixture_row():
    rows = load_source_data_readiness_fixture()

    result = evaluate_source_data_readiness(rows)

    assert result.input_row_count == len(rows)
    assert result.output_row_count == len(rows)
    assert len(result.records) == len(rows)


def test_source_data_readiness_preserves_row_identity_and_traceability():
    rows = load_source_data_readiness_fixture()

    result = evaluate_source_data_readiness(rows)

    assert result.preserved_row_ids == tuple(
        row["source_record_id"] for row in rows
    )
    assert result.provenance_fixture_name == SOURCE_DATA_READINESS_FIXTURE_NAME
    assert {
        record.provenance_fixture_name for record in result.records
    } == {SOURCE_DATA_READINESS_FIXTURE_NAME}


def test_source_data_readiness_repeated_runs_are_deterministic():
    first = evaluate_source_data_readiness()
    second = evaluate_source_data_readiness()

    assert first == second


def test_missing_source_data_values_remain_explicit_and_not_zero():
    result = evaluate_source_data_readiness()
    review_required_records = [
        record
        for record in result.records
        if record.status is SourceDataStatus.REVIEW_REQUIRED
    ]

    assert review_required_records
    for record in review_required_records:
        assert record.metric_value == ""
        assert record.missing_fields == ("metric_value",)
        assert record.missing_value_policy == "missing_not_zero"
        assert record.metric_value != "0"


def test_source_data_readiness_states_do_not_imply_investment_quality():
    result = evaluate_source_data_readiness()

    assert {record.status for record in result.records} == {
        SourceDataStatus.AVAILABLE,
        SourceDataStatus.REVIEW_REQUIRED,
    }
    for record in result.records:
        assert not hasattr(record, "quality_score")
        assert not hasattr(record, "investment_quality")


def test_source_data_readiness_contains_no_decision_or_allocation_fields():
    result = evaluate_source_data_readiness()
    result_fields = set(result.__dataclass_fields__)
    record_fields = {
        field
        for record in result.records
        for field in record.__dataclass_fields__
    }

    assert result_fields.isdisjoint(FORBIDDEN_SOURCE_DATA_FIELDS)
    assert record_fields.isdisjoint(FORBIDDEN_SOURCE_DATA_FIELDS)


def test_no_buy_sell_hold_decision_states_are_implemented():
    readiness_values = {status.value for status in SourceDataStatus}

    assert "BUY" not in readiness_values
    assert "SELL" not in readiness_values
    assert "HOLD" not in readiness_values


def test_source_data_readiness_imports_no_network_or_provider_modules():
    watched_modules_before = {
        module_name
        for module_name in sys.modules
        if module_name.split(".", maxsplit=1)[0] in NETWORK_MODULE_NAMES
    }

    importlib.import_module("market_scanner.fundamentals.source_data_readiness")
    evaluate_source_data_readiness()

    watched_modules_after = {
        module_name
        for module_name in sys.modules
        if module_name.split(".", maxsplit=1)[0] in NETWORK_MODULE_NAMES
    }

    assert watched_modules_after == watched_modules_before


def test_source_data_readiness_does_not_import_legacy_scripts():
    scripts_modules_before = {
        module_name
        for module_name in sys.modules
        if module_name == "scripts" or module_name.startswith("scripts.")
    }

    importlib.import_module("market_scanner.fundamentals.source_data_readiness")
    evaluate_source_data_readiness()

    scripts_modules_after = {
        module_name
        for module_name in sys.modules
        if module_name == "scripts" or module_name.startswith("scripts.")
    }

    assert scripts_modules_after == scripts_modules_before


def test_source_data_readiness_import_and_run_has_no_filesystem_side_effects(
    tmp_path, monkeypatch
):
    before = set(Path(tmp_path).iterdir())
    monkeypatch.chdir(tmp_path)

    importlib.import_module("market_scanner.fundamentals.source_data_readiness")
    evaluate_source_data_readiness()

    after = set(Path(tmp_path).iterdir())

    assert after == before


def test_source_data_readiness_creates_no_generated_outputs():
    watched_roots = [Path("reports"), Path("data/processed"), Path("data/logs")]
    before = {
        root.as_posix(): sorted(path.as_posix() for path in root.rglob("*"))
        for root in watched_roots
    }

    evaluate_source_data_readiness()

    after = {
        root.as_posix(): sorted(path.as_posix() for path in root.rglob("*"))
        for root in watched_roots
    }

    assert after == before


def test_decision_engine_behavior_remains_unchanged():
    pipeline_result = run_minimal_pipeline_from_fixtures()
    before = run_decision_engine(pipeline_result)

    evaluate_source_data_readiness()

    after = run_decision_engine(pipeline_result)

    assert after == before


def test_reporting_behavior_remains_unchanged():
    decision_result = run_decision_engine(run_minimal_pipeline_from_fixtures())
    before = build_communication_report(decision_result)

    evaluate_source_data_readiness()

    after = build_communication_report(decision_result)

    assert after == before
