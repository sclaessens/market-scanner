import importlib
import sys
from pathlib import Path

from market_scanner.decisions.decision_engine import run_decision_engine
from market_scanner.decisions.decision_records import DecisionState
from market_scanner.orchestration.pipeline_core import (
    run_minimal_pipeline_from_fixtures,
)
from market_scanner.reporting.reporting_engine import build_communication_report


FORBIDDEN_REPORTING_FIELDS = {
    "allocation",
    "allocation_amount",
    "position_size",
    "rank",
    "score",
    "urgency",
    "tradeability",
    "conviction",
    "execution_instruction",
    "priority",
}


def _build_decision_result():
    return run_decision_engine(run_minimal_pipeline_from_fixtures())


def test_reporting_accepts_decision_engine_result():
    decision_result = _build_decision_result()

    report_result = build_communication_report(decision_result)

    assert report_result.input_decision_count == decision_result.output_row_count
    assert report_result.layers_consumed == decision_result.layers_consumed
    assert report_result.fixture_source_names == decision_result.fixture_source_names


def test_reporting_emits_one_record_per_decision_record():
    decision_result = _build_decision_result()

    report_result = build_communication_report(decision_result)

    assert report_result.output_record_count == decision_result.output_row_count
    assert len(report_result.records) == len(decision_result.decision_records)


def test_reporting_preserves_row_identity_and_order():
    decision_result = _build_decision_result()

    report_result = build_communication_report(decision_result)

    assert report_result.preserved_row_ids == decision_result.preserved_row_ids
    assert tuple(record.row_id for record in report_result.records) == (
        decision_result.preserved_row_ids
    )


def test_reporting_preserves_decision_state_and_rationale_exactly():
    decision_result = _build_decision_result()

    report_result = build_communication_report(decision_result)

    for decision_record, report_record in zip(
        decision_result.decision_records,
        report_result.records,
        strict=True,
    ):
        assert report_record.communicated_action is decision_record.final_action
        assert report_record.communicated_rationale == decision_record.rationale


def test_reporting_does_not_filter_suppress_or_add_records():
    decision_result = _build_decision_result()

    report_result = build_communication_report(decision_result)

    assert [record.row_id for record in report_result.records] == [
        record.row_id for record in decision_result.decision_records
    ]
    assert report_result.output_record_count == report_result.input_decision_count


def test_reporting_repeated_runs_are_identical():
    decision_result = _build_decision_result()

    first = build_communication_report(decision_result)
    second = build_communication_report(decision_result)

    assert first == second


def test_reporting_import_and_run_has_no_filesystem_side_effects(
    tmp_path, monkeypatch
):
    before = set(Path(tmp_path).iterdir())
    monkeypatch.chdir(tmp_path)

    importlib.import_module("market_scanner.reporting.reporting_engine")
    build_communication_report(_build_decision_result())

    after = set(Path(tmp_path).iterdir())

    assert after == before


def test_reporting_writes_no_report_files():
    reports_root = Path("reports")
    before = sorted(path.as_posix() for path in reports_root.rglob("*"))

    build_communication_report(_build_decision_result())

    after = sorted(path.as_posix() for path in reports_root.rglob("*"))

    assert after == before


def test_reporting_does_not_import_telegram_or_legacy_scripts():
    watched_modules_before = {
        module_name
        for module_name in sys.modules
        if module_name == "scripts"
        or module_name.startswith("scripts.")
        or "telegram" in module_name.lower()
    }

    importlib.import_module("market_scanner.reporting.reporting_engine")
    build_communication_report(_build_decision_result())

    watched_modules_after = {
        module_name
        for module_name in sys.modules
        if module_name == "scripts"
        or module_name.startswith("scripts.")
        or "telegram" in module_name.lower()
    }

    assert watched_modules_after == watched_modules_before


def test_reporting_does_not_create_buy_sell_hold_states():
    report_result = build_communication_report(_build_decision_result())
    communicated_values = {
        record.communicated_action.value for record in report_result.records
    }

    assert communicated_values == {DecisionState.REVIEW.value}
    assert "BUY" not in communicated_values
    assert "SELL" not in communicated_values
    assert "HOLD" not in communicated_values


def test_reporting_contains_no_allocation_tradeability_or_priority_fields():
    report_result = build_communication_report(_build_decision_result())
    result_fields = set(report_result.__dataclass_fields__)
    record_fields = {
        field
        for record in report_result.records
        for field in record.__dataclass_fields__
    }

    assert result_fields.isdisjoint(FORBIDDEN_REPORTING_FIELDS)
    assert record_fields.isdisjoint(FORBIDDEN_REPORTING_FIELDS)


def test_reporting_summary_lines_are_deterministic_communication_only():
    report_result = build_communication_report(_build_decision_result())

    assert report_result.summary_lines == tuple(
        record.line for record in report_result.records
    )
    for line, record in zip(
        report_result.summary_lines,
        report_result.records,
        strict=True,
    ):
        assert record.row_id in line
        assert record.source_name in line
        assert record.communicated_action.value in line
        assert record.communicated_rationale in line


def test_reporting_keeps_missing_source_data_as_explicit_decision_metadata():
    decision_result = _build_decision_result()
    report_result = build_communication_report(decision_result)
    source_records = [
        record
        for record in report_result.records
        if record.source_name == "synthetic_source_data_readiness"
    ]

    assert source_records
    assert all(
        record.communicated_action is DecisionState.REVIEW
        for record in source_records
    )
    assert all(
        record.communicated_rationale == "review_only_scaffold"
        for record in source_records
    )
