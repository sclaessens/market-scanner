import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from market_scanner.fundamentals import fundamentals_persistence
from market_scanner.fundamentals.fundamentals_persistence import (
    PersistenceIssueCode,
    prepare_persistence_batch,
    validate_normalized_fundamental_record,
    validate_raw_evidence_record,
    validate_readiness_record,
    write_synthetic_persistence_batch,
)


FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "fundamentals" / "persistence"
)


def _payload(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _raw(name: str) -> dict:
    return _payload(name)["raw_evidence"]


def _normalized(name: str) -> list[dict]:
    return _payload(name)["expected_contract"]["normalized_fundamentals"]["records"]


def _readiness(name: str) -> dict:
    return _payload(name)["expected_contract"]["readiness"]


def _batch(name: str):
    return prepare_persistence_batch(
        (_raw(name),),
        tuple(_normalized(name)),
        (_readiness(name),),
    )


def _issue_codes(issues):
    return {issue.issue_code for issue in issues}


def test_complete_synthetic_records_validate_successfully():
    assert validate_raw_evidence_record(_raw("raw_complete_source.json")) == ()
    for record in _normalized("raw_complete_source.json"):
        assert validate_normalized_fundamental_record(record) == ()
    assert validate_readiness_record(_readiness("raw_complete_source.json")) == ()

    batch = _batch("raw_complete_source.json")

    assert batch.batch_status == "valid"
    assert batch.issues == ()


def test_partial_records_remain_neutral_and_explicit():
    batch = _batch("raw_partial_source.json")
    missing_record = next(
        record
        for record in batch.normalized_records
        if record["metric_value_status"] == "missing_source_field"
    )

    assert batch.batch_status == "valid"
    assert missing_record["metric_value"] is None
    assert missing_record["metric_value_status"] == "missing_source_field"
    assert batch.readiness_records[0]["readiness_state"] == "partial"
    assert batch.readiness_records[0]["missing_fundamentals_count"] == 2


def test_invalid_records_fail_closed_without_silent_normalization():
    batch = _batch("raw_invalid_source.json")

    assert batch.batch_status == "fail_closed"
    assert PersistenceIssueCode.FAIL_CLOSED_SOURCE_DATA_STATE in _issue_codes(
        batch.issues
    )
    assert batch.readiness_records[0]["readiness_state"] == "fail_closed"
    assert batch.readiness_records[0]["invalid_data_count"] == 1
    assert batch.normalized_records[0]["metric_value"] is None
    assert batch.normalized_records[0]["metric_value_status"] == "invalid_unparseable"


def test_provenance_gap_records_fail_closed():
    batch = _batch("raw_provenance_gap_source.json")

    assert batch.batch_status == "fail_closed"
    assert PersistenceIssueCode.MISSING_REQUIRED_VALUE in _issue_codes(batch.issues)
    assert PersistenceIssueCode.MISSING_PROVENANCE in _issue_codes(batch.issues)
    assert batch.readiness_records[0]["readiness_state"] == "fail_closed"
    assert batch.readiness_records[0]["provenance_status"] == "incomplete"


def test_missing_values_are_not_converted_to_zero():
    record = dict(_normalized("raw_partial_source.json")[1])
    record["metric_value"] = "0"

    issues = validate_normalized_fundamental_record(record)

    assert PersistenceIssueCode.MISSING_VALUE_ZERO_SUBSTITUTION in _issue_codes(issues)


def test_raw_missing_values_are_preserved_as_explicit_none():
    raw = _raw("raw_partial_source.json")

    assert raw["raw_fields"]["GrossProfit"] is None
    assert raw["raw_fields"]["FreeCashFlow"] is None
    assert validate_raw_evidence_record(raw) == ()


def test_forbidden_semantics_are_rejected_from_approved_records():
    raw = dict(_raw("raw_complete_source.json"))
    raw["allocation"] = "synthetic forbidden output"
    normalized = dict(_normalized("raw_complete_source.json")[0])
    normalized["target_price"] = "999"
    readiness = dict(_readiness("raw_complete_source.json"))
    readiness["tradeability"] = "synthetic forbidden output"

    raw_issues = validate_raw_evidence_record(raw)
    normalized_issues = validate_normalized_fundamental_record(normalized)
    readiness_issues = validate_readiness_record(readiness)

    assert PersistenceIssueCode.FORBIDDEN_FIELD in _issue_codes(raw_issues)
    assert PersistenceIssueCode.FORBIDDEN_FIELD in _issue_codes(normalized_issues)
    assert PersistenceIssueCode.FORBIDDEN_FIELD in _issue_codes(readiness_issues)


def test_controlled_forbidden_fixture_outputs_are_flagged_but_not_authority():
    batch = _batch("raw_forbidden_semantics_source.json")

    assert batch.batch_status == "fail_closed"
    assert PersistenceIssueCode.FAIL_CLOSED_SOURCE_DATA_STATE in _issue_codes(
        batch.issues
    )
    assert batch.readiness_records[0]["readiness_state"] == "neutral_review_required"
    assert "forbidden_test_input" not in batch.raw_records[0]
    assert all("allocation" not in record for record in batch.normalized_records)


def test_persistence_batch_preserves_record_family_separation():
    batch = _batch("raw_complete_source.json")

    assert batch.raw_records[0]["raw_evidence_id"].startswith("synthetic-raw-")
    assert batch.normalized_records[0]["normalized_record_id"].startswith("norm-")
    assert batch.readiness_records[0]["readiness_record_id"].startswith("readiness-")
    assert "raw_fields" not in batch.normalized_records[0]
    assert "metric_value" not in batch.raw_records[0]
    assert "readiness_state" not in batch.raw_records[0]


def test_synthetic_writes_go_only_to_tmp_path(tmp_path):
    output_root = tmp_path / "persistence"
    batch = _batch("raw_complete_source.json")

    result = write_synthetic_persistence_batch(batch, output_root)

    assert result.batch_status == "written"
    assert result.output_root == str(output_root)
    assert result.raw_record_count == 1
    assert result.normalized_record_count == 2
    assert result.readiness_record_count == 1
    assert (output_root / "raw_source_evidence").is_dir()
    assert (output_root / "normalized_fundamentals").is_dir()
    assert (output_root / "source_data_readiness").is_dir()
    assert len(result.write_records) == 4
    assert all(Path(record.output_path).is_file() for record in result.write_records)


def test_write_result_metadata_is_deterministic(tmp_path):
    batch = _batch("raw_complete_source.json")

    first = write_synthetic_persistence_batch(batch, tmp_path / "first")
    second = write_synthetic_persistence_batch(batch, tmp_path / "first")

    assert first == second
    assert tuple(record.record_family for record in first.write_records) == (
        "raw",
        "normalized",
        "normalized",
        "readiness",
    )


def test_invalid_batch_does_not_write_files(tmp_path):
    batch = _batch("raw_provenance_gap_source.json")

    result = write_synthetic_persistence_batch(batch, tmp_path / "persistence")

    assert result.batch_status == "fail_closed"
    assert result.write_records == ()
    assert result.issues == batch.issues
    assert not (tmp_path / "persistence").exists()


@pytest.mark.parametrize(
    "forbidden_root",
    [
        Path("data"),
        Path("data/raw"),
        Path("data/processed/fundamentals"),
        Path("reports"),
        Path("reports/daily"),
        Path("reports/daily/telegram_message.txt"),
        Path(".github/workflows"),
    ],
)
def test_production_report_telegram_and_workflow_paths_are_rejected(forbidden_root):
    batch = _batch("raw_complete_source.json")

    result = write_synthetic_persistence_batch(batch, forbidden_root)

    assert result.batch_status == "fail_closed"
    assert result.write_records == ()
    assert PersistenceIssueCode.FORBIDDEN_OUTPUT_PATH in _issue_codes(result.issues)


def test_persistence_module_has_no_provider_network_pipeline_or_decision_imports():
    source = Path(fundamentals_persistence.__file__).read_text(encoding="utf-8")

    for forbidden in (
        "fundamentals_provider_adapter",
        "fundamentals_real_source_smoke",
        "decision_engine",
        "requests",
        "urllib",
        "httpx",
        "aiohttp",
        "yfinance",
        "telegram",
        "run_full_pipeline",
    ):
        assert forbidden not in source


def test_persistence_records_are_immutable_metadata():
    batch = _batch("raw_complete_source.json")

    with pytest.raises(FrozenInstanceError):
        batch.batch_status = "changed"
