from __future__ import annotations

import copy
import io
import json
from datetime import date
from pathlib import Path

import pytest

from market_engine.data import operator_fundamental_metric_package as package
from market_engine.data.operator_fundamental_metric_package_command import run_command
from market_engine.data.validated_fundamental_metric_sourcing import _load_and_validate_operator_import


def _record(metric: str = "revenue_growth_yoy", value: float = 12.5) -> dict[str, object]:
    return {
        "ticker": "AAA",
        "company_identity": {"name": "AAA Corporation", "instrument_id": "equity:aaa"},
        "canonical_metric": metric,
        "value": value,
        "unit": "percent",
        "currency": None,
        "period_type": "quarter",
        "period_start": "2026-04-01",
        "period_end": "2026-06-30",
        "fiscal_year": 2026,
        "fiscal_period": "Q2",
        "provenance": {
            "source_name": "operator-approved-primary-source",
            "source_reference": "local-evidence://AAA/2026-Q2/revenue",
            "raw_source_field": "reportedRevenueGrowthYoY",
            "source_date": "2026-07-01",
            "observed_at": "2026-07-01T09:00:00Z",
            "acquired_at": "2026-07-02T10:00:00Z",
            "parser_version": "operator-parser-v1",
        },
    }


def _payload(records: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "schema_version": package.INPUT_SCHEMA_VERSION,
        "package_id": "fixture-package-2026-q2",
        "created_at": "2026-07-02T10:00:00Z",
        "records": records if records is not None else [_record()],
    }


def _validate(payload: object):
    return package.validate_and_normalize_operator_input(payload, input_sha256="a" * 64)


def _reason_codes(report: dict[str, object]) -> set[str]:
    return {item["reason_code"] for item in report["errors"]}


def test_happy_path_is_data07_compatible_deterministic_and_canonically_ordered() -> None:
    records = [_record("operating_margin", 17.0), _record("revenue_growth_yoy", 12.5)]
    records[0]["provenance"] = {**records[0]["provenance"], "source_reference": "local-evidence://AAA/2026-Q2/common"}
    records[1]["provenance"] = {**records[1]["provenance"], "source_reference": "local-evidence://AAA/2026-Q2/common"}
    first, report = _validate(_payload(records))
    second, second_report = _validate(_payload(list(reversed(records))))

    assert first == second
    assert report["status"] == second_report["status"] == "accepted"
    assert first["schema_version"] == "market-engine-data07-operator-fundamental-metrics-v1"
    assert list(first["records"][0]["metrics"]) == ["revenue_growth_yoy", "operating_margin"]
    assert first["records"][0]["metrics"]["revenue_growth_yoy"]["value"] == 12.5
    assert report["counts"] == {
        "input_metrics": 2,
        "accepted_metrics": 2,
        "normalized_metrics": 2,
        "warning_count": 2,
        "rejected_metrics": 0,
        "error_count": 0,
    }
    assert report["downstream_consumability"] == "eligible_for_explicit_me_data07_operator_import"


def test_prepare_writes_stable_artifacts_without_mutating_input(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    original = json.dumps(_payload(), indent=4)
    source.write_text(original, encoding="utf-8")
    first_package = tmp_path / "first" / "package.json"
    first_report = tmp_path / "first" / "report.json"
    second_package = tmp_path / "second" / "package.json"
    second_report = tmp_path / "second" / "report.json"

    package.prepare_operator_fundamental_metric_package(source, package_output_path=first_package, report_output_path=first_report)
    package.prepare_operator_fundamental_metric_package(source, package_output_path=second_package, report_output_path=second_report)

    assert source.read_text(encoding="utf-8") == original
    assert first_package.read_bytes() == second_package.read_bytes()
    first_validation = json.loads(first_report.read_text(encoding="utf-8"))
    second_validation = json.loads(second_report.read_text(encoding="utf-8"))
    first_validation["artifacts"] = second_validation["artifacts"]
    assert first_validation == second_validation


def test_synthetic_fixture_prepares_and_passes_existing_data07_validator(tmp_path: Path) -> None:
    source = Path("tests/fixtures/market_engine/data/me_data08_operator_fundamental_metric_input.json")
    accepted = tmp_path / "accepted.json"
    output, report = package.prepare_operator_fundamental_metric_package(
        source,
        package_output_path=accepted,
        report_output_path=tmp_path / "report.json",
    )
    records, data07_report = _load_and_validate_operator_import(
        accepted,
        mappings={"AAA": {"mapping_status": "mapped", "provider_symbol": "AAA"}},
        instruments={"AAA": {"instrument_id": "equity:aaa"}},
        as_of=date(2026, 7, 19),
    )
    assert output is not None and report["status"] == "accepted"
    assert data07_report["validation_status"] == "passed"
    assert records[0]["metrics"]["revenue_growth_yoy"] == 0.125


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda value: value.update(schema_version="unknown-v9"), "UNSUPPORTED_SCHEMA_VERSION"),
        (lambda value: value.pop("package_id"), "REQUIRED_FIELD_MISSING"),
        (lambda value: value.update(records=[]), "EMPTY_METRIC_SET"),
        (lambda value: value["records"][0].update(canonical_metric="unknown_metric"), "METRIC_NOT_ALLOWLISTED"),
        (lambda value: value["records"][0].update(canonical_metric="margin"), "AMBIGUOUS_METRIC_ALIAS"),
        (lambda value: value["records"][0].pop("provenance"), "PROVENANCE_INCOMPLETE"),
        (lambda value: value["records"][0]["provenance"].update(acquired_at="yesterday"), "INVALID_TIMESTAMP"),
        (lambda value: value["records"][0].update(period_end="2026/06/30"), "INVALID_REPORTING_PERIOD"),
        (lambda value: value["records"][0].update(period_type="instant"), "INCOMPATIBLE_PERIOD_TYPE"),
        (lambda value: value["records"][0].update(unit="USD"), "INVALID_UNIT"),
        (lambda value: value["records"][0].update(currency="USD"), "CURRENCY_NOT_APPLICABLE"),
        (lambda value: value["records"][0]["company_identity"].update(instrument_id="equity:bbb"), "COMPANY_TICKER_MISMATCH"),
        (lambda value: value["records"][0].update(value="12.5"), "VALUE_NOT_NUMERIC"),
        (lambda value: value["records"][0].update(scale="millions"), "AMBIGUOUS_SCALE"),
        (lambda value: value["records"][0].update(recommendation="buy"), "AUTHORITY_FIELD_FORBIDDEN"),
        (lambda value: value["records"][0].update(uncontracted_note="guess"), "UNKNOWN_FIELD"),
    ],
)
def test_fail_closed_categories(mutate, reason: str) -> None:
    payload = _payload()
    mutate(payload)
    output, report = _validate(payload)

    assert output is None
    assert report["status"] == "rejected"
    assert report["downstream_consumability"] == "not_consumable"
    assert reason in _reason_codes(report)
    assert report["counts"]["accepted_metrics"] == 0


@pytest.mark.parametrize(("different", "reason"), [(False, "DUPLICATE_METRIC_RECORD"), (True, "DUPLICATE_METRIC_CONFLICT")])
def test_duplicate_and_conflicting_duplicate_fail_closed(different: bool, reason: str) -> None:
    first = _record()
    second = copy.deepcopy(first)
    if different:
        second["value"] = 13.0
    output, report = _validate(_payload([first, second]))

    assert output is None
    assert reason in _reason_codes(report)


def test_partial_validity_rejects_whole_package() -> None:
    valid = _record()
    invalid = _record("gross_margin")
    invalid["provenance"] = {}
    output, report = _validate(_payload([valid, invalid]))
    assert output is None
    assert report["counts"]["rejected_metrics"] == 2


def test_malformed_json_has_operator_error_and_no_traceback(tmp_path: Path) -> None:
    source = tmp_path / "bad.json"
    source.write_text('{"records": [NaN]}', encoding="utf-8")
    stdout, stderr = io.StringIO(), io.StringIO()
    code = run_command(["--input", str(source), "--package-output", str(tmp_path / "package.json"), "--report-output", str(tmp_path / "report.json")], stdout=stdout, stderr=stderr)
    assert code == 2
    assert "ME-DATA08 input error" in stderr.getvalue()
    assert "Traceback" not in stderr.getvalue()
    assert not (tmp_path / "package.json").exists()


def test_cli_exit_codes_and_rejection_never_write_accepted_package(tmp_path: Path) -> None:
    valid = tmp_path / "valid.json"
    invalid = tmp_path / "invalid.json"
    valid.write_text(json.dumps(_payload()), encoding="utf-8")
    rejected = _payload()
    rejected["records"][0]["unit"] = "USD"
    invalid.write_text(json.dumps(rejected), encoding="utf-8")

    valid_out, valid_err = io.StringIO(), io.StringIO()
    assert run_command(["--input", str(valid), "--package-output", str(tmp_path / "accepted.json"), "--report-output", str(tmp_path / "accepted-report.json")], stdout=valid_out, stderr=valid_err) == 0
    assert json.loads(valid_out.getvalue())["status"] == "accepted"

    invalid_out, invalid_err = io.StringIO(), io.StringIO()
    assert run_command(["--input", str(invalid), "--package-output", str(tmp_path / "must-not-exist.json"), "--report-output", str(tmp_path / "rejected-report.json")], stdout=invalid_out, stderr=invalid_err) == 1
    assert json.loads(invalid_out.getvalue())["status"] == "rejected"
    assert not (tmp_path / "must-not-exist.json").exists()


def test_json_nan_and_infinity_are_rejected_before_validation() -> None:
    for constant in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(ValueError, match="forbidden JSON numeric constant"):
            json.loads(f'{{"value": {constant}}}', parse_constant=package._reject_json_constant)
