from importlib import reload
from pathlib import Path

from market_scanner.fundamentals import fundamentals_normalization_contracts
from market_scanner.fundamentals.fundamentals_normalization_contracts import (
    FundamentalsNormalizationDatasetRole,
    FundamentalsNormalizationIssue,
    FundamentalsNormalizationIssueCode,
    forbidden_normalized_fundamentals_fields,
    generated_fundamentals_dataset_roles,
    normalized_fundamentals_dataset_roles,
    raw_source_dataset_roles,
    reporting_display_dataset_roles,
    required_normalized_fundamentals_fields,
    required_raw_source_fields,
    required_source_readiness_fields,
    validate_normalized_fundamentals_shape,
    validate_raw_source_shape,
    validate_source_readiness_shape,
)


def _raw_source_record(**overrides):
    record = {
        "source_provider": "synthetic_provider",
        "source_record_id": "raw-001",
        "ticker": "ASML",
        "fiscal_period": "FY",
        "fiscal_year": "2025",
        "captured_at": "2026-06-03T00:00:00Z",
        "source_reference": "synthetic/raw/asml/2025",
        "raw_payload_hash": "sha256:synthetic",
    }
    record.update(overrides)
    return record


def _normalized_record(**overrides):
    record = {
        "ticker": "ASML",
        "fiscal_period": "FY",
        "fiscal_year": "2025",
        "metric_name": "revenue",
        "metric_value": "123.45",
        "metric_unit": "EUR million",
        "currency": "EUR",
        "normalized_at": "2026-06-03T00:00:00Z",
        "source_provider": "synthetic_provider",
        "source_reference": "synthetic/raw/asml/2025",
        "source_record_identity": "raw-001",
    }
    record.update(overrides)
    return record


def _readiness_record(**overrides):
    record = {
        "ticker": "ASML",
        "fiscal_period": "FY",
        "readiness_state": "partial",
        "source_data_status": "partial_data",
        "missing_fundamentals_count": "1",
        "partial_data_count": "2",
        "stale_data_count": "0",
        "source_reference": "synthetic/readiness/asml",
    }
    record.update(overrides)
    return record


def test_raw_source_capture_role_is_separate_from_normalized_input():
    assert raw_source_dataset_roles() == (
        FundamentalsNormalizationDatasetRole.RAW_SOURCE_CAPTURE,
    )
    assert FundamentalsNormalizationDatasetRole.NORMALIZED_FUNDAMENTALS_INPUT not in (
        raw_source_dataset_roles()
    )


def test_normalized_input_is_separate_from_generated_outputs():
    assert normalized_fundamentals_dataset_roles() == (
        FundamentalsNormalizationDatasetRole.NORMALIZED_FUNDAMENTALS_INPUT,
        FundamentalsNormalizationDatasetRole.SOURCE_DATA_READINESS,
    )
    assert generated_fundamentals_dataset_roles() == (
        FundamentalsNormalizationDatasetRole.GENERATED_FUNDAMENTAL_QUALITY,
        FundamentalsNormalizationDatasetRole.GENERATED_FUNDAMENTAL_ANALYSIS,
    )
    assert set(normalized_fundamentals_dataset_roles()).isdisjoint(
        generated_fundamentals_dataset_roles()
    )


def test_source_data_readiness_role_is_not_investment_quality():
    assert FundamentalsNormalizationDatasetRole.SOURCE_DATA_READINESS in (
        normalized_fundamentals_dataset_roles()
    )
    assert "investment" not in FundamentalsNormalizationDatasetRole.SOURCE_DATA_READINESS


def test_generated_and_reporting_roles_are_not_source_of_truth():
    assert FundamentalsNormalizationDatasetRole.RAW_SOURCE_CAPTURE not in (
        generated_fundamentals_dataset_roles()
    )
    assert reporting_display_dataset_roles() == (
        FundamentalsNormalizationDatasetRole.REPORTING_DISPLAY_INPUT,
    )
    assert FundamentalsNormalizationDatasetRole.RAW_SOURCE_CAPTURE not in (
        reporting_display_dataset_roles()
    )


def test_required_raw_source_fields_include_provenance():
    assert required_raw_source_fields() == (
        "source_provider",
        "source_record_id",
        "ticker",
        "fiscal_period",
        "fiscal_year",
        "captured_at",
        "source_reference",
        "raw_payload_hash",
    )
    assert validate_raw_source_shape(_raw_source_record()) == ()


def test_required_normalized_fields_include_source_traceability():
    assert {"source_provider", "source_reference", "source_record_identity"}.issubset(
        required_normalized_fundamentals_fields()
    )
    assert validate_normalized_fundamentals_shape(_normalized_record()) == ()


def test_required_source_readiness_fields_include_status_counts_and_traceability():
    assert required_source_readiness_fields() == (
        "ticker",
        "fiscal_period",
        "readiness_state",
        "source_data_status",
        "missing_fundamentals_count",
        "partial_data_count",
        "stale_data_count",
        "source_reference",
    )
    assert validate_source_readiness_shape(_readiness_record()) == ()


def test_missing_raw_source_fields_are_reported_explicitly():
    record = _raw_source_record()
    record.pop("source_record_id")

    assert validate_raw_source_shape(record) == (
        FundamentalsNormalizationIssue(
            field_name="source_record_id",
            issue_code=FundamentalsNormalizationIssueCode.MISSING_REQUIRED_FIELD,
            observed_value=None,
        ),
    )


def test_missing_metric_values_are_not_converted_to_zero():
    assert validate_normalized_fundamentals_shape(
        _normalized_record(metric_value="")
    ) == (
        FundamentalsNormalizationIssue(
            field_name="metric_value",
            issue_code=FundamentalsNormalizationIssueCode.MISSING_REQUIRED_VALUE,
            observed_value="",
        ),
    )


def test_missing_count_fields_are_not_converted_to_zero():
    assert validate_source_readiness_shape(
        _readiness_record(missing_fundamentals_count="")
    ) == (
        FundamentalsNormalizationIssue(
            field_name="missing_fundamentals_count",
            issue_code=FundamentalsNormalizationIssueCode.MISSING_REQUIRED_VALUE,
            observed_value="",
        ),
    )


def test_partial_stale_source_missing_unavailable_and_invalid_states_are_explicit():
    for readiness_state in (
        "partial",
        "stale",
        "source_missing",
        "unavailable",
        "invalid",
    ):
        assert validate_source_readiness_shape(
            _readiness_record(readiness_state=readiness_state)
        ) == ()


def test_invalid_readiness_state_is_reported_without_quality_inference():
    assert validate_source_readiness_shape(
        _readiness_record(readiness_state="excellent_business")
    ) == (
        FundamentalsNormalizationIssue(
            field_name="readiness_state",
            issue_code=FundamentalsNormalizationIssueCode.INVALID_READINESS_STATE,
            observed_value="excellent_business",
        ),
    )


def test_forbidden_fields_are_reported_explicitly():
    assert validate_normalized_fundamentals_shape(
        _normalized_record(target_price="EUR 900")
    ) == (
        FundamentalsNormalizationIssue(
            field_name="target_price",
            issue_code=FundamentalsNormalizationIssueCode.FORBIDDEN_FIELD,
            observed_value="EUR 900",
        ),
    )


def test_normalized_contract_has_no_final_action_or_quality_authority_fields():
    normalized_fields = set(required_normalized_fundamentals_fields())
    readiness_fields = set(required_source_readiness_fields())

    for field_name in forbidden_normalized_fundamentals_fields():
        assert field_name not in normalized_fields
        assert field_name not in readiness_fields


def test_contract_module_does_not_import_legacy_or_network_modules():
    source = Path(fundamentals_normalization_contracts.__file__).read_text(
        encoding="utf-8"
    )

    for forbidden in (
        "scripts",
        "requests",
        "urllib",
        "httpx",
        "aiohttp",
        "yfinance",
        "EDGAR",
    ):
        assert forbidden not in source


def test_contract_import_and_helpers_create_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    reload(fundamentals_normalization_contracts)
    validate_raw_source_shape(_raw_source_record(captured_at=""))
    validate_normalized_fundamentals_shape(_normalized_record(metric_value=None))
    validate_source_readiness_shape(_readiness_record(stale_data_count=""))

    assert list(tmp_path.iterdir()) == []
    assert not Path("reports/daily/telegram_message.txt").exists()
