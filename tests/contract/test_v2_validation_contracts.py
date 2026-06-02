from pathlib import Path

from market_scanner.validation import validation_contracts
from market_scanner.validation.validation_contracts import (
    ValidationIssueCode,
    ValidationState,
    candidate_identity_fields,
    forbidden_upstream_decision_fields,
    required_candidate_fields,
    validate_candidate_record_shape,
    validation_classification_fields,
)


def _complete_candidate(**overrides):
    candidate = {
        "ticker": "ALFA",
        "date": "2026-05-06",
        "primary_setup": "BREAKOUT",
        "rr": "2.5",
        "close": "102.0",
        "ma20": "100.0",
        "ma50": "90.0",
        "ma200": "80.0",
        "high_20d": "104.0",
        "low_20d": "95.0",
        "atr14": "2.0",
        "volume_ratio": "1.5",
        "extension_atr": "1.0",
    }
    candidate.update(overrides)
    return candidate


def test_validation_lifecycle_states_are_contract_metadata_only():
    assert {state.name for state in ValidationState} == {
        "COHERENT",
        "BROKEN",
        "INCOMPLETE",
        "REVIEW_REQUIRED",
    }


def test_required_candidate_fields_translate_legacy_validation_inputs():
    assert required_candidate_fields() == (
        "ticker",
        "date",
        "primary_setup",
        "rr",
        "close",
        "ma20",
        "ma50",
        "ma200",
        "high_20d",
        "low_20d",
        "atr14",
        "volume_ratio",
        "extension_atr",
    )


def test_validation_contract_preserves_row_identity_concept():
    assert candidate_identity_fields() == ("ticker", "date")
    assert set(candidate_identity_fields()).issubset(required_candidate_fields())


def test_validation_classification_fields_do_not_include_final_action_semantics():
    output_fields = set(validation_classification_fields())

    for field_name in forbidden_upstream_decision_fields():
        assert field_name not in output_fields


def test_complete_candidate_shape_has_no_contract_issues():
    assert validate_candidate_record_shape(_complete_candidate()) == ()


def test_missing_required_fields_are_reported_explicitly():
    candidate = _complete_candidate()
    candidate.pop("volume_ratio")

    issues = validate_candidate_record_shape(candidate)

    assert issues == (
        validation_contracts.ValidationIssue(
            field_name="volume_ratio",
            issue_code=ValidationIssueCode.MISSING_REQUIRED_FIELD,
            observed_value=None,
        ),
    )


def test_missing_numeric_values_are_not_converted_to_zero():
    issues = validate_candidate_record_shape(_complete_candidate(close=""))

    assert issues == (
        validation_contracts.ValidationIssue(
            field_name="close",
            issue_code=ValidationIssueCode.MISSING_REQUIRED_VALUE,
            observed_value="",
        ),
    )


def test_invalid_numeric_values_are_reported_without_filtering_row():
    issues = validate_candidate_record_shape(_complete_candidate(rr="not-a-number"))

    assert issues == (
        validation_contracts.ValidationIssue(
            field_name="rr",
            issue_code=ValidationIssueCode.INVALID_NUMERIC_VALUE,
            observed_value="not-a-number",
        ),
    )


def test_required_positive_metrics_remain_data_coherence_checks():
    issues = validate_candidate_record_shape(_complete_candidate(atr14="0"))

    assert issues == (
        validation_contracts.ValidationIssue(
            field_name="atr14",
            issue_code=ValidationIssueCode.INVALID_NON_POSITIVE_VALUE,
            observed_value="0",
        ),
    )


def test_forbidden_upstream_decision_fields_are_reported_as_contract_issues():
    issues = validate_candidate_record_shape(_complete_candidate(final_action="REVIEW"))

    assert issues == (
        validation_contracts.ValidationIssue(
            field_name="final_action",
            issue_code=ValidationIssueCode.FORBIDDEN_FIELD,
            observed_value="REVIEW",
        ),
    )


def test_validation_contract_module_does_not_import_legacy_scripts():
    source = Path(validation_contracts.__file__).read_text(encoding="utf-8")

    assert "scripts" not in source


def test_validation_contract_helpers_create_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    validate_candidate_record_shape(_complete_candidate(close=None))

    assert list(tmp_path.iterdir()) == []
