from pathlib import Path

from market_scanner.fundamentals import fundamental_contracts
from market_scanner.fundamentals.fundamental_contracts import (
    FundamentalContractIssue,
    FundamentalContractIssueCode,
    FundamentalDatasetRole,
    SourceDataReadinessState,
    forbidden_fundamental_upstream_fields,
    fundamental_history_identity_fields,
    generated_dataset_roles,
    required_fundamental_history_fields,
    required_source_readiness_fields,
    source_dataset_roles,
    source_readiness_identity_fields,
    validate_fundamental_history_shape,
    validate_source_readiness_shape,
)
from market_scanner.shared.data_contracts import (
    APPROVED_FIXTURE_CONTRACTS,
    read_fixture_rows,
)


def _complete_source_readiness(**overrides):
    record = {
        "source_record_id": "src-001",
        "symbol": "ALFA",
        "source_name": "synthetic_provider",
        "metric_name": "revenue_growth",
        "metric_value": "0.12",
        "metric_unit": "ratio",
        "as_of_date": "2025-12-31",
        "readiness_state": "available",
        "missing_value_policy": "not_applicable",
        "review_required_reason": "",
    }
    record.update(overrides)
    return record


def _complete_history(**overrides):
    record = {
        "ticker": "ALFA",
        "fiscal_year": "2025",
        "fiscal_period": "FY",
        "period_end_date": "2025-12-31",
        "report_date": "2026-02-15",
        "currency": "EUR",
        "revenue": "1000",
        "gross_profit": "600",
        "operating_income": "300",
        "net_income": "200",
        "diluted_eps": "5",
        "total_debt": "150",
        "total_equity": "500",
        "free_cash_flow": "250",
        "source_name": "synthetic filing",
        "source_reference": "synthetic FY2025 filing",
        "source_freshness_date": "2026-05-08",
        "extraction_date": "2026-05-08",
        "notes": "",
    }
    record.update(overrides)
    return record


def test_source_data_readiness_states_are_explicit_and_not_quality_states():
    assert {state.name for state in SourceDataReadinessState} == {
        "AVAILABLE",
        "MISSING",
        "SOURCE_MISSING",
        "ROW_MISSING",
        "PARTIAL",
        "STALE",
        "INVALID",
        "UNAVAILABLE",
        "REVIEW_REQUIRED",
    }

    assert "QUALITY" not in {state.name for state in SourceDataReadinessState}


def test_fundamental_dataset_roles_separate_raw_normalized_and_generated_data():
    assert source_dataset_roles() == (
        FundamentalDatasetRole.RAW_SOURCE_CAPTURE,
        FundamentalDatasetRole.NORMALIZED_SOURCE_READINESS,
        FundamentalDatasetRole.NORMALIZED_FUNDAMENTAL_HISTORY,
    )
    assert generated_dataset_roles() == (
        FundamentalDatasetRole.GENERATED_FUNDAMENTAL_CLASSIFICATION,
        FundamentalDatasetRole.GENERATED_FUNDAMENTAL_ANALYSIS,
    )
    assert set(source_dataset_roles()).isdisjoint(generated_dataset_roles())


def test_source_readiness_required_fields_match_approved_v2_fixture():
    contract = next(
        contract
        for contract in APPROVED_FIXTURE_CONTRACTS
        if contract.name == "synthetic_source_data_readiness"
    )

    assert required_source_readiness_fields() == contract.required_columns
    assert validate_source_readiness_shape(read_fixture_rows(contract)[0]) == ()


def test_fundamental_history_required_fields_include_period_and_provenance():
    assert required_fundamental_history_fields() == (
        "ticker",
        "fiscal_year",
        "fiscal_period",
        "period_end_date",
        "report_date",
        "currency",
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "diluted_eps",
        "total_debt",
        "total_equity",
        "free_cash_flow",
        "source_name",
        "source_reference",
        "source_freshness_date",
        "extraction_date",
        "notes",
    )


def test_source_data_identity_fields_are_explicit():
    assert source_readiness_identity_fields() == ("source_record_id",)
    assert fundamental_history_identity_fields() == (
        "ticker",
        "fiscal_year",
        "fiscal_period",
    )


def test_missing_source_readiness_fields_are_reported_explicitly():
    record = _complete_source_readiness()
    record.pop("source_name")

    issues = validate_source_readiness_shape(record)

    assert issues == (
        FundamentalContractIssue(
            field_name="source_name",
            issue_code=FundamentalContractIssueCode.MISSING_REQUIRED_FIELD,
            observed_value=None,
        ),
    )


def test_missing_source_data_values_are_not_converted_to_zero():
    issues = validate_source_readiness_shape(
        _complete_source_readiness(metric_value="")
    )

    assert issues == (
        FundamentalContractIssue(
            field_name="metric_value",
            issue_code=FundamentalContractIssueCode.MISSING_REQUIRED_VALUE,
            observed_value="",
        ),
    )


def test_invalid_readiness_states_are_reported_without_quality_inference():
    issues = validate_source_readiness_shape(
        _complete_source_readiness(readiness_state="excellent_company")
    )

    assert issues == (
        FundamentalContractIssue(
            field_name="readiness_state",
            issue_code=FundamentalContractIssueCode.INVALID_READINESS_STATE,
            observed_value="excellent_company",
        ),
    )


def test_missing_history_numeric_values_remain_missing_not_zero():
    issues = validate_fundamental_history_shape(_complete_history(revenue=""))

    assert issues == (
        FundamentalContractIssue(
            field_name="revenue",
            issue_code=FundamentalContractIssueCode.MISSING_REQUIRED_VALUE,
            observed_value="",
        ),
    )


def test_invalid_history_numeric_values_are_reported_without_scoring():
    issues = validate_fundamental_history_shape(
        _complete_history(gross_profit="not-a-number")
    )

    assert issues == (
        FundamentalContractIssue(
            field_name="gross_profit",
            issue_code=FundamentalContractIssueCode.INVALID_NUMERIC_VALUE,
            observed_value="not-a-number",
        ),
    )


def test_forbidden_fundamental_authority_fields_are_reported_as_issues():
    issues = validate_source_readiness_shape(
        _complete_source_readiness(quality_score="100")
    )

    assert issues == (
        FundamentalContractIssue(
            field_name="quality_score",
            issue_code=FundamentalContractIssueCode.FORBIDDEN_FIELD,
            observed_value="100",
        ),
    )


def test_contract_fields_exclude_final_action_and_investment_quality_authority():
    readiness_fields = set(required_source_readiness_fields())
    history_fields = set(required_fundamental_history_fields())

    for field_name in forbidden_fundamental_upstream_fields():
        assert field_name not in readiness_fields
        assert field_name not in history_fields


def test_fundamental_contract_module_does_not_import_legacy_or_network_modules():
    source = Path(fundamental_contracts.__file__).read_text(encoding="utf-8")

    for forbidden in (
        "scripts",
        "requests",
        "urllib",
        "httpx",
        "aiohttp",
        "yfinance",
    ):
        assert forbidden not in source


def test_fundamental_contract_helpers_create_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    validate_source_readiness_shape(_complete_source_readiness(metric_value=None))
    validate_fundamental_history_shape(_complete_history(total_debt=""))

    assert list(tmp_path.iterdir()) == []
