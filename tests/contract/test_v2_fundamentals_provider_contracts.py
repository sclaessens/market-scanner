from importlib import reload
from pathlib import Path

from market_scanner.fundamentals import fundamentals_provider_contracts
from market_scanner.fundamentals.fundamentals_provider_contracts import (
    ProviderCategory,
    ProviderContractIssue,
    ProviderContractIssueCode,
    ProviderPriorYearGrowthEvidenceRecord,
    ProviderRawFieldEvidence,
    ProviderSourceResponse,
    ProviderSourceStatus,
    approved_provider_categories,
    forbidden_provider_contract_fields,
    provider_source_statuses,
    required_provider_response_fields,
    validate_provider_source_response_shape,
)


FORBIDDEN_AUTHORITY_FIELDS = {
    "allocation",
    "conviction",
    "final_action",
    "investment_quality",
    "recommendation",
    "score",
    "target_price",
    "threshold_price",
    "tradeability",
    "urgency",
}


def _field(name: str, value: object = "1000") -> ProviderRawFieldEvidence:
    return ProviderRawFieldEvidence(
        original_field_name=name,
        original_field_value=value,
        original_currency="USD",
        original_unit="USD",
    )


def _response(**overrides) -> ProviderSourceResponse:
    response = ProviderSourceResponse(
        provider_name="SEC Companyfacts",
        provider_category=ProviderCategory.REGULATORY_FILING.value,
        provider_record_id="CIK0000320193-FY-2025",
        original_source_reference="sec-companyfacts:CIK0000320193/FY/2025",
        ticker="AAPL",
        symbol="AAPL",
        entity_identifier="CIK0000320193",
        source_timestamp="2026-02-15",
        retrieval_timestamp="2026-06-03T00:00:00Z",
        reported_period="FY",
        fiscal_year="2025",
        fiscal_quarter="",
        raw_fields={
            "Revenues": _field("Revenues", "1000"),
            "NetIncomeLoss": _field("NetIncomeLoss", "155"),
        },
        provider_status=ProviderSourceStatus.AVAILABLE.value,
        provider_error_status="",
        missing_field_evidence=(),
        provenance_metadata="official SEC companyfacts response metadata",
        raw_payload_hash="sha256:provider-contract-fixture",
        capture_version="v2-provider-contract-v1",
    )
    values = {
        field_name: getattr(response, field_name)
        for field_name in response.__dataclass_fields__
    }
    values.update(overrides)
    return ProviderSourceResponse(**values)


def test_approved_provider_categories_are_primary_and_provenance_first():
    assert approved_provider_categories() == (
        ProviderCategory.OFFICIAL_COMPANY_FILING,
        ProviderCategory.REGULATORY_FILING,
        ProviderCategory.OFFICIAL_COMPANY_INVESTOR_RELATIONS,
        ProviderCategory.TRACEABLE_PROVIDER_API,
        ProviderCategory.GOVERNED_MANUAL_FILE,
    )


def test_provider_source_statuses_are_neutral_data_states():
    assert provider_source_statuses() == (
        ProviderSourceStatus.AVAILABLE,
        ProviderSourceStatus.PARTIAL_DATA,
        ProviderSourceStatus.STALE_DATA,
        ProviderSourceStatus.INVALID_DATA,
        ProviderSourceStatus.SOURCE_MISSING,
        ProviderSourceStatus.PROVIDER_ERROR,
    )


def test_required_provider_response_fields_include_provenance_and_capture_metadata():
    assert required_provider_response_fields() == (
        "provider_name",
        "provider_category",
        "provider_record_id",
        "original_source_reference",
        "ticker",
        "symbol",
        "source_timestamp",
        "retrieval_timestamp",
        "reported_period",
        "fiscal_year",
        "raw_fields",
        "provider_status",
        "missing_field_evidence",
        "provenance_metadata",
        "raw_payload_hash",
        "capture_version",
    )
    assert validate_provider_source_response_shape(_response()) == ()


def test_missing_provider_response_values_are_reported_explicitly():
    assert validate_provider_source_response_shape(
        _response(provider_record_id="")
    ) == (
        ProviderContractIssue(
            field_name="provider_record_id",
            issue_code=ProviderContractIssueCode.MISSING_REQUIRED_VALUE,
            observed_value="",
        ),
    )


def test_invalid_provider_category_is_reported_without_fallback_approval():
    assert validate_provider_source_response_shape(
        _response(provider_category="anonymous_scraped_page")
    ) == (
        ProviderContractIssue(
            field_name="provider_category",
            issue_code=ProviderContractIssueCode.INVALID_PROVIDER_CATEGORY,
            observed_value="anonymous_scraped_page",
        ),
    )


def test_invalid_provider_status_is_reported_without_quality_inference():
    assert validate_provider_source_response_shape(
        _response(provider_status="excellent_company")
    ) == (
        ProviderContractIssue(
            field_name="provider_status",
            issue_code=ProviderContractIssueCode.INVALID_PROVIDER_STATUS,
            observed_value="excellent_company",
        ),
    )


def test_forbidden_authority_fields_are_reported_by_provider_contracts():
    assert validate_provider_source_response_shape(
        {
            "provider_name": "SEC Companyfacts",
            "provider_category": ProviderCategory.REGULATORY_FILING.value,
            "provider_record_id": "CIK0000320193-FY-2025",
            "original_source_reference": "sec-companyfacts:CIK0000320193",
            "ticker": "AAPL",
            "symbol": "AAPL",
            "source_timestamp": "2026-02-15",
            "retrieval_timestamp": "2026-06-03T00:00:00Z",
            "reported_period": "FY",
            "fiscal_year": "2025",
            "raw_fields": {"Revenues": _field("Revenues")},
            "provider_status": ProviderSourceStatus.AVAILABLE.value,
            "missing_field_evidence": (),
            "provenance_metadata": "official SEC companyfacts response metadata",
            "raw_payload_hash": "sha256:provider-contract-fixture",
            "capture_version": "v2-provider-contract-v1",
            "score": "100",
        }
    ) == (
        ProviderContractIssue(
            field_name="score",
            issue_code=ProviderContractIssueCode.FORBIDDEN_FIELD,
            observed_value="100",
        ),
    )


def test_provider_contract_records_do_not_define_investment_authority_fields():
    contract_fields = (
        set(ProviderSourceResponse.__dataclass_fields__)
        | set(ProviderRawFieldEvidence.__dataclass_fields__)
        | set(ProviderPriorYearGrowthEvidenceRecord.__dataclass_fields__)
    )

    assert contract_fields.isdisjoint(FORBIDDEN_AUTHORITY_FIELDS)
    assert set(forbidden_provider_contract_fields()).issuperset(
        FORBIDDEN_AUTHORITY_FIELDS
    )


def test_provider_contract_module_does_not_import_legacy_or_network_modules():
    source = Path(fundamentals_provider_contracts.__file__).read_text(
        encoding="utf-8"
    )

    for forbidden in (
        "scripts",
        "requests",
        "urllib",
        "httpx",
        "aiohttp",
        "yfinance",
    ):
        assert forbidden not in source


def test_provider_contract_import_and_validation_create_no_files(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    reload(fundamentals_provider_contracts)
    validate_provider_source_response_shape(_response(raw_payload_hash=""))

    assert list(tmp_path.iterdir()) == []
    assert not Path("data/raw").exists()
    assert not Path("data/normalized").exists()
    assert not Path("data/generated").exists()
    assert not Path("reports/daily/telegram_message.txt").exists()
