"""Provider-source contracts for governed v2 fundamentals ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping


class ProviderCategory(StrEnum):
    """Approved provider categories for primary-first fundamentals sources."""

    OFFICIAL_COMPANY_FILING = "official_company_filing"
    REGULATORY_FILING = "regulatory_filing"
    OFFICIAL_COMPANY_INVESTOR_RELATIONS = "official_company_investor_relations"
    TRACEABLE_PROVIDER_API = "traceable_provider_api"
    GOVERNED_MANUAL_FILE = "governed_manual_file"


class ProviderSourceStatus(StrEnum):
    """Neutral provider/source availability statuses only."""

    AVAILABLE = "available"
    PARTIAL_DATA = "partial_data"
    STALE_DATA = "stale_data"
    INVALID_DATA = "invalid_data"
    SOURCE_MISSING = "source_missing"
    PROVIDER_ERROR = "provider_error"


class ProviderContractIssueCode(StrEnum):
    """Issue codes for provider-source contract checks."""

    MISSING_REQUIRED_FIELD = "missing_required_field"
    MISSING_REQUIRED_VALUE = "missing_required_value"
    FORBIDDEN_FIELD = "forbidden_field"
    INVALID_PROVIDER_CATEGORY = "invalid_provider_category"
    INVALID_PROVIDER_STATUS = "invalid_provider_status"


@dataclass(frozen=True)
class ProviderContractIssue:
    """Explicit provider contract issue without scoring or decisions."""

    field_name: str
    issue_code: ProviderContractIssueCode
    observed_value: object


@dataclass(frozen=True)
class ProviderRawFieldEvidence:
    """Original provider field evidence preserved before normalization."""

    original_field_name: str
    original_field_value: object
    original_currency: str
    original_unit: str
    reported_period: str = ""
    fiscal_year: str = ""
    fiscal_quarter: str = ""


@dataclass(frozen=True)
class ProviderSourceResponse:
    """Provider/source response supplied explicitly by an injected source client."""

    provider_name: str
    provider_category: str
    provider_record_id: str
    original_source_reference: str
    ticker: str
    symbol: str
    entity_identifier: str
    source_timestamp: str
    retrieval_timestamp: str
    reported_period: str
    fiscal_year: str
    fiscal_quarter: str
    raw_fields: Mapping[str, ProviderRawFieldEvidence]
    provider_status: str
    provider_error_status: str
    missing_field_evidence: tuple[str, ...]
    provenance_metadata: str
    raw_payload_hash: str
    capture_version: str


@dataclass(frozen=True)
class ProviderRawEvidenceRecord:
    """Immutable raw evidence captured from a governed provider response."""

    provider_name: str
    provider_category: str
    provider_record_id: str
    original_source_reference: str
    ticker: str
    symbol: str
    entity_identifier: str
    source_timestamp: str
    retrieval_timestamp: str
    reported_period: str
    fiscal_year: str
    fiscal_quarter: str
    raw_fields: tuple[ProviderRawFieldEvidence, ...]
    provider_status: str
    provider_error_status: str
    missing_field_evidence: tuple[str, ...]
    provenance_metadata: str
    raw_payload_hash: str
    capture_version: str


@dataclass(frozen=True)
class ProviderNormalizedFundamentalRecord:
    """Program-ready normalized provider metric with raw evidence traceability."""

    ticker: str
    fiscal_period: str
    fiscal_year: str
    fiscal_quarter: str
    metric_name: str
    metric_value: object
    metric_unit: str
    currency: str
    normalized_at: str
    source_provider: str
    source_reference: str
    source_record_identity: str
    original_field_name: str
    normalization_status: str
    validation_status: str
    derivation_formula: str = ""
    source_field_names: tuple[str, ...] = ()
    validation_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProviderSourceDataReadinessRecord:
    """Neutral readiness metadata for provider-source ingestion."""

    ticker: str
    fiscal_period: str
    fiscal_year: str
    readiness_state: str
    source_data_status: str
    missing_fundamentals_count: int
    partial_data_count: int
    stale_data_count: int
    source_reference: str
    provider_status: str
    provider_error_status: str
    readiness_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProviderPriorYearGrowthEvidenceRecord:
    """Governed current/prior growth evidence with explicit provenance."""

    ticker: str
    metric_name: str
    current_period_value: object
    prior_period_value: object
    current_period_reference: str
    prior_period_reference: str
    current_fiscal_year: str
    prior_fiscal_year: str
    fiscal_period: str
    fiscal_quarter: str
    currency: str
    unit: str
    current_source_reference: str
    prior_source_reference: str
    current_source_record_identity: str
    prior_source_record_identity: str
    current_source_field_names: tuple[str, ...]
    prior_source_field_names: tuple[str, ...]
    comparison_formula: str
    growth_rate: object
    growth_status: str
    validation_warnings: tuple[str, ...] = ()


REQUIRED_PROVIDER_RESPONSE_FIELDS: tuple[str, ...] = (
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

OPTIONAL_EMPTY_PROVIDER_RESPONSE_FIELDS: tuple[str, ...] = (
    "missing_field_evidence",
)

FORBIDDEN_PROVIDER_CONTRACT_FIELDS: tuple[str, ...] = (
    "final_action",
    "allocation",
    "allocation_amount",
    "execution_instruction",
    "urgency",
    "conviction",
    "tradeability",
    "tradeable_setup",
    "rank",
    "ranking",
    "score",
    "recommendation",
    "target_price",
    "threshold_price",
    "investment_quality",
    "investment_quality_score",
    "quality_score",
    "telegram_message",
    "report_message",
)


def approved_provider_categories() -> tuple[ProviderCategory, ...]:
    """Return provider categories approved by governance for future use."""

    return tuple(ProviderCategory)


def provider_source_statuses() -> tuple[ProviderSourceStatus, ...]:
    """Return neutral provider/source statuses."""

    return tuple(ProviderSourceStatus)


def required_provider_response_fields() -> tuple[str, ...]:
    """Return required fields for provider/source responses."""

    return REQUIRED_PROVIDER_RESPONSE_FIELDS


def forbidden_provider_contract_fields() -> tuple[str, ...]:
    """Return provider contract fields that would introduce forbidden authority."""

    return FORBIDDEN_PROVIDER_CONTRACT_FIELDS


def validate_provider_source_response_shape(
    response: ProviderSourceResponse | Mapping[str, object],
) -> tuple[ProviderContractIssue, ...]:
    """Validate provider response metadata without source calls or file IO."""

    record = _response_mapping(response)
    issues: list[ProviderContractIssue] = []

    for field_name in REQUIRED_PROVIDER_RESPONSE_FIELDS:
        if field_name not in record:
            issues.append(
                ProviderContractIssue(
                    field_name=field_name,
                    issue_code=ProviderContractIssueCode.MISSING_REQUIRED_FIELD,
                    observed_value=None,
                )
            )
            continue

        value = record[field_name]
        if value is None or value == "" or (
            value == () and field_name not in OPTIONAL_EMPTY_PROVIDER_RESPONSE_FIELDS
        ):
            issues.append(
                ProviderContractIssue(
                    field_name=field_name,
                    issue_code=ProviderContractIssueCode.MISSING_REQUIRED_VALUE,
                    observed_value=value,
                )
            )

    if "provider_category" in record and record.get("provider_category") not in {
        category.value for category in ProviderCategory
    }:
        issues.append(
            ProviderContractIssue(
                field_name="provider_category",
                issue_code=ProviderContractIssueCode.INVALID_PROVIDER_CATEGORY,
                observed_value=record.get("provider_category"),
            )
        )

    if "provider_status" in record and record.get("provider_status") not in {
        status.value for status in ProviderSourceStatus
    }:
        issues.append(
            ProviderContractIssue(
                field_name="provider_status",
                issue_code=ProviderContractIssueCode.INVALID_PROVIDER_STATUS,
                observed_value=record.get("provider_status"),
            )
        )

    for field_name in FORBIDDEN_PROVIDER_CONTRACT_FIELDS:
        if field_name in record:
            issues.append(
                ProviderContractIssue(
                    field_name=field_name,
                    issue_code=ProviderContractIssueCode.FORBIDDEN_FIELD,
                    observed_value=record[field_name],
                )
            )

    return tuple(issues)


def _response_mapping(
    response: ProviderSourceResponse | Mapping[str, object],
) -> Mapping[str, object]:
    if hasattr(response, "__dataclass_fields__"):
        return {
            field_name: getattr(response, field_name)
            for field_name in response.__dataclass_fields__
        }
    return response
