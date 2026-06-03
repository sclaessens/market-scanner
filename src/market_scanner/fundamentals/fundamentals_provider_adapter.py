"""V2 provider adapter boundary for governed fundamentals ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from market_scanner.fundamentals.fundamental_contracts import (
    SourceDataReadinessState,
)
from market_scanner.fundamentals.fundamentals_normalization_contracts import (
    FundamentalsNormalizationIssue,
    validate_normalized_fundamentals_shape,
    validate_source_readiness_shape,
)
from market_scanner.fundamentals.fundamentals_provider_contracts import (
    ProviderContractIssue,
    ProviderRawEvidenceRecord,
    ProviderRawFieldEvidence,
    ProviderSourceDataReadinessRecord,
    ProviderSourceResponse,
    ProviderSourceStatus,
    ProviderNormalizedFundamentalRecord,
    validate_provider_source_response_shape,
)


class FundamentalsProviderClient(Protocol):
    """Dependency-injected provider boundary; implementations may fetch manually."""

    def fetch_fundamentals(self, ticker: str) -> ProviderSourceResponse:
        """Return one provider/source response for a ticker."""


@dataclass(frozen=True)
class ProviderFundamentalsIngestionResult:
    """Pure in-memory provider ingestion result."""

    raw_evidence: ProviderRawEvidenceRecord
    normalized_records: tuple[ProviderNormalizedFundamentalRecord, ...]
    readiness_record: ProviderSourceDataReadinessRecord
    issues: tuple[ProviderContractIssue | FundamentalsNormalizationIssue, ...]


DEFAULT_PROVIDER_METRIC_MAPPINGS: Mapping[str, tuple[str, ...]] = {
    "revenue": ("Revenues", "Revenue", "SalesRevenueNet", "revenue"),
    "gross_profit": ("GrossProfit", "GrossProfitLoss", "gross_profit"),
    "operating_income": (
        "OperatingIncomeLoss",
        "OperatingIncome",
        "operating_income",
    ),
    "net_income": ("NetIncomeLoss", "NetIncome", "net_income"),
    "eps_diluted": (
        "EarningsPerShareDiluted",
        "DilutedEarningsPerShare",
        "eps_diluted",
    ),
    "total_assets": ("Assets", "TotalAssets", "total_assets"),
    "total_liabilities": (
        "Liabilities",
        "TotalLiabilities",
        "total_liabilities",
    ),
    "shareholders_equity": (
        "StockholdersEquity",
        "ShareholdersEquity",
        "shareholders_equity",
    ),
    "operating_cash_flow": (
        "NetCashProvidedByUsedInOperatingActivities",
        "OperatingCashFlow",
        "operating_cash_flow",
    ),
    "capital_expenditures": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpenditures",
        "capital_expenditures",
    ),
    "free_cash_flow": ("FreeCashFlow", "free_cash_flow"),
}


def ingest_provider_fundamentals_from_client(
    client: FundamentalsProviderClient,
    *,
    ticker: str,
    metric_mappings: Mapping[str, Sequence[str]] = DEFAULT_PROVIDER_METRIC_MAPPINGS,
) -> ProviderFundamentalsIngestionResult:
    """Fetch through an injected client and ingest without hidden side effects."""

    return ingest_provider_fundamentals(
        client.fetch_fundamentals(ticker),
        metric_mappings=metric_mappings,
    )


def ingest_provider_fundamentals(
    response: ProviderSourceResponse,
    *,
    metric_mappings: Mapping[str, Sequence[str]] = DEFAULT_PROVIDER_METRIC_MAPPINGS,
) -> ProviderFundamentalsIngestionResult:
    """Capture raw provider evidence and normalize mapped fundamentals in memory."""

    raw_evidence = capture_provider_raw_evidence(response)
    normalized_records, normalization_issues = normalize_provider_raw_evidence(
        raw_evidence,
        metric_mappings=metric_mappings,
    )
    readiness_record = build_provider_source_data_readiness(
        raw_evidence,
        normalized_records=normalized_records,
        has_contract_issues=bool(validate_provider_source_response_shape(response)),
    )
    readiness_issues = validate_source_readiness_shape(
        _readiness_contract_shape(readiness_record)
    )

    return ProviderFundamentalsIngestionResult(
        raw_evidence=raw_evidence,
        normalized_records=normalized_records,
        readiness_record=readiness_record,
        issues=(
            *validate_provider_source_response_shape(response),
            *normalization_issues,
            *readiness_issues,
        ),
    )


def capture_provider_raw_evidence(
    response: ProviderSourceResponse,
) -> ProviderRawEvidenceRecord:
    """Preserve provider/source response values as immutable raw evidence."""

    return ProviderRawEvidenceRecord(
        provider_name=response.provider_name,
        provider_category=response.provider_category,
        provider_record_id=response.provider_record_id,
        original_source_reference=response.original_source_reference,
        ticker=response.ticker,
        symbol=response.symbol,
        entity_identifier=response.entity_identifier,
        source_timestamp=response.source_timestamp,
        retrieval_timestamp=response.retrieval_timestamp,
        reported_period=response.reported_period,
        fiscal_year=response.fiscal_year,
        fiscal_quarter=response.fiscal_quarter,
        raw_fields=tuple(response.raw_fields.values()),
        provider_status=response.provider_status,
        provider_error_status=response.provider_error_status,
        missing_field_evidence=tuple(response.missing_field_evidence),
        provenance_metadata=response.provenance_metadata,
        raw_payload_hash=response.raw_payload_hash,
        capture_version=response.capture_version,
    )


def normalize_provider_raw_evidence(
    raw_evidence: ProviderRawEvidenceRecord,
    *,
    metric_mappings: Mapping[str, Sequence[str]] = DEFAULT_PROVIDER_METRIC_MAPPINGS,
) -> tuple[
    tuple[ProviderNormalizedFundamentalRecord, ...],
    tuple[FundamentalsNormalizationIssue, ...],
]:
    """Map raw provider evidence into program-ready records without scoring."""

    normalized_records: list[ProviderNormalizedFundamentalRecord] = []
    issues: list[FundamentalsNormalizationIssue] = []
    raw_fields_by_name = {
        field.original_field_name: field for field in raw_evidence.raw_fields
    }

    for metric_name, candidate_field_names in metric_mappings.items():
        raw_field = _first_matching_field(raw_fields_by_name, candidate_field_names)
        normalized_record = _normalized_record_for(
            raw_evidence,
            metric_name=metric_name,
            raw_field=raw_field,
        )
        normalized_records.append(normalized_record)
        issues.extend(
            validate_normalized_fundamentals_shape(
                _normalized_contract_shape(normalized_record)
            )
        )

    return tuple(normalized_records), tuple(issues)


def build_provider_source_data_readiness(
    raw_evidence: ProviderRawEvidenceRecord,
    *,
    normalized_records: Sequence[ProviderNormalizedFundamentalRecord],
    has_contract_issues: bool = False,
) -> ProviderSourceDataReadinessRecord:
    """Build neutral source-data readiness from provider metadata only."""

    missing_count = sum(
        1
        for record in normalized_records
        if record.metric_value is None or record.metric_value == ""
    )
    readiness_state = _readiness_state_for(
        raw_evidence,
        metric_count=len(normalized_records),
        missing_count=missing_count,
        has_contract_issues=has_contract_issues,
    )

    return ProviderSourceDataReadinessRecord(
        ticker=raw_evidence.ticker,
        fiscal_period=raw_evidence.reported_period,
        fiscal_year=raw_evidence.fiscal_year,
        readiness_state=readiness_state.value,
        source_data_status=readiness_state.value,
        missing_fundamentals_count=missing_count,
        partial_data_count=missing_count,
        stale_data_count=(
            1
            if raw_evidence.provider_status == ProviderSourceStatus.STALE_DATA.value
            else 0
        ),
        source_reference=raw_evidence.original_source_reference,
        provider_status=raw_evidence.provider_status,
        provider_error_status=raw_evidence.provider_error_status,
    )


def _first_matching_field(
    raw_fields_by_name: Mapping[str, ProviderRawFieldEvidence],
    candidate_field_names: Sequence[str],
) -> ProviderRawFieldEvidence | None:
    for field_name in candidate_field_names:
        if field_name in raw_fields_by_name:
            return raw_fields_by_name[field_name]
    return None


def _normalized_record_for(
    raw_evidence: ProviderRawEvidenceRecord,
    *,
    metric_name: str,
    raw_field: ProviderRawFieldEvidence | None,
) -> ProviderNormalizedFundamentalRecord:
    return ProviderNormalizedFundamentalRecord(
        ticker=raw_evidence.ticker,
        fiscal_period=raw_evidence.reported_period,
        fiscal_year=raw_evidence.fiscal_year,
        fiscal_quarter=raw_evidence.fiscal_quarter,
        metric_name=metric_name,
        metric_value=(
            raw_field.original_field_value if raw_field is not None else None
        ),
        metric_unit=raw_field.original_unit if raw_field is not None else "",
        currency=raw_field.original_currency if raw_field is not None else "",
        normalized_at=raw_evidence.retrieval_timestamp,
        source_provider=raw_evidence.provider_name,
        source_reference=raw_evidence.original_source_reference,
        source_record_identity=raw_evidence.provider_record_id,
        original_field_name=(
            raw_field.original_field_name if raw_field is not None else ""
        ),
        normalization_status=(
            "mapped_from_raw_evidence" if raw_field is not None else "missing_source_field"
        ),
        validation_status=(
            "valid" if raw_field is not None and raw_field.original_field_value not in (None, "") else "review_required"
        ),
    )


def _readiness_state_for(
    raw_evidence: ProviderRawEvidenceRecord,
    *,
    metric_count: int,
    missing_count: int,
    has_contract_issues: bool,
) -> SourceDataReadinessState:
    if raw_evidence.provider_status == ProviderSourceStatus.PROVIDER_ERROR.value:
        return SourceDataReadinessState.INVALID
    if raw_evidence.provider_status == ProviderSourceStatus.INVALID_DATA.value:
        return SourceDataReadinessState.INVALID
    if has_contract_issues:
        return SourceDataReadinessState.INVALID
    if raw_evidence.provider_status == ProviderSourceStatus.SOURCE_MISSING.value:
        return SourceDataReadinessState.SOURCE_MISSING
    if raw_evidence.provider_status == ProviderSourceStatus.STALE_DATA.value:
        return SourceDataReadinessState.STALE
    if metric_count == 0 or missing_count == metric_count:
        return SourceDataReadinessState.MISSING
    if missing_count > 0 or raw_evidence.missing_field_evidence:
        return SourceDataReadinessState.PARTIAL
    return SourceDataReadinessState.AVAILABLE


def _normalized_contract_shape(
    record: ProviderNormalizedFundamentalRecord,
) -> dict[str, object]:
    return {
        "ticker": record.ticker,
        "fiscal_period": record.fiscal_period,
        "fiscal_year": record.fiscal_year,
        "metric_name": record.metric_name,
        "metric_value": record.metric_value,
        "metric_unit": record.metric_unit,
        "currency": record.currency,
        "normalized_at": record.normalized_at,
        "source_provider": record.source_provider,
        "source_reference": record.source_reference,
        "source_record_identity": record.source_record_identity,
    }


def _readiness_contract_shape(
    record: ProviderSourceDataReadinessRecord,
) -> dict[str, object]:
    return {
        "ticker": record.ticker,
        "fiscal_period": record.fiscal_period,
        "readiness_state": record.readiness_state,
        "source_data_status": record.source_data_status,
        "missing_fundamentals_count": record.missing_fundamentals_count,
        "partial_data_count": record.partial_data_count,
        "stale_data_count": record.stale_data_count,
        "source_reference": record.source_reference,
    }
