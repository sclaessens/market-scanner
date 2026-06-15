from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from market_engine.source_intake.models import SourceIntakeError, TickerSourceResult
from market_engine.source_intake.provider_boundary import ProviderSourceResponse
from market_engine.source_intake.readiness import SourceReadinessStatus
from market_engine.source_intake.sec_companyfacts_fields import (
    SEC_COMPANYFACTS_PROVIDER_NAME,
    SEC_COMPANYFACTS_REQUIRED_FIELDS,
    SecCompanyFactsMappedField,
    map_sec_companyfacts_fields,
)


SOURCE_DATA_ONLY_MODE = "source-data-only"


@dataclass(frozen=True)
class FundamentalSourceContext:
    ticker: str
    provider: str
    source_status: SourceReadinessStatus
    canonical_fields: dict[str, Any | None]
    missing_canonical_fields: tuple[str, ...]
    provenance: dict[str, SecCompanyFactsMappedField] = field(default_factory=dict)
    period_metadata: dict[str, dict[str, Any | None]] = field(default_factory=dict)
    provider_error_category: str | None = None
    provider_error_message: str | None = None
    mode: str = SOURCE_DATA_ONLY_MODE


def build_sec_fundamental_source_context(
    *,
    ticker: str,
    response: ProviderSourceResponse | None = None,
    source_result: TickerSourceResult | None = None,
) -> FundamentalSourceContext:
    provider = (
        source_result.provider_name
        if source_result is not None
        else SEC_COMPANYFACTS_PROVIDER_NAME
    )
    if source_result is not None and source_result.readiness_status in {
        SourceReadinessStatus.UNSUPPORTED,
        SourceReadinessStatus.INVALID_TICKER,
        SourceReadinessStatus.PROVIDER_ERROR,
    }:
        return _context_from_terminal_result(ticker=ticker, provider=provider, source_result=source_result)

    raw_evidence = response.raw_evidence if response is not None else None
    if isinstance(raw_evidence, dict):
        mapped_fields = map_sec_companyfacts_fields(raw_evidence)
        canonical_fields = {
            field_name: mapped_field.raw_value if mapped_field is not None else None
            for field_name, mapped_field in mapped_fields.items()
        }
        provenance = {
            field_name: mapped_field
            for field_name, mapped_field in mapped_fields.items()
            if mapped_field is not None
        }
    else:
        canonical_fields = _empty_canonical_fields()
        provenance = {}

    missing = tuple(
        field_name
        for field_name in SEC_COMPANYFACTS_REQUIRED_FIELDS
        if canonical_fields.get(field_name) is None
    )
    return FundamentalSourceContext(
        ticker=ticker,
        provider=provider,
        source_status=_readiness_from_canonical_fields(canonical_fields),
        canonical_fields=canonical_fields,
        missing_canonical_fields=missing,
        provenance=provenance,
        period_metadata={
            field_name: _period_metadata(mapped_field)
            for field_name, mapped_field in provenance.items()
        },
    )


def _context_from_terminal_result(
    *,
    ticker: str,
    provider: str,
    source_result: TickerSourceResult,
) -> FundamentalSourceContext:
    error = source_result.error
    return FundamentalSourceContext(
        ticker=ticker,
        provider=provider,
        source_status=source_result.readiness_status,
        canonical_fields=_empty_canonical_fields(),
        missing_canonical_fields=SEC_COMPANYFACTS_REQUIRED_FIELDS,
        provider_error_category=_error_category(error),
        provider_error_message=error.message if error is not None else None,
    )


def _readiness_from_canonical_fields(
    canonical_fields: dict[str, Any | None],
) -> SourceReadinessStatus:
    available_count = sum(
        1
        for field_name in SEC_COMPANYFACTS_REQUIRED_FIELDS
        if canonical_fields.get(field_name) is not None
    )
    if available_count == len(SEC_COMPANYFACTS_REQUIRED_FIELDS):
        return SourceReadinessStatus.AVAILABLE
    if available_count > 0:
        return SourceReadinessStatus.PARTIAL
    return SourceReadinessStatus.MISSING


def _empty_canonical_fields() -> dict[str, Any | None]:
    return {field_name: None for field_name in SEC_COMPANYFACTS_REQUIRED_FIELDS}


def _period_metadata(mapped_field: SecCompanyFactsMappedField) -> dict[str, Any | None]:
    return {
        "fiscal_year": mapped_field.fiscal_year,
        "fiscal_period": mapped_field.fiscal_period,
        "filing_form": mapped_field.filing_form,
        "filing_date": mapped_field.filing_date,
        "period_start_date": mapped_field.period_start_date,
        "period_end_date": mapped_field.period_end_date,
        "accession_number": mapped_field.accession_number,
        "frame": mapped_field.frame,
    }


def _error_category(error: SourceIntakeError | None) -> str | None:
    return error.error_type if error is not None else None
