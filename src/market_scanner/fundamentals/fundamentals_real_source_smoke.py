"""Manual-only v2 smoke harness for controlled fundamentals source checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Sequence

from market_scanner.fundamentals.fundamentals_provider_adapter import (
    DEFAULT_PROVIDER_METRIC_MAPPINGS,
    FundamentalsProviderClient,
    ProviderFundamentalsIngestionResult,
    ingest_provider_fundamentals,
    ingest_provider_fundamentals_from_client,
)
from market_scanner.fundamentals.fundamentals_provider_contracts import (
    ProviderSourceResponse,
)


class RealSourceSmokeStatus(StrEnum):
    """Neutral controlled-smoke statuses, not investment conclusions."""

    PASSED = "passed"
    REVIEW_REQUIRED = "review_required"
    PROVIDER_ERROR = "provider_error"


@dataclass(frozen=True)
class ControlledRealSourceSmokeResult:
    """In-memory smoke-test result for a manually invoked source check."""

    ticker: str
    provider_name: str
    smoke_status: str
    provenance_summary: str
    missing_field_summary: tuple[str, ...]
    warnings: tuple[str, ...]
    ingestion_result: ProviderFundamentalsIngestionResult | None


def run_controlled_real_source_smoke_test(
    client: FundamentalsProviderClient,
    *,
    ticker: str,
    metric_mappings: Mapping[str, Sequence[str]] = DEFAULT_PROVIDER_METRIC_MAPPINGS,
) -> ControlledRealSourceSmokeResult:
    """Explicitly invoke an injected client and return an in-memory smoke result."""

    try:
        ingestion_result = ingest_provider_fundamentals_from_client(
            client,
            ticker=ticker,
            metric_mappings=metric_mappings,
        )
    except Exception as exc:
        return ControlledRealSourceSmokeResult(
            ticker=ticker,
            provider_name="",
            smoke_status=RealSourceSmokeStatus.PROVIDER_ERROR.value,
            provenance_summary="",
            missing_field_summary=(),
            warnings=(f"provider_source_error:{type(exc).__name__}",),
            ingestion_result=None,
        )

    return _smoke_result_from_ingestion(ticker, ingestion_result)


def review_injected_source_response(
    response: ProviderSourceResponse,
    *,
    metric_mappings: Mapping[str, Sequence[str]] = DEFAULT_PROVIDER_METRIC_MAPPINGS,
) -> ControlledRealSourceSmokeResult:
    """Review an explicitly supplied source response without client calls."""

    return _smoke_result_from_ingestion(
        response.ticker,
        ingest_provider_fundamentals(response, metric_mappings=metric_mappings),
    )


def _smoke_result_from_ingestion(
    ticker: str,
    ingestion_result: ProviderFundamentalsIngestionResult,
) -> ControlledRealSourceSmokeResult:
    raw = ingestion_result.raw_evidence
    smoke_status = (
        RealSourceSmokeStatus.REVIEW_REQUIRED.value
        if ingestion_result.issues
        or ingestion_result.readiness_record.missing_fundamentals_count > 0
        else RealSourceSmokeStatus.PASSED.value
    )

    return ControlledRealSourceSmokeResult(
        ticker=ticker,
        provider_name=raw.provider_name,
        smoke_status=smoke_status,
        provenance_summary="|".join(
            (
                raw.provider_name,
                raw.original_source_reference,
                raw.source_timestamp,
                raw.retrieval_timestamp,
                raw.reported_period,
                raw.fiscal_year,
            )
        ),
        missing_field_summary=raw.missing_field_evidence,
        warnings=_warnings_for(ingestion_result),
        ingestion_result=ingestion_result,
    )


def _warnings_for(
    ingestion_result: ProviderFundamentalsIngestionResult,
) -> tuple[str, ...]:
    warnings = [
        f"issue:{issue.field_name}:{issue.issue_code.value}"
        for issue in ingestion_result.issues
    ]

    readiness = ingestion_result.readiness_record
    if readiness.missing_fundamentals_count > 0:
        warnings.append(
            f"missing_fundamentals:{readiness.missing_fundamentals_count}"
        )
    if readiness.stale_data_count > 0:
        warnings.append(f"stale_source_data:{readiness.stale_data_count}")

    return tuple(warnings)
