from __future__ import annotations

from collections import Counter
from typing import Iterable

from market_engine.source_intake.models import (
    BatchSourceIntakeSummary,
    SourceIntakeError,
    TickerSourceResult,
)
from market_engine.source_intake.provider_boundary import (
    SourceProvider,
    SourceProviderError,
)
from market_engine.source_intake.readiness import (
    SourceReadinessStatus,
    readiness_from_response,
    split_available_and_missing_fields,
)


def run_source_intake(
    tickers: Iterable[str],
    provider: SourceProvider,
    required_fields: Iterable[str],
) -> BatchSourceIntakeSummary:
    required = tuple(required_fields)
    results: list[TickerSourceResult] = []
    status_counts: Counter[SourceReadinessStatus] = Counter()
    missing_field_frequency: Counter[str] = Counter()

    for ticker in tuple(tickers):
        result = _process_ticker(ticker=ticker, provider=provider, required_fields=required)
        results.append(result)
        status_counts[result.readiness_status] += 1
        missing_field_frequency.update(result.missing_fields)

    success_count = sum(1 for result in results if result.intake_success)
    return BatchSourceIntakeSummary(
        provider_name=provider.name,
        required_fields=required,
        results=tuple(results),
        status_counts=dict(status_counts),
        missing_field_frequency=dict(missing_field_frequency),
        total_tickers=len(results),
        intake_success_count=success_count,
        intake_failure_count=len(results) - success_count,
    )


def _process_ticker(
    ticker: str,
    provider: SourceProvider,
    required_fields: tuple[str, ...],
) -> TickerSourceResult:
    try:
        response = provider.fetch_source(ticker)
    except SourceProviderError as error:
        status = readiness_from_response(response=None, required_fields=required_fields, error=error)
        return TickerSourceResult(
            ticker=ticker,
            provider_name=provider.name,
            readiness_status=status,
            missing_fields=required_fields,
            error=SourceIntakeError(error_type=type(error).__name__, message=str(error)),
            intake_success=False,
        )

    status = readiness_from_response(response=response, required_fields=required_fields)
    fields = response.fields if response is not None else {}
    available, missing = split_available_and_missing_fields(fields, required_fields)
    return TickerSourceResult(
        ticker=ticker,
        provider_name=provider.name,
        readiness_status=status,
        available_fields=available,
        missing_fields=missing,
        raw_evidence_present=response is not None and response.raw_evidence is not None,
        raw_evidence_summary=response.raw_evidence_summary if response is not None else None,
        normalized_data=dict(fields),
        intake_success=status in {SourceReadinessStatus.AVAILABLE, SourceReadinessStatus.PARTIAL},
    )
