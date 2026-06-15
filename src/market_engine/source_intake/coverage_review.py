from __future__ import annotations

from dataclasses import dataclass

from market_engine.source_intake.models import BatchSourceIntakeSummary
from market_engine.source_intake.readiness import SourceReadinessStatus


@dataclass(frozen=True)
class SourceCoverageReview:
    provider_name: str
    ticker_count: int
    readiness_counts: dict[str, int]
    missing_field_frequency: dict[str, int]
    provider_error_count: int
    unsupported_count: int
    invalid_ticker_count: int
    top_missing_fields: tuple[tuple[str, int], ...]
    failed_or_unsupported_tickers: tuple[str, ...]
    note: str


def build_source_coverage_review(summary: BatchSourceIntakeSummary) -> SourceCoverageReview:
    failed_statuses = {
        SourceReadinessStatus.PROVIDER_ERROR,
        SourceReadinessStatus.UNSUPPORTED,
        SourceReadinessStatus.INVALID_TICKER,
    }
    failed_tickers = tuple(
        result.ticker
        for result in summary.results
        if result.readiness_status in failed_statuses
    )
    return SourceCoverageReview(
        provider_name=summary.provider_name,
        ticker_count=summary.total_tickers,
        readiness_counts={
            status.value: count
            for status, count in sorted(summary.status_counts.items(), key=lambda item: item[0].value)
        },
        missing_field_frequency=dict(sorted(summary.missing_field_frequency.items())),
        provider_error_count=summary.status_counts.get(SourceReadinessStatus.PROVIDER_ERROR, 0),
        unsupported_count=summary.status_counts.get(SourceReadinessStatus.UNSUPPORTED, 0),
        invalid_ticker_count=summary.status_counts.get(SourceReadinessStatus.INVALID_TICKER, 0),
        top_missing_fields=tuple(
            sorted(
                summary.missing_field_frequency.items(),
                key=lambda item: (-item[1], item[0]),
            )[:5]
        ),
        failed_or_unsupported_tickers=failed_tickers,
        note="Source coverage evidence only. Not analysis.",
    )


def format_source_coverage_review(review: SourceCoverageReview) -> str:
    readiness = ", ".join(
        f"{status}={count}"
        for status, count in review.readiness_counts.items()
    )
    missing = ", ".join(
        f"{field}={count}"
        for field, count in review.missing_field_frequency.items()
    )
    failed = ", ".join(review.failed_or_unsupported_tickers)
    return (
        f"provider={review.provider_name}\n"
        f"tickers={review.ticker_count}\n"
        f"readiness={readiness or 'none'}\n"
        f"missing_fields={missing or 'none'}\n"
        f"provider_errors={review.provider_error_count}\n"
        f"unsupported={review.unsupported_count}\n"
        f"invalid_tickers={review.invalid_ticker_count}\n"
        f"failed_or_unsupported_tickers={failed or 'none'}\n"
        f"note={review.note}"
    )
