"""Synthetic in-memory adapter for fundamentals raw-to-normalized flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from market_scanner.fundamentals.fundamental_contracts import (
    SourceDataReadinessState,
)
from market_scanner.fundamentals.fundamentals_normalization_contracts import (
    FundamentalsNormalizationIssue,
    validate_normalized_fundamentals_shape,
    validate_raw_source_shape,
    validate_source_readiness_shape,
)


@dataclass(frozen=True)
class SyntheticRawFundamentalRecord:
    """Synthetic raw source evidence supplied explicitly by tests or callers."""

    source_provider: str
    source_record_id: str
    ticker: str
    fiscal_period: str
    fiscal_year: str
    captured_at: str
    source_reference: str
    raw_payload_hash: str
    metrics: Mapping[str, object]
    metric_units: Mapping[str, str]
    currency: str = ""
    stale_metric_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class SyntheticNormalizedFundamentalRecord:
    """Program-ready synthetic metric record with raw source traceability."""

    ticker: str
    fiscal_period: str
    fiscal_year: str
    metric_name: str
    metric_value: object
    metric_unit: str
    currency: str
    normalized_at: str
    source_provider: str
    source_reference: str
    source_record_identity: str


@dataclass(frozen=True)
class SyntheticSourceDataReadinessRecord:
    """Synthetic source-data readiness metadata, not investment quality."""

    ticker: str
    fiscal_period: str
    fiscal_year: str
    readiness_state: str
    source_data_status: str
    missing_fundamentals_count: int
    partial_data_count: int
    stale_data_count: int
    source_reference: str


@dataclass(frozen=True)
class SyntheticFundamentalsNormalizationResult:
    """Result of synthetic raw-to-normalized mapping."""

    normalized_records: tuple[SyntheticNormalizedFundamentalRecord, ...]
    readiness_records: tuple[SyntheticSourceDataReadinessRecord, ...]
    issues: tuple[FundamentalsNormalizationIssue, ...]


def normalize_synthetic_fundamentals(
    raw_records: Sequence[SyntheticRawFundamentalRecord],
) -> SyntheticFundamentalsNormalizationResult:
    """Map supplied synthetic raw records to normalized and readiness records."""

    normalized_records: list[SyntheticNormalizedFundamentalRecord] = []
    readiness_records: list[SyntheticSourceDataReadinessRecord] = []
    issues: list[FundamentalsNormalizationIssue] = []

    for raw_record in raw_records:
        raw_shape = _raw_record_shape(raw_record)
        raw_issues = validate_raw_source_shape(raw_shape)
        issues.extend(raw_issues)

        for metric_name, metric_value in raw_record.metrics.items():
            normalized_record = SyntheticNormalizedFundamentalRecord(
                ticker=raw_record.ticker,
                fiscal_period=raw_record.fiscal_period,
                fiscal_year=raw_record.fiscal_year,
                metric_name=metric_name,
                metric_value=metric_value,
                metric_unit=raw_record.metric_units.get(metric_name, ""),
                currency=raw_record.currency,
                normalized_at=raw_record.captured_at,
                source_provider=raw_record.source_provider,
                source_reference=raw_record.source_reference,
                source_record_identity=raw_record.source_record_id,
            )
            normalized_records.append(normalized_record)
            issues.extend(
                validate_normalized_fundamentals_shape(
                    _normalized_record_shape(normalized_record)
                )
            )

        readiness_record = _readiness_record_for(raw_record, has_raw_issues=bool(raw_issues))
        readiness_records.append(readiness_record)
        issues.extend(
            validate_source_readiness_shape(_readiness_record_shape(readiness_record))
        )

    return SyntheticFundamentalsNormalizationResult(
        normalized_records=tuple(normalized_records),
        readiness_records=tuple(readiness_records),
        issues=tuple(issues),
    )


def _raw_record_shape(record: SyntheticRawFundamentalRecord) -> dict[str, object]:
    return {
        "source_provider": record.source_provider,
        "source_record_id": record.source_record_id,
        "ticker": record.ticker,
        "fiscal_period": record.fiscal_period,
        "fiscal_year": record.fiscal_year,
        "captured_at": record.captured_at,
        "source_reference": record.source_reference,
        "raw_payload_hash": record.raw_payload_hash,
    }


def _normalized_record_shape(
    record: SyntheticNormalizedFundamentalRecord,
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


def _readiness_record_shape(
    record: SyntheticSourceDataReadinessRecord,
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


def _readiness_record_for(
    raw_record: SyntheticRawFundamentalRecord,
    *,
    has_raw_issues: bool,
) -> SyntheticSourceDataReadinessRecord:
    missing_metric_names = tuple(
        metric_name
        for metric_name, metric_value in raw_record.metrics.items()
        if metric_value is None or metric_value == ""
    )
    stale_metric_names = tuple(raw_record.stale_metric_names)

    readiness_state = _readiness_state_for(
        metric_count=len(raw_record.metrics),
        missing_count=len(missing_metric_names),
        stale_count=len(stale_metric_names),
        has_raw_issues=has_raw_issues,
    )

    return SyntheticSourceDataReadinessRecord(
        ticker=raw_record.ticker,
        fiscal_period=raw_record.fiscal_period,
        fiscal_year=raw_record.fiscal_year,
        readiness_state=readiness_state.value,
        source_data_status=readiness_state.value,
        missing_fundamentals_count=len(missing_metric_names),
        partial_data_count=len(missing_metric_names),
        stale_data_count=len(stale_metric_names),
        source_reference=raw_record.source_reference,
    )


def _readiness_state_for(
    *,
    metric_count: int,
    missing_count: int,
    stale_count: int,
    has_raw_issues: bool,
) -> SourceDataReadinessState:
    if has_raw_issues:
        return SourceDataReadinessState.INVALID
    if metric_count == 0:
        return SourceDataReadinessState.SOURCE_MISSING
    if missing_count == metric_count:
        return SourceDataReadinessState.MISSING
    if missing_count > 0:
        return SourceDataReadinessState.PARTIAL
    if stale_count > 0:
        return SourceDataReadinessState.STALE
    return SourceDataReadinessState.AVAILABLE
