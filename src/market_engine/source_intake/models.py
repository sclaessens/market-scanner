from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from market_engine.source_intake.readiness import SourceReadinessStatus


@dataclass(frozen=True)
class SourceIntakeError:
    error_type: str
    message: str


@dataclass(frozen=True)
class TickerSourceResult:
    ticker: str
    provider_name: str
    readiness_status: SourceReadinessStatus
    available_fields: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    raw_evidence_present: bool = False
    raw_evidence_summary: str | None = None
    normalized_data: dict[str, Any] = field(default_factory=dict)
    error: SourceIntakeError | None = None
    intake_success: bool = False


@dataclass(frozen=True)
class BatchSourceIntakeSummary:
    provider_name: str
    required_fields: tuple[str, ...]
    results: tuple[TickerSourceResult, ...]
    status_counts: dict[SourceReadinessStatus, int]
    missing_field_frequency: dict[str, int]
    total_tickers: int
    intake_success_count: int
    intake_failure_count: int
