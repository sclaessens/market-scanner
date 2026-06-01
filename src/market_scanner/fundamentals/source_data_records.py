"""Source-data readiness records for the v2 fundamentals scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SourceDataStatus(StrEnum):
    """Descriptive source-data readiness states only."""

    AVAILABLE = "AVAILABLE"
    MISSING = "MISSING"
    PARTIAL = "PARTIAL"
    STALE = "STALE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


@dataclass(frozen=True)
class SourceDataReadinessRecord:
    """Descriptive source-data readiness metadata for one source row."""

    row_id: str
    symbol: str
    source_name: str
    metric_name: str
    metric_value: str
    metric_unit: str
    as_of_date: str
    status: SourceDataStatus
    missing_fields: tuple[str, ...]
    missing_value_policy: str
    review_required_reason: str
    provenance_fixture_name: str


@dataclass(frozen=True)
class SourceDataReadinessResult:
    """Deterministic source-data readiness result."""

    input_row_count: int
    output_row_count: int
    preserved_row_ids: tuple[str, ...]
    provenance_fixture_name: str
    records: tuple[SourceDataReadinessRecord, ...]
