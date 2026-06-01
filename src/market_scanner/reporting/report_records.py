"""Communication-only reporting records for the v2 scaffold."""

from __future__ import annotations

from dataclasses import dataclass

from market_scanner.decisions.decision_records import DecisionState


@dataclass(frozen=True)
class ReportRecord:
    """In-memory communication record preserving Decision Engine output."""

    row_id: str
    source_name: str
    communicated_action: DecisionState
    communicated_rationale: str
    line: str


@dataclass(frozen=True)
class ReportResult:
    """In-memory communication result for Decision Engine output."""

    input_decision_count: int
    output_record_count: int
    preserved_row_ids: tuple[str, ...]
    fixture_source_names: tuple[str, ...]
    layers_consumed: tuple[str, ...]
    records: tuple[ReportRecord, ...]
    summary_lines: tuple[str, ...]
