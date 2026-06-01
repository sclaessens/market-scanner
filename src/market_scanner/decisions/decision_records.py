"""Decision Engine authority records for the v2 scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DecisionState(StrEnum):
    """RESET-6 review-only placeholder state."""

    REVIEW = "REVIEW"


@dataclass(frozen=True)
class DecisionRecord:
    """Review-only Decision Engine output for one pipeline record."""

    row_id: str
    source_name: str
    final_action: DecisionState
    rationale: str


@dataclass(frozen=True)
class DecisionEngineResult:
    """Metadata-only result for the RESET-6 Decision Engine scaffold."""

    input_row_count: int
    output_row_count: int
    preserved_row_ids: tuple[str, ...]
    layers_consumed: tuple[str, ...]
    fixture_source_names: tuple[str, ...]
    decision_records: tuple[DecisionRecord, ...]
