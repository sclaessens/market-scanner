"""Neutral record types for the v2 pipeline scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class PipelineRecord:
    """Contract-safe fixture row wrapper with stable identity."""

    source_name: str
    row_id: str
    values: Mapping[str, str]

    @classmethod
    def from_fixture_row(
        cls,
        *,
        source_name: str,
        row_id_column: str,
        row: Mapping[str, str],
    ) -> "PipelineRecord":
        return cls(
            source_name=source_name,
            row_id=row[row_id_column],
            values=MappingProxyType(dict(row)),
        )


@dataclass(frozen=True)
class PipelineStageResult:
    """Metadata-only result for one pass-through pipeline stage."""

    stage_name: str
    input_row_count: int
    output_row_count: int
    preserved_row_ids: tuple[str, ...]


@dataclass(frozen=True)
class PipelineResult:
    """Metadata-only result for the minimal v2 pipeline scaffold."""

    input_row_count: int
    output_row_count: int
    preserved_row_ids: tuple[str, ...]
    fixture_source_names: tuple[str, ...]
    layers_visited: tuple[str, ...]
    stage_results: tuple[PipelineStageResult, ...]
    records: tuple[PipelineRecord, ...]
