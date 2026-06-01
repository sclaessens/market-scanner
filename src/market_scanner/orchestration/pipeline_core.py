"""Minimal deterministic v2 pipeline scaffold."""

from __future__ import annotations

from collections.abc import Iterable

from market_scanner.shared.data_contracts import (
    APPROVED_FIXTURE_CONTRACTS,
    FixtureContract,
    read_fixture_rows,
)
from market_scanner.shared.records import (
    PipelineRecord,
    PipelineResult,
    PipelineStageResult,
)


FIXTURE_ROW_ID_COLUMNS = {
    "synthetic_universe_candidates": "candidate_id",
    "synthetic_portfolio_transactions": "transaction_id",
    "synthetic_source_data_readiness": "source_record_id",
}

MINIMAL_PIPELINE_LAYER_ORDER = (
    "input_contracts",
    "discovery",
    "validation",
    "context",
    "fundamentals",
    "timing",
    "portfolio",
)


def load_fixture_records(
    fixture_contracts: Iterable[FixtureContract] = APPROVED_FIXTURE_CONTRACTS,
) -> tuple[PipelineRecord, ...]:
    records: list[PipelineRecord] = []

    for contract in fixture_contracts:
        row_id_column = FIXTURE_ROW_ID_COLUMNS[contract.name]
        for row in read_fixture_rows(contract):
            records.append(
                PipelineRecord.from_fixture_row(
                    source_name=contract.name,
                    row_id_column=row_id_column,
                    row=row,
                )
            )

    return tuple(records)


def run_minimal_pipeline(
    records: Iterable[PipelineRecord],
) -> PipelineResult:
    current_records = tuple(records)
    initial_row_ids = tuple(record.row_id for record in current_records)
    stage_results: list[PipelineStageResult] = []

    for layer_name in MINIMAL_PIPELINE_LAYER_ORDER:
        current_records, stage_result = _pass_through_stage(
            layer_name=layer_name,
            records=current_records,
        )
        stage_results.append(stage_result)

    final_row_ids = tuple(record.row_id for record in current_records)
    if final_row_ids != initial_row_ids:
        raise ValueError("Pipeline scaffold must preserve row identity.")

    fixture_source_names = tuple(
        dict.fromkeys(record.source_name for record in current_records)
    )

    return PipelineResult(
        input_row_count=len(initial_row_ids),
        output_row_count=len(current_records),
        preserved_row_ids=final_row_ids,
        fixture_source_names=fixture_source_names,
        layers_visited=MINIMAL_PIPELINE_LAYER_ORDER,
        stage_results=tuple(stage_results),
        records=current_records,
    )


def run_minimal_pipeline_from_fixtures() -> PipelineResult:
    return run_minimal_pipeline(load_fixture_records())


def _pass_through_stage(
    *,
    layer_name: str,
    records: tuple[PipelineRecord, ...],
) -> tuple[tuple[PipelineRecord, ...], PipelineStageResult]:
    row_ids = tuple(record.row_id for record in records)
    return records, PipelineStageResult(
        stage_name=layer_name,
        input_row_count=len(records),
        output_row_count=len(records),
        preserved_row_ids=row_ids,
    )
