"""Review-only v2 Decision Engine authority scaffold."""

from __future__ import annotations

from market_scanner.decisions.decision_records import (
    DecisionEngineResult,
    DecisionRecord,
    DecisionState,
)
from market_scanner.shared.records import PipelineResult


REVIEW_ONLY_RATIONALE = "review_only_scaffold"


def run_decision_engine(pipeline_result: PipelineResult) -> DecisionEngineResult:
    decision_records = tuple(
        DecisionRecord(
            row_id=record.row_id,
            source_name=record.source_name,
            final_action=DecisionState.REVIEW,
            rationale=REVIEW_ONLY_RATIONALE,
        )
        for record in pipeline_result.records
    )
    preserved_row_ids = tuple(record.row_id for record in decision_records)

    if preserved_row_ids != pipeline_result.preserved_row_ids:
        raise ValueError("Decision Engine scaffold must preserve row identity.")

    return DecisionEngineResult(
        input_row_count=pipeline_result.output_row_count,
        output_row_count=len(decision_records),
        preserved_row_ids=preserved_row_ids,
        layers_consumed=pipeline_result.layers_visited,
        fixture_source_names=pipeline_result.fixture_source_names,
        decision_records=decision_records,
    )
