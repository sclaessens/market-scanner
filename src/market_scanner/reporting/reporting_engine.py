"""Communication-only v2 reporting scaffold."""

from __future__ import annotations

from market_scanner.decisions.decision_records import DecisionEngineResult
from market_scanner.reporting.report_records import ReportRecord, ReportResult


def build_communication_report(
    decision_result: DecisionEngineResult,
) -> ReportResult:
    records = tuple(
        ReportRecord(
            row_id=record.row_id,
            source_name=record.source_name,
            communicated_action=record.final_action,
            communicated_rationale=record.rationale,
            line=format_report_line(
                row_id=record.row_id,
                source_name=record.source_name,
                action=record.final_action.value,
                rationale=record.rationale,
            ),
        )
        for record in decision_result.decision_records
    )
    preserved_row_ids = tuple(record.row_id for record in records)

    if preserved_row_ids != decision_result.preserved_row_ids:
        raise ValueError("Reporting scaffold must preserve row identity.")

    return ReportResult(
        input_decision_count=decision_result.output_row_count,
        output_record_count=len(records),
        preserved_row_ids=preserved_row_ids,
        fixture_source_names=decision_result.fixture_source_names,
        layers_consumed=decision_result.layers_consumed,
        records=records,
        summary_lines=tuple(record.line for record in records),
    )


def format_report_line(
    *,
    row_id: str,
    source_name: str,
    action: str,
    rationale: str,
) -> str:
    return (
        f"row_id={row_id} | source={source_name} | "
        f"decision_state={action} | rationale={rationale}"
    )
