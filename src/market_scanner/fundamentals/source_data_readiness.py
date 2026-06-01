"""Fixture-only source-data readiness scaffold."""

from __future__ import annotations

from market_scanner.fundamentals.source_data_records import (
    SourceDataReadinessRecord,
    SourceDataReadinessResult,
    SourceDataStatus,
)
from market_scanner.shared.data_contracts import (
    APPROVED_FIXTURE_CONTRACTS,
    FixtureContract,
    read_fixture_rows,
)


SOURCE_DATA_READINESS_FIXTURE_NAME = "synthetic_source_data_readiness"

_STATUS_BY_FIXTURE_VALUE = {
    "available": SourceDataStatus.AVAILABLE,
    "missing": SourceDataStatus.MISSING,
    "partial": SourceDataStatus.PARTIAL,
    "stale": SourceDataStatus.STALE,
    "review_required": SourceDataStatus.REVIEW_REQUIRED,
}


def load_source_data_readiness_fixture() -> tuple[dict[str, str], ...]:
    contract = _source_data_readiness_contract()
    return tuple(read_fixture_rows(contract))


def evaluate_source_data_readiness(
    rows: tuple[dict[str, str], ...] | None = None,
) -> SourceDataReadinessResult:
    fixture_rows = load_source_data_readiness_fixture() if rows is None else rows
    records = tuple(_to_readiness_record(row) for row in fixture_rows)
    preserved_row_ids = tuple(record.row_id for record in records)

    return SourceDataReadinessResult(
        input_row_count=len(fixture_rows),
        output_row_count=len(records),
        preserved_row_ids=preserved_row_ids,
        provenance_fixture_name=SOURCE_DATA_READINESS_FIXTURE_NAME,
        records=records,
    )


def _source_data_readiness_contract() -> FixtureContract:
    return next(
        contract
        for contract in APPROVED_FIXTURE_CONTRACTS
        if contract.name == SOURCE_DATA_READINESS_FIXTURE_NAME
    )


def _to_readiness_record(row: dict[str, str]) -> SourceDataReadinessRecord:
    missing_fields = tuple(
        field_name
        for field_name in ("metric_value",)
        if row[field_name] == ""
    )

    return SourceDataReadinessRecord(
        row_id=row["source_record_id"],
        symbol=row["symbol"],
        source_name=row["source_name"],
        metric_name=row["metric_name"],
        metric_value=row["metric_value"],
        metric_unit=row["metric_unit"],
        as_of_date=row["as_of_date"],
        status=_STATUS_BY_FIXTURE_VALUE[row["readiness_state"]],
        missing_fields=missing_fields,
        missing_value_policy=row["missing_value_policy"],
        review_required_reason=row["review_required_reason"],
        provenance_fixture_name=SOURCE_DATA_READINESS_FIXTURE_NAME,
    )
