from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.data_sources.common import (
    PrefillAudit,
    atomic_write_csv,
    clean_text,
    common_parser,
    ensure_output_path,
    normalize_ticker,
    parse_iso_date,
    print_audit,
    read_provider_export,
    reject_blank_tickers,
    reject_duplicate_identity,
    require_governed_output_path,
    require_columns,
    utc_timestamp,
    validate_no_forbidden_columns,
)

OUTPUT_PATH = Path("data/raw/fundamentals.csv")
STALE_THRESHOLD_DAYS = 120

IDENTITY_COLUMNS = ["ticker", "as_of_date"]
METADATA_COLUMNS = ["source_name", "source_last_updated", "report_period", "currency"]
NUMERIC_COLUMNS = [
    "revenue_growth_yoy",
    "eps_growth_yoy",
    "gross_margin",
    "operating_margin",
    "debt_to_equity",
]
BOOLEAN_COLUMNS = ["free_cash_flow_positive"]
DATA_COLUMNS = NUMERIC_COLUMNS + BOOLEAN_COLUMNS
OUTPUT_COLUMNS = IDENTITY_COLUMNS + METADATA_COLUMNS + DATA_COLUMNS


@dataclass(frozen=True)
class FundamentalsPrefillResult:
    output_df: pd.DataFrame
    audit: PrefillAudit


def _is_number(value: Any) -> bool:
    text = clean_text(value)
    if text == "":
        return False
    try:
        float(text)
    except ValueError:
        return False
    return True


def _is_boolean(value: Any) -> bool:
    return clean_text(value) in {"true", "false", "TRUE", "FALSE", "1", "0"}


def prepare_fundamentals_prefill(
    input_path: Path,
    output_path: Path = OUTPUT_PATH,
    source_label: str = "operator_provided",
    as_of_date: str | None = None,
    write: bool = False,
    allow_overwrite: bool = False,
) -> FundamentalsPrefillResult:
    run_timestamp = utc_timestamp()
    require_governed_output_path(output_path, ("data", "raw", "fundamentals.csv"))
    validation_date = parse_iso_date(as_of_date or run_timestamp[:10], "as_of_date")
    input_df = read_provider_export(input_path)
    validate_no_forbidden_columns(list(input_df.columns), "fundamentals provider export")
    require_columns(input_df, OUTPUT_COLUMNS, "fundamentals provider export")

    output_df = reject_blank_tickers(input_df[OUTPUT_COLUMNS].copy(), "fundamentals provider export")
    output_df["as_of_date"] = output_df["as_of_date"].map(lambda value: parse_iso_date(value, "as_of_date").isoformat())
    output_df["source_last_updated"] = output_df["source_last_updated"].map(
        lambda value: parse_iso_date(value, "source_last_updated").isoformat()
    )
    reject_duplicate_identity(output_df, IDENTITY_COLUMNS, "fundamentals provider export")

    invalid_rows = 0
    partial_rows = 0
    stale_rows = 0
    for _, row in output_df.iterrows():
        invalid_fields: list[str] = []
        partial_fields: list[str] = []
        if clean_text(row["source_name"]) == "":
            invalid_fields.append("source_name")
        if clean_text(row["report_period"]) == "":
            partial_fields.append("report_period")
        if clean_text(row["currency"]) == "":
            partial_fields.append("currency")
        for column in NUMERIC_COLUMNS:
            if clean_text(row[column]) == "":
                partial_fields.append(column)
            elif not _is_number(row[column]):
                invalid_fields.append(column)
        for column in BOOLEAN_COLUMNS:
            if clean_text(row[column]) == "":
                partial_fields.append(column)
            elif not _is_boolean(row[column]):
                invalid_fields.append(column)

        source_last_updated = parse_iso_date(row["source_last_updated"], "source_last_updated")
        freshness_days = (validation_date - source_last_updated).days
        if freshness_days < 0:
            invalid_fields.append("source_last_updated")
        elif freshness_days > STALE_THRESHOLD_DAYS:
            stale_rows += 1

        if invalid_fields:
            invalid_rows += 1
        elif partial_fields:
            partial_rows += 1

    if invalid_rows:
        validation_status = "FAILED"
        failure_reason = "fundamentals provider export contains invalid records"
    else:
        validation_status = "VALIDATED_WITH_DIAGNOSTICS" if partial_rows or stale_rows else "VALIDATED"
        failure_reason = ""

    if write and invalid_rows:
        raise ValueError(failure_reason)
    if write:
        ensure_output_path(output_path, allow_overwrite=allow_overwrite)
        atomic_write_csv(output_df, output_path)

    audit = PrefillAudit(
        run_timestamp=run_timestamp,
        provider_source_label=source_label,
        requested_ticker_count=int(output_df["ticker"].nunique()),
        matched_ticker_count=int(output_df["ticker"].nunique()),
        missing_ticker_count=0,
        written_row_count=int(len(output_df)) if write else 0,
        stale_row_count=int(stale_rows),
        invalid_row_count=int(invalid_rows),
        partial_row_count=int(partial_rows),
        duplicate_detection_result="PASSED",
        artifact_write_path=str(output_path),
        validation_status=validation_status,
        failure_reason=failure_reason,
        refresh_mode="provider_assisted_prefill",
        source_artifact_target=str(output_path),
        dry_run=not write,
    )
    return FundamentalsPrefillResult(output_df=output_df, audit=audit)


def main(argv: list[str] | None = None) -> int:
    parser = common_parser("Validate or prefill governed fundamentals source artifact.", OUTPUT_PATH)
    args = parser.parse_args(argv)
    write = bool(args.write and not args.dry_run)
    try:
        result = prepare_fundamentals_prefill(
            input_path=args.input,
            output_path=args.output,
            source_label=args.source_label,
            as_of_date=args.as_of_date,
            write=write,
            allow_overwrite=args.allow_overwrite,
        )
        print_audit(result.audit)
    except Exception as exc:
        audit = PrefillAudit(
            run_timestamp=utc_timestamp(),
            provider_source_label=args.source_label,
            requested_ticker_count=0,
            matched_ticker_count=0,
            missing_ticker_count=0,
            written_row_count=0,
            stale_row_count=0,
            invalid_row_count=0,
            partial_row_count=0,
            duplicate_detection_result="FAILED",
            artifact_write_path=str(args.output),
            validation_status="FAILED",
            failure_reason=str(exc),
            refresh_mode="provider_assisted_prefill",
            source_artifact_target=str(args.output),
            dry_run=not write,
        )
        print_audit(audit)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
