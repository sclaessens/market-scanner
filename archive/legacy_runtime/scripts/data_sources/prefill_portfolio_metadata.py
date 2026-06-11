from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.data_sources.common import (
    PrefillAudit,
    atomic_write_csv,
    clean_text,
    common_parser,
    ensure_output_path,
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

OUTPUT_PATH = Path("data/portfolio/portfolio_metadata.csv")
STALE_THRESHOLD_DAYS = 365
ACCEPTED_ASSET_CLASSES = {"Equity", "ETF", "Cash", "Other"}

REQUIRED_COLUMNS = [
    "ticker",
    "sector",
    "industry",
    "asset_class",
    "currency",
    "metadata_source",
    "metadata_last_updated",
]
OPTIONAL_COLUMNS = [
    "sector_taxonomy",
    "industry_group",
    "country",
    "region",
    "exchange",
    "notes",
]
OUTPUT_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


@dataclass(frozen=True)
class PortfolioMetadataPrefillResult:
    output_df: pd.DataFrame
    audit: PrefillAudit


def prepare_portfolio_metadata_prefill(
    input_path: Path,
    output_path: Path = OUTPUT_PATH,
    source_label: str = "operator_provided",
    as_of_date: str | None = None,
    write: bool = False,
    allow_overwrite: bool = False,
) -> PortfolioMetadataPrefillResult:
    run_timestamp = utc_timestamp()
    require_governed_output_path(output_path, ("data", "portfolio", "portfolio_metadata.csv"))
    validation_date = parse_iso_date(as_of_date or run_timestamp[:10], "as_of_date")
    input_df = read_provider_export(input_path)
    validate_no_forbidden_columns(list(input_df.columns), "portfolio metadata provider export")
    require_columns(input_df, REQUIRED_COLUMNS, "portfolio metadata provider export")

    output_columns = [column for column in OUTPUT_COLUMNS if column in input_df.columns]
    output_df = reject_blank_tickers(input_df[output_columns].copy(), "portfolio metadata provider export")
    for column in OPTIONAL_COLUMNS:
        if column not in output_df.columns:
            output_df[column] = ""
    output_df = output_df[OUTPUT_COLUMNS]
    reject_duplicate_identity(output_df, ["ticker"], "portfolio metadata provider export")

    invalid_rows = 0
    partial_rows = 0
    stale_rows = 0
    for _, row in output_df.iterrows():
        invalid_fields: list[str] = []
        partial_fields: list[str] = []
        for column in ["sector", "industry", "currency", "metadata_source"]:
            if clean_text(row[column]) == "":
                partial_fields.append(column)
        if clean_text(row["asset_class"]) == "":
            partial_fields.append("asset_class")
        elif clean_text(row["asset_class"]) not in ACCEPTED_ASSET_CLASSES:
            invalid_fields.append("asset_class")

        metadata_last_updated = parse_iso_date(row["metadata_last_updated"], "metadata_last_updated")
        freshness_days = (validation_date - metadata_last_updated).days
        if freshness_days < 0:
            invalid_fields.append("metadata_last_updated")
        elif freshness_days > STALE_THRESHOLD_DAYS:
            stale_rows += 1

        if invalid_fields:
            invalid_rows += 1
        elif partial_fields:
            partial_rows += 1

    if invalid_rows:
        validation_status = "FAILED"
        failure_reason = "portfolio metadata provider export contains invalid records"
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
    return PortfolioMetadataPrefillResult(output_df=output_df, audit=audit)


def main(argv: list[str] | None = None) -> int:
    parser = common_parser("Validate or prefill governed portfolio metadata source artifact.", OUTPUT_PATH)
    args = parser.parse_args(argv)
    write = bool(args.write and not args.dry_run)
    try:
        result = prepare_portfolio_metadata_prefill(
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
