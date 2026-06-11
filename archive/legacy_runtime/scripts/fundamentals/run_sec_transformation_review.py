from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.fundamentals.build_history_intake import REQUIRED_COLUMNS
from scripts.fundamentals.sec_companyfacts_transform import transform_companyfacts_file
from scripts.fundamentals.sec_ticker_cik_index import (
    build_cik_coverage,
    build_ticker_cik_index,
    read_project_tickers,
)

REVIEW_COLUMNS = [
    "mapping_status",
    "transformation_status",
    "review_required",
    "review_reason",
    "missing_fields",
    "derived_fields_status",
]
OUTPUT_COLUMNS = REQUIRED_COLUMNS + REVIEW_COLUMNS


def _blank_history_row(ticker: str, *, source_freshness_date: str, extraction_date: str) -> dict[str, str]:
    row = {column: "" for column in REQUIRED_COLUMNS}
    row.update(
        {
            "ticker": ticker,
            "source_name": "SEC Company Facts",
            "source_freshness_date": source_freshness_date,
            "extraction_date": extraction_date,
        }
    )
    return row


def _notes(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _review_row(
    *,
    ticker: str,
    mapping_status: str,
    transformation_status: str,
    review_reason: str,
    source_freshness_date: str,
    extraction_date: str,
    missing_fields: str = "",
    derived_fields_status: str = "NOT_EVALUATED",
    source_reference: str = "",
) -> dict[str, str]:
    row = _blank_history_row(ticker, source_freshness_date=source_freshness_date, extraction_date=extraction_date)
    row["source_reference"] = source_reference
    row["notes"] = _notes({"review_reason": review_reason})
    row.update(
        {
            "mapping_status": mapping_status,
            "transformation_status": transformation_status,
            "review_required": "true",
            "review_reason": review_reason,
            "missing_fields": missing_fields,
            "derived_fields_status": derived_fields_status,
        }
    )
    return row


def _companyfacts_path(companyfacts_dir: str | Path, cik_padded: str) -> Path | None:
    directory = Path(companyfacts_dir)
    candidates = [
        directory / f"CIK{cik_padded}.json",
        directory / f"{cik_padded}.json",
        directory / f"{int(cik_padded)}.json",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _missing_fields(row: pd.Series) -> str:
    fields = [
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "diluted_eps",
        "total_debt",
        "total_equity",
        "free_cash_flow",
    ]
    return "|".join(field for field in fields if str(row.get(field, "")).strip() == "")


def _derived_fields_status(row: pd.Series) -> str:
    total_debt_present = str(row.get("total_debt", "")).strip() != ""
    free_cash_flow_present = str(row.get("free_cash_flow", "")).strip() != ""
    if total_debt_present and free_cash_flow_present:
        return "DERIVED_FIELDS_PRESENT"
    if total_debt_present or free_cash_flow_present:
        return "DERIVED_FIELDS_PARTIAL"
    return "DERIVED_FIELDS_MISSING_OR_REVIEW_REQUIRED"


def _review_reason(row: pd.Series) -> str:
    missing = _missing_fields(row)
    if not missing:
        return "transformed local Company Facts fixture with all reviewed fields present"
    return f"transformed local Company Facts fixture with missing fields: {missing}"


def build_sec_transformation_review(
    *,
    project_tickers_path: str | Path,
    ticker_cik_source_path: str | Path,
    companyfacts_dir: str | Path,
    source_freshness_date: str,
    extraction_date: str,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    project_tickers = read_project_tickers(project_tickers_path)
    index_df = build_ticker_cik_index(ticker_cik_source_path)
    coverage_df = build_cik_coverage(project_tickers, index_df)

    output_rows: list[dict[str, Any]] = []
    for _, coverage_row in coverage_df.iterrows():
        ticker = str(coverage_row["ticker"])
        mapping_status = str(coverage_row["mapping_status"])
        if mapping_status != "CIK_MATCHED":
            output_rows.append(
                _review_row(
                    ticker=ticker,
                    mapping_status=mapping_status,
                    transformation_status="CIK_REVIEW_REQUIRED",
                    review_reason=str(coverage_row["mapping_reason"]),
                    source_freshness_date=source_freshness_date,
                    extraction_date=extraction_date,
                    missing_fields="cik_padded",
                    source_reference=str(coverage_row["source_reference"]),
                )
            )
            continue

        cik_padded = str(coverage_row["cik_padded"])
        companyfacts_path = _companyfacts_path(companyfacts_dir, cik_padded)
        if companyfacts_path is None:
            output_rows.append(
                _review_row(
                    ticker=ticker,
                    mapping_status=mapping_status,
                    transformation_status="COMPANYFACTS_MISSING",
                    review_reason=f"local Company Facts JSON not found for CIK{cik_padded}",
                    source_freshness_date=source_freshness_date,
                    extraction_date=extraction_date,
                    missing_fields="companyfacts_json",
                    source_reference=str(coverage_row["source_reference"]),
                )
            )
            continue

        try:
            transformed = transform_companyfacts_file(
                companyfacts_path,
                ticker=ticker,
                cik=cik_padded,
                source_freshness_date=source_freshness_date,
                extraction_date=extraction_date,
            )
        except Exception as exc:
            output_rows.append(
                _review_row(
                    ticker=ticker,
                    mapping_status=mapping_status,
                    transformation_status="TRANSFORM_REVIEW_REQUIRED",
                    review_reason=f"local Company Facts transform failed: {exc}",
                    source_freshness_date=source_freshness_date,
                    extraction_date=extraction_date,
                    missing_fields="transformable_companyfacts",
                    source_reference=str(companyfacts_path),
                )
            )
            continue

        if transformed.empty:
            output_rows.append(
                _review_row(
                    ticker=ticker,
                    mapping_status=mapping_status,
                    transformation_status="NO_TRANSFORMABLE_FACTS",
                    review_reason="local Company Facts JSON produced no transformable periods",
                    source_freshness_date=source_freshness_date,
                    extraction_date=extraction_date,
                    missing_fields="transformable_facts",
                    source_reference=str(companyfacts_path),
                )
            )
            continue

        for _, transformed_row in transformed.iterrows():
            row = transformed_row.to_dict()
            missing = _missing_fields(transformed_row)
            derived_status = _derived_fields_status(transformed_row)
            review_required = "true" if missing else "false"
            row.update(
                {
                    "mapping_status": mapping_status,
                    "transformation_status": "TRANSFORMED",
                    "review_required": review_required,
                    "review_reason": _review_reason(transformed_row),
                    "missing_fields": missing,
                    "derived_fields_status": derived_status,
                }
            )
            output_rows.append(row)

    review_df = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        review_df.to_csv(output, index=False)
    return review_df


def summarize_review(review_df: pd.DataFrame, output_path: str | Path | None = None) -> dict[str, Any]:
    return {
        "status": "VALID",
        "row_count": int(len(review_df)),
        "review_required_count": int((review_df["review_required"] == "true").sum()),
        "output_path": str(output_path) if output_path is not None else "",
        "transformation_status_distribution": {
            str(key): int(value)
            for key, value in review_df["transformation_status"].value_counts().sort_index().items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run controlled local SEC transformation review from explicit files.")
    parser.add_argument("--project-tickers", required=True, type=Path, help="Local project ticker list.")
    parser.add_argument("--ticker-cik-source", required=True, type=Path, help="Local ticker/CIK source file.")
    parser.add_argument("--companyfacts-dir", required=True, type=Path, help="Local Company Facts-like JSON directory.")
    parser.add_argument("--output", type=Path, help="Explicit review CSV output path.")
    parser.add_argument("--source-freshness-date", required=True, help="Source freshness date to preserve in review rows.")
    parser.add_argument("--extraction-date", required=True, help="Extraction date to preserve in review rows.")
    parser.add_argument("--validate-only", action="store_true", help="Validate and summarize without writing output.")
    args = parser.parse_args(argv)

    if args.output is None and not args.validate_only:
        raise SystemExit("--output is required unless --validate-only is supplied.")

    output_path = None if args.validate_only else args.output
    review_df = build_sec_transformation_review(
        project_tickers_path=args.project_tickers,
        ticker_cik_source_path=args.ticker_cik_source,
        companyfacts_dir=args.companyfacts_dir,
        source_freshness_date=args.source_freshness_date,
        extraction_date=args.extraction_date,
        output_path=output_path,
    )
    print(json.dumps(summarize_review(review_df, output_path), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
