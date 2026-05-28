from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = [
    "ticker",
    "fiscal_year",
    "fiscal_period",
    "period_end_date",
    "report_date",
    "currency",
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "diluted_eps",
    "total_debt",
    "total_equity",
    "free_cash_flow",
    "source_name",
    "source_reference",
    "source_freshness_date",
    "extraction_date",
    "notes",
]

KEY_COLUMNS = ["ticker", "fiscal_year", "fiscal_period"]
IDENTITY_AND_SOURCE_VALUE_COLUMNS = [
    "ticker",
    "fiscal_year",
    "fiscal_period",
    "currency",
    "source_name",
    "source_reference",
]
DATE_COLUMNS = [
    "period_end_date",
    "report_date",
    "source_freshness_date",
    "extraction_date",
]
NUMERIC_COLUMNS = [
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "diluted_eps",
    "total_debt",
    "total_equity",
    "free_cash_flow",
]
SUPPORTED_FISCAL_PERIODS = {"FY", "Q1", "Q2", "Q3", "Q4", "TTM"}
FORBIDDEN_COLUMNS = {
    "buy",
    "sell",
    "action",
    "final_action",
    "decision",
    "allocation",
    "position_size",
    "urgency",
    "conviction",
    "tradeability",
    "eligible",
    "eligibility",
    "ranking",
    "score",
    "priority",
    "entry",
    "stop",
    "target",
}


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _invalid_value_rows(df: pd.DataFrame, column: str) -> list[int]:
    mask = df[column].map(_clean_text) == ""
    return [int(index) + 2 for index in df.index[mask]]


def _invalid_fiscal_year_rows(df: pd.DataFrame) -> list[int]:
    invalid_rows: list[int] = []
    for index, value in df["fiscal_year"].items():
        text = _clean_text(value)
        try:
            year = int(text)
        except ValueError:
            invalid_rows.append(int(index) + 2)
            continue
        if year < 1900 or year > 2200:
            invalid_rows.append(int(index) + 2)
    return invalid_rows


def _invalid_fiscal_period_rows(df: pd.DataFrame) -> list[int]:
    invalid_rows: list[int] = []
    for index, value in df["fiscal_period"].items():
        text = _clean_text(value).upper()
        if text == "" or text not in SUPPORTED_FISCAL_PERIODS:
            invalid_rows.append(int(index) + 2)
    return invalid_rows


def _invalid_date_rows(df: pd.DataFrame, column: str) -> list[int]:
    invalid_rows: list[int] = []
    values = df[column].map(_clean_text)
    for index, value in values.items():
        if value == "":
            continue
        if pd.isna(pd.to_datetime(value, errors="coerce")):
            invalid_rows.append(int(index) + 2)
    return invalid_rows


def _invalid_numeric_rows(df: pd.DataFrame, column: str) -> list[int]:
    invalid_rows: list[int] = []
    values = df[column].map(_clean_text)
    for index, value in values.items():
        if value == "":
            continue
        try:
            float(value)
        except ValueError:
            invalid_rows.append(int(index) + 2)
    return invalid_rows


def _duplicate_key_count(df: pd.DataFrame) -> int:
    normalized_keys = df[KEY_COLUMNS].copy()
    normalized_keys["ticker"] = normalized_keys["ticker"].map(lambda value: _clean_text(value).upper())
    normalized_keys["fiscal_year"] = normalized_keys["fiscal_year"].map(_clean_text)
    normalized_keys["fiscal_period"] = normalized_keys["fiscal_period"].map(lambda value: _clean_text(value).upper())
    return int(normalized_keys.duplicated(subset=KEY_COLUMNS, keep=False).sum())


def _base_result(row_count: int) -> dict[str, Any]:
    return {
        "status": "VALID",
        "row_count": row_count,
        "issue_count": 0,
        "missing_required_columns": [],
        "forbidden_columns": [],
        "duplicate_key_count": 0,
        "invalid_fiscal_year_rows": [],
        "invalid_fiscal_period_rows": [],
        "invalid_date_columns": {},
        "invalid_numeric_columns": {},
        "missing_required_value_columns": {},
    }


def _invalid_result(row_count: int, **updates: Any) -> dict[str, Any]:
    result = _base_result(row_count)
    result["status"] = "INVALID"
    result.update(updates)
    result["issue_count"] = (
        len(result["missing_required_columns"])
        + len(result["forbidden_columns"])
        + int(result["duplicate_key_count"])
        + len(result["invalid_fiscal_year_rows"])
        + len(result["invalid_fiscal_period_rows"])
        + sum(len(rows) for rows in result["invalid_date_columns"].values())
        + sum(len(rows) for rows in result["invalid_numeric_columns"].values())
        + sum(len(rows) for rows in result["missing_required_value_columns"].values())
    )
    return result


def validate_fundamentals_history(input_path: str | Path) -> dict[str, Any]:
    path = Path(input_path)
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return _invalid_result(0, missing_required_columns=REQUIRED_COLUMNS)

    missing_required_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_required_columns:
        return _invalid_result(len(df), missing_required_columns=missing_required_columns)

    forbidden_columns = sorted(column for column in df.columns if column.strip().lower() in FORBIDDEN_COLUMNS)
    if forbidden_columns:
        return _invalid_result(len(df), forbidden_columns=forbidden_columns)

    duplicate_key_count = _duplicate_key_count(df)
    if duplicate_key_count:
        return _invalid_result(len(df), duplicate_key_count=duplicate_key_count)

    missing_required_value_columns = {
        column: rows for column in IDENTITY_AND_SOURCE_VALUE_COLUMNS if (rows := _invalid_value_rows(df, column))
    }
    invalid_fiscal_year_rows = _invalid_fiscal_year_rows(df)
    invalid_fiscal_period_rows = _invalid_fiscal_period_rows(df)
    invalid_date_columns = {column: rows for column in DATE_COLUMNS if (rows := _invalid_date_rows(df, column))}
    invalid_numeric_columns = {column: rows for column in NUMERIC_COLUMNS if (rows := _invalid_numeric_rows(df, column))}

    if (
        missing_required_value_columns
        or invalid_fiscal_year_rows
        or invalid_fiscal_period_rows
        or invalid_date_columns
        or invalid_numeric_columns
    ):
        return _invalid_result(
            len(df),
            invalid_fiscal_year_rows=invalid_fiscal_year_rows,
            invalid_fiscal_period_rows=invalid_fiscal_period_rows,
            invalid_date_columns=invalid_date_columns,
            invalid_numeric_columns=invalid_numeric_columns,
            missing_required_value_columns=missing_required_value_columns,
        )

    return _base_result(len(df))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate raw fundamentals history schema and values.")
    parser.add_argument("input_path", help="Path to a fundamentals_history.csv-style input file.")
    parser.add_argument(
        "--report-path",
        help="Optional path for a JSON validation report. No files are written unless this is supplied.",
    )
    args = parser.parse_args()

    result = validate_fundamentals_history(args.input_path)
    rendered = json.dumps(result, sort_keys=True, indent=2)
    print(rendered)

    if args.report_path:
        Path(args.report_path).write_text(rendered + "\n", encoding="utf-8")

    return 0 if result["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
