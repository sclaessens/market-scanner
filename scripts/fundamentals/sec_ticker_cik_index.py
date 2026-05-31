from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

MAPPING_STATUSES = {
    "CIK_MATCHED",
    "CIK_MISSING",
    "CIK_AMBIGUOUS",
    "CIK_REVIEW_REQUIRED",
    "CIK_NOT_SEC_REPORTER",
}

INDEX_COLUMNS = [
    "ticker",
    "cik",
    "cik_padded",
    "company_name",
    "exchange",
    "mapping_status",
    "mapping_reason",
    "source_reference",
]

COVERAGE_COLUMNS = [
    "ticker",
    "cik",
    "cik_padded",
    "company_name",
    "mapping_status",
    "mapping_reason",
    "source_reference",
    "review_required",
]


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalize_ticker(value: Any) -> str:
    return clean_text(value).upper()


def normalize_cik(value: Any) -> str:
    text = clean_text(value)
    if text.endswith(".0"):
        text = text[:-2]
    if text == "":
        raise ValueError("CIK is missing.")
    if not text.isdigit():
        raise ValueError(f"CIK must contain only digits: {text}")
    if len(text) > 10:
        raise ValueError(f"CIK must not exceed 10 digits: {text}")
    if int(text) <= 0:
        raise ValueError(f"CIK must be a positive integer: {text}")
    return text.zfill(10)


def _extract_source_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if "rows" in payload and isinstance(payload["rows"], list):
            return [row for row in payload["rows"] if isinstance(row, dict)]
        if all(isinstance(value, dict) for value in payload.values()):
            return [value for _, value in sorted(payload.items(), key=lambda item: str(item[0]))]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError("SEC ticker/CIK source JSON must contain row objects.")


def _read_source_rows(path: str | Path) -> list[dict[str, Any]]:
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"SEC ticker/CIK source not found: {source_path}")
    if source_path.suffix.lower() == ".json":
        return _extract_source_rows(json.loads(source_path.read_text(encoding="utf-8")))
    try:
        df = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"SEC ticker/CIK source is empty: {source_path}") from exc
    return df.to_dict(orient="records")


def _row_value(row: dict[str, Any], candidates: tuple[str, ...]) -> str:
    lowered = {key.lower(): value for key, value in row.items()}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return clean_text(lowered[candidate.lower()])
    return ""


def build_ticker_cik_index(source_path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    source_reference = str(source_path)
    for row in _read_source_rows(source_path):
        ticker = normalize_ticker(_row_value(row, ("ticker", "symbol")))
        company_name = _row_value(row, ("company_name", "title", "name"))
        exchange = _row_value(row, ("exchange", "exchange_name"))
        raw_cik = _row_value(row, ("cik", "cik_str", "cik_str_int"))

        mapping_status = "CIK_MATCHED"
        mapping_reason = "ticker and CIK source row available"
        cik_padded = ""
        cik = clean_text(raw_cik)

        if ticker == "":
            mapping_status = "CIK_REVIEW_REQUIRED"
            mapping_reason = "ticker is missing"
        elif raw_cik == "":
            mapping_status = "CIK_REVIEW_REQUIRED"
            mapping_reason = "CIK is missing in source row"
        else:
            try:
                cik_padded = normalize_cik(raw_cik)
                cik = str(int(cik_padded))
            except ValueError as exc:
                mapping_status = "CIK_REVIEW_REQUIRED"
                mapping_reason = str(exc)

        rows.append(
            {
                "ticker": ticker,
                "cik": cik,
                "cik_padded": cik_padded,
                "company_name": company_name,
                "exchange": exchange,
                "mapping_status": mapping_status,
                "mapping_reason": mapping_reason,
                "source_reference": source_reference,
            }
        )

    return pd.DataFrame(rows, columns=INDEX_COLUMNS)


def read_project_tickers(path: str | Path) -> list[str]:
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"project ticker input not found: {source_path}")
    if source_path.suffix.lower() == ".json":
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            values = payload.get("tickers", payload.get("rows", []))
        else:
            values = payload
        if not isinstance(values, list):
            raise ValueError("project ticker JSON must contain a list or a 'tickers' list.")
        tickers = [normalize_ticker(row.get("ticker", "")) if isinstance(row, dict) else normalize_ticker(row) for row in values]
    elif source_path.suffix.lower() == ".csv":
        df = pd.read_csv(source_path, dtype=str, keep_default_na=False)
        if "ticker" not in df.columns:
            raise ValueError("project ticker CSV must contain a ticker column.")
        tickers = [normalize_ticker(value) for value in df["ticker"].tolist()]
    else:
        tickers = [normalize_ticker(line) for line in source_path.read_text(encoding="utf-8").splitlines()]

    return [ticker for ticker in tickers if ticker]


def _coverage_row_for_ticker(ticker: str, matches: pd.DataFrame) -> dict[str, str]:
    if matches.empty:
        return {
            "ticker": ticker,
            "cik": "",
            "cik_padded": "",
            "company_name": "",
            "mapping_status": "CIK_MISSING",
            "mapping_reason": "ticker not found in SEC ticker/CIK index",
            "source_reference": "",
            "review_required": "true",
        }

    if len(matches) > 1:
        return {
            "ticker": ticker,
            "cik": "|".join(matches["cik"].astype(str).tolist()),
            "cik_padded": "|".join(matches["cik_padded"].astype(str).tolist()),
            "company_name": "|".join(matches["company_name"].astype(str).tolist()),
            "mapping_status": "CIK_AMBIGUOUS",
            "mapping_reason": "multiple SEC ticker/CIK rows matched ticker",
            "source_reference": "|".join(sorted(set(matches["source_reference"].astype(str).tolist()))),
            "review_required": "true",
        }

    match = matches.iloc[0]
    status = clean_text(match["mapping_status"])
    review_required = "false" if status == "CIK_MATCHED" else "true"
    return {
        "ticker": ticker,
        "cik": clean_text(match["cik"]),
        "cik_padded": clean_text(match["cik_padded"]),
        "company_name": clean_text(match["company_name"]),
        "mapping_status": status,
        "mapping_reason": clean_text(match["mapping_reason"]),
        "source_reference": clean_text(match["source_reference"]),
        "review_required": review_required,
    }


def build_cik_coverage(project_tickers: list[str], ticker_cik_index: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in INDEX_COLUMNS if column not in ticker_cik_index.columns]
    if missing_columns:
        raise ValueError(f"ticker CIK index is missing required columns: {missing_columns}")

    normalized_index = ticker_cik_index.copy()
    normalized_index["ticker"] = normalized_index["ticker"].map(normalize_ticker)
    rows = []
    for ticker in project_tickers:
        normalized_ticker = normalize_ticker(ticker)
        matches = normalized_index[normalized_index["ticker"] == normalized_ticker]
        rows.append(_coverage_row_for_ticker(normalized_ticker, matches))
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def build_coverage_from_files(
    *,
    ticker_source_path: str | Path,
    project_tickers_path: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    index_df = build_ticker_cik_index(ticker_source_path)
    project_tickers = read_project_tickers(project_tickers_path)
    coverage_df = build_cik_coverage(project_tickers, index_df)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        coverage_df.to_csv(output, index=False)
    return coverage_df


def summarize_index(index_df: pd.DataFrame) -> dict[str, int]:
    return {
        "row_count": int(len(index_df)),
        "matched_count": int((index_df["mapping_status"] == "CIK_MATCHED").sum()),
        "review_required_count": int((index_df["mapping_status"] == "CIK_REVIEW_REQUIRED").sum()),
    }


def summarize_coverage(coverage_df: pd.DataFrame) -> dict[str, int]:
    return {
        "row_count": int(len(coverage_df)),
        "matched_count": int((coverage_df["mapping_status"] == "CIK_MATCHED").sum()),
        "missing_count": int((coverage_df["mapping_status"] == "CIK_MISSING").sum()),
        "ambiguous_count": int((coverage_df["mapping_status"] == "CIK_AMBIGUOUS").sum()),
        "review_required_count": int((coverage_df["review_required"] == "true").sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SEC ticker/CIK index and project ticker coverage.")
    parser.add_argument("--ticker-source", required=True, type=Path, help="Local SEC-supported ticker/CIK source file.")
    parser.add_argument("--project-tickers", type=Path, help="Project ticker list file for coverage analysis.")
    parser.add_argument("--output", type=Path, help="Optional generated coverage CSV output path.")
    parser.add_argument("--validate-only", action="store_true", help="Validate/index the ticker source without coverage output.")
    args = parser.parse_args()

    index_df = build_ticker_cik_index(args.ticker_source)
    if args.validate_only:
        print(json.dumps(summarize_index(index_df), sort_keys=True, indent=2))
        return 0
    if args.project_tickers is None:
        raise SystemExit("--project-tickers is required unless --validate-only is supplied.")

    coverage_df = build_cik_coverage(read_project_tickers(args.project_tickers), index_df)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        coverage_df.to_csv(args.output, index=False)
    print(json.dumps(summarize_coverage(coverage_df), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
