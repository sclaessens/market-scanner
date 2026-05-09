from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

CONTEXT_PATH = Path("data/processed/context_strength.csv")
OUTPUT_PATH = Path("data/processed/fundamental_quality.csv")
LOG_PATH = Path("data/logs/fundamental_layer_log.csv")

INPUT_REQUIRED_COLUMNS = ["ticker", "date"]
OUTPUT_COLUMNS = [
    "ticker",
    "date",
    "quality_state",
    "quality_reason",
    "profitability_profile",
    "balance_sheet_profile",
    "earnings_quality_profile",
    "capital_efficiency_profile",
    "cashflow_profile",
    "stability_profile",
    "quality_metadata_status",
    "source_data_status",
    "source_timestamp",
    "generated_at",
]
LOG_COLUMNS = [
    "generated_at",
    "input_row_count",
    "output_row_count",
    "unique_ticker_date_count",
    "duplicate_ticker_date_count",
    "missing_fundamentals_count",
    "partial_data_count",
    "stale_data_count",
    "quality_state_distribution",
    "quality_metadata_status_distribution",
    "source_data_status_distribution",
]


def _load_required_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{label} is empty: {path}") from exc
    if df.empty:
        raise ValueError(f"{label} is empty: {path}")
    return df


def _validate_columns(df: pd.DataFrame, required_columns: list[str], label: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _validate_required_values(df: pd.DataFrame, label: str) -> None:
    for column in INPUT_REQUIRED_COLUMNS:
        missing_mask = df[column].isna() | (df[column].astype(str).str.strip() == "")
        if missing_mask.any():
            rows = df.loc[missing_mask, INPUT_REQUIRED_COLUMNS].to_dict(orient="records")
            raise ValueError(f"{label} contains missing {column} values: {rows}")


def _count_duplicate_keys(df: pd.DataFrame) -> int:
    return int(df.duplicated(subset=INPUT_REQUIRED_COLUMNS, keep=False).sum())


def _validate_no_duplicate_keys(df: pd.DataFrame, label: str) -> None:
    duplicate_mask = df.duplicated(subset=INPUT_REQUIRED_COLUMNS, keep=False)
    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, INPUT_REQUIRED_COLUMNS].to_dict(orient="records")
        raise ValueError(f"{label} contains duplicate ticker/date rows: {duplicates}")


def _distribution(series: pd.Series) -> str:
    counts = {str(key): int(value) for key, value in series.value_counts(dropna=False).sort_index().items()}
    return json.dumps(counts, sort_keys=True, separators=(",", ":"))


def _build_unavailable_output(context_df: pd.DataFrame, generated_at: str) -> pd.DataFrame:
    output_df = context_df[INPUT_REQUIRED_COLUMNS].copy()
    output_df["quality_state"] = "INSUFFICIENT_DATA"
    output_df["quality_reason"] = "fundamental data unavailable"
    output_df["profitability_profile"] = "UNAVAILABLE"
    output_df["balance_sheet_profile"] = "UNAVAILABLE"
    output_df["earnings_quality_profile"] = "UNAVAILABLE"
    output_df["capital_efficiency_profile"] = "UNAVAILABLE"
    output_df["cashflow_profile"] = "UNAVAILABLE"
    output_df["stability_profile"] = "UNAVAILABLE"
    output_df["quality_metadata_status"] = "source_missing"
    output_df["source_data_status"] = "source_missing"
    output_df["source_timestamp"] = ""
    output_df["generated_at"] = generated_at
    return output_df[OUTPUT_COLUMNS]


def _write_log(context_df: pd.DataFrame, output_df: pd.DataFrame, generated_at: str, duplicate_count: int) -> None:
    log_row = {
        "generated_at": generated_at,
        "input_row_count": int(len(context_df)),
        "output_row_count": int(len(output_df)),
        "unique_ticker_date_count": int(context_df[INPUT_REQUIRED_COLUMNS].drop_duplicates().shape[0]),
        "duplicate_ticker_date_count": int(duplicate_count),
        "missing_fundamentals_count": int(len(output_df)),
        "partial_data_count": 0,
        "stale_data_count": 0,
        "quality_state_distribution": _distribution(output_df["quality_state"]),
        "quality_metadata_status_distribution": _distribution(output_df["quality_metadata_status"]),
        "source_data_status_distribution": _distribution(output_df["source_data_status"]),
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([log_row], columns=LOG_COLUMNS).to_csv(LOG_PATH, index=False)


def build_fundamental_layer(generated_at: str | None = None) -> pd.DataFrame:
    context_df = _load_required_csv(CONTEXT_PATH, "context_strength.csv")
    _validate_columns(context_df, INPUT_REQUIRED_COLUMNS, "context_strength.csv")
    _validate_required_values(context_df, "context_strength.csv")
    duplicate_count = _count_duplicate_keys(context_df)
    _validate_no_duplicate_keys(context_df, "context_strength.csv")

    run_timestamp = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_df = _build_unavailable_output(context_df, run_timestamp)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    _write_log(context_df, output_df, run_timestamp, duplicate_count)
    print(f"Fundamental layer written to: {OUTPUT_PATH}")
    print(f"Fundamental layer log written to: {LOG_PATH}")
    return output_df


if __name__ == "__main__":
    build_fundamental_layer()
