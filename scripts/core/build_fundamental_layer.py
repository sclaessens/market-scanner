from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

CONTEXT_PATH = Path("data/processed/context_strength.csv")
RAW_FUNDAMENTALS_PATH = Path("data/raw/fundamentals.csv")
OUTPUT_PATH = Path("data/processed/fundamental_quality.csv")
LOG_PATH = Path("data/logs/fundamental_layer_log.csv")
STALE_THRESHOLD_DAYS = 120

INPUT_REQUIRED_COLUMNS = ["ticker", "date"]
RAW_IDENTITY_COLUMNS = ["ticker", "as_of_date"]
RAW_METADATA_COLUMNS = ["source_name", "source_last_updated", "report_period", "currency"]
RAW_NUMERIC_COLUMNS = [
    "revenue_growth_yoy",
    "eps_growth_yoy",
    "gross_margin",
    "operating_margin",
    "debt_to_equity",
]
RAW_BOOLEAN_COLUMNS = ["free_cash_flow_positive"]
RAW_DATA_COLUMNS = RAW_NUMERIC_COLUMNS + RAW_BOOLEAN_COLUMNS
RAW_REQUIRED_COLUMNS = RAW_IDENTITY_COLUMNS + RAW_METADATA_COLUMNS + RAW_DATA_COLUMNS
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
    "source_name",
    "source_last_updated",
    "source_freshness_days",
    "missing_required_fields",
    "partial_data_reason",
    "stale_data_reason",
    "invalid_data_reason",
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


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_ticker(value: Any) -> str:
    return _clean_text(value).upper()


def _parse_iso_date(value: Any) -> date | None:
    text = _clean_text(value)
    if len(text) < 10:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _parse_number(value: Any) -> bool:
    text = _clean_text(value)
    if text == "":
        return False
    try:
        float(text)
    except ValueError:
        return False
    return True


def _parse_boolean(value: Any) -> bool:
    text = _clean_text(value)
    return text in {"true", "false", "TRUE", "FALSE", "1", "0"}


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
    output_df["source_name"] = ""
    output_df["source_last_updated"] = ""
    output_df["source_freshness_days"] = ""
    output_df["missing_required_fields"] = ""
    output_df["partial_data_reason"] = ""
    output_df["stale_data_reason"] = ""
    output_df["invalid_data_reason"] = ""
    output_df["generated_at"] = generated_at
    return output_df[OUTPUT_COLUMNS]


def _load_raw_fundamentals(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        raw_df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"fundamentals.csv is empty: {path}") from exc
    _validate_columns(raw_df, RAW_REQUIRED_COLUMNS, "fundamentals.csv")
    return raw_df


def _normalize_raw_fundamentals(raw_df: pd.DataFrame) -> pd.DataFrame:
    normalized = raw_df.copy()
    normalized["_normalized_ticker"] = normalized["ticker"].map(_normalize_ticker)
    normalized["_as_of_date"] = normalized["as_of_date"].map(_parse_iso_date)

    missing_ticker = normalized["_normalized_ticker"] == ""
    if missing_ticker.any():
        rows = raw_df.loc[missing_ticker, RAW_IDENTITY_COLUMNS].to_dict(orient="records")
        raise ValueError(f"fundamentals.csv contains missing ticker values: {rows}")

    invalid_as_of = normalized["_as_of_date"].isna()
    if invalid_as_of.any():
        rows = raw_df.loc[invalid_as_of, RAW_IDENTITY_COLUMNS].to_dict(orient="records")
        raise ValueError(f"fundamentals.csv contains invalid as_of_date values: {rows}")

    duplicate_mask = normalized.duplicated(subset=["_normalized_ticker", "_as_of_date"], keep=False)
    if duplicate_mask.any():
        duplicates = raw_df.loc[duplicate_mask, RAW_IDENTITY_COLUMNS].to_dict(orient="records")
        raise ValueError(f"fundamentals.csv contains duplicate ticker/as_of_date rows: {duplicates}")

    return normalized.sort_values(["_normalized_ticker", "_as_of_date"], kind="mergesort").reset_index(drop=True)


def _metadata_output(
    row: pd.Series,
    generated_at: str,
    quality_state: str,
    quality_reason: str,
    profile_state: str,
    quality_metadata_status: str,
    source_data_status: str,
    source_timestamp: str = "",
    source_name: str = "",
    source_last_updated: str = "",
    source_freshness_days: int | str = "",
    missing_required_fields: list[str] | None = None,
    partial_data_reason: str = "",
    stale_data_reason: str = "",
    invalid_data_reason: str = "",
) -> dict[str, Any]:
    return {
        "ticker": row["ticker"],
        "date": row["date"],
        "quality_state": quality_state,
        "quality_reason": quality_reason,
        "profitability_profile": profile_state,
        "balance_sheet_profile": profile_state,
        "earnings_quality_profile": profile_state,
        "capital_efficiency_profile": profile_state,
        "cashflow_profile": profile_state,
        "stability_profile": profile_state,
        "quality_metadata_status": quality_metadata_status,
        "source_data_status": source_data_status,
        "source_timestamp": source_timestamp,
        "source_name": source_name,
        "source_last_updated": source_last_updated,
        "source_freshness_days": source_freshness_days,
        "missing_required_fields": "|".join(missing_required_fields or []),
        "partial_data_reason": partial_data_reason,
        "stale_data_reason": stale_data_reason,
        "invalid_data_reason": invalid_data_reason,
        "generated_at": generated_at,
    }


def _classify_raw_match(context_row: pd.Series, raw_row: pd.Series, opportunity_date: date, generated_at: str) -> dict[str, Any]:
    source_name = _clean_text(raw_row.get("source_name"))
    source_last_updated = _clean_text(raw_row.get("source_last_updated"))
    report_period = _clean_text(raw_row.get("report_period"))
    currency = _clean_text(raw_row.get("currency"))
    source_last_updated_date = _parse_iso_date(source_last_updated)

    invalid_reasons: list[str] = []
    if source_name == "":
        invalid_reasons.append("source_name is missing")
    if source_last_updated_date is None:
        invalid_reasons.append("source_last_updated is invalid")
    if report_period == "":
        invalid_reasons.append("report_period is missing")
    if currency == "":
        invalid_reasons.append("currency is missing")

    if source_last_updated_date is None or invalid_reasons:
        return _metadata_output(
            context_row,
            generated_at,
            quality_state="INSUFFICIENT_DATA",
            quality_reason="fundamental source metadata invalid",
            profile_state="UNAVAILABLE",
            quality_metadata_status="invalid",
            source_data_status="invalid_data",
            source_timestamp=source_last_updated,
            source_name=source_name,
            source_last_updated=source_last_updated,
            invalid_data_reason="; ".join(invalid_reasons),
        )

    source_freshness_days = (opportunity_date - source_last_updated_date).days
    if source_freshness_days < 0:
        return _metadata_output(
            context_row,
            generated_at,
            quality_state="INSUFFICIENT_DATA",
            quality_reason="fundamental source metadata invalid",
            profile_state="UNAVAILABLE",
            quality_metadata_status="invalid",
            source_data_status="invalid_data",
            source_timestamp=source_last_updated,
            source_name=source_name,
            source_last_updated=source_last_updated,
            source_freshness_days=source_freshness_days,
            invalid_data_reason="source_last_updated is after opportunity date",
        )

    missing_fields = [column for column in RAW_DATA_COLUMNS if _clean_text(raw_row.get(column)) == ""]
    invalid_fields: list[str] = []
    for column in RAW_NUMERIC_COLUMNS:
        if _clean_text(raw_row.get(column)) != "" and not _parse_number(raw_row.get(column)):
            invalid_fields.append(column)
    for column in RAW_BOOLEAN_COLUMNS:
        if _clean_text(raw_row.get(column)) != "" and not _parse_boolean(raw_row.get(column)):
            invalid_fields.append(column)

    valid_data_count = 0
    for column in RAW_NUMERIC_COLUMNS:
        if _parse_number(raw_row.get(column)):
            valid_data_count += 1
    for column in RAW_BOOLEAN_COLUMNS:
        if _parse_boolean(raw_row.get(column)):
            valid_data_count += 1

    invalid_data_reason = ""
    if invalid_fields:
        invalid_data_reason = f"invalid required fields: {'|'.join(invalid_fields)}"

    if valid_data_count == 0:
        reason = "all required fundamental data fields are missing or invalid"
        return _metadata_output(
            context_row,
            generated_at,
            quality_state="INSUFFICIENT_DATA",
            quality_reason="fundamental data insufficient",
            profile_state="UNAVAILABLE",
            quality_metadata_status="invalid" if invalid_fields else "partial",
            source_data_status="invalid_data" if invalid_fields else "partial_data",
            source_timestamp=source_last_updated,
            source_name=source_name,
            source_last_updated=source_last_updated,
            source_freshness_days=source_freshness_days,
            missing_required_fields=missing_fields,
            partial_data_reason=reason,
            invalid_data_reason=invalid_data_reason,
        )

    if source_freshness_days > STALE_THRESHOLD_DAYS:
        stale_reason = f"source_last_updated is older than {STALE_THRESHOLD_DAYS} days"
        return _metadata_output(
            context_row,
            generated_at,
            quality_state="STALE_DATA",
            quality_reason="fundamental source data stale",
            profile_state="STALE",
            quality_metadata_status="stale",
            source_data_status="stale_data",
            source_timestamp=source_last_updated,
            source_name=source_name,
            source_last_updated=source_last_updated,
            source_freshness_days=source_freshness_days,
            missing_required_fields=missing_fields,
            stale_data_reason=stale_reason,
            invalid_data_reason=invalid_data_reason,
        )

    if invalid_fields:
        return _metadata_output(
            context_row,
            generated_at,
            quality_state="PARTIAL_DATA",
            quality_reason="fundamental source data partially invalid",
            profile_state="PARTIAL",
            quality_metadata_status="invalid",
            source_data_status="invalid_data",
            source_timestamp=source_last_updated,
            source_name=source_name,
            source_last_updated=source_last_updated,
            source_freshness_days=source_freshness_days,
            missing_required_fields=missing_fields,
            partial_data_reason="one or more required fundamental fields are invalid",
            invalid_data_reason=invalid_data_reason,
        )

    if missing_fields:
        return _metadata_output(
            context_row,
            generated_at,
            quality_state="PARTIAL_DATA",
            quality_reason="fundamental source data partially available",
            profile_state="PARTIAL",
            quality_metadata_status="partial",
            source_data_status="partial_data",
            source_timestamp=source_last_updated,
            source_name=source_name,
            source_last_updated=source_last_updated,
            source_freshness_days=source_freshness_days,
            missing_required_fields=missing_fields,
            partial_data_reason="one or more required fundamental fields are missing",
        )

    return _metadata_output(
        context_row,
        generated_at,
        quality_state="SUFFICIENT_DATA",
        quality_reason="fundamental source data available",
        profile_state="OBSERVED",
        quality_metadata_status="complete",
        source_data_status="source_available",
        source_timestamp=source_last_updated,
        source_name=source_name,
        source_last_updated=source_last_updated,
        source_freshness_days=source_freshness_days,
    )


def _build_from_raw_fundamentals(context_df: pd.DataFrame, raw_df: pd.DataFrame, generated_at: str) -> pd.DataFrame:
    normalized_raw = _normalize_raw_fundamentals(raw_df)
    output_rows: list[dict[str, Any]] = []

    for _, context_row in context_df.iterrows():
        opportunity_date = _parse_iso_date(context_row["date"])
        if opportunity_date is None:
            output_rows.append(
                _metadata_output(
                    context_row,
                    generated_at,
                    quality_state="INSUFFICIENT_DATA",
                    quality_reason="opportunity date invalid",
                    profile_state="UNAVAILABLE",
                    quality_metadata_status="invalid",
                    source_data_status="invalid_data",
                    invalid_data_reason="opportunity date is invalid",
                )
            )
            continue

        ticker = _normalize_ticker(context_row["ticker"])
        candidates = normalized_raw[
            (normalized_raw["_normalized_ticker"] == ticker)
            & (normalized_raw["_as_of_date"] <= opportunity_date)
        ]
        if candidates.empty:
            output_rows.append(
                _metadata_output(
                    context_row,
                    generated_at,
                    quality_state="INSUFFICIENT_DATA",
                    quality_reason="fundamental source row unavailable",
                    profile_state="UNAVAILABLE",
                    quality_metadata_status="row_missing",
                    source_data_status="row_missing",
                )
            )
            continue

        selected_row = candidates.iloc[-1]
        output_rows.append(_classify_raw_match(context_row, selected_row, opportunity_date, generated_at))

    return pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)


def _write_log(context_df: pd.DataFrame, output_df: pd.DataFrame, generated_at: str, duplicate_count: int) -> None:
    log_row = {
        "generated_at": generated_at,
        "input_row_count": int(len(context_df)),
        "output_row_count": int(len(output_df)),
        "unique_ticker_date_count": int(context_df[INPUT_REQUIRED_COLUMNS].drop_duplicates().shape[0]),
        "duplicate_ticker_date_count": int(duplicate_count),
        "missing_fundamentals_count": int((output_df["quality_state"] == "INSUFFICIENT_DATA").sum()),
        "partial_data_count": int((output_df["quality_state"] == "PARTIAL_DATA").sum()),
        "stale_data_count": int((output_df["quality_state"] == "STALE_DATA").sum()),
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
    raw_df = _load_raw_fundamentals(RAW_FUNDAMENTALS_PATH)
    if raw_df is None:
        output_df = _build_unavailable_output(context_df, run_timestamp)
    else:
        output_df = _build_from_raw_fundamentals(context_df, raw_df, run_timestamp)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    _write_log(context_df, output_df, run_timestamp, duplicate_count)
    print(f"Fundamental layer written to: {OUTPUT_PATH}")
    print(f"Fundamental layer log written to: {LOG_PATH}")
    return output_df


if __name__ == "__main__":
    build_fundamental_layer()
