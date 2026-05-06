from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

INPUT_PATH = Path("data/processed/scanner_ranked.csv")
OUTPUT_PATH = Path("data/processed/validation_layer.csv")
LOG_PATH = Path("data/logs/validation_layer_log.csv")

REQUIRED_COLUMNS = [
    "ticker",
    "date",
    "primary_setup",
    "rr",
    "close",
    "ma20",
    "ma50",
    "high_20d",
    "volume_ratio",
    "extension_atr",
]

NUMERIC_COLUMNS = [
    "rr",
    "close",
    "ma20",
    "ma50",
    "high_20d",
    "volume_ratio",
    "extension_atr",
]

OUTPUT_COLUMNS = [
    "ticker",
    "date",
    "valid_setup",
    "tradeable_setup",
    "validation_reason",
]

LOG_COLUMNS = [
    "run_date",
    "total_rows",
    "valid_count",
    "invalid_count",
    "breakout_valid_count",
    "pullback_valid_count",
    "vcp_valid_count",
    "invalid_rr_count",
    "weak_trend_count",
    "missing_data_count",
]

VALIDATION_REASON_ENUM = {
    "valid_breakout",
    "valid_pullback",
    "valid_vcp",
    "invalid_rr",
    "invalid_structure",
    "weak_trend",
    "missing_data",
    "no_setup",
}


def _is_missing(value: Any) -> bool:
    if pd.isna(value):
        return True

    if isinstance(value, str) and value.strip() == "":
        return True

    return False


def _normalize_setup(value: Any) -> str:
    if _is_missing(value):
        return ""

    return str(value).strip().upper()


def _load_scanner_ranked() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"scanner_ranked.csv not found: {INPUT_PATH}")

    try:
        df = pd.read_csv(INPUT_PATH)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"scanner_ranked.csv is empty: {INPUT_PATH}") from exc

    if df.empty:
        raise ValueError(f"scanner_ranked.csv is empty: {INPUT_PATH}")

    return df


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing_columns:
        raise ValueError(
            "scanner_ranked.csv is missing required columns: "
            f"{missing_columns}"
        )


def _validate_no_duplicate_ticker_date(df: pd.DataFrame) -> None:
    duplicate_mask = df.duplicated(subset=["ticker", "date"], keep=False)

    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, ["ticker", "date"]].to_dict(
            orient="records"
        )
        raise ValueError(
            "scanner_ranked.csv contains duplicate ticker/date rows: "
            f"{duplicates}"
        )


def _validate_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    validated_df = df.copy()

    for column in NUMERIC_COLUMNS:
        try:
            validated_df[column] = pd.to_numeric(
                validated_df[column],
                errors="raise",
            )
        except Exception as exc:
            raise ValueError(
                f"scanner_ranked.csv column '{column}' must be numeric"
            ) from exc

    return validated_df


def _validate_scanner_contract(df: pd.DataFrame) -> pd.DataFrame:
    _validate_required_columns(df)
    _validate_no_duplicate_ticker_date(df)
    return _validate_numeric_columns(df)


def _has_missing_required_row_data(row: pd.Series) -> bool:
    for column in REQUIRED_COLUMNS:
        if column == "primary_setup":
            continue

        if _is_missing(row.get(column)):
            return True

    return False


def evaluate_valid_setup(row: pd.Series) -> tuple[bool, str]:
    primary_setup = _normalize_setup(row.get("primary_setup"))

    if not primary_setup:
        return False, "no_setup"

    if _has_missing_required_row_data(row):
        return False, "missing_data"

    rr = float(row["rr"])
    close = float(row["close"])
    ma20 = float(row["ma20"])
    ma50 = float(row["ma50"])
    high_20d = float(row["high_20d"])
    volume_ratio = float(row["volume_ratio"])

    if rr < 1.8:
        return False, "invalid_rr"

    if close <= ma50:
        return False, "weak_trend"

    if high_20d == 0 or ma20 == 0:
        return False, "missing_data"

    if primary_setup == "BREAKOUT":
        distance_high = (high_20d - close) / high_20d
        extension_atr = float(row["extension_atr"])

        valid_breakout = (
                distance_high <= 0.08
                and distance_high <= 0.03
                and volume_ratio >= 1.3
                and close > ma20
                and close > ma50
                and extension_atr <= 2.5
        )

        if valid_breakout:
            return True, "valid_breakout"

        return False, "invalid_structure"

    if primary_setup == "PULLBACK":
        distance_ma20 = (close - ma20) / ma20

        valid_pullback = (
            -0.08 <= distance_ma20 <= 0.03
            and close > ma50
        )

        if valid_pullback:
            return True, "valid_pullback"

        return False, "invalid_structure"

    if primary_setup == "VCP":
        valid_vcp = (
            close > ma20
            and ma20 > ma50
            and close >= 0.80 * high_20d
        )

        if valid_vcp:
            return True, "valid_vcp"

        return False, "invalid_structure"

    return False, "invalid_structure"


def _build_output_row(row: pd.Series) -> dict[str, Any]:
    valid_setup, validation_reason = evaluate_valid_setup(row)

    if validation_reason not in VALIDATION_REASON_ENUM:
        raise ValueError(f"Invalid validation_reason generated: {validation_reason}")

    return {
        "ticker": str(row["ticker"]).strip().upper(),
        "date": row["date"],
        "valid_setup": bool(valid_setup),
        "tradeable_setup": bool(valid_setup),
        "validation_reason": validation_reason,
    }


def _write_validation_log(validation_df: pd.DataFrame) -> None:
    reason_counts = validation_df["validation_reason"].value_counts().to_dict()

    log_row = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_rows": int(len(validation_df)),
        "valid_count": int(validation_df["valid_setup"].sum()),
        "invalid_count": int((validation_df["valid_setup"] == False).sum()),
        "breakout_valid_count": int(reason_counts.get("valid_breakout", 0)),
        "pullback_valid_count": int(reason_counts.get("valid_pullback", 0)),
        "vcp_valid_count": int(reason_counts.get("valid_vcp", 0)),
        "invalid_rr_count": int(reason_counts.get("invalid_rr", 0)),
        "weak_trend_count": int(reason_counts.get("weak_trend", 0)),
        "missing_data_count": int(reason_counts.get("missing_data", 0)),
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    new_log_df = pd.DataFrame([log_row], columns=LOG_COLUMNS)

    if LOG_PATH.exists():
        existing_log_df = pd.read_csv(LOG_PATH)

        missing_log_columns = [
            column for column in LOG_COLUMNS if column not in existing_log_df.columns
        ]
        if missing_log_columns:
            raise ValueError(
                "validation_layer_log.csv is missing required columns: "
                f"{missing_log_columns}"
            )

        new_log_df = pd.concat(
            [existing_log_df[LOG_COLUMNS], new_log_df],
            ignore_index=True,
        )

    new_log_df.to_csv(LOG_PATH, index=False)


def build_validation_layer() -> pd.DataFrame:
    scanner_df = _load_scanner_ranked()
    scanner_df = _validate_scanner_contract(scanner_df)

    output_rows = [_build_output_row(row) for _, row in scanner_df.iterrows()]

    validation_df = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)

    if list(validation_df.columns) != OUTPUT_COLUMNS:
        raise ValueError(
            "validation_layer.csv output schema mismatch: "
            f"{list(validation_df.columns)}"
        )

    if not validation_df["validation_reason"].isin(VALIDATION_REASON_ENUM).all():
        invalid_values = sorted(
            set(validation_df["validation_reason"]) - VALIDATION_REASON_ENUM
        )
        raise ValueError(f"Invalid validation_reason values: {invalid_values}")

    if not (
        validation_df["tradeable_setup"].astype(bool)
        == validation_df["valid_setup"].astype(bool)
    ).all():
        raise ValueError("tradeable_setup must equal valid_setup")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    validation_df.to_csv(OUTPUT_PATH, index=False)

    _write_validation_log(validation_df)

    print(f"Validation layer written to: {OUTPUT_PATH}")
    print(f"Validation layer log written to: {LOG_PATH}")

    return validation_df


if __name__ == "__main__":
    build_validation_layer()