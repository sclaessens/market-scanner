from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from config.settings import ENTRY_QUALITY_CONFIG

INPUT_PATH = Path("data/processed/scanner_ranked.csv")
OUTPUT_PATH = Path("data/processed/validation_layer.csv")
ENTRY_QUALITY_OUTPUT_PATH = Path("data/processed/entry_quality_metrics.csv")
LOG_PATH = Path("data/logs/validation_layer_log.csv")

VALIDATION_REQUIRED_COLUMNS = [
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

ENTRY_QUALITY_REQUIRED_COLUMNS = [
    "ticker",
    "date",
    "primary_setup",
    "rr",
    "close",
    "ma20",
    "ma50",
    "high_20d",
    "low_20d",
    "atr14",
    "volume_ratio",
    "extension_atr",
]

NUMERIC_COLUMNS = [
    "rr",
    "close",
    "ma20",
    "ma50",
    "high_20d",
    "low_20d",
    "atr14",
    "volume_ratio",
    "extension_atr",
]

VALIDATION_OUTPUT_COLUMNS = [
    "ticker",
    "date",
    "valid_setup",
    "tradeable_setup",
    "validation_reason",
]

ENTRY_QUALITY_OUTPUT_COLUMNS = [
    "ticker",
    "date",
    "distance_to_breakout_pct",
    "breakout_extension_atr",
    "extension_atr",
    "distance_ma20_pct",
    "volume_ratio",
    "range_atr",
    "entry_quality_flag",
    "entry_quality_reason",
]

LOG_COLUMNS = [
    "run_date",
    "total_rows",
    "valid_setups_before",
    "valid_setups_after",
    "rejected_by_entry_quality",
    "avg_extension_atr",
    "avg_volume_ratio",
    "median_range_atr",
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

ENTRY_QUALITY_REASON_ENUM = {
    "ok",
    "too_far_from_breakout",
    "overextended_atr",
    "overextended_ma20",
    "weak_volume",
    "excessive_volume",
    "range_expansion",
    "invalid_structure",
    "missing_data",
}


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
    required_columns = sorted(
        set(VALIDATION_REQUIRED_COLUMNS + ENTRY_QUALITY_REQUIRED_COLUMNS)
    )
    missing_columns = [column for column in required_columns if column not in df.columns]

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


def _normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = df.copy()

    normalized_df["ticker"] = (
        normalized_df["ticker"].astype(str).str.strip().str.upper()
    )
    normalized_df["date"] = pd.to_datetime(
        normalized_df["date"],
        errors="raise",
    ).dt.strftime("%Y-%m-%d")
    normalized_df["primary_setup"] = (
        normalized_df["primary_setup"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if normalized_df["ticker"].isna().any() or (normalized_df["ticker"] == "").any():
        raise ValueError("scanner_ranked.csv contains empty ticker values")

    return normalized_df


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


def _validate_fail_fast_values(df: pd.DataFrame) -> None:
    if (df["atr14"] <= 0).any():
        bad_rows = df.loc[df["atr14"] <= 0, ["ticker", "date", "atr14"]].to_dict(
            orient="records"
        )
        raise ValueError(f"atr14 must be > 0 for all rows: {bad_rows}")

    if (df["volume_ratio"] < 0).any():
        bad_rows = df.loc[
            df["volume_ratio"] < 0,
            ["ticker", "date", "volume_ratio"],
        ].to_dict(orient="records")
        raise ValueError(f"volume_ratio must be >= 0 for all rows: {bad_rows}")


def _validate_scanner_contract(df: pd.DataFrame) -> pd.DataFrame:
    _validate_required_columns(df)
    df = _normalize_input(df)
    _validate_no_duplicate_ticker_date(df)
    df = _validate_numeric_columns(df)
    _validate_fail_fast_values(df)
    return df


def _build_validation_layer_df(scanner_df: pd.DataFrame) -> pd.DataFrame:
    df = scanner_df.copy()

    missing_required_data = df[VALIDATION_REQUIRED_COLUMNS].isna().any(axis=1)
    no_setup = df["primary_setup"] == ""

    rr_invalid = df["rr"] < 1.8
    weak_trend = df["close"] <= df["ma50"]
    missing_price_reference = (df["high_20d"] == 0) | (df["ma20"] == 0)

    distance_high = (df["high_20d"] - df["close"]) / df["high_20d"]
    distance_ma20 = (df["close"] - df["ma20"]) / df["ma20"]

    valid_breakout = (
        (df["primary_setup"] == "BREAKOUT")
        & (distance_high <= 0.08)
        & (distance_high <= 0.03)
        & (df["volume_ratio"] >= 1.3)
        & (df["close"] > df["ma20"])
        & (df["close"] > df["ma50"])
        & (df["extension_atr"] <= 2.5)
    )

    valid_pullback = (
        (df["primary_setup"] == "PULLBACK")
        & (distance_ma20 >= -0.08)
        & (distance_ma20 <= 0.03)
        & (df["close"] > df["ma50"])
    )

    valid_vcp = (
        (df["primary_setup"] == "VCP")
        & (df["close"] > df["ma20"])
        & (df["ma20"] > df["ma50"])
        & (df["close"] >= 0.80 * df["high_20d"])
    )

    valid_setup = valid_breakout | valid_pullback | valid_vcp

    validation_reason = pd.Series("invalid_structure", index=df.index, dtype="object")
    validation_reason = validation_reason.mask(valid_breakout, "valid_breakout")
    validation_reason = validation_reason.mask(valid_pullback, "valid_pullback")
    validation_reason = validation_reason.mask(valid_vcp, "valid_vcp")
    validation_reason = validation_reason.mask(weak_trend, "weak_trend")
    validation_reason = validation_reason.mask(rr_invalid, "invalid_rr")
    validation_reason = validation_reason.mask(missing_price_reference, "missing_data")
    validation_reason = validation_reason.mask(missing_required_data, "missing_data")
    validation_reason = validation_reason.mask(no_setup, "no_setup")

    valid_setup = valid_setup & ~(
        no_setup
        | missing_required_data
        | missing_price_reference
        | rr_invalid
        | weak_trend
    )

    validation_df = pd.DataFrame(
        {
            "ticker": df["ticker"],
            "date": df["date"],
            "valid_setup": valid_setup.astype(bool),
            "tradeable_setup": valid_setup.astype(bool),
            "validation_reason": validation_reason,
        },
        columns=VALIDATION_OUTPUT_COLUMNS,
    )

    if list(validation_df.columns) != VALIDATION_OUTPUT_COLUMNS:
        raise ValueError(
            "validation_layer.csv output schema mismatch: "
            f"{list(validation_df.columns)}"
        )

    if not validation_df["validation_reason"].isin(VALIDATION_REASON_ENUM).all():
        invalid_values = sorted(
            set(validation_df["validation_reason"]) - VALIDATION_REASON_ENUM
        )
        raise ValueError(f"Invalid validation_reason values: {invalid_values}")

    return validation_df


def _build_entry_quality_metrics_df(scanner_df: pd.DataFrame) -> pd.DataFrame:
    cfg = ENTRY_QUALITY_CONFIG
    df = scanner_df.copy()

    breakout_level = df["high_20d"]

    missing_data = (
        breakout_level.isna()
        | (breakout_level == 0)
        | df["ma20"].isna()
        | (df["ma20"] == 0)
        | df["volume_ratio"].isna()
    )

    if "avg_vol_20" in df.columns:
        avg_volume_20 = pd.to_numeric(df["avg_vol_20"], errors="coerce")
        missing_data = missing_data | avg_volume_20.isna() | (avg_volume_20 == 0)

    distance_to_breakout_pct = (df["close"] - breakout_level) / breakout_level * 100
    breakout_extension_atr = (df["close"] - breakout_level) / df["atr14"]
    extension_atr = (df["close"] - df["ma20"]) / df["atr14"]
    distance_ma20_pct = (df["close"] - df["ma20"]) / df["ma20"] * 100
    volume_ratio = df["volume_ratio"]
    range_atr = (df["high_20d"] - df["low_20d"]) / df["atr14"]
    range_atr = range_atr.mask(df["high_20d"] == df["low_20d"], 0)

    invalid_structure = breakout_extension_atr < 0

    too_far_from_breakout = (
        distance_to_breakout_pct > cfg["max_distance_breakout_pct"]
    )
    overextended_atr = (
        breakout_extension_atr > cfg["max_breakout_extension_atr"]
    )
    overextended_ma20 = extension_atr > cfg["max_extension_atr"]
    weak_volume = volume_ratio < cfg["min_volume_ratio"]
    excessive_volume = volume_ratio > cfg["max_volume_ratio"]
    range_expansion = range_atr > cfg["max_range_atr"]

    entry_quality_reason = pd.Series("ok", index=df.index, dtype="object")
    entry_quality_reason = entry_quality_reason.mask(
        too_far_from_breakout,
        "too_far_from_breakout",
    )
    entry_quality_reason = entry_quality_reason.mask(
        ~too_far_from_breakout & overextended_atr,
        "overextended_atr",
    )
    entry_quality_reason = entry_quality_reason.mask(
        ~too_far_from_breakout & ~overextended_atr & overextended_ma20,
        "overextended_ma20",
    )
    entry_quality_reason = entry_quality_reason.mask(
        ~too_far_from_breakout
        & ~overextended_atr
        & ~overextended_ma20
        & weak_volume,
        "weak_volume",
    )
    entry_quality_reason = entry_quality_reason.mask(
        ~too_far_from_breakout
        & ~overextended_atr
        & ~overextended_ma20
        & ~weak_volume
        & excessive_volume,
        "excessive_volume",
    )
    entry_quality_reason = entry_quality_reason.mask(
        ~too_far_from_breakout
        & ~overextended_atr
        & ~overextended_ma20
        & ~weak_volume
        & ~excessive_volume
        & range_expansion,
        "range_expansion",
    )
    entry_quality_reason = entry_quality_reason.mask(
        invalid_structure,
        "invalid_structure",
    )
    entry_quality_reason = entry_quality_reason.mask(missing_data, "missing_data")

    entry_quality_flag = entry_quality_reason == "ok"

    metrics_df = pd.DataFrame(
        {
            "ticker": df["ticker"],
            "date": df["date"],
            "distance_to_breakout_pct": distance_to_breakout_pct,
            "breakout_extension_atr": breakout_extension_atr,
            "extension_atr": extension_atr,
            "distance_ma20_pct": distance_ma20_pct,
            "volume_ratio": volume_ratio,
            "range_atr": range_atr,
            "entry_quality_flag": entry_quality_flag.astype(bool),
            "entry_quality_reason": entry_quality_reason,
        },
        columns=ENTRY_QUALITY_OUTPUT_COLUMNS,
    )

    float_columns = [
        "distance_to_breakout_pct",
        "breakout_extension_atr",
        "extension_atr",
        "distance_ma20_pct",
        "volume_ratio",
        "range_atr",
    ]
    metrics_df[float_columns] = metrics_df[float_columns].round(4)

    if list(metrics_df.columns) != ENTRY_QUALITY_OUTPUT_COLUMNS:
        raise ValueError(
            "entry_quality_metrics.csv output schema mismatch: "
            f"{list(metrics_df.columns)}"
        )

    if not metrics_df["entry_quality_reason"].isin(ENTRY_QUALITY_REASON_ENUM).all():
        invalid_values = sorted(
            set(metrics_df["entry_quality_reason"]) - ENTRY_QUALITY_REASON_ENUM
        )
        raise ValueError(f"Invalid entry_quality_reason values: {invalid_values}")

    return metrics_df


def _validate_data_contracts(
    validation_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> None:
    if list(validation_df.columns) != VALIDATION_OUTPUT_COLUMNS:
        raise ValueError("validation_layer.csv schema changed unexpectedly")

    if len(metrics_df) != len(validation_df):
        raise ValueError(
            "entry_quality_metrics row count must equal validation_layer row count"
        )

    if validation_df.duplicated(subset=["ticker", "date"]).any():
        raise ValueError("validation_layer contains duplicate ticker/date rows")

    if metrics_df.duplicated(subset=["ticker", "date"]).any():
        raise ValueError("entry_quality_metrics contains duplicate ticker/date rows")

    validation_keys = validation_df[["ticker", "date"]].sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    metrics_keys = metrics_df[["ticker", "date"]].sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    if not validation_keys.equals(metrics_keys):
        raise ValueError("Every validation row must exist in entry_quality_metrics")


def _write_validation_log(
    validation_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> None:
    rejected_by_entry_quality = int(
        (
            (validation_df["valid_setup"] == True)
            & (metrics_df["entry_quality_flag"] == False)
        ).sum()
    )

    log_row = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_rows": int(len(validation_df)),
        "valid_setups_before": int(validation_df["valid_setup"].sum()),
        "valid_setups_after": int(validation_df["valid_setup"].sum()),
        "rejected_by_entry_quality": rejected_by_entry_quality,
        "avg_extension_atr": round(float(metrics_df["extension_atr"].mean()), 4),
        "avg_volume_ratio": round(float(metrics_df["volume_ratio"].mean()), 4),
        "median_range_atr": round(float(metrics_df["range_atr"].median()), 4),
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    new_log_df = pd.DataFrame([log_row], columns=LOG_COLUMNS)

    if LOG_PATH.exists():
        existing_log_df = pd.read_csv(LOG_PATH)
        new_log_df = pd.concat(
            [existing_log_df.reindex(columns=LOG_COLUMNS), new_log_df],
            ignore_index=True,
        )

    new_log_df.to_csv(LOG_PATH, index=False)


def build_validation_layer() -> pd.DataFrame:
    scanner_df = _load_scanner_ranked()
    scanner_df = _validate_scanner_contract(scanner_df)

    validation_df = _build_validation_layer_df(scanner_df)
    metrics_df = _build_entry_quality_metrics_df(scanner_df)

    _validate_data_contracts(validation_df, metrics_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENTRY_QUALITY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    validation_df.to_csv(OUTPUT_PATH, index=False)
    metrics_df.to_csv(ENTRY_QUALITY_OUTPUT_PATH, index=False)

    _write_validation_log(validation_df, metrics_df)

    print(f"Validation layer written to: {OUTPUT_PATH}")
    print(f"Entry quality metrics written to: {ENTRY_QUALITY_OUTPUT_PATH}")
    print(f"Validation layer log written to: {LOG_PATH}")

    return validation_df


if __name__ == "__main__":
    build_validation_layer()