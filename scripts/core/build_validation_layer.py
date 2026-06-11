from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/scanner_ranked.csv")
OUTPUT_PATH = Path("data/processed/validation_layer.csv")
ENTRY_QUALITY_OUTPUT_PATH = Path("data/processed/entry_quality_metrics.csv")
LOG_PATH = Path("data/logs/validation_layer_log.csv")

VALIDATION_REQUIRED_COLUMNS = [
    "ticker", "date", "primary_setup", "close", "ma20", "ma50", "ma200",
    "high_20d", "low_20d", "atr14", "volume_ratio", "extension_atr",
]

NUMERIC_COLUMNS = [
    "close", "ma20", "ma50", "ma200", "high_20d", "low_20d",
    "atr14", "volume_ratio", "extension_atr",
]

VALIDATION_OUTPUT_COLUMNS = [
    "ticker", "date", "structure_state", "structure_reason", "setup_type",
    "valid_setup", "validation_reason",
]

ENTRY_QUALITY_OUTPUT_COLUMNS = [
    "ticker", "date", "distance_to_breakout_pct", "breakout_extension_atr",
    "extension_atr", "distance_ma20_pct", "volume_ratio", "range_atr",
    "entry_quality_state", "entry_quality_reason",
]

LOG_COLUMNS = [
    "run_date", "total_rows", "coherent_count", "broken_count", "incomplete_count", "avg_extension_atr",
    "avg_volume_ratio", "median_range_atr",
]

STRUCTURE_REASON_ENUM = {
    "coherent_breakout", "coherent_pullback", "coherent_vcp",
    "structure_broken", "missing_data", "no_setup", "unknown_setup",
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
    missing = [column for column in VALIDATION_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"scanner_ranked.csv is missing required columns: {missing}")


def _validate_no_duplicate_ticker_date(df: pd.DataFrame) -> None:
    duplicate_mask = df.duplicated(subset=["ticker", "date"], keep=False)
    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, ["ticker", "date"]].to_dict(orient="records")
        raise ValueError(f"scanner_ranked.csv contains duplicate ticker/date rows: {duplicates}")


def _normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = df.copy()
    normalized_df["ticker"] = normalized_df["ticker"].astype(str).str.strip().str.upper()
    normalized_df["date"] = pd.to_datetime(normalized_df["date"], errors="raise").dt.strftime("%Y-%m-%d")
    normalized_df["primary_setup"] = normalized_df["primary_setup"].fillna("").astype(str).str.strip().str.upper()
    if normalized_df["ticker"].isna().any() or (normalized_df["ticker"] == "").any():
        raise ValueError("scanner_ranked.csv contains empty ticker values")
    return normalized_df


def _validate_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    validated_df = df.copy()
    for column in NUMERIC_COLUMNS:
        try:
            validated_df[column] = pd.to_numeric(validated_df[column], errors="raise")
        except Exception as exc:
            raise ValueError(f"scanner_ranked.csv column '{column}' must be numeric") from exc
    return validated_df


def _validate_fail_fast_values(df: pd.DataFrame) -> None:
    if (df["atr14"] <= 0).any():
        bad_rows = df.loc[df["atr14"] <= 0, ["ticker", "date", "atr14"]].to_dict(orient="records")
        raise ValueError(f"atr14 must be > 0 for all rows: {bad_rows}")
    if (df["high_20d"] < df["low_20d"]).any():
        bad_rows = df.loc[df["high_20d"] < df["low_20d"], ["ticker", "date", "high_20d", "low_20d"]].to_dict(orient="records")
        raise ValueError(f"high_20d must be >= low_20d for all rows: {bad_rows}")


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
    known_setup = df["primary_setup"].isin(["BREAKOUT", "PULLBACK", "VCP"])

    data_integrity_valid = ~missing_required_data & (df["atr14"] > 0) & (df["high_20d"] >= df["low_20d"])
    trend_structure_valid = (df["close"] > 0) & (df["ma20"] > 0) & (df["ma50"] > 0) & (df["ma200"] > 0)

    breakout_structure = (df["primary_setup"] == "BREAKOUT") & data_integrity_valid & trend_structure_valid & (df["close"] >= df["ma20"])
    pullback_structure = (df["primary_setup"] == "PULLBACK") & data_integrity_valid & trend_structure_valid & (df["close"] >= df["ma50"])
    vcp_structure = (df["primary_setup"] == "VCP") & data_integrity_valid & trend_structure_valid & (df["ma20"] >= df["ma50"])

    valid_setup = breakout_structure | pullback_structure | vcp_structure

    structure_reason = pd.Series("unknown_setup", index=df.index, dtype="object")
    structure_reason = structure_reason.mask(breakout_structure, "coherent_breakout")
    structure_reason = structure_reason.mask(pullback_structure, "coherent_pullback")
    structure_reason = structure_reason.mask(vcp_structure, "coherent_vcp")
    structure_reason = structure_reason.mask(known_setup & ~valid_setup, "structure_broken")
    structure_reason = structure_reason.mask(missing_required_data, "missing_data")
    structure_reason = structure_reason.mask(no_setup, "no_setup")

    structure_state = pd.Series("BROKEN", index=df.index, dtype="object")
    structure_state = structure_state.mask(valid_setup, "COHERENT")
    structure_state = structure_state.mask(missing_required_data | no_setup, "INCOMPLETE")

    validation_df = pd.DataFrame({
        "ticker": df["ticker"],
        "date": df["date"],
        "structure_state": structure_state,
        "structure_reason": structure_reason,
        "setup_type": df["primary_setup"],
        # Deprecated compatibility alias: structure coherence only, never capital eligibility.
        "valid_setup": valid_setup.astype(bool),
        "validation_reason": structure_reason,
    }, columns=VALIDATION_OUTPUT_COLUMNS)

    if not validation_df["structure_reason"].isin(STRUCTURE_REASON_ENUM).all():
        invalid_values = sorted(set(validation_df["structure_reason"]) - STRUCTURE_REASON_ENUM)
        raise ValueError(f"Invalid structure_reason values: {invalid_values}")
    return validation_df


def _build_entry_quality_metrics_df(scanner_df: pd.DataFrame) -> pd.DataFrame:
    df = scanner_df.copy()
    breakout_level = df["high_20d"]
    distance_to_breakout_pct = (df["close"] - breakout_level) / breakout_level * 100
    breakout_extension_atr = (df["close"] - breakout_level) / df["atr14"]
    extension_atr = (df["close"] - df["ma20"]) / df["atr14"]
    distance_ma20_pct = (df["close"] - df["ma20"]) / df["ma20"] * 100
    range_atr = (df["high_20d"] - df["low_20d"]) / df["atr14"]

    state = pd.Series("BALANCED", index=df.index, dtype="object")
    state = state.mask(extension_atr >= 2.0, "EXTENDED")
    state = state.mask(extension_atr <= -1.0, "PULLBACK")
    state = state.mask(range_atr >= 6.0, "WIDE_RANGE")

    reason = pd.Series("balanced_structure", index=df.index, dtype="object")
    reason = reason.mask(state == "EXTENDED", "extended_vs_ma20")
    reason = reason.mask(state == "PULLBACK", "below_ma20")
    reason = reason.mask(state == "WIDE_RANGE", "wide_recent_range")

    metrics_df = pd.DataFrame({
        "ticker": df["ticker"],
        "date": df["date"],
        "distance_to_breakout_pct": distance_to_breakout_pct,
        "breakout_extension_atr": breakout_extension_atr,
        "extension_atr": extension_atr,
        "distance_ma20_pct": distance_ma20_pct,
        "volume_ratio": df["volume_ratio"],
        "range_atr": range_atr,
        "entry_quality_state": state,
        "entry_quality_reason": reason,
    }, columns=ENTRY_QUALITY_OUTPUT_COLUMNS)

    float_columns = [
        "distance_to_breakout_pct", "breakout_extension_atr", "extension_atr",
        "distance_ma20_pct", "volume_ratio", "range_atr",
    ]
    metrics_df[float_columns] = metrics_df[float_columns].round(4)
    return metrics_df


def _validate_data_contracts(validation_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    if list(validation_df.columns) != VALIDATION_OUTPUT_COLUMNS:
        raise ValueError("validation_layer.csv schema changed unexpectedly")
    if len(metrics_df) != len(validation_df):
        raise ValueError("entry_quality_metrics row count must equal validation_layer row count")
    if validation_df.duplicated(subset=["ticker", "date"]).any():
        raise ValueError("validation_layer contains duplicate ticker/date rows")
    if metrics_df.duplicated(subset=["ticker", "date"]).any():
        raise ValueError("entry_quality_metrics contains duplicate ticker/date rows")


def _write_validation_log(validation_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    counts = validation_df["structure_state"].value_counts().to_dict()
    log_row = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_rows": int(len(validation_df)),
        "coherent_count": int(counts.get("COHERENT", 0)),
        "broken_count": int(counts.get("BROKEN", 0)),
        "incomplete_count": int(counts.get("INCOMPLETE", 0)),
        "avg_extension_atr": round(float(metrics_df["extension_atr"].mean()), 4),
        "avg_volume_ratio": round(float(metrics_df["volume_ratio"].mean()), 4),
        "median_range_atr": round(float(metrics_df["range_atr"].median()), 4),
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_log_df = pd.DataFrame([log_row], columns=LOG_COLUMNS)
    if LOG_PATH.exists():
        existing_log_df = pd.read_csv(LOG_PATH)
        new_log_df = pd.concat([existing_log_df.reindex(columns=LOG_COLUMNS), new_log_df], ignore_index=True)
    new_log_df.to_csv(LOG_PATH, index=False)


def build_validation_layer() -> pd.DataFrame:
    scanner_df = _validate_scanner_contract(_load_scanner_ranked())
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
    raise SystemExit(
        "FAIL_CLOSED: scripts/core/build_validation_layer.py is a legacy script-era module. "
        "Use the canonical market_scanner runtime instead."
    )
