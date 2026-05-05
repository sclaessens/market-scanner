from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

SCANNER_PATH = Path("data/processed/scanner_ranked.csv")
VALIDATION_PATH = Path("data/processed/validation_layer.csv")
SECTOR_RS_PATH = Path("data/processed/sector_relative_strength.csv")

OUTPUT_PATH = Path("data/processed/context_strength.csv")
LOG_PATH = Path("data/logs/context_layer_log.csv")

NEUTRAL_BAND = 0.25

SCANNER_REQUIRED_COLUMNS = ["ticker", "date", "rs_20d_pct", "sector"]
VALIDATION_REQUIRED_COLUMNS = [
    "ticker",
    "date",
    "valid_setup",
    "tradeable_setup",
    "validation_reason",
]
SECTOR_REQUIRED_COLUMNS = ["sector", "date", "sector_rs_20d_pct"]

OUTPUT_COLUMNS = [
    "ticker",
    "date",
    "rs_20d",
    "rs_vs_sector",
    "context_strength",
    "context_reason",
    "context_tradeable",
    "context_tradeable_reason",
]

LOG_COLUMNS = [
    "run_date",
    "total_rows",
    "valid_setups",
    "tradeable_setups_before",
    "tradeable_setups_after",
    "weak_count",
    "neutral_count",
    "strong_count",
    "leading_count",
    "unknown_count",
    "missing_sector_count",
]


def is_nan(value: Any) -> bool:
    return pd.isna(value)


def classify_context(rs_20d: Any, rs_vs_sector: Any) -> tuple[str, str]:
    if rs_20d is None or is_nan(rs_20d):
        return "UNKNOWN", "missing_rs_20d"

    if abs(rs_20d) <= NEUTRAL_BAND:
        return "NEUTRAL", "neutral_rs"

    if rs_20d < -NEUTRAL_BAND:
        return "WEAK", "negative_rs"

    if rs_20d > NEUTRAL_BAND:
        if (
            rs_vs_sector is not None
            and not is_nan(rs_vs_sector)
            and rs_vs_sector > NEUTRAL_BAND
        ):
            return "LEADING", "market_and_sector_outperformance"
        return "STRONG", "market_outperformance"

    return "UNKNOWN", "fallback"


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _load_required_csv(path: Path, label: str) -> pd.DataFrame:
    _require_file(path, label)

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


def _normalize_ticker_date(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.strftime("%Y-%m-%d")
    return df


def _normalize_sector(value: Any) -> str | None:
    if pd.isna(value):
        return None

    normalized = str(value).strip().upper()
    if normalized == "":
        return None

    return normalized


def _validate_no_duplicate_keys(
    df: pd.DataFrame,
    key_columns: list[str],
    label: str,
) -> None:
    duplicate_mask = df.duplicated(subset=key_columns, keep=False)

    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, key_columns].to_dict(orient="records")
        raise ValueError(f"{label} contains duplicate keys {key_columns}: {duplicates}")


def _validate_scanner(scanner_df: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(scanner_df, SCANNER_REQUIRED_COLUMNS, "scanner_ranked.csv")

    scanner_df = _normalize_ticker_date(scanner_df, "scanner_ranked.csv")
    scanner_df["sector"] = scanner_df["sector"].apply(_normalize_sector)
    scanner_df["rs_20d_pct"] = pd.to_numeric(
        scanner_df["rs_20d_pct"],
        errors="raise",
    )

    if scanner_df["ticker"].isna().any() or (scanner_df["ticker"] == "").any():
        raise ValueError("scanner_ranked.csv contains empty ticker values")

    _validate_no_duplicate_keys(scanner_df, ["ticker", "date"], "scanner_ranked.csv")

    return scanner_df


def _validate_validation_layer(validation_df: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(
        validation_df,
        VALIDATION_REQUIRED_COLUMNS,
        "validation_layer.csv",
    )

    validation_df = _normalize_ticker_date(validation_df, "validation_layer.csv")

    if validation_df["ticker"].isna().any() or (validation_df["ticker"] == "").any():
        raise ValueError("validation_layer.csv contains empty ticker values")

    if not validation_df["valid_setup"].isin([True, False]).all():
        raise ValueError("validation_layer.csv valid_setup must be boolean")

    _validate_no_duplicate_keys(
        validation_df,
        ["ticker", "date"],
        "validation_layer.csv",
    )

    return validation_df


def _load_sector_relative_strength() -> pd.DataFrame:
    if not SECTOR_RS_PATH.exists():
        return pd.DataFrame(columns=SECTOR_REQUIRED_COLUMNS)

    sector_df = pd.read_csv(SECTOR_RS_PATH)

    if sector_df.empty:
        return pd.DataFrame(columns=SECTOR_REQUIRED_COLUMNS)

    _validate_columns(
        sector_df,
        SECTOR_REQUIRED_COLUMNS,
        "sector_relative_strength.csv",
    )

    sector_df = sector_df.copy()
    sector_df["sector"] = sector_df["sector"].apply(_normalize_sector)
    sector_df["date"] = pd.to_datetime(
        sector_df["date"],
        errors="raise",
    ).dt.strftime("%Y-%m-%d")
    sector_df["sector_rs_20d_pct"] = pd.to_numeric(
        sector_df["sector_rs_20d_pct"],
        errors="raise",
    )

    _validate_no_duplicate_keys(
        sector_df,
        ["sector", "date"],
        "sector_relative_strength.csv",
    )

    return sector_df


def _context_tradeable_reason(valid_setup: bool, context_strength: str) -> str:
    if not valid_setup:
        return "invalid_setup"

    if context_strength in {"STRONG", "LEADING"}:
        return "tradeable_context"

    if context_strength == "WEAK":
        return "weak_context"

    if context_strength == "NEUTRAL":
        return "neutral_context"

    return "unknown_context"


def _write_log(output_df: pd.DataFrame, validation_df: pd.DataFrame) -> None:
    counts = output_df["context_strength"].value_counts().to_dict()

    log_row = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_rows": int(len(output_df)),
        "valid_setups": int(validation_df["valid_setup"].sum()),
        "tradeable_setups_before": int(validation_df["tradeable_setup"].sum()),
        "tradeable_setups_after": int(output_df["context_tradeable"].sum()),
        "weak_count": int(counts.get("WEAK", 0)),
        "neutral_count": int(counts.get("NEUTRAL", 0)),
        "strong_count": int(counts.get("STRONG", 0)),
        "leading_count": int(counts.get("LEADING", 0)),
        "unknown_count": int(counts.get("UNKNOWN", 0)),
        "missing_sector_count": int(output_df["rs_vs_sector"].isna().sum()),
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    log_df = pd.DataFrame([log_row], columns=LOG_COLUMNS)

    if LOG_PATH.exists():
        existing_log_df = pd.read_csv(LOG_PATH)
        log_df = pd.concat([existing_log_df, log_df], ignore_index=True)

    log_df.to_csv(LOG_PATH, index=False)


def build_context_layer() -> pd.DataFrame:
    scanner_df = _load_required_csv(SCANNER_PATH, "scanner_ranked.csv")
    validation_df = _load_required_csv(VALIDATION_PATH, "validation_layer.csv")

    scanner_df = _validate_scanner(scanner_df)
    validation_df = _validate_validation_layer(validation_df)
    sector_df = _load_sector_relative_strength()

    merged_df = validation_df.merge(
        scanner_df[["ticker", "date", "rs_20d_pct", "sector"]],
        on=["ticker", "date"],
        how="inner",
        validate="one_to_one",
    )

    if len(merged_df) != len(validation_df):
        raise ValueError(
            "Row count mismatch after scanner/validation merge: "
            f"expected {len(validation_df)}, got {len(merged_df)}"
        )

    merged_df = merged_df.merge(
        sector_df,
        on=["sector", "date"],
        how="left",
        validate="many_to_one",
    )

    merged_df["rs_20d"] = merged_df["rs_20d_pct"]
    merged_df["rs_vs_sector"] = (
        merged_df["rs_20d"] - merged_df["sector_rs_20d_pct"]
    )

    classifications = merged_df.apply(
        lambda row: classify_context(row["rs_20d"], row["rs_vs_sector"]),
        axis=1,
    )

    merged_df["context_strength"] = classifications.apply(lambda item: item[0])
    merged_df["context_reason"] = classifications.apply(lambda item: item[1])

    merged_df.loc[merged_df["sector"].isna(), "context_strength"] = "UNKNOWN"
    merged_df.loc[merged_df["sector"].isna(), "context_reason"] = "missing_sector"

    merged_df["context_tradeable"] = (
        (merged_df["valid_setup"] == True)
        & (merged_df["context_strength"].isin(["STRONG", "LEADING"]))
    )

    merged_df["context_tradeable_reason"] = merged_df.apply(
        lambda row: _context_tradeable_reason(
            bool(row["valid_setup"]),
            str(row["context_strength"]),
        ),
        axis=1,
    )

    output_df = merged_df[
        [
            "ticker",
            "date",
            "rs_20d",
            "rs_vs_sector",
            "context_strength",
            "context_reason",
            "context_tradeable",
            "context_tradeable_reason",
        ]
    ].copy()

    output_df = output_df[OUTPUT_COLUMNS]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    _write_log(output_df, validation_df)

    print(f"Context layer written to: {OUTPUT_PATH}")
    print(f"Context layer log written to: {LOG_PATH}")

    return output_df


if __name__ == "__main__":
    build_context_layer()