from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

SCANNER_PATH = Path("data/processed/scanner_ranked.csv")
SECTOR_RS_PATH = Path("data/processed/sector_relative_strength.csv")
OUTPUT_PATH = Path("data/processed/context_strength.csv")
LOG_PATH = Path("data/logs/context_layer_log.csv")

SCANNER_REQUIRED_COLUMNS = ["ticker", "date", "rs_20d_pct", "sector"]
SECTOR_REQUIRED_COLUMNS = ["sector", "date", "sector_rs_20d_pct"]
OUTPUT_COLUMNS = [
    "ticker", "date", "rs_score", "rs_percentile", "rs_rank", "rs_vs_market",
    "rs_vs_sector", "context_strength", "context_reason", "leadership_state",
]
LOG_COLUMNS = [
    "run_date", "total_rows", "weak_count", "neutral_count", "strong_count",
    "leading_count", "missing_sector_count", "top_decile_count", "median_rs_score",
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


def _normalize_ticker_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.strftime("%Y-%m-%d")
    return df


def _normalize_sector(value: Any) -> str | None:
    if pd.isna(value):
        return None
    normalized = str(value).strip().upper()
    return normalized if normalized else None


def _validate_no_duplicate_keys(df: pd.DataFrame, key_columns: list[str], label: str) -> None:
    duplicate_mask = df.duplicated(subset=key_columns, keep=False)
    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, key_columns].to_dict(orient="records")
        raise ValueError(f"{label} contains duplicate keys {key_columns}: {duplicates}")


def _validate_scanner(scanner_df: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(scanner_df, SCANNER_REQUIRED_COLUMNS, "scanner_ranked.csv")
    scanner_df = _normalize_ticker_date(scanner_df)
    scanner_df["sector"] = scanner_df["sector"].apply(_normalize_sector)
    scanner_df["rs_20d_pct"] = pd.to_numeric(scanner_df["rs_20d_pct"], errors="raise")
    if scanner_df["ticker"].isna().any() or (scanner_df["ticker"] == "").any():
        raise ValueError("scanner_ranked.csv contains empty ticker values")
    _validate_no_duplicate_keys(scanner_df, ["ticker", "date"], "scanner_ranked.csv")
    return scanner_df


def _load_sector_relative_strength() -> pd.DataFrame:
    if not SECTOR_RS_PATH.exists():
        return pd.DataFrame(columns=SECTOR_REQUIRED_COLUMNS)
    sector_df = pd.read_csv(SECTOR_RS_PATH)
    if sector_df.empty:
        return pd.DataFrame(columns=SECTOR_REQUIRED_COLUMNS)
    _validate_columns(sector_df, SECTOR_REQUIRED_COLUMNS, "sector_relative_strength.csv")
    sector_df = sector_df.copy()
    sector_df["sector"] = sector_df["sector"].apply(_normalize_sector)
    sector_df["date"] = pd.to_datetime(sector_df["date"], errors="raise").dt.strftime("%Y-%m-%d")
    sector_df["sector_rs_20d_pct"] = pd.to_numeric(sector_df["sector_rs_20d_pct"], errors="raise")
    _validate_no_duplicate_keys(sector_df, ["sector", "date"], "sector_relative_strength.csv")
    return sector_df


def _classify_from_percentile(percentile: float) -> tuple[str, str]:
    if pd.isna(percentile):
        return "UNKNOWN", "missing_percentile"
    if percentile >= 90:
        return "LEADING", "top_decile_leadership"
    if percentile >= 75:
        return "STRONG", "upper_quartile_leadership"
    if percentile >= 40:
        return "NEUTRAL", "middle_distribution"
    return "WEAK", "lower_distribution"


def _write_log(output_df: pd.DataFrame) -> None:
    counts = output_df["context_strength"].value_counts().to_dict()
    log_row = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_rows": int(len(output_df)),
        "weak_count": int(counts.get("WEAK", 0)),
        "neutral_count": int(counts.get("NEUTRAL", 0)),
        "strong_count": int(counts.get("STRONG", 0)),
        "leading_count": int(counts.get("LEADING", 0)),
        "missing_sector_count": int(output_df["rs_vs_sector"].isna().sum()),
        "top_decile_count": int((output_df["rs_percentile"] >= 90).sum()),
        "median_rs_score": round(float(output_df["rs_score"].median()), 4) if not output_df.empty else 0.0,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_df = pd.DataFrame([log_row], columns=LOG_COLUMNS)
    if LOG_PATH.exists():
        existing_log_df = pd.read_csv(LOG_PATH)
        log_df = pd.concat([existing_log_df.reindex(columns=LOG_COLUMNS), log_df], ignore_index=True)
    log_df.to_csv(LOG_PATH, index=False)


def build_context_layer() -> pd.DataFrame:
    scanner_df = _validate_scanner(_load_required_csv(SCANNER_PATH, "scanner_ranked.csv"))
    sector_df = _load_sector_relative_strength()

    merged_df = scanner_df[["ticker", "date", "rs_20d_pct", "sector"]].copy()
    merged_df = merged_df.merge(sector_df, on=["sector", "date"], how="left", validate="many_to_one")
    merged_df["rs_score"] = merged_df["rs_20d_pct"]
    merged_df["rs_rank"] = merged_df["rs_score"].rank(method="first", ascending=False).astype(int)
    merged_df["rs_percentile"] = merged_df["rs_score"].rank(method="average", pct=True) * 100
    merged_df["rs_vs_market"] = merged_df["rs_score"]
    merged_df["rs_vs_sector"] = merged_df["rs_score"] - merged_df["sector_rs_20d_pct"]

    classifications = merged_df["rs_percentile"].apply(_classify_from_percentile)
    merged_df["context_strength"] = classifications.apply(lambda item: item[0])
    merged_df["context_reason"] = classifications.apply(lambda item: item[1])
    merged_df["leadership_state"] = merged_df["context_strength"]

    output_df = merged_df[OUTPUT_COLUMNS].copy()
    output_df["rs_score"] = output_df["rs_score"].round(4)
    output_df["rs_percentile"] = output_df["rs_percentile"].round(2)
    output_df["rs_vs_market"] = output_df["rs_vs_market"].round(4)
    output_df["rs_vs_sector"] = output_df["rs_vs_sector"].round(4)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    _write_log(output_df)
    print(f"Context layer written to: {OUTPUT_PATH}")
    print(f"Context layer log written to: {LOG_PATH}")
    return output_df


if __name__ == "__main__":
    raise SystemExit(
        "FAIL_CLOSED: scripts/core/build_context_layer.py is a legacy script-era module. "
        "Use the canonical market_scanner runtime instead."
    )
