from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR, SCANS_LOG_FILE


PROCESSED_DIR = DATA_DIR / "processed"
VALIDATION_RESULTS_FILE = DATA_DIR / "processed" / "validation_results.csv"

MAX_WAIT_DAYS_FOR_ENTRY = 5
VALIDATION_HORIZON_DAYS = 20


OUTPUT_COLUMNS = [
    "scan_date",
    "ticker",
    "setup_type",
    "grade",
    "regime",
    "entry",
    "stop",
    "target",
    "rr",
    "entry_hit",
    "entry_date",
    "outcome",
    "outcome_date",
    "days_to_entry",
    "days_to_outcome",
    "max_gain_pct",
    "max_drawdown_pct",
]


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def safe_float(value) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def load_price_data(ticker: str) -> pd.DataFrame:
    candidates = [
        PROCESSED_DIR / f"{ticker}.csv",
        PROCESSED_DIR / f"{ticker}_indicators.csv",
    ]

    for path in candidates:
        df = read_csv_safe(path)
        if not df.empty:
            break
    else:
        return pd.DataFrame()

    if "Date" not in df.columns:
        # Sommige processed files zijn opgeslagen zonder Date-kolom.
        # Gebruik dan de index als fallback, maar validation op echte datums is dan niet betrouwbaar.
        return pd.DataFrame()

    required = {"Date", "High", "Low", "Close"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["High"] = pd.to_numeric(df["High"], errors="coerce")
    df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = df.dropna(subset=["Date", "High", "Low", "Close"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def validate_one_signal(row: pd.Series) -> dict:
    ticker = str(row.get("ticker", "")).upper().strip()
    scan_date = pd.to_datetime(row.get("scan_date"), errors="coerce")

    setup_type = str(row.get("primary_setup", row.get("setup", ""))).upper().strip()
    grade = str(row.get("grade", "")).upper().strip()
    regime = str(row.get("regime", "")).upper().strip()

    entry = safe_float(row.get("entry"))
    stop = safe_float(row.get("stop"))
    target = safe_float(row.get("target"))
    rr = safe_float(row.get("rr"))

    base = {
        "scan_date": row.get("scan_date"),
        "ticker": ticker,
        "setup_type": setup_type,
        "grade": grade,
        "regime": regime,
        "entry": entry,
        "stop": stop,
        "target": target,
        "rr": rr,
        "entry_hit": False,
        "entry_date": None,
        "outcome": "INVALID",
        "outcome_date": None,
        "days_to_entry": None,
        "days_to_outcome": None,
        "max_gain_pct": None,
        "max_drawdown_pct": None,
    }

    if not ticker or pd.isna(scan_date) or entry is None or stop is None or target is None:
        return base

    df = load_price_data(ticker)
    if df.empty:
        base["outcome"] = "NO_PRICE_DATA"
        return base

    future = df[df["Date"] > scan_date].copy()

    if future.empty:
        base["outcome"] = "NO_FUTURE_DATA"
        return base

    entry_window = future.head(MAX_WAIT_DAYS_FOR_ENTRY)
    entry_hits = entry_window[entry_window["High"] >= entry]

    if entry_hits.empty:
        base["outcome"] = "NOT_TRIGGERED"
        return base

    entry_row = entry_hits.iloc[0]
    entry_date = entry_row["Date"]

    base["entry_hit"] = True
    base["entry_date"] = entry_date.strftime("%Y-%m-%d")
    base["days_to_entry"] = int((entry_date.normalize() - scan_date.normalize()).days)

    trade_window = df[df["Date"] >= entry_date].head(VALIDATION_HORIZON_DAYS).copy()

    if trade_window.empty:
        base["outcome"] = "NO_TRADE_WINDOW"
        return base

    max_high = trade_window["High"].max()
    min_low = trade_window["Low"].min()

    base["max_gain_pct"] = round(((max_high - entry) / entry) * 100, 2)
    base["max_drawdown_pct"] = round(((min_low - entry) / entry) * 100, 2)

    outcome = "OPEN"
    outcome_date = None

    for _, day in trade_window.iterrows():
        hit_stop = day["Low"] <= stop
        hit_target = day["High"] >= target

        if hit_stop and hit_target:
            outcome = "STOP_FIRST_SAME_DAY"
            outcome_date = day["Date"]
            break

        if hit_stop:
            outcome = "STOP"
            outcome_date = day["Date"]
            break

        if hit_target:
            outcome = "TARGET"
            outcome_date = day["Date"]
            break

    if outcome == "OPEN":
        outcome = "TIMEOUT"

    base["outcome"] = outcome

    if outcome_date is not None:
        base["outcome_date"] = outcome_date.strftime("%Y-%m-%d")
        base["days_to_outcome"] = int((outcome_date.normalize() - entry_date.normalize()).days)

    return base


def validate_scans() -> pd.DataFrame:
    VALIDATION_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    scans_df = read_csv_safe(SCANS_LOG_FILE)

    if scans_df.empty:
        empty_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        empty_df.to_csv(VALIDATION_RESULTS_FILE, index=False)
        return empty_df

    results = [validate_one_signal(row) for _, row in scans_df.iterrows()]
    result_df = pd.DataFrame(results)

    for col in OUTPUT_COLUMNS:
        if col not in result_df.columns:
            result_df[col] = None

    result_df = result_df[OUTPUT_COLUMNS]
    result_df.to_csv(VALIDATION_RESULTS_FILE, index=False)

    return result_df


def main() -> None:
    df = validate_scans()
    print(f"Validation results written to: {VALIDATION_RESULTS_FILE}")
    print(f"Rows: {len(df)}")

    if not df.empty:
        print(df[["scan_date", "ticker", "setup_type", "grade", "outcome"]].tail(20).to_string(index=False))


if __name__ == "__main__":
    main()