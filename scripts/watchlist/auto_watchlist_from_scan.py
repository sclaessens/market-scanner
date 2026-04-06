from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.settings import DATA_DIR, SCANS_LOG_FILE


WATCHLIST_TRANSACTIONS_FILE = DATA_DIR / "watchlist" / "watchlist_transactions.csv"

REQUIRED_SCAN_COLUMNS = [
    "scan_date",
    "ticker",
    "primary_setup",
    "grade",
]

REQUIRED_WATCHLIST_COLUMNS = [
    "timestamp",
    "ticker",
    "action",
    "setup_type",
    "source",
    "note",
]


def ensure_dirs() -> None:
    WATCHLIST_TRANSACTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)


def empty_watchlist_df() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_WATCHLIST_COLUMNS)


def load_scans_log() -> pd.DataFrame:
    if not SCANS_LOG_FILE.exists():
        return pd.DataFrame(columns=REQUIRED_SCAN_COLUMNS)

    try:
        df = pd.read_csv(SCANS_LOG_FILE)
    except Exception:
        return pd.DataFrame(columns=REQUIRED_SCAN_COLUMNS)

    if df.empty:
        return pd.DataFrame(columns=REQUIRED_SCAN_COLUMNS)

    for col in REQUIRED_SCAN_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["primary_setup"] = df["primary_setup"].astype(str).str.upper().str.strip()
    df["grade"] = df["grade"].astype(str).str.upper().str.strip()
    df["scan_date"] = pd.to_datetime(df["scan_date"], errors="coerce")

    df = df.dropna(subset=["scan_date"])
    df = df[df["ticker"] != ""]

    return df


def load_watchlist_transactions() -> pd.DataFrame:
    if not WATCHLIST_TRANSACTIONS_FILE.exists():
        return empty_watchlist_df()

    try:
        df = pd.read_csv(WATCHLIST_TRANSACTIONS_FILE)
    except Exception:
        return empty_watchlist_df()

    if df.empty:
        return empty_watchlist_df()

    for col in REQUIRED_WATCHLIST_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[REQUIRED_WATCHLIST_COLUMNS].copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["action"] = df["action"].astype(str).str.upper().str.strip()
    df["setup_type"] = df["setup_type"].fillna("").astype(str).str.upper().str.strip()
    df["source"] = df["source"].fillna("").astype(str).str.strip()
    df["note"] = df["note"].fillna("").astype(str).str.strip()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["timestamp"])
    df = df[df["ticker"] != ""]
    df = df[df["action"].isin(["WATCH", "UNWATCH"])]

    return df


def get_latest_scan_a_setups(scans_df: pd.DataFrame) -> pd.DataFrame:
    if scans_df.empty:
        return pd.DataFrame(columns=REQUIRED_SCAN_COLUMNS)

    latest_scan_date = scans_df["scan_date"].max()
    latest_df = scans_df[scans_df["scan_date"] == latest_scan_date].copy()
    latest_df = latest_df[latest_df["grade"] == "A"].copy()

    if latest_df.empty:
        return pd.DataFrame(columns=REQUIRED_SCAN_COLUMNS)

    latest_df = latest_df.sort_values(["ticker", "scan_date"]).drop_duplicates(
        subset=["ticker"], keep="last"
    )

    return latest_df


def get_currently_active_tickers(transactions_df: pd.DataFrame) -> set[str]:
    if transactions_df.empty:
        return set()

    active_tickers: set[str] = set()

    grouped = transactions_df.sort_values("timestamp").groupby("ticker", sort=False)

    for ticker, group in grouped:
        last_action = str(group.iloc[-1]["action"]).upper().strip()
        if last_action == "WATCH":
            active_tickers.add(ticker)

    return active_tickers


def build_new_watch_rows(
    latest_a_df: pd.DataFrame,
    active_tickers: set[str],
) -> list[dict]:
    rows: list[dict] = []

    if latest_a_df.empty:
        return rows

    now_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    for _, row in latest_a_df.iterrows():
        ticker = str(row["ticker"]).upper().strip()
        setup_type = str(row["primary_setup"]).upper().strip() or "PULLBACK"

        if ticker in active_tickers:
            continue

        rows.append(
            {
                "timestamp": now_str,
                "ticker": ticker,
                "action": "WATCH",
                "setup_type": setup_type,
                "source": "auto_scan",
                "note": "auto-added from A setup",
            }
        )

    return rows


def save_transactions(transactions_df: pd.DataFrame) -> None:
    output_df = transactions_df.copy()

    if not output_df.empty:
        output_df["timestamp"] = pd.to_datetime(output_df["timestamp"], errors="coerce")
        output_df = output_df.sort_values(["timestamp", "ticker"]).copy()
        output_df["timestamp"] = output_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    WATCHLIST_TRANSACTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(WATCHLIST_TRANSACTIONS_FILE, index=False)


def main() -> None:
    ensure_dirs()

    scans_df = load_scans_log()
    transactions_df = load_watchlist_transactions()

    latest_a_df = get_latest_scan_a_setups(scans_df)
    active_tickers = get_currently_active_tickers(transactions_df)

    new_rows = build_new_watch_rows(latest_a_df, active_tickers)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if transactions_df.empty:
            combined_df = new_df[REQUIRED_WATCHLIST_COLUMNS].copy()
        else:
            combined_df = pd.concat([transactions_df, new_df], ignore_index=True)
    else:
        combined_df = transactions_df.copy()

    if combined_df.empty:
        combined_df = empty_watchlist_df()

    save_transactions(combined_df)

    print(f"Watchlist transactions updated: {WATCHLIST_TRANSACTIONS_FILE}")
    print(f"Latest A setups found: {len(latest_a_df)}")
    print(f"New WATCH rows added: {len(new_rows)}")


if __name__ == "__main__":
    main()