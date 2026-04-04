from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.settings import DATA_DIR


WATCHLIST_TRANSACTIONS_FILE = DATA_DIR / "watchlist" / "watchlist_transactions.csv"
WATCHLIST_ACTIVE_FILE = DATA_DIR / "watchlist" / "watchlist_active.csv"


REQUIRED_COLUMNS = [
    "timestamp",
    "ticker",
    "action",
    "setup_type",
    "source",
    "note",
]


def ensure_parent_dirs() -> None:
    WATCHLIST_TRANSACTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    WATCHLIST_ACTIVE_FILE.parent.mkdir(parents=True, exist_ok=True)


def empty_transactions_df() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_COLUMNS)


def load_transactions() -> pd.DataFrame:
    if not WATCHLIST_TRANSACTIONS_FILE.exists():
        return empty_transactions_df()

    try:
        df = pd.read_csv(WATCHLIST_TRANSACTIONS_FILE)
    except Exception:
        return empty_transactions_df()

    if df.empty:
        return empty_transactions_df()

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[REQUIRED_COLUMNS].copy()

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["action"] = df["action"].astype(str).str.upper().str.strip()
    df["setup_type"] = df["setup_type"].fillna("").astype(str).str.upper().str.strip()
    df["source"] = df["source"].fillna("").astype(str).str.strip()
    df["note"] = df["note"].fillna("").astype(str).str.strip()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["timestamp"])
    df = df[df["ticker"] != ""]
    df = df[df["action"].isin(["WATCH", "UNWATCH"])]

    return df.sort_values(["timestamp", "ticker"]).reset_index(drop=True)


def build_active_watchlist(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "is_active",
                "setup_type",
                "added_at",
                "last_action",
                "last_action_at",
                "source",
                "note",
            ]
        )

    rows: list[dict] = []

    grouped = transactions.groupby("ticker", sort=True)

    for ticker, group in grouped:
        group = group.sort_values("timestamp").reset_index(drop=True)

        last_row = group.iloc[-1]
        last_action = str(last_row["action"]).upper().strip()

        watch_rows = group[group["action"] == "WATCH"].copy()

        added_at = None
        setup_type = ""
        source = ""
        note = ""

        if not watch_rows.empty:
            latest_watch = watch_rows.iloc[-1]
            added_at = latest_watch["timestamp"]
            setup_type = str(latest_watch["setup_type"]).upper().strip()
            source = str(latest_watch["source"]).strip()
            note = str(latest_watch["note"]).strip()

        is_active = last_action == "WATCH"

        if not is_active:
            continue

        if not setup_type:
            setup_type = "PULLBACK"

        rows.append(
            {
                "ticker": ticker,
                "is_active": True,
                "setup_type": setup_type,
                "added_at": added_at.strftime("%Y-%m-%d %H:%M:%S")
                if pd.notna(added_at)
                else "",
                "last_action": last_action,
                "last_action_at": last_row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                if pd.notna(last_row["timestamp"])
                else "",
                "source": source,
                "note": note,
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "is_active",
                "setup_type",
                "added_at",
                "last_action",
                "last_action_at",
                "source",
                "note",
            ]
        )

    return result.sort_values(["ticker"]).reset_index(drop=True)


def main() -> None:
    ensure_parent_dirs()

    transactions = load_transactions()
    active_df = build_active_watchlist(transactions)

    WATCHLIST_ACTIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    active_df.to_csv(WATCHLIST_ACTIVE_FILE, index=False)

    print(f"Watchlist active file written to: {WATCHLIST_ACTIVE_FILE}")
    print(f"Active tickers: {len(active_df)}")


if __name__ == "__main__":
    main()