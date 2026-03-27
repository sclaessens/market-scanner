import os
from datetime import datetime
from typing import List

import pandas as pd


WATCHLIST_DIR = "data/watchlist"
TRANSACTIONS_FILE = os.path.join(WATCHLIST_DIR, "watchlist_transactions.csv")
ACTIVE_FILE = os.path.join(WATCHLIST_DIR, "watchlist_active.csv")

TRANSACTION_COLUMNS: List[str] = [
    "timestamp",
    "ticker",
    "action",
    "setup_type",
    "source",
    "note",
]

ACTIVE_COLUMNS: List[str] = [
    "ticker",
    "is_active",
    "setup_type",
    "added_at",
    "last_action",
    "last_action_at",
    "source",
    "note",
]


def ensure_directories() -> None:
    os.makedirs(WATCHLIST_DIR, exist_ok=True)


def write_empty_active_file() -> pd.DataFrame:
    empty_df = pd.DataFrame(columns=ACTIVE_COLUMNS)
    empty_df.to_csv(ACTIVE_FILE, index=False)
    print(f"Created empty active watchlist: {ACTIVE_FILE}")
    return empty_df


def normalize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Zorg dat alle verwachte kolommen bestaan
    for col in TRANSACTION_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[TRANSACTION_COLUMNS]

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["action"] = df["action"].astype(str).str.upper().str.strip()
    df["setup_type"] = (
        df["setup_type"]
        .fillna("GENERAL")
        .astype(str)
        .str.upper()
        .str.strip()
        .replace("", "GENERAL")
    )
    df["source"] = df["source"].fillna("").astype(str).str.strip()
    df["note"] = df["note"].fillna("").astype(str).str.strip()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Verwijder ongeldige rijen
    df = df[df["ticker"] != ""]
    df = df[df["action"].isin(["WATCH", "UNWATCH"])]
    df = df.dropna(subset=["timestamp"])

    return df


def load_transactions() -> pd.DataFrame:
    if not os.path.exists(TRANSACTIONS_FILE):
        print(f"No transactions file found: {TRANSACTIONS_FILE}")
        return pd.DataFrame(columns=TRANSACTION_COLUMNS)

    try:
        df = pd.read_csv(TRANSACTIONS_FILE)
    except pd.errors.EmptyDataError:
        print(f"Transactions file is empty: {TRANSACTIONS_FILE}")
        return pd.DataFrame(columns=TRANSACTION_COLUMNS)

    if df.empty:
        print("Transactions file exists but contains no rows.")
        return pd.DataFrame(columns=TRANSACTION_COLUMNS)

    df = normalize_transactions(df)
    return df


def build_watchlist() -> pd.DataFrame:
    ensure_directories()

    transactions = load_transactions()

    # Geen transacties -> toch lege active file aanmaken
    if transactions.empty:
        return write_empty_active_file()

    # Sorteer op tijd zodat "laatste actie per ticker" correct is
    transactions = transactions.sort_values(["ticker", "timestamp"])

    active_rows = []

    for ticker, group in transactions.groupby("ticker", sort=True):
        group = group.sort_values("timestamp").reset_index(drop=True)

        first_watch_rows = group[group["action"] == "WATCH"]
        if first_watch_rows.empty:
            # Alleen UNWATCH zonder eerdere WATCH -> negeren
            continue

        latest_row = group.iloc[-1]
        latest_action = latest_row["action"]

        # Alleen tickers met laatste actie WATCH blijven actief
        if latest_action != "WATCH":
            continue

        first_watch = first_watch_rows.iloc[0]

        setup_type = latest_row["setup_type"]
        if not setup_type or setup_type == "GENERAL":
            # Als laatste row geen bruikbaar setup_type heeft, neem eerste WATCH
            setup_type = first_watch["setup_type"] or "GENERAL"

        active_rows.append(
            {
                "ticker": ticker,
                "is_active": True,
                "setup_type": setup_type,
                "added_at": first_watch["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "last_action": latest_row["action"],
                "last_action_at": latest_row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "source": latest_row["source"],
                "note": latest_row["note"],
            }
        )

    if not active_rows:
        return write_empty_active_file()

    active_df = pd.DataFrame(active_rows, columns=ACTIVE_COLUMNS)
    active_df = active_df.sort_values("ticker").reset_index(drop=True)

    active_df.to_csv(ACTIVE_FILE, index=False)

    print(f"Active watchlist saved: {len(active_df)} tickers")
    print(active_df.to_string(index=False))

    return active_df


if __name__ == "__main__":
    build_watchlist()
