from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.settings import DATA_DIR


WATCHLIST_STATUS_FILE = DATA_DIR / "watchlist" / "watchlist_status.csv"
WATCHLIST_TRANSACTIONS_FILE = DATA_DIR / "watchlist" / "watchlist_transactions.csv"


REQUIRED_TX_COLUMNS = [
    "timestamp",
    "ticker",
    "action",
    "setup_type",
    "source",
    "note",
]


def load_status() -> pd.DataFrame:
    if not WATCHLIST_STATUS_FILE.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(WATCHLIST_STATUS_FILE)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["status"] = df["status"].astype(str).str.upper().str.strip()
    df["setup_type"] = df["setup_type"].astype(str).str.upper().str.strip()

    return df


def load_transactions() -> pd.DataFrame:
    if not WATCHLIST_TRANSACTIONS_FILE.exists():
        return pd.DataFrame(columns=REQUIRED_TX_COLUMNS)

    try:
        df = pd.read_csv(WATCHLIST_TRANSACTIONS_FILE)
    except Exception:
        return pd.DataFrame(columns=REQUIRED_TX_COLUMNS)

    if df.empty:
        return pd.DataFrame(columns=REQUIRED_TX_COLUMNS)

    for col in REQUIRED_TX_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[REQUIRED_TX_COLUMNS].copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["action"] = df["action"].astype(str).str.upper().str.strip()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


def last_action_map(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}

    df_sorted = df.sort_values("timestamp")
    result = {}

    for ticker, group in df_sorted.groupby("ticker"):
        last = group.iloc[-1]
        result[ticker] = str(last["action"]).upper().strip()

    return result


def build_action_rows(status_df: pd.DataFrame, last_actions: dict) -> list[dict]:
    rows = []
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    for _, row in status_df.iterrows():
        ticker = row["ticker"]
        status = row["status"]
        setup_type = row.get("setup_type", "PULLBACK")

        last_action = last_actions.get(ticker, "")

        # 🔴 REJECTED → UNWATCH
        if status == "REJECTED":
            if last_action != "UNWATCH":
                rows.append({
                    "timestamp": now,
                    "ticker": ticker,
                    "action": "UNWATCH",
                    "setup_type": setup_type,
                    "source": "decision_engine",
                    "note": "auto-removed (REJECTED)"
                })

        # 🟢 READY → BUY SIGNAL (nog geen echte trade)
        elif status == "READY":
            rows.append({
                "timestamp": now,
                "ticker": ticker,
                "action": "SIGNAL",
                "setup_type": setup_type,
                "source": "decision_engine",
                "note": "ready-to-buy signal"
            })

    return rows


def save_transactions(existing: pd.DataFrame, new_rows: list[dict]) -> None:
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = existing.copy()

    if combined.empty:
        combined = pd.DataFrame(columns=REQUIRED_TX_COLUMNS)

    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
    combined = combined.sort_values(["timestamp", "ticker"])
    combined["timestamp"] = combined["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    WATCHLIST_TRANSACTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(WATCHLIST_TRANSACTIONS_FILE, index=False)


def main() -> None:
    status_df = load_status()
    tx_df = load_transactions()

    if status_df.empty:
        print("No watchlist status found.")
        return

    last_actions = last_action_map(tx_df)
    new_rows = build_action_rows(status_df, last_actions)

    save_transactions(tx_df, new_rows)

    print(f"Decision engine processed.")
    print(f"New actions created: {len(new_rows)}")


if __name__ == "__main__":
    main()