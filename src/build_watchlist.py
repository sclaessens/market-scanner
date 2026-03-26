import pandas as pd
import os

TRANSACTIONS_FILE = "data/watchlist/watchlist_transactions.csv"
OUTPUT_FILE = "data/watchlist/watchlist_active.csv"

os.makedirs("data/watchlist", exist_ok=True)


def build_watchlist():
    if not os.path.exists(TRANSACTIONS_FILE):
        print("No transactions file found.")
        return

    df = pd.read_csv(TRANSACTIONS_FILE)

    if df.empty:
        print("No transactions.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    latest = df.groupby("ticker").last().reset_index()

    active = latest[latest["action"] == "WATCH"].copy()

    active["is_active"] = True

    active = active[[
        "ticker",
        "is_active",
        "setup_type",
        "timestamp",
        "action"
    ]]

    active.rename(columns={
        "timestamp": "last_action_at",
        "action": "last_action"
    }, inplace=True)

    active.to_csv(OUTPUT_FILE, index=False)

    print(f"Active watchlist saved: {len(active)} tickers")


if __name__ == "__main__":
    build_watchlist()
