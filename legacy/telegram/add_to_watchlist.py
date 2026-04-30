import sys
from datetime import datetime
import pandas as pd
from pathlib import Path

WATCHLIST_FILE = Path("data/watchlist/watchlist_active.csv")


def load_watchlist():
    if WATCHLIST_FILE.exists():
        return pd.read_csv(WATCHLIST_FILE)
    return pd.DataFrame(columns=[
        "ticker", "is_active", "setup_type", "added_at",
        "source", "created_at", "notes"
    ])


def save_watchlist(df):
    WATCHLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(WATCHLIST_FILE, index=False)


def add_ticker(ticker: str, setup_type: str):
    ticker = ticker.upper()
    setup_type = setup_type.upper()

    df = load_watchlist()

    # Check of ticker al bestaat
    if ticker in df["ticker"].values:
        print(f"{ticker} staat al op de watchlist.")
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_row = {
        "ticker": ticker,
        "is_active": True,
        "setup_type": setup_type,
        "added_at": now,
        "source": "telegram",
        "created_at": now,
        "notes": ""
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_watchlist(df)

    print(f"{ticker} toegevoegd als {setup_type}.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Gebruik: python add_to_watchlist.py TICKER SETUP_TYPE")
        sys.exit(1)

    ticker = sys.argv[1]
    setup_type = sys.argv[2]

    add_ticker(ticker, setup_type)