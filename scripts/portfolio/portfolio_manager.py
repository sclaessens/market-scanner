import os
from datetime import datetime
import pandas as pd

# === PATHS ===
TRANSACTIONS_FILE = "data/portfolio/portfolio_transactions.csv"
POSITIONS_FILE = "data/portfolio/portfolio_positions.csv"


# === HELPERS ===

def ensure_directories():
    os.makedirs(os.path.dirname(TRANSACTIONS_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(POSITIONS_FILE), exist_ok=True)


def load_transactions() -> pd.DataFrame:
    if not os.path.exists(TRANSACTIONS_FILE):
        return pd.DataFrame(columns=["timestamp", "ticker", "side", "quantity", "price"])

    df = pd.read_csv(TRANSACTIONS_FILE)
    return df


def save_transactions(df: pd.DataFrame):
    df.to_csv(TRANSACTIONS_FILE, index=False)


def save_positions(df: pd.DataFrame):
    df.to_csv(POSITIONS_FILE, index=False)


# === CORE FUNCTIONS ===

def log_trade(ticker: str, side: str, quantity: float, price: float):
    """
    Log a BUY or SELL transaction.
    """
    ensure_directories()

    side = side.upper()
    if side not in ["BUY", "SELL"]:
        raise ValueError("side must be BUY or SELL")

    new_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker.upper(),
        "side": side,
        "quantity": float(quantity),
        "price": float(price),
    }

    df = load_transactions()
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    save_transactions(df)


def build_positions():
    """
    Rebuild portfolio positions from transactions.
    """
    ensure_directories()

    df = load_transactions()

    if df.empty:
        empty_df = pd.DataFrame(columns=[
            "ticker",
            "quantity",
            "avg_cost",
            "status",
            "last_action",
            "last_action_at"
        ])
        save_positions(empty_df)
        return empty_df

    # Sort chronologisch
    df = df.sort_values("timestamp")

    positions = {}

    for _, row in df.iterrows():
        ticker = row["ticker"]
        side = row["side"]
        qty = float(row["quantity"])
        price = float(row["price"])
        timestamp = row["timestamp"]

        if ticker not in positions:
            positions[ticker] = {
                "quantity": 0.0,
                "avg_cost": 0.0,
                "last_action": None,
                "last_action_at": None
            }

        pos = positions[ticker]

        if side == "BUY":
            total_cost = pos["avg_cost"] * pos["quantity"] + price * qty
            new_qty = pos["quantity"] + qty

            pos["quantity"] = new_qty
            pos["avg_cost"] = total_cost / new_qty if new_qty > 0 else 0.0


        elif side == "SELL":

            if qty > pos["quantity"]:
                raise ValueError(f"Cannot sell more than position size for {ticker}")

            pos["quantity"] -= qty

        pos["last_action"] = side
        pos["last_action_at"] = timestamp

    # Convert naar DataFrame
    rows = []
    for ticker, pos in positions.items():
        status = "OPEN" if pos["quantity"] > 0 else "CLOSED"

        rows.append({
            "ticker": ticker,
            "quantity": round(pos["quantity"], 4),
            "avg_cost": round(pos["avg_cost"], 4),
            "status": status,
            "last_action": pos["last_action"],
            "last_action_at": pos["last_action_at"]
        })

    positions_df = pd.DataFrame(rows)

    # Sorteer op actieve posities eerst
    positions_df["status_order"] = positions_df["status"].map({
        "OPEN": 0,
        "CLOSED": 1
    })

    positions_df = positions_df.sort_values(
        by=["status_order", "ticker"]
    ).drop(columns=["status_order"])

    save_positions(positions_df)

    return positions_df


# === QUICK TEST / MANUAL RUN ===

if __name__ == "__main__":
    print("Rebuilding portfolio...")
    df = build_positions()
    print(df)