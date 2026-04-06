import os
import sys
from typing import Optional, Dict, List

import pandas as pd

# Zorg dat project root in path zit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

TRANSACTIONS_FILE = "data/portfolio/portfolio_transactions.csv"
POSITIONS_FILE = "data/portfolio/portfolio_positions.csv"
PROCESSED_DIR = "data/processed"


# === HELPERS ===

def ensure_directories() -> None:
    """Maak benodigde mappen aan indien nodig."""
    os.makedirs(os.path.dirname(TRANSACTIONS_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(POSITIONS_FILE), exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_transactions() -> pd.DataFrame:
    """
    Laad transacties uit CSV.
    Verwachte kolommen:
    timestamp, ticker, side, quantity, price
    """
    if not os.path.exists(TRANSACTIONS_FILE):
        return pd.DataFrame(columns=["timestamp", "ticker", "side", "quantity", "price"])

    df = pd.read_csv(TRANSACTIONS_FILE)

    if df.empty:
        return pd.DataFrame(columns=["timestamp", "ticker", "side", "quantity", "price"])

    required_cols = {"timestamp", "ticker", "side", "quantity", "price"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in transactions file: {sorted(missing)}")

    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["side"] = df["side"].astype(str).str.upper()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=["timestamp", "ticker", "side", "quantity", "price"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def get_processed_file_path(ticker: str) -> Optional[str]:
    """
    Zoek een bruikbaar processed bestand voor de ticker.
    Ondersteunt beide naamgevingen:
    - data/processed/{ticker}.csv
    - data/processed/{ticker}_indicators.csv
    """
    candidates = [
        os.path.join(PROCESSED_DIR, f"{ticker}.csv"),
        os.path.join(PROCESSED_DIR, f"{ticker}_indicators.csv"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def get_last_price(ticker: str) -> Optional[float]:
    """
    Haal de meest recente Close op uit processed data.
    Returnt None als bestand/kolom/data ontbreekt.
    """
    path = get_processed_file_path(ticker)
    if path is None:
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if df.empty:
        return None

    close_col = None
    for candidate in ["Close", "close"]:
        if candidate in df.columns:
            close_col = candidate
            break

    if close_col is None:
        return None

    series = pd.to_numeric(df[close_col], errors="coerce").dropna()
    if series.empty:
        return None

    return float(series.iloc[-1])


def safe_round(value: Optional[float], digits: int = 4) -> Optional[float]:
    """Rond floats af, maar laat None ongemoeid."""
    if value is None:
        return None
    return round(float(value), digits)


# === CORE BUILD LOGIC ===

def build_portfolio() -> pd.DataFrame:
    """
    Bouw actuele portfolio-posities op basis van transacties
    en verrijk met marktdata uit processed files.
    """
    ensure_directories()

    tx = load_transactions()

    if tx.empty:
        empty_df = pd.DataFrame(columns=[
            "ticker",
            "quantity",
            "avg_cost",
            "last_price",
            "market_value",
            "unrealized_pnl",
            "pnl_pct",
            "status",
            "last_action",
            "last_action_at",
        ])
        empty_df.to_csv(POSITIONS_FILE, index=False)
        return empty_df

    positions: Dict[str, Dict[str, object]] = {}

    for _, row in tx.iterrows():
        ticker = row["ticker"]
        side = row["side"]
        qty = float(row["quantity"])
        price = float(row["price"])
        timestamp = str(row["timestamp"])

        if qty <= 0:
            raise ValueError(f"Invalid quantity for {ticker}: {qty}")
        if price <= 0:
            raise ValueError(f"Invalid price for {ticker}: {price}")
        if side not in {"BUY", "SELL"}:
            raise ValueError(f"Invalid side for {ticker}: {side}")

        if ticker not in positions:
            positions[ticker] = {
                "ticker": ticker,
                "quantity": 0.0,
                "avg_cost": 0.0,
                "last_action": None,
                "last_action_at": None,
            }

        pos = positions[ticker]

        if side == "BUY":
            current_qty = float(pos["quantity"])
            current_avg = float(pos["avg_cost"])

            total_cost_before = current_qty * current_avg
            total_cost_after = total_cost_before + (qty * price)
            new_qty = current_qty + qty

            pos["quantity"] = new_qty
            pos["avg_cost"] = total_cost_after / new_qty if new_qty > 0 else 0.0

        elif side == "SELL":
            current_qty = float(pos["quantity"])

            if qty > current_qty:
                raise ValueError(
                    f"Cannot sell more than current position for {ticker}: "
                    f"trying to sell {qty}, current quantity is {current_qty}"
                )

            new_qty = current_qty - qty
            pos["quantity"] = new_qty

            # Als positie volledig dicht is, avg_cost resetten
            if new_qty == 0:
                pos["avg_cost"] = 0.0

        pos["last_action"] = side
        pos["last_action_at"] = timestamp

    rows: List[Dict[str, object]] = []

    for ticker, pos in positions.items():
        quantity = float(pos["quantity"])
        avg_cost = float(pos["avg_cost"])
        last_price = get_last_price(ticker)

        status = "OPEN" if quantity > 0 else "CLOSED"

        if last_price is not None and quantity > 0:
            market_value = quantity * last_price
            unrealized_pnl = (last_price - avg_cost) * quantity
            pnl_pct = ((last_price - avg_cost) / avg_cost * 100) if avg_cost > 0 else None
        else:
            market_value = 0.0 if quantity == 0 else None
            unrealized_pnl = 0.0 if quantity == 0 else None
            pnl_pct = None

        rows.append({
            "ticker": ticker,
            "quantity": safe_round(quantity, 4),
            "avg_cost": safe_round(avg_cost, 4),
            "last_price": safe_round(last_price, 4),
            "market_value": safe_round(market_value, 4),
            "unrealized_pnl": safe_round(unrealized_pnl, 4),
            "pnl_pct": safe_round(pnl_pct, 4),
            "status": status,
            "last_action": pos["last_action"],
            "last_action_at": pos["last_action_at"],
        })

    positions_df = pd.DataFrame(rows)

    # OPEN eerst, dan CLOSED
    positions_df["status_order"] = positions_df["status"].map({
        "OPEN": 0,
        "CLOSED": 1,
    })

    positions_df = positions_df.sort_values(
        by=["status_order", "ticker"],
        ascending=[True, True]
    ).drop(columns=["status_order"]).reset_index(drop=True)

    positions_df.to_csv(POSITIONS_FILE, index=False)
    return positions_df


if __name__ == "__main__":
    print("Building portfolio...")
    portfolio_df = build_portfolio()
    print("\n=== PORTFOLIO POSITIONS ===")
    print(portfolio_df)