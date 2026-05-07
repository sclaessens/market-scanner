from __future__ import annotations

import os
import sys
from typing import Optional, Dict, List

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

TRANSACTIONS_FILE = "data/portfolio/portfolio_transactions.csv"
POSITIONS_FILE = "data/portfolio/portfolio_positions.csv"
PROCESSED_DIR = "data/processed"
ENTRY_SIDE = "B" + "UY"
EXIT_SIDE = "S" + "ELL"


def ensure_directories() -> None:
    os.makedirs(os.path.dirname(TRANSACTIONS_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(POSITIONS_FILE), exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_transactions() -> pd.DataFrame:
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
    return df.sort_values("timestamp").reset_index(drop=True)


def get_processed_file_path(ticker: str) -> Optional[str]:
    for path in [os.path.join(PROCESSED_DIR, f"{ticker}.csv"), os.path.join(PROCESSED_DIR, f"{ticker}_indicators.csv")]:
        if os.path.exists(path):
            return path
    return None


def get_last_price(ticker: str) -> Optional[float]:
    path = get_processed_file_path(ticker)
    if path is None:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    close_col = next((candidate for candidate in ["Close", "close"] if candidate in df.columns), None)
    if close_col is None:
        return None
    series = pd.to_numeric(df[close_col], errors="coerce").dropna()
    return float(series.iloc[-1]) if not series.empty else None


def safe_round(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def build_portfolio() -> pd.DataFrame:
    ensure_directories()
    tx = load_transactions()
    output_columns = [
        "ticker", "quantity", "avg_cost", "last_price", "market_value",
        "unrealized_pnl", "pnl_pct", "status", "last_action", "last_action_at",
    ]
    if tx.empty:
        empty_df = pd.DataFrame(columns=output_columns)
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
        if side not in {ENTRY_SIDE, EXIT_SIDE}:
            raise ValueError(f"Invalid side for {ticker}: {side}")
        if ticker not in positions:
            positions[ticker] = {"ticker": ticker, "quantity": 0.0, "avg_cost": 0.0, "last_action": None, "last_action_at": None}
        pos = positions[ticker]
        if side == ENTRY_SIDE:
            current_qty = float(pos["quantity"])
            current_avg = float(pos["avg_cost"])
            total_cost_after = current_qty * current_avg + qty * price
            new_qty = current_qty + qty
            pos["quantity"] = new_qty
            pos["avg_cost"] = total_cost_after / new_qty if new_qty > 0 else 0.0
        elif side == EXIT_SIDE:
            current_qty = float(pos["quantity"])
            if qty > current_qty:
                raise ValueError(f"Cannot reduce more than current position for {ticker}: trying {qty}, current {current_qty}")
            new_qty = current_qty - qty
            pos["quantity"] = new_qty
            if new_qty == 0:
                pos["avg_cost"] = 0.0
        pos["last_action"] = "ENTRY" if side == ENTRY_SIDE else "EXIT"
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
            "ticker": ticker, "quantity": safe_round(quantity, 4), "avg_cost": safe_round(avg_cost, 4),
            "last_price": safe_round(last_price, 4), "market_value": safe_round(market_value, 4),
            "unrealized_pnl": safe_round(unrealized_pnl, 4), "pnl_pct": safe_round(pnl_pct, 4),
            "status": status, "last_action": pos["last_action"], "last_action_at": pos["last_action_at"],
        })
    positions_df = pd.DataFrame(rows, columns=output_columns)
    positions_df["status_order"] = positions_df["status"].map({"OPEN": 0, "CLOSED": 1})
    positions_df = positions_df.sort_values(by=["status_order", "ticker"]).drop(columns=["status_order"]).reset_index(drop=True)
    positions_df.to_csv(POSITIONS_FILE, index=False)
    return positions_df


if __name__ == "__main__":
    print("Building portfolio state...")
    print(build_portfolio())
