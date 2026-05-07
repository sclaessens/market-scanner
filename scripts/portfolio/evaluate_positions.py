from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

POSITIONS_FILE = "data/portfolio/portfolio_positions.csv"
REVIEW_FILE = "data/portfolio/portfolio_review.csv"
PROCESSED_DIR = "data/processed"

EXTENDED_ABOVE_MA20_PCT = 10.0


def ensure_directories() -> None:
    os.makedirs(os.path.dirname(REVIEW_FILE), exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_positions() -> pd.DataFrame:
    base_columns = [
        "ticker", "quantity", "avg_cost", "last_price", "market_value",
        "unrealized_pnl", "pnl_pct", "status", "last_action", "last_action_at",
    ]
    if not os.path.exists(POSITIONS_FILE):
        return pd.DataFrame(columns=base_columns)
    df = pd.read_csv(POSITIONS_FILE)
    if df.empty:
        return pd.DataFrame(columns=base_columns)
    required_cols = {"ticker", "quantity", "avg_cost", "last_price", "pnl_pct", "status"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in portfolio_positions.csv: {sorted(missing)}")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    for column in ["quantity", "avg_cost", "last_price", "pnl_pct"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def get_processed_file_path(ticker: str) -> Optional[str]:
    for path in [os.path.join(PROCESSED_DIR, f"{ticker}.csv"), os.path.join(PROCESSED_DIR, f"{ticker}_indicators.csv")]:
        if os.path.exists(path):
            return path
    return None


def get_latest_indicator_values(ticker: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    path = get_processed_file_path(ticker)
    if path is None:
        return None, None, None, None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None, None, None, None
    if df.empty:
        return None, None, None, None

    def get_last_numeric(column_candidates):
        for col in column_candidates:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                if not series.empty:
                    return float(series.iloc[-1])
        return None

    return (
        get_last_numeric(["Close", "close"]),
        get_last_numeric(["MA20", "ma20"]),
        get_last_numeric(["MA50", "ma50"]),
        get_last_numeric(["MA200", "ma200"]),
    )


def safe_round(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def evaluate_position_state(ticker: str, quantity: float, avg_cost: float, last_price: Optional[float], pnl_pct: Optional[float]) -> dict:
    close, ma20, ma50, ma200 = get_latest_indicator_values(ticker)
    reviewed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    price = close if close is not None else last_price

    base = {
        "ticker": ticker,
        "quantity": safe_round(quantity),
        "avg_cost": safe_round(avg_cost),
        "last_price": safe_round(price if price is not None else last_price),
        "pnl_pct": safe_round(pnl_pct),
        "ma20": safe_round(ma20),
        "ma50": safe_round(ma50),
        "ma200": safe_round(ma200),
        "reviewed_at": reviewed_at,
    }

    if price is None:
        return {**base, "exposure_state": "OPEN", "drawdown_state": "UNKNOWN", "risk_state": "DATA_GAP", "portfolio_reason": "missing_price_data"}
    if ma20 is None or ma50 is None or ma200 is None:
        return {**base, "exposure_state": "OPEN", "drawdown_state": "UNKNOWN", "risk_state": "DATA_GAP", "portfolio_reason": "missing_moving_average_data"}

    drawdown_state = "GAIN" if pnl_pct is not None and pnl_pct >= 0 else "DRAWDOWN"
    risk_state = "NORMAL"
    reason = "healthy_structure"

    if price < ma50 and price < ma200:
        risk_state = "STRUCTURE_BROKEN"
        reason = "below_ma50_and_ma200"
    elif price < ma50 and price >= ma200:
        risk_state = "STRUCTURE_WEAKENING"
        reason = "below_ma50_above_ma200"
    else:
        extended_pct = ((price - ma20) / ma20) * 100 if ma20 > 0 else None
        if extended_pct is not None and extended_pct >= EXTENDED_ABOVE_MA20_PCT:
            risk_state = "EXTENDED_PROFIT"
            reason = f"extended_{safe_round(extended_pct, 2)}pct_above_ma20"

    return {**base, "exposure_state": "OPEN", "drawdown_state": drawdown_state, "risk_state": risk_state, "portfolio_reason": reason}


def evaluate_positions() -> pd.DataFrame:
    ensure_directories()
    positions_df = load_positions()
    output_columns = [
        "ticker", "quantity", "avg_cost", "last_price", "pnl_pct", "ma20", "ma50",
        "ma200", "exposure_state", "drawdown_state", "risk_state", "portfolio_reason", "reviewed_at",
    ]
    if positions_df.empty:
        empty_df = pd.DataFrame(columns=output_columns)
        empty_df.to_csv(REVIEW_FILE, index=False)
        return empty_df
    open_positions = positions_df[positions_df["status"] == "OPEN"].copy()
    if open_positions.empty:
        empty_df = pd.DataFrame(columns=output_columns)
        empty_df.to_csv(REVIEW_FILE, index=False)
        return empty_df
    reviews = [evaluate_position_state(row["ticker"], float(row["quantity"]), float(row["avg_cost"]), row["last_price"] if pd.notna(row["last_price"]) else None, row["pnl_pct"] if pd.notna(row["pnl_pct"]) else None) for _, row in open_positions.iterrows()]
    review_df = pd.DataFrame(reviews, columns=output_columns)
    review_df = review_df.sort_values(by=["risk_state", "ticker"], ascending=[True, True]).reset_index(drop=True)
    review_df.to_csv(REVIEW_FILE, index=False)
    return review_df


if __name__ == "__main__":
    print("Evaluating open portfolio position states...")
    print(evaluate_positions())
