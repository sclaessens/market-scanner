import os
import sys
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd

# Zorg dat project root in path zit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

POSITIONS_FILE = "data/portfolio/portfolio_positions.csv"
REVIEW_FILE = "data/portfolio/portfolio_review.csv"
PROCESSED_DIR = "data/processed"

EXTENDED_ABOVE_MA20_PCT = 10.0  # MVP-threshold


# === HELPERS ===

def ensure_directories() -> None:
    os.makedirs(os.path.dirname(REVIEW_FILE), exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_positions() -> pd.DataFrame:
    if not os.path.exists(POSITIONS_FILE):
        return pd.DataFrame(columns=[
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

    df = pd.read_csv(POSITIONS_FILE)

    if df.empty:
        return pd.DataFrame(columns=[
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

    required_cols = {"ticker", "quantity", "avg_cost", "last_price", "pnl_pct", "status"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in portfolio_positions.csv: {sorted(missing)}")

    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["avg_cost"] = pd.to_numeric(df["avg_cost"], errors="coerce")
    df["last_price"] = pd.to_numeric(df["last_price"], errors="coerce")
    df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce")

    return df


def get_processed_file_path(ticker: str) -> Optional[str]:
    candidates = [
        os.path.join(PROCESSED_DIR, f"{ticker}.csv"),
        os.path.join(PROCESSED_DIR, f"{ticker}_indicators.csv"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def get_latest_indicator_values(ticker: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Returnt tuple:
    (close, ma20, ma50, ma200)
    """
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

    close = get_last_numeric(["Close", "close"])
    ma20 = get_last_numeric(["MA20", "ma20"])
    ma50 = get_last_numeric(["MA50", "ma50"])
    ma200 = get_last_numeric(["MA200", "ma200"])

    return close, ma20, ma50, ma200


def safe_round(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


# === DECISION LOGIC ===

def evaluate_position(
    ticker: str,
    quantity: float,
    avg_cost: float,
    last_price: Optional[float],
    pnl_pct: Optional[float],
) -> dict:
    """
    Bepaal HOLD / TRIM / SELL / REVIEW voor één open positie.
    """
    close, ma20, ma50, ma200 = get_latest_indicator_values(ticker)

    reviewed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Gebruik bij voorkeur close uit processed file; fallback naar last_price uit positions
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

    # 1. Ontbrekende prijsdata
    if price is None:
        return {
            **base,
            "decision": "REVIEW",
            "reason": "missing_price_data",
            "risk_flag": "HIGH",
        }

    # 2. Ontbrekende MA-data
    if ma20 is None or ma50 is None or ma200 is None:
        return {
            **base,
            "decision": "REVIEW",
            "reason": "missing_moving_average_data",
            "risk_flag": "MEDIUM",
        }

    # 3. Echte trendbreuk: onder MA50 én onder MA200
    if price < ma50 and price < ma200:
        return {
            **base,
            "decision": "SELL",
            "reason": "below_ma50_and_ma200",
            "risk_flag": "HIGH",
        }

    # 4. Onder MA50 maar nog boven MA200 = verzwakking, nog geen harde SELL
    if price < ma50 and price >= ma200:
        return {
            **base,
            "decision": "REVIEW",
            "reason": "below_ma50_above_ma200",
            "risk_flag": "MEDIUM",
        }

    # 5. Sterk extended boven MA20
    extended_pct = ((price - ma20) / ma20) * 100 if ma20 > 0 else None
    if extended_pct is not None and extended_pct >= EXTENDED_ABOVE_MA20_PCT:
        return {
            **base,
            "decision": "TRIM",
            "reason": f"extended_{safe_round(extended_pct, 2)}pct_above_ma20",
            "risk_flag": "MEDIUM",
        }

    # 6. Gezonde structuur
    if price > ma20 and price > ma50:
        return {
            **base,
            "decision": "HOLD",
            "reason": "above_ma20_and_ma50",
            "risk_flag": "LOW",
        }

    # 7. Restcategorie
    return {
        **base,
        "decision": "REVIEW",
        "reason": "mixed_structure",
        "risk_flag": "MEDIUM",
    }


def evaluate_positions() -> pd.DataFrame:
    """
    Evalueer alle OPEN posities en schrijf portfolio_review.csv weg.
    """
    ensure_directories()
    positions_df = load_positions()

    output_columns = [
        "ticker",
        "quantity",
        "avg_cost",
        "last_price",
        "pnl_pct",
        "ma20",
        "ma50",
        "ma200",
        "decision",
        "reason",
        "risk_flag",
        "reviewed_at",
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

    reviews = []
    for _, row in open_positions.iterrows():
        review = evaluate_position(
            ticker=row["ticker"],
            quantity=float(row["quantity"]),
            avg_cost=float(row["avg_cost"]),
            last_price=row["last_price"] if pd.notna(row["last_price"]) else None,
            pnl_pct=row["pnl_pct"] if pd.notna(row["pnl_pct"]) else None,
        )
        reviews.append(review)

    review_df = pd.DataFrame(reviews)

    decision_order = {
        "SELL": 0,
        "TRIM": 1,
        "REVIEW": 2,
        "HOLD": 3,
    }

    review_df["decision_order"] = review_df["decision"].map(decision_order).fillna(99)
    review_df = review_df.sort_values(
        by=["decision_order", "ticker"],
        ascending=[True, True]
    ).drop(columns=["decision_order"]).reset_index(drop=True)

    review_df.to_csv(REVIEW_FILE, index=False)
    return review_df


if __name__ == "__main__":
    print("Evaluating open portfolio positions...")
    df = evaluate_positions()
    print("\n=== PORTFOLIO REVIEW ===")
    print(df)