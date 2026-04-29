from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.settings import DATA_DIR


WATCHLIST_TRANSACTIONS_FILE = DATA_DIR / "watchlist" / "watchlist_transactions.csv"
WATCHLIST_ACTIVE_FILE = DATA_DIR / "watchlist" / "watchlist_active.csv"
PORTFOLIO_POSITIONS_FILE = DATA_DIR / "portfolio" / "portfolio_positions.csv"
PROCESSED_DIR = DATA_DIR / "processed"

REQUIRED_COLUMNS = [
    "timestamp",
    "ticker",
    "action",
    "setup_type",
    "source",
    "note",
]

VALID_SETUP_TYPES = {"PULLBACK", "BREAKOUT", "VCP"}


# =========================
# BASIC IO
# =========================

def ensure_parent_dirs() -> None:
    WATCHLIST_TRANSACTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_transactions() -> pd.DataFrame:
    df = read_csv_safe(WATCHLIST_TRANSACTIONS_FILE)
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df[REQUIRED_COLUMNS].copy()


def save_transactions(df: pd.DataFrame) -> None:
    ensure_parent_dirs()
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df[REQUIRED_COLUMNS].to_csv(WATCHLIST_TRANSACTIONS_FILE, index=False)


def append_transaction(row: dict) -> None:
    df = load_transactions()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_transactions(df)


# =========================
# DATA HELPERS
# =========================

def to_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def round_or_none(value: float) -> Optional[float]:
    if pd.isna(value):
        return None
    return round(float(value), 2)


def load_indicator_file(ticker: str) -> pd.DataFrame:
    possible_paths = [
        PROCESSED_DIR / f"{ticker}_indicators.csv",
        PROCESSED_DIR / f"{ticker}.csv",
    ]

    for path in possible_paths:
        if path.exists():
            df = read_csv_safe(path)
            if not df.empty:
                return df

    return pd.DataFrame()


def build_metric_alias_map(columns: list[str]) -> dict[str, Optional[str]]:
    normalized = {str(col).strip().upper(): col for col in columns}
    aliases = {
        "CLOSE": ["CLOSE", "close"],
        "MA20": ["MA20", "ma20"],
        "MA50": ["MA50", "ma50"],
        "MA200": ["MA200", "ma200"],
        "20D_HIGH": ["20D_HIGH", "HIGH_20D", "20d_high", "high_20d"],
        "20D_LOW": ["20D_LOW", "LOW_20D", "20d_low", "low_20d"],
        "VOLUME": ["VOLUME", "volume"],
        "AVG_VOL_20": ["AVG_VOL_20", "AVG_VOLUME_20D", "avg_vol_20", "avg_volume_20d"],
    }

    resolved: dict[str, Optional[str]] = {}
    for canonical, options in aliases.items():
        resolved[canonical] = None
        for option in options:
            key = str(option).strip().upper()
            if key in normalized:
                resolved[canonical] = normalized[key]
                break
    return resolved


def get_latest_metrics(ticker: str) -> Optional[dict]:
    df = load_indicator_file(ticker)
    if df.empty:
        return None

    alias_map = build_metric_alias_map(list(df.columns))
    required = ["CLOSE", "MA20", "MA50", "MA200", "20D_HIGH"]
    if any(alias_map.get(name) is None for name in required):
        return None

    latest = df.iloc[-1]

    close = to_float(latest.get(alias_map["CLOSE"]))
    ma20 = to_float(latest.get(alias_map["MA20"]))
    ma50 = to_float(latest.get(alias_map["MA50"]))
    ma200 = to_float(latest.get(alias_map["MA200"]))
    high_20d = to_float(latest.get(alias_map["20D_HIGH"]))
    low_20d = to_float(latest.get(alias_map["20D_LOW"])) if alias_map.get("20D_LOW") else float("nan")
    volume = to_float(latest.get(alias_map["VOLUME"])) if alias_map.get("VOLUME") else float("nan")
    avg_vol_20 = to_float(latest.get(alias_map["AVG_VOL_20"])) if alias_map.get("AVG_VOL_20") else float("nan")

    if any(pd.isna(x) for x in [close, ma20, ma50, ma200, high_20d]):
        return None

    distance_to_ma20_pct = (close - ma20) / ma20 if ma20 else float("nan")
    distance_to_high_pct = (high_20d - close) / high_20d if high_20d else float("nan")
    breakout_above_high_pct = (close - high_20d) / high_20d if high_20d else float("nan")

    return {
        "close": round(close, 2),
        "ma20": round(ma20, 2),
        "ma50": round(ma50, 2),
        "ma200": round(ma200, 2),
        "high_20d": round(high_20d, 2),
        "low_20d": round_or_none(low_20d),
        "volume": round_or_none(volume),
        "avg_vol_20": round_or_none(avg_vol_20),
        "distance_to_ma20_pct": distance_to_ma20_pct,
        "distance_to_high_pct": distance_to_high_pct,
        "breakout_above_high_pct": breakout_above_high_pct,
    }


def ticker_in_portfolio(ticker: str) -> bool:
    df = read_csv_safe(PORTFOLIO_POSITIONS_FILE)
    if df.empty or "ticker" not in df.columns:
        return False

    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.upper().str.strip()
        df = df[df["status"].isin(["OPEN", "", "NAN"])]

    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
        df = df[df["quantity"] > 0]

    return ticker.upper().strip() in set(df["ticker"].tolist())


def ticker_already_active(ticker: str) -> bool:
    df = read_csv_safe(WATCHLIST_ACTIVE_FILE)
    if df.empty or "ticker" not in df.columns:
        return False

    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    if "is_active" in df.columns:
        active_values = df["is_active"].astype(str).str.upper().str.strip()
        df = df[active_values.isin(["TRUE", "1", "YES"])]

    return ticker.upper().strip() in set(df["ticker"].tolist())


# =========================
# CLASSIFICATION LOGIC
# =========================

def classify_watch_candidate(ticker: str) -> dict:
    ticker = ticker.upper().strip()

    if ticker_in_portfolio(ticker):
        return {
            "ticker": ticker,
            "accepted": False,
            "setup_type": "",
            "action_now": "PORTFOLIO",
            "trigger_type": "none",
            "trigger_price": None,
            "reason": "Ticker zit al in portfolio. Portfolio heeft prioriteit boven watchlist.",
        }

    metrics = get_latest_metrics(ticker)
    if metrics is None:
        return {
            "ticker": ticker,
            "accepted": False,
            "setup_type": "",
            "action_now": "REJECTED",
            "trigger_type": "none",
            "trigger_price": None,
            "reason": "Geen geldige indicatorendata gevonden.",
        }

    close = metrics["close"]
    ma20 = metrics["ma20"]
    ma50 = metrics["ma50"]
    high_20d = metrics["high_20d"]
    dist_ma20 = metrics["distance_to_ma20_pct"]
    dist_high = metrics["distance_to_high_pct"]
    breakout_above_high = metrics["breakout_above_high_pct"]

    # Zwakke trend: niet opvolgen als nieuwe koopkans.
    if close < ma50:
        return {
            "ticker": ticker,
            "accepted": False,
            "setup_type": "REJECTED",
            "action_now": "REJECTED",
            "trigger_type": "none",
            "trigger_price": None,
            "reason": "Koers staat onder MA50. Trend is te zwak voor een nieuwe watchlist-entry.",
        }

    # Boven de breakout-trigger: nooit nog een stop-buy onder de actuele koers voorstellen.
    # Net boven de trigger mag nog BUY NOW zijn; verder erboven wordt automatisch PULLBACK.
    if breakout_above_high >= 0.002:
        if breakout_above_high <= 0.03:
            return {
                "ticker": ticker,
                "accepted": True,
                "setup_type": "BREAKOUT",
                "action_now": "BUY NOW",
                "trigger_type": "buy_now",
                "trigger_price": close,
                "reason": "Koers is net gecontroleerd boven de 20D high uitgebroken.",
            }

        return {
            "ticker": ticker,
            "accepted": True,
            "setup_type": "PULLBACK",
            "action_now": "SET LIMIT BUY",
            "trigger_type": "limit_buy",
            "trigger_price": ma20,
            "reason": "Breakout is al gebeurd. Niet najagen; wacht op pullback richting MA20.",
        }

    # Vlak onder 20D high: breakout kandidaat met stop buy.
    if 0 <= dist_high <= 0.015 and close >= ma20:
        return {
            "ticker": ticker,
            "accepted": True,
            "setup_type": "BREAKOUT",
            "action_now": "SET STOP BUY",
            "trigger_type": "buy_above",
            "trigger_price": high_20d,
            "reason": "Koers staat vlak onder de 20D high. Breakout-kandidaat.",
        }

    # Gezonde trend dicht bij MA20: pullback kandidaat.
    if close >= ma20 and dist_ma20 <= 0.025:
        return {
            "ticker": ticker,
            "accepted": True,
            "setup_type": "PULLBACK",
            "action_now": "BUY NOW",
            "trigger_type": "buy_now",
            "trigger_price": close,
            "reason": "Koers zit dicht bij MA20 in een gezonde trend.",
        }

    # Gezonde trend maar te ver boven MA20: pullback afwachten.
    if close > ma20 and dist_ma20 > 0.025:
        return {
            "ticker": ticker,
            "accepted": True,
            "setup_type": "PULLBACK",
            "action_now": "SET LIMIT BUY",
            "trigger_type": "limit_buy",
            "trigger_price": ma20,
            "reason": "Trend is gezond, maar koers staat te ver boven MA20. Wacht op betere pullback-entry.",
        }

    # Onder MA20 maar boven MA50: nog geen entry, wel opvolgen als pullback.
    return {
        "ticker": ticker,
        "accepted": True,
        "setup_type": "PULLBACK",
        "action_now": "WAIT",
        "trigger_type": "buy_above",
        "trigger_price": ma20,
        "reason": "Koers zit boven MA50 maar nog onder MA20. Eerst reclaim van MA20 afwachten.",
    }


def build_note(classification: dict) -> str:
    trigger_price = classification.get("trigger_price")
    trigger_price_text = "" if trigger_price is None else str(round(float(trigger_price), 2))

    parts = [
        f"auto_classified=1",
        f"action_now={classification.get('action_now', '')}",
        f"trigger_type={classification.get('trigger_type', '')}",
        f"trigger_price={trigger_price_text}",
        f"reason={classification.get('reason', '')}",
    ]
    return "; ".join(parts)


# =========================
# COMMAND PARSING
# =========================

def parse_command(args: list[str]) -> tuple[str, str, Optional[str]]:
    if not args:
        raise ValueError("Gebruik: python scripts/watchlist/parse_watchlist_commands.py WATCH NVDA")

    action = args[0].upper().strip()
    if action not in {"WATCH", "UNWATCH"}:
        raise ValueError("Alleen WATCH en UNWATCH worden ondersteund.")

    if len(args) < 2:
        raise ValueError("Ticker ontbreekt. Voorbeeld: WATCH NVDA")

    ticker = args[1].upper().strip()
    manual_setup = args[2].upper().strip() if len(args) >= 3 else None

    if manual_setup and manual_setup not in VALID_SETUP_TYPES:
        raise ValueError(f"Ongeldig setup_type: {manual_setup}. Gebruik PULLBACK, BREAKOUT of VCP.")

    return action, ticker, manual_setup


def handle_watch(ticker: str, manual_setup: Optional[str] = None) -> None:
    classification = classify_watch_candidate(ticker)

    if manual_setup:
        classification["accepted"] = True
        classification["setup_type"] = manual_setup
        classification["reason"] = f"Handmatig setup_type gekozen: {manual_setup}."

    if not classification["accepted"]:
        print(f"Niet toegevoegd: {ticker}")
        print(f"Reden: {classification['reason']}")
        return

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "action": "WATCH",
        "setup_type": classification["setup_type"],
        "source": "manual",
        "note": build_note(classification),
    }
    append_transaction(row)

    print(f"WATCH toegevoegd: {ticker}")
    print(f"Setup type: {classification['setup_type']}")
    print(f"Actie nu: {classification['action_now']}")
    print(f"Trigger: {classification['trigger_type']} {classification.get('trigger_price')}")
    print(f"Waarom: {classification['reason']}")


def handle_unwatch(ticker: str) -> None:
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "action": "UNWATCH",
        "setup_type": "",
        "source": "manual",
        "note": "manual unwatch",
    }
    append_transaction(row)

    print(f"UNWATCH toegevoegd: {ticker}")


def main() -> None:
    try:
        action, ticker, manual_setup = parse_command(sys.argv[1:])
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)

    if action == "WATCH":
        handle_watch(ticker, manual_setup)
    elif action == "UNWATCH":
        handle_unwatch(ticker)


if __name__ == "__main__":
    main()
