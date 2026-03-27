import os
from datetime import datetime
from typing import Dict, Tuple, Optional

import pandas as pd


WATCHLIST_FILE = "data/watchlist/watchlist_active.csv"
MARKET_REGIME_FILE = "data/processed/market_regime.csv"
OUTPUT_FILE = "data/watchlist/watchlist_status.csv"
PROCESSED_DIR = "data/features"

# Simpele defaults voor MVP.
# Later kan dit naar config/thresholds.yaml.
DEFAULT_MAX_WATCHLIST_AGE_DAYS = 20
DEFAULT_PULLBACK_READY_DISTANCE_TO_MA20_PCT = 2.0
DEFAULT_PULLBACK_MAX_EXTENDED_FROM_MA20_PCT = 6.0
DEFAULT_BREAKOUT_READY_BUFFER_PCT = 1.0
DEFAULT_VCP_READY_BUFFER_PCT = 1.0


def ensure_directories() -> None:
    os.makedirs("data/watchlist", exist_ok=True)


def load_watchlist() -> pd.DataFrame:
    if not os.path.exists(WATCHLIST_FILE):
        raise FileNotFoundError(f"Missing watchlist file: {WATCHLIST_FILE}")

    df = pd.read_csv(WATCHLIST_FILE)

    if df.empty:
        return df

    required_cols = {
        "ticker",
        "is_active",
        "setup_type",
        "last_action_at",
        "last_action",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"watchlist_active.csv missing columns: {sorted(missing)}")

    # added_at is gewenst volgens je docs, maar build_watchlist.py gebruikte eerder last_action_at.
    # Daarom maken we added_at aan als die ontbreekt.
    if "added_at" not in df.columns:
        df["added_at"] = df["last_action_at"]

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["setup_type"] = df["setup_type"].fillna("GENERAL").astype(str).str.upper().str.strip()
    df["is_active"] = df["is_active"].astype(str).str.lower().isin(["true", "1", "yes"])
    df["added_at"] = pd.to_datetime(df["added_at"], errors="coerce")
    df["last_action_at"] = pd.to_datetime(df["last_action_at"], errors="coerce")

    return df[df["is_active"]].copy()


def load_market_regime() -> str:
    if not os.path.exists(MARKET_REGIME_FILE):
        return "NEUTRAL"

    df = pd.read_csv(MARKET_REGIME_FILE)
    if df.empty:
        return "NEUTRAL"

    # flexibel omgaan met kolomnamen
    for col in ["regime", "Regime", "market_regime"]:
        if col in df.columns:
            value = str(df.iloc[-1][col]).upper().strip()
            if value:
                return value

    return "NEUTRAL"


def resolve_indicator_file(ticker: str) -> Optional[str]:
    candidates = [
        os.path.join(PROCESSED_DIR, f"{ticker}_indicators.csv"),
        os.path.join(PROCESSED_DIR, f"{ticker}.csv"),
        os.path.join(PROCESSED_DIR, f"{ticker}_processed.csv"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}

    for col in df.columns:
        c = col.strip()

        if c == "Close":
            rename_map[col] = "close"
        elif c == "MA20":
            rename_map[col] = "ma20"
        elif c == "MA50":
            rename_map[col] = "ma50"
        elif c == "MA200":
            rename_map[col] = "ma200"
        elif c in ["20D_High", "high_20d", "High_20d", "20d_high"]:
            rename_map[col] = "high_20d"
        elif c in ["20D_Low", "low_20d", "Low_20d", "20d_low"]:
            rename_map[col] = "low_20d"
        elif c in ["ATR14", "atr14", "ATR_14"]:
            rename_map[col] = "atr14"
        elif c in ["Volume", "volume"]:
            rename_map[col] = "volume"
        elif c in ["AvgVolume20", "avg_volume_20d", "avg_volume20", "AverageVolume20"]:
            rename_map[col] = "avg_volume_20d"
        elif c in ["ret_5d", "Ret_5d"]:
            rename_map[col] = "ret_5d"
        elif c in ["ret_10d", "Ret_10d"]:
            rename_map[col] = "ret_10d"
        elif c in ["Date", "date"]:
            rename_map[col] = "date"
        elif c in ["20D_HIGH", "20d_high", "High_20d"]:
            rename_map[col] = "high_20d"
        elif c in ["20D_LOW", "20d_low", "Low_20d"]:
            rename_map[col] = "low_20d"
        elif c in ["AVG_VOL_20", "avg_vol_20", "avg_volume_20d"]:
            rename_map[col] = "avg_volume_20d"

    return df.rename(columns=rename_map)


def load_latest_indicators(ticker: str) -> Optional[pd.Series]:
    path = resolve_indicator_file(ticker)
    if path is None:
        return None

    df = pd.read_csv(path)
    if df.empty:
        return None

    df = normalize_columns(df)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")

    return df.iloc[-1]


def calc_pct_distance(a: float, b: float) -> Optional[float]:
    if b is None or pd.isna(b) or b == 0:
        return None
    if a is None or pd.isna(a):
        return None
    return ((a - b) / b) * 100.0


def evaluate_pullback(latest: pd.Series, regime: str) -> Tuple[str, str, str]:
    close = latest.get("close")
    ma20 = latest.get("ma20")
    ma50 = latest.get("ma50")
    high_20d = latest.get("high_20d")

    if pd.isna(close) or pd.isna(ma20) or pd.isna(ma50):
        return "WAIT", "wait", "missing core pullback indicators"

    if close < ma50:
        return "REJECTED", "avoid", "trend below MA50"

    if regime == "BEARISH":
        return "WAIT", "wait", "market regime bearish"

    dist_ma20 = calc_pct_distance(close, ma20)
    if dist_ma20 is None:
        return "WAIT", "wait", "cannot calculate distance to MA20"

    # Klaar voor koopzone: dicht bij MA20, trend intact
    if abs(dist_ma20) <= DEFAULT_PULLBACK_READY_DISTANCE_TO_MA20_PCT:
        return "READY", "buy", "pullback near MA20 with trend intact"

    # Nog te ver weg maar trend is nog goed
    if dist_ma20 > DEFAULT_PULLBACK_READY_DISTANCE_TO_MA20_PCT:
        if dist_ma20 > DEFAULT_PULLBACK_MAX_EXTENDED_FROM_MA20_PCT:
            return "WAIT", "wait", "too extended above MA20"
        return "WAIT", "wait", "still above ideal pullback zone"

    # Onder MA20 maar nog boven MA50 = nog mogelijk, maar geen directe entry
    if close >= ma50:
        if high_20d is not None and not pd.isna(high_20d):
            return "WAIT", "wait", "pullback deeper than MA20 but still above MA50"
        return "WAIT", "wait", "below MA20 but trend not broken"

    return "REJECTED", "avoid", "pullback structure broken"


def evaluate_breakout(latest: pd.Series, regime: str) -> Tuple[str, str, str]:
    close = latest.get("close")
    ma20 = latest.get("ma20")
    ma50 = latest.get("ma50")
    high_20d = latest.get("high_20d")

    if pd.isna(close) or pd.isna(ma20) or pd.isna(ma50) or pd.isna(high_20d):
        return "WAIT", "wait", "missing breakout indicators"

    if close < ma50:
        return "REJECTED", "avoid", "breakout candidate below MA50"

    if regime == "BEARISH":
        return "WAIT", "wait", "market regime bearish"

    breakout_trigger = high_20d * (1 - DEFAULT_BREAKOUT_READY_BUFFER_PCT / 100.0)

    if close >= breakout_trigger:
        return "READY", "buy", "near or through 20D breakout trigger"

    return "WAIT", "wait", "not yet above breakout trigger"


def evaluate_vcp(latest, regime):
    close = latest.get("close")
    ma20 = latest.get("ma20")
    ma50 = latest.get("ma50")
    high_20d = latest.get("high_20d")

    if pd.isna(close) or pd.isna(ma20) or pd.isna(ma50) or pd.isna(high_20d):
        return "WAIT", "wait", "missing basic indicators"

    if close < ma50:
        return "REJECTED", "avoid", "below MA50"

    if regime == "BEARISH":
        return "WAIT", "wait", "bearish regime"

    # simpele breakout logic
    trigger = high_20d * 0.99

    if close >= trigger:
        return "READY", "buy", "near breakout"

    return "WAIT", "wait", "VCP not yet at trigger"


def evaluate_general(latest: pd.Series, regime: str) -> Tuple[str, str, str]:
    close = latest.get("close")
    ma20 = latest.get("ma20")
    ma50 = latest.get("ma50")

    if pd.isna(close) or pd.isna(ma20) or pd.isna(ma50):
        return "WAIT", "wait", "missing general indicators"

    if close < ma50:
        return "REJECTED", "avoid", "trend below MA50"

    if regime == "BEARISH":
        return "WAIT", "wait", "market regime bearish"

    if abs(calc_pct_distance(close, ma20) or 999) <= DEFAULT_PULLBACK_READY_DISTANCE_TO_MA20_PCT:
        return "READY", "buy", "in acceptable buy zone"

    return "WAIT", "wait", "watching for better entry"


def evaluate_setup(setup_type: str, latest: pd.Series, regime: str) -> Tuple[str, str, str]:
    setup_type = (setup_type or "GENERAL").upper().strip()

    if setup_type == "PULLBACK":
        return evaluate_pullback(latest, regime)

    if setup_type == "BREAKOUT":
        return evaluate_breakout(latest, regime)

    if setup_type == "VCP":
        return evaluate_vcp(latest, regime)

    return evaluate_general(latest, regime)


def check_expired(added_at: pd.Timestamp) -> bool:
    if pd.isna(added_at):
        return False
    age_days = (pd.Timestamp.now() - added_at).days
    return age_days > DEFAULT_MAX_WATCHLIST_AGE_DAYS


def build_status_row(row: pd.Series, regime: str) -> Dict:
    ticker = row["ticker"]
    setup_type = row.get("setup_type", "GENERAL")
    added_at = row.get("added_at")

    if check_expired(added_at):
        return {
            "ticker": ticker,
            "is_active": True,
            "setup_type": setup_type,
            "status": "EXPIRED",
            "entry_bias": "avoid",
            "added_at": added_at,
            "last_reviewed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reason": "watchlist item expired without trigger",
        }

    latest = load_latest_indicators(ticker)
    if latest is None:
        return {
            "ticker": ticker,
            "is_active": True,
            "setup_type": setup_type,
            "status": "WAIT",
            "entry_bias": "wait",
            "added_at": added_at,
            "last_reviewed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reason": "missing indicators file",
        }

    status, entry_bias, reason = evaluate_setup(setup_type, latest, regime)

    return {
        "ticker": ticker,
        "is_active": True,
        "setup_type": setup_type,
        "status": status,
        "entry_bias": entry_bias,
        "added_at": added_at,
        "last_reviewed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reason": reason,
    }


def evaluate_watchlist() -> pd.DataFrame:
    ensure_directories()

    watchlist = load_watchlist()

    if watchlist.empty:
        empty = pd.DataFrame(columns=[
            "ticker",
            "is_active",
            "setup_type",
            "status",
            "entry_bias",
            "added_at",
            "last_reviewed_at",
            "reason",
        ])
        empty.to_csv(OUTPUT_FILE, index=False)
        print("No active watchlist items.")
        return empty

    regime = load_market_regime()

    rows = []
    for _, row in watchlist.iterrows():
        rows.append(build_status_row(row, regime))

    status_df = pd.DataFrame(rows)

    # nette sortering: READY eerst, dan WAIT, dan REJECTED, dan EXPIRED
    priority = {
        "READY": 0,
        "WAIT": 1,
        "REJECTED": 2,
        "EXPIRED": 3,
        "BOUGHT": 4,
    }
    status_df["_sort"] = status_df["status"].map(priority).fillna(99)
    status_df = status_df.sort_values(["_sort", "ticker"]).drop(columns=["_sort"])

    status_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Watchlist evaluated: {len(status_df)} tickers")
    print(status_df[["ticker", "setup_type", "status", "reason"]].to_string(index=False))

    return status_df


if __name__ == "__main__":
    evaluate_watchlist()
