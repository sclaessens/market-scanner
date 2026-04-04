from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

from config.settings import DATA_DIR


WATCHLIST_ACTIVE_FILE = DATA_DIR / "watchlist" / "watchlist_active.csv"
WATCHLIST_STATUS_FILE = DATA_DIR / "watchlist" / "watchlist_status.csv"
MARKET_REGIME_FILE = DATA_DIR / "processed" / "market_regime.csv"
PROCESSED_DIR = DATA_DIR / "processed"
THRESHOLDS_FILE = PROJECT_ROOT / "config" / "thresholds.yaml"


DEFAULT_THRESHOLDS = {
    "watchlist": {
        "ready_distance_to_ma20_pct": 0.025,
        "neutral_ready_distance_to_ma20_pct": 0.015,
        "max_extended_from_ma20_pct": 0.06,
        "ready_near_high_pct": 0.015,
        "reject_below_ma50": True,
        "expire_after_days": 20,
    }
}


def load_thresholds() -> dict:
    if yaml is None or not THRESHOLDS_FILE.exists():
        return DEFAULT_THRESHOLDS

    try:
        with THRESHOLDS_FILE.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return DEFAULT_THRESHOLDS

    merged = DEFAULT_THRESHOLDS.copy()
    merged_watchlist = dict(DEFAULT_THRESHOLDS["watchlist"])
    merged_watchlist.update(data.get("watchlist", {}))
    merged["watchlist"] = merged_watchlist
    return merged


def load_market_regime() -> str:
    if not MARKET_REGIME_FILE.exists():
        return "NEUTRAL"

    try:
        df = pd.read_csv(MARKET_REGIME_FILE)
    except Exception:
        return "NEUTRAL"

    if df.empty:
        return "NEUTRAL"

    last_row = df.iloc[-1]

    for col in ["regime", "Regime", "market_regime"]:
        if col in df.columns:
            value = str(last_row[col]).strip().upper()
            if value:
                return value

    return "NEUTRAL"


def load_watchlist_active() -> pd.DataFrame:
    if not WATCHLIST_ACTIVE_FILE.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(WATCHLIST_ACTIVE_FILE)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    if "setup_type" in df.columns:
        df["setup_type"] = df["setup_type"].astype(str).str.upper().str.strip()
    else:
        df["setup_type"] = "PULLBACK"

    if "is_active" not in df.columns:
        df["is_active"] = True

    return df


def load_indicator_file(ticker: str) -> pd.DataFrame:
    possible_paths = [
        PROCESSED_DIR / f"{ticker}_indicators.csv",
        PROCESSED_DIR / f"{ticker}.csv",
    ]

    for path in possible_paths:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                continue

    return pd.DataFrame()


def to_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def parse_timestamp(value) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    try:
        return pd.to_datetime(value)
    except Exception:
        return pd.NaT


def get_latest_metrics(df: pd.DataFrame) -> Optional[dict]:
    required_cols = {"Close", "MA20", "MA50", "MA200", "20D_HIGH"}
    if df.empty or not required_cols.issubset(df.columns):
        return None

    latest = df.iloc[-1]

    close = to_float(latest.get("Close"))
    ma20 = to_float(latest.get("MA20"))
    ma50 = to_float(latest.get("MA50"))
    ma200 = to_float(latest.get("MA200"))
    high_20d = to_float(latest.get("20D_HIGH"))
    avg_vol_20 = to_float(latest.get("AVG_VOL_20")) if "AVG_VOL_20" in df.columns else float("nan")
    volume = to_float(latest.get("Volume")) if "Volume" in df.columns else float("nan")

    if any(pd.isna(x) for x in [close, ma20, ma50, ma200, high_20d]):
        return None

    distance_to_ma20_pct = (close - ma20) / ma20 if ma20 else float("nan")
    distance_to_high_pct = (high_20d - close) / high_20d if high_20d else float("nan")

    return {
        "close": round(close, 2),
        "ma20": round(ma20, 2),
        "ma50": round(ma50, 2),
        "ma200": round(ma200, 2),
        "high_20d": round(high_20d, 2),
        "avg_vol_20": round(avg_vol_20, 2) if not pd.isna(avg_vol_20) else None,
        "volume": round(volume, 2) if not pd.isna(volume) else None,
        "distance_to_ma20_pct": distance_to_ma20_pct,
        "distance_to_high_pct": distance_to_high_pct,
    }


def evaluate_pullback(metrics: dict, regime: str, cfg: dict) -> tuple[str, str, str]:
    close = metrics["close"]
    ma20 = metrics["ma20"]
    ma50 = metrics["ma50"]
    dist_ma20 = metrics["distance_to_ma20_pct"]

    reject_below_ma50 = cfg["reject_below_ma50"]
    ready_distance = cfg["ready_distance_to_ma20_pct"]
    neutral_ready_distance = cfg["neutral_ready_distance_to_ma20_pct"]
    max_extended = cfg["max_extended_from_ma20_pct"]

    if reject_below_ma50 and close < ma50:
        return "REJECTED", "none", "below_ma50"

    if pd.isna(dist_ma20):
        return "WAIT", "wait", "missing_ma20_distance"

    if regime == "BEARISH":
        return "WAIT", "wait", "bearish_regime"

    if dist_ma20 > max_extended:
        return "WAIT", "wait", "too_extended_above_ma20"

    if close < ma20:
        return "WAIT", "wait", "still_below_ma20"

    if regime == "BULLISH":
        if -ready_distance <= dist_ma20 <= ready_distance and close >= ma50:
            return "READY", "buy", "pullback_near_ma20_bullish"
        return "WAIT", "wait", "trend_ok_but_not_in_buy_zone"

    if regime == "NEUTRAL":
        if -neutral_ready_distance <= dist_ma20 <= neutral_ready_distance and close >= ma50:
            return "READY", "buy", "pullback_near_ma20_neutral"
        return "WAIT", "wait", "neutral_regime_wait_for_better_entry"

    return "WAIT", "wait", "unsupported_regime"


def evaluate_breakout(metrics: dict, regime: str, cfg: dict) -> tuple[str, str, str]:
    close = metrics["close"]
    ma20 = metrics["ma20"]
    ma50 = metrics["ma50"]
    dist_high = metrics["distance_to_high_pct"]

    reject_below_ma50 = cfg["reject_below_ma50"]
    ready_near_high = cfg["ready_near_high_pct"]

    if reject_below_ma50 and close < ma50:
        return "REJECTED", "none", "below_ma50"

    if pd.isna(dist_high):
        return "WAIT", "wait", "missing_high_distance"

    if regime == "BEARISH":
        return "WAIT", "wait", "bearish_regime"

    if close < ma20:
        return "WAIT", "wait", "below_ma20"

    if regime == "NEUTRAL":
        return "WAIT", "wait", "neutral_regime_wait_for_breakout_confirmation"

    if regime == "BULLISH" and dist_high <= ready_near_high and close >= ma20 and close >= ma50:
        return "READY", "buy", "near_breakout_trigger_bullish"

    return "WAIT", "wait", "not_close_enough_to_breakout"


def evaluate_vcp(metrics: dict, regime: str, cfg: dict) -> tuple[str, str, str]:
    close = metrics["close"]
    ma20 = metrics["ma20"]
    ma50 = metrics["ma50"]
    dist_high = metrics["distance_to_high_pct"]

    reject_below_ma50 = cfg["reject_below_ma50"]
    ready_near_high = cfg["ready_near_high_pct"]

    if reject_below_ma50 and close < ma50:
        return "REJECTED", "none", "below_ma50"

    if pd.isna(dist_high):
        return "WAIT", "wait", "missing_high_distance"

    if regime == "BEARISH":
        return "WAIT", "wait", "bearish_regime"

    if close < ma20:
        return "WAIT", "wait", "below_ma20"

    if regime == "NEUTRAL":
        return "WAIT", "wait", "neutral_regime_wait_for_vcp_confirmation"

    if regime == "BULLISH" and dist_high <= ready_near_high and close >= ma20 and close >= ma50:
        return "READY", "buy", "vcp_near_pivot_bullish"

    return "WAIT", "wait", "vcp_not_ready"


def apply_expiry(status: str, added_at: pd.Timestamp, expire_after_days: int) -> tuple[str, Optional[str]]:
    if pd.isna(added_at):
        return status, None

    days_active = (pd.Timestamp.now().normalize() - added_at.normalize()).days
    if days_active > expire_after_days and status not in {"READY", "REJECTED"}:
        return "EXPIRED", f"expired_after_{days_active}_days"

    return status, None


def evaluate_row(row: pd.Series, regime: str, cfg: dict) -> dict:
    ticker = str(row.get("ticker", "")).upper().strip()
    setup_type = str(row.get("setup_type", "PULLBACK")).upper().strip()
    is_active = bool(row.get("is_active", True))
    added_at = parse_timestamp(row.get("added_at"))

    base_result = {
        "ticker": ticker,
        "is_active": is_active,
        "setup_type": setup_type,
        "status": "WAIT",
        "entry_bias": "wait",
        "added_at": row.get("added_at"),
        "last_reviewed_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reason": "",
        "regime": regime,
        "close": None,
        "ma20": None,
        "ma50": None,
        "ma200": None,
        "high_20d": None,
    }

    if not is_active:
        base_result["status"] = "INACTIVE"
        base_result["entry_bias"] = "none"
        base_result["reason"] = "not_active"
        return base_result

    df_ind = load_indicator_file(ticker)
    metrics = get_latest_metrics(df_ind)

    if metrics is None:
        base_result["status"] = "REJECTED"
        base_result["entry_bias"] = "none"
        base_result["reason"] = "missing_indicator_data"
        return base_result

    base_result["close"] = metrics["close"]
    base_result["ma20"] = metrics["ma20"]
    base_result["ma50"] = metrics["ma50"]
    base_result["ma200"] = metrics["ma200"]
    base_result["high_20d"] = metrics["high_20d"]

    if setup_type == "BREAKOUT":
        status, entry_bias, reason = evaluate_breakout(metrics, regime, cfg)
    elif setup_type == "VCP":
        status, entry_bias, reason = evaluate_vcp(metrics, regime, cfg)
    else:
        status, entry_bias, reason = evaluate_pullback(metrics, regime, cfg)

    status, expiry_reason = apply_expiry(status, added_at, cfg["expire_after_days"])
    if expiry_reason:
        reason = expiry_reason
        entry_bias = "none"

    base_result["status"] = status
    base_result["entry_bias"] = entry_bias
    base_result["reason"] = reason
    return base_result


def main() -> None:
    thresholds = load_thresholds()
    cfg = thresholds["watchlist"]
    regime = load_market_regime()

    active_df = load_watchlist_active()

    WATCHLIST_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)

    if active_df.empty:
        empty_cols = [
            "ticker",
            "is_active",
            "setup_type",
            "status",
            "entry_bias",
            "added_at",
            "last_reviewed_at",
            "reason",
            "regime",
            "close",
            "ma20",
            "ma50",
            "ma200",
            "high_20d",
        ]
        pd.DataFrame(columns=empty_cols).to_csv(WATCHLIST_STATUS_FILE, index=False)
        print(f"No active watchlist found. Empty status file written to: {WATCHLIST_STATUS_FILE}")
        return

    rows = []
    for _, row in active_df.iterrows():
        rows.append(evaluate_row(row, regime, cfg))

    result_df = pd.DataFrame(rows)

    sort_order = {"READY": 0, "WAIT": 1, "REJECTED": 2, "EXPIRED": 3, "INACTIVE": 4}
    result_df["sort_key"] = result_df["status"].map(sort_order).fillna(99)
    result_df = result_df.sort_values(["sort_key", "ticker"]).drop(columns=["sort_key"])

    result_df.to_csv(WATCHLIST_STATUS_FILE, index=False)
    print(f"Watchlist status written to: {WATCHLIST_STATUS_FILE}")


if __name__ == "__main__":
    main()