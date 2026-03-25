from __future__ import annotations

from typing import Optional

import pandas as pd

from config.settings import (
    MIN_PRICE,
    MIN_AVG_VOLUME,
    MIN_RR,
    TOP_SETUPS_PER_SECTION,
    VCP_LOOKBACK_DAYS,
    VCP_NEAR_HIGH_THRESHOLD,
    VCP_CONTRACTION_THRESHOLD,
)


def _has_required_columns(df: pd.DataFrame, columns: set[str]) -> bool:
    return not df.empty and columns.issubset(df.columns)


def _to_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _safe_pct(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b) or b == 0:
        return float("nan")
    return (a / b) - 1.0


def _grade_from_score(score: float) -> str:
    if score >= 8:
        return "A"
    if score >= 5:
        return "B"
    return "C"


def is_liquid_leader(df: pd.DataFrame) -> bool:
    required = {"Close", "AVG_VOL_20"}
    if not _has_required_columns(df, required) or len(df) < 60:
        return False

    latest = df.iloc[-1]
    close = latest.get("Close")
    avg_vol = latest.get("AVG_VOL_20")

    if pd.isna(close) or pd.isna(avg_vol):
        return False

    return close >= MIN_PRICE and avg_vol >= MIN_AVG_VOLUME


def detect_vcp(ticker: str, df: pd.DataFrame, regime: str = "NEUTRAL") -> Optional[dict]:
    required = {"Close", "High", "Low", "MA20", "MA50", "MA200", "AVG_VOL_20"}
    if not _has_required_columns(df, required):
        return None

    if len(df) < VCP_LOOKBACK_DAYS:
        return None

    if regime == "BEARISH":
        return None

    if not is_liquid_leader(df):
        return None

    recent = df.tail(VCP_LOOKBACK_DAYS).copy()
    latest = recent.iloc[-1]

    close = _to_float(latest["Close"])
    ma20 = _to_float(latest["MA20"])
    ma50 = _to_float(latest["MA50"])
    ma200 = _to_float(latest["MA200"])
    avg_vol = _to_float(latest["AVG_VOL_20"])

    if any(pd.isna(x) for x in [close, ma20, ma50, ma200, avg_vol]):
        return None

    recent_high = _to_float(recent["High"].max())
    recent_low = _to_float(recent["Low"].min())

    if pd.isna(recent_high) or pd.isna(recent_low) or recent_high <= 0:
        return None

    near_high = close >= recent_high * VCP_NEAR_HIGH_THRESHOLD
    contraction = ((recent_high - recent_low) / recent_high) <= VCP_CONTRACTION_THRESHOLD
    trend_aligned = close > ma20 > ma50 > ma200

    if not (near_high and contraction and trend_aligned):
        return None

    score_components = _score_common_components(df)
    score = (
        score_components["trend"]
        + score_components["momentum"]
        + score_components["position"]
        + 2.0
    )

    grade = _grade_from_score(score)

    return {
        "ticker": ticker,
        "setup": "VCP",
        "primary_setup": "VCP",
        "score": round(score, 2),
        "grade": grade,
        "score_trend": round(score_components["trend"], 2),
        "score_momentum": round(score_components["momentum"], 2),
        "score_position": round(score_components["position"] + 2.0, 2),
        "close": round(close, 2),
        "ma20": round(ma20, 2),
        "ma50": round(ma50, 2),
        "ma200": round(ma200, 2),
        "recent_high": round(recent_high, 2),
        "recent_low": round(recent_low, 2),
        "avg_vol": int(avg_vol),
        "summary": (
            f"{ticker} | VCP | grade {grade} | score {score:.2f} | "
            f"close {close:.2f} | {VCP_LOOKBACK_DAYS}d high {recent_high:.2f}"
        ),
    }


def _score_common_components(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]

    close = _to_float(latest["Close"])
    ma20 = _to_float(latest["MA20"])
    ma50 = _to_float(latest["MA50"])
    ma200 = _to_float(latest["MA200"])
    high_20 = _to_float(latest["20D_HIGH"])
    avg_vol = _to_float(latest["AVG_VOL_20"])

    score_trend = 0.0
    score_momentum = 0.0
    score_position = 0.0

    if close > ma20:
        score_trend += 1.0
    if ma20 > ma50:
        score_trend += 1.0
    if ma50 > ma200:
        score_trend += 1.0
    if close > ma50:
        score_trend += 1.0

    if len(df) >= 6:
        ret_5d = _safe_pct(close, _to_float(df["Close"].iloc[-6]))
        if not pd.isna(ret_5d):
            if ret_5d > 0.05:
                score_momentum += 2.0
            elif ret_5d > 0.0:
                score_momentum += 1.0

    if len(df) >= 11:
        ret_10d = _safe_pct(close, _to_float(df["Close"].iloc[-11]))
        if not pd.isna(ret_10d):
            if ret_10d > 0.08:
                score_momentum += 2.0
            elif ret_10d > 0.0:
                score_momentum += 1.0

    if not pd.isna(high_20) and high_20 > 0:
        distance_from_high = (high_20 - close) / high_20
        if distance_from_high <= 0.02:
            score_position += 3.0
        elif distance_from_high <= 0.05:
            score_position += 2.0
        elif distance_from_high <= 0.10:
            score_position += 1.0

    if not pd.isna(avg_vol):
        if avg_vol >= 5_000_000:
            score_position += 1.0
        elif avg_vol >= 2_000_000:
            score_position += 0.5

    return {
        "trend": score_trend,
        "momentum": score_momentum,
        "position": score_position,
    }


def build_tradeplan(df: pd.DataFrame, primary_setup: str) -> dict:
    required = {"Close", "MA20", "MA50", "ATR14", "20D_HIGH", "20D_LOW"}
    if not _has_required_columns(df, required):
        return {}

    latest = df.iloc[-1]

    close = _to_float(latest["Close"])
    ma20 = _to_float(latest["MA20"])
    ma50 = _to_float(latest["MA50"])
    atr = _to_float(latest["ATR14"])
    high_20 = _to_float(latest["20D_HIGH"])
    low_20 = _to_float(latest["20D_LOW"])

    if any(pd.isna(x) for x in [close, ma20, ma50, atr, high_20, low_20]):
        return {}

    if atr <= 0:
        return {}

    if primary_setup == "BREAKOUT":
        entry = max(close, high_20 * 1.005)
        stop = max(ma20, entry - (1.5 * atr))
    elif primary_setup == "VCP":
        entry = max(close, high_20 * 1.003)
        stop = max(ma20, entry - (1.3 * atr))
    else:
        entry = ma20
        stop = ma50 - (0.5 * atr)

    risk = entry - stop
    if risk <= 0:
        return {}

    target = entry + (2.2 * risk)
    rr = (target - entry) / risk

    if rr < MIN_RR:
        return {}

    return {
        "entry": round(float(entry), 2),
        "stop": round(float(stop), 2),
        "target": round(float(target), 2),
        "rr": round(float(rr), 2),
    }


def scan_ticker(ticker: str, df: pd.DataFrame, regime: str) -> Optional[dict]:
    required = {
        "Close",
        "High",
        "Low",
        "Volume",
        "MA20",
        "MA50",
        "MA200",
        "ATR14",
        "20D_HIGH",
        "20D_LOW",
        "AVG_VOL_20",
    }

    if not _has_required_columns(df, required):
        return None

    if len(df) < 220:
        return None

    latest = df.iloc[-1]

    close = _to_float(latest["Close"])
    ma20 = _to_float(latest["MA20"])
    ma50 = _to_float(latest["MA50"])
    ma200 = _to_float(latest["MA200"])
    atr = _to_float(latest["ATR14"])
    high_20 = _to_float(latest["20D_HIGH"])
    low_20 = _to_float(latest["20D_LOW"])
    volume = _to_float(latest["Volume"])
    avg_vol = _to_float(latest["AVG_VOL_20"])

    if any(pd.isna(x) for x in [close, ma20, ma50, ma200, atr, high_20, low_20, volume, avg_vol]):
        return None

    if close < MIN_PRICE or avg_vol < MIN_AVG_VOLUME:
        return None

    if regime == "BEARISH":
        return None

    distance_ma20 = (close - ma20) / ma20 if ma20 else float("nan")
    distance_ma50 = (close - ma50) / ma50 if ma50 else float("nan")
    distance_high = (high_20 - close) / high_20 if high_20 else float("nan")

    trend_ok = close > ma50 and ma50 > ma200
    strong_trend = close > ma20 > ma50 > ma200
    momentum_ok = len(df) >= 11 and close > df["Close"].iloc[-6] and close > df["Close"].iloc[-11]

    pullback = (
        trend_ok
        and not pd.isna(distance_ma20)
        and -0.08 <= distance_ma20 <= 0.04
        and not pd.isna(distance_ma50)
        and distance_ma50 >= 0.0
        and close >= 0.85 * high_20
    )

    breakout = (
        trend_ok
        and not pd.isna(distance_high)
        and distance_high <= 0.03
        and volume >= 1.35 * avg_vol
        and close > ma20
    )

    vcp_result = detect_vcp(ticker, df, regime)
    vcp = vcp_result is not None

    if breakout:
        primary_setup = "BREAKOUT"
    elif pullback:
        primary_setup = "PULLBACK"
    elif vcp:
        primary_setup = "VCP"
    else:
        return None

    setup_types = []
    if pullback:
        setup_types.append("PULLBACK")
    if breakout:
        setup_types.append("BREAKOUT")
    if vcp:
        setup_types.append("VCP")

    score_components = _score_common_components(df)

    if pullback:
        score_components["position"] += 1.0
    if breakout:
        score_components["momentum"] += 1.0
    if vcp:
        score_components["position"] += 2.0
    if strong_trend:
        score_components["trend"] += 1.0
    if regime == "BULLISH":
        score_components["trend"] += 1.0

    total_score = (
        score_components["trend"]
        + score_components["momentum"]
        + score_components["position"]
    )

    grade = _grade_from_score(total_score)

    tradeplan = build_tradeplan(df, primary_setup)
    if not tradeplan:
        return None

    result = {
        "ticker": ticker,
        "setup": ", ".join(setup_types),
        "primary_setup": primary_setup,
        "grade": grade,
        "score": round(float(total_score), 2),
        "score_trend": round(float(score_components["trend"]), 2),
        "score_momentum": round(float(score_components["momentum"]), 2),
        "score_position": round(float(score_components["position"]), 2),
        "trend_ok": trend_ok,
        "momentum_ok": momentum_ok,
        "regime_ok": regime != "BEARISH",
        "close": round(float(close), 2),
        "ma20": round(float(ma20), 2),
        "ma50": round(float(ma50), 2),
        "ma200": round(float(ma200), 2),
        "atr14": round(float(atr), 2),
        "high_20d": round(float(high_20), 2),
        "low_20d": round(float(low_20), 2),
        "avg_vol_20": int(avg_vol),
    }

    result.update(tradeplan)
    return result


def rank_setups(setups: list[dict], top_n: int = TOP_SETUPS_PER_SECTION) -> list[dict]:
    grade_priority = {"A": 3, "B": 2, "C": 1}

    ranked = sorted(
        setups,
        key=lambda x: (
            grade_priority.get(x.get("grade", "C"), 0),
            x.get("score", 0),
            x.get("rr", 0),
            x.get("ticker", ""),
        ),
        reverse=True,
    )

    return ranked[:top_n]
