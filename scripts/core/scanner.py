from __future__ import annotations

from typing import Optional

import pandas as pd

from config.settings import (
    MIN_PRICE,
    MIN_AVG_VOLUME,
    MIN_RR,
    VCP_LOOKBACK_DAYS,
    VCP_NEAR_HIGH_THRESHOLD,
    VCP_CONTRACTION_THRESHOLD,
)


MIN_HISTORY_ROWS = 220


def _has_required_columns(df: pd.DataFrame, columns: set[str]) -> bool:
    return not df.empty and columns.issubset(df.columns)


def _to_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _safe_pct(current: float, reference: float) -> float:
    if pd.isna(current) or pd.isna(reference) or reference == 0:
        return float("nan")
    return (current / reference) - 1.0


def _recent_swing_low(df: pd.DataFrame, lookback: int = 10) -> float:
    if df.empty or "Low" not in df.columns:
        return float("nan")
    recent = df.tail(lookback)
    if recent.empty:
        return float("nan")
    return _to_float(recent["Low"].min())


def _recent_swing_high(df: pd.DataFrame, lookback: int = 10) -> float:
    if df.empty or "High" not in df.columns:
        return float("nan")
    recent = df.tail(lookback)
    if recent.empty:
        return float("nan")
    return _to_float(recent["High"].max())


def _compute_return_20d(df: pd.DataFrame) -> float:
    if df.empty or "Close" not in df.columns or len(df) < 21:
        return float("nan")

    latest_close = _to_float(df["Close"].iloc[-1])
    ref_close = _to_float(df["Close"].iloc[-21])

    if pd.isna(latest_close) or pd.isna(ref_close) or ref_close == 0:
        return float("nan")

    return (latest_close / ref_close) - 1.0


def is_liquid_leader(df: pd.DataFrame) -> bool:
    required = {"Close", "AVG_VOL_20"}
    if not _has_required_columns(df, required) or len(df) < 60:
        return False

    latest = df.iloc[-1]
    close = _to_float(latest.get("Close"))
    avg_vol = _to_float(latest.get("AVG_VOL_20"))

    if pd.isna(close) or pd.isna(avg_vol):
        return False

    return close >= MIN_PRICE and avg_vol >= MIN_AVG_VOLUME


def _score_common_components(df: pd.DataFrame, qqq_return_20d: float) -> dict:
    latest = df.iloc[-1]

    close = _to_float(latest["Close"])
    ma20 = _to_float(latest["MA20"])
    ma50 = _to_float(latest["MA50"])
    ma200 = _to_float(latest["MA200"])
    high_20 = _to_float(latest["20D_HIGH"])
    low_20 = _to_float(latest["20D_LOW"])
    avg_vol = _to_float(latest["AVG_VOL_20"])
    atr = _to_float(latest["ATR14"])

    score_trend = 0.0
    score_momentum = 0.0
    score_position = 0.0
    score_relative_strength = 0.0

    if close > ma20:
        score_trend += 1.0
    if close > ma50:
        score_trend += 1.0
    if ma20 > ma50:
        score_trend += 1.0
    if ma50 > ma200:
        score_trend += 1.0

    ret_5d = float("nan")
    ret_10d = float("nan")
    ret_20d = _compute_return_20d(df)

    if len(df) >= 6:
        ret_5d = _safe_pct(close, _to_float(df["Close"].iloc[-6]))
        if not pd.isna(ret_5d):
            if ret_5d > 0.04:
                score_momentum += 2.0
            elif ret_5d > 0.00:
                score_momentum += 1.0
            elif ret_5d < -0.03:
                score_momentum -= 1.0

    if len(df) >= 11:
        ret_10d = _safe_pct(close, _to_float(df["Close"].iloc[-11]))
        if not pd.isna(ret_10d):
            if ret_10d > 0.08:
                score_momentum += 2.0
            elif ret_10d > 0.02:
                score_momentum += 1.0
            elif ret_10d < -0.05:
                score_momentum -= 1.0

    rs_20d = float("nan")
    if not pd.isna(ret_20d) and not pd.isna(qqq_return_20d):
        rs_20d = ret_20d - qqq_return_20d

        if rs_20d >= 0.08:
            score_relative_strength += 2.0
        elif rs_20d >= 0.03:
            score_relative_strength += 1.0
        elif rs_20d <= -0.05:
            score_relative_strength -= 2.0
        elif rs_20d < 0:
            score_relative_strength -= 1.0

    if not pd.isna(high_20) and high_20 > 0:
        distance_from_high = (high_20 - close) / high_20
        if distance_from_high <= 0.02:
            score_position += 2.0
        elif distance_from_high <= 0.05:
            score_position += 1.0

    if not pd.isna(low_20) and high_20 > low_20:
        range_position = (close - low_20) / (high_20 - low_20)
        if range_position >= 0.80:
            score_position += 1.5
        elif range_position >= 0.65:
            score_position += 1.0

    if not pd.isna(avg_vol):
        if avg_vol >= 5_000_000:
            score_position += 1.0
        elif avg_vol >= 2_000_000:
            score_position += 0.5

    atr_pct = float("nan")
    if not pd.isna(atr) and close > 0:
        atr_pct = atr / close

        if atr_pct > 0.08:
            score_position -= 0.5

        if atr_pct < 0.03:
            score_position -= 1.0
        if atr_pct < 0.02:
            score_trend *= 0.7
            score_momentum *= 0.7
            score_position *= 0.7

    return {
        "trend": round(score_trend, 2),
        "momentum": round(score_momentum, 2),
        "position": round(score_position, 2),
        "relative_strength": round(score_relative_strength, 2),
        "ret_20d": round(ret_20d, 4) if not pd.isna(ret_20d) else None,
        "rs_20d": round(rs_20d, 4) if not pd.isna(rs_20d) else None,
        "atr_pct": round(atr_pct, 4) if not pd.isna(atr_pct) else None,
    }


def detect_vcp(ticker: str, df: pd.DataFrame, regime: str = "NEUTRAL") -> Optional[dict]:
    required = {
        "Close",
        "High",
        "Low",
        "MA20",
        "MA50",
        "MA200",
        "AVG_VOL_20",
        "ATR14",
        "Volume",
    }
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

    recent_vol = _to_float(recent.tail(5)["Volume"].mean())
    prior_vol = _to_float(recent.head(max(len(recent) - 5, 1))["Volume"].mean())
    volume_dry = (
        not pd.isna(recent_vol)
        and not pd.isna(prior_vol)
        and recent_vol <= prior_vol
    )

    if not (near_high and contraction and trend_aligned and volume_dry):
        return None

    return {
        "ticker": ticker,
        "setup": "VCP",
        "primary_setup": "VCP",
    }


def build_tradeplan(df: pd.DataFrame, primary_setup: str) -> dict:
    required = {"Close", "MA20", "MA50", "ATR14", "20D_HIGH", "20D_LOW", "High", "Low"}
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

    swing_low_5 = _recent_swing_low(df, lookback=5)
    swing_low_10 = _recent_swing_low(df, lookback=10)
    swing_high_5 = _recent_swing_high(df, lookback=5)

    if primary_setup == "BREAKOUT":
        entry = max(close, high_20 * 1.002)
        stop = min(
            x for x in [swing_low_5, ma20, entry - (1.5 * atr)] if not pd.isna(x)
        )
        target_multiple = 2.6

    elif primary_setup == "VCP":
        entry = max(
            close,
            swing_high_5 * 1.001 if not pd.isna(swing_high_5) else close,
        )
        stop = min(
            x for x in [swing_low_5, ma20, entry - (1.3 * atr)] if not pd.isna(x)
        )
        target_multiple = 3.0

    else:
        entry = max(x for x in [ma20, close] if not pd.isna(x))
        stop = min(
            x for x in [swing_low_10, ma50, entry - (1.2 * atr)] if not pd.isna(x)
        )
        target_multiple = 2.0

    risk = entry - stop
    if risk <= 0:
        return {}

    target = entry + (target_multiple * risk)
    rr = (target - entry) / risk

    if rr < MIN_RR:
        return {}

    return {
        "entry": round(float(entry), 2),
        "stop": round(float(stop), 2),
        "target": round(float(target), 2),
        "rr": round(float(rr), 2),
    }


def scan_ticker(
    ticker: str,
    df: pd.DataFrame,
    regime: str,
    qqq_return_20d: float,
) -> Optional[dict]:
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

    if len(df) < MIN_HISTORY_ROWS:
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

    if any(
        pd.isna(x)
        for x in [close, ma20, ma50, ma200, atr, high_20, low_20, volume, avg_vol]
    ):
        return None

    if close < MIN_PRICE or avg_vol < MIN_AVG_VOLUME:
        return None

    if regime == "BEARISH":
        return None

    distance_ma20 = (close - ma20) / ma20 if ma20 else float("nan")
    distance_ma50 = (close - ma50) / ma50 if ma50 else float("nan")
    distance_high = (high_20 - close) / high_20 if high_20 else float("nan")
    atr_pct = atr / close if close > 0 else float("nan")

    trend_ok = close > ma50 and ma50 > ma200
    strong_trend = close > ma20 > ma50 > ma200

    ret_5d = (
        _safe_pct(close, _to_float(df["Close"].iloc[-6]))
        if len(df) >= 6
        else float("nan")
    )
    ret_10d = (
        _safe_pct(close, _to_float(df["Close"].iloc[-11]))
        if len(df) >= 11
        else float("nan")
    )
    momentum_ok = (
        not pd.isna(ret_5d)
        and not pd.isna(ret_10d)
        and ret_5d > -0.02
        and ret_10d > 0.00
    )

    pullback = (
        trend_ok
        and momentum_ok
        and not pd.isna(distance_ma20)
        and -0.08 <= distance_ma20 <= 0.04
        and not pd.isna(distance_ma50)
        and distance_ma50 >= -0.02
        and close >= 0.80 * high_20
        and not pd.isna(atr_pct)
        and atr_pct <= 0.10
    )

    breakout = (
        trend_ok
        and not pd.isna(distance_high)
        and distance_high <= 0.08
        and volume >= 1.10 * avg_vol
        and close > ma20
        and (pd.isna(ret_5d) or ret_5d >= -0.01)
    )

    vcp_result = detect_vcp(ticker, df, regime)
    vcp = vcp_result is not None

    if breakout:
        primary_setup = "BREAKOUT"
    elif vcp:
        primary_setup = "VCP"
    elif pullback:
        primary_setup = "PULLBACK"
    else:
        return None

    setup_types = []
    if pullback:
        setup_types.append("PULLBACK")
    if breakout:
        setup_types.append("BREAKOUT")
    if vcp:
        setup_types.append("VCP")

    score_components = _score_common_components(df, qqq_return_20d)

    if pullback:
        score_components["position"] += 0.5

    if breakout:
        score_components["momentum"] += 1.5
        score_components["position"] += 0.5

    if vcp:
        rs_20d = score_components.get("rs_20d")

        if rs_20d is not None and rs_20d >= 0.03:
            score_components["position"] += 2.0
            score_components["trend"] += 0.5
        else:
            score_components["position"] += 0.5

    if strong_trend:
        score_components["trend"] += 1.0

    if regime == "BULLISH":
        score_components["trend"] += 1.0
    elif regime == "NEUTRAL":
        score_components["trend"] -= 0.5

    raw_total_score = (
        score_components["trend"]
        + score_components["momentum"]
        + score_components["position"]
        + score_components["relative_strength"]
    )

    tradeplan = build_tradeplan(df, primary_setup)

    if not tradeplan:
        tradeplan = {
            "entry": None,
            "stop": None,
            "target": None,
            "rr": None,
        }

    return {
        "ticker": ticker,
        "setup": ", ".join(setup_types),
        "primary_setup": primary_setup,
        "raw_score": round(float(raw_total_score), 2),
        "score": round(float(raw_total_score), 2),
        "score_trend": round(float(score_components["trend"]), 2),
        "score_momentum": round(float(score_components["momentum"]), 2),
        "score_position": round(float(score_components["position"]), 2),
        "score_relative_strength": round(
            float(score_components["relative_strength"]), 2
        ),
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
        "ret_20d_pct": (
            round(float(score_components["ret_20d"] * 100), 2)
            if score_components["ret_20d"] is not None
            else None
        ),
        "rs_20d_pct": (
            round(float(score_components["rs_20d"] * 100), 2)
            if score_components["rs_20d"] is not None
            else None
        ),
        "atr_pct": (
            round(float(score_components["atr_pct"] * 100), 2)
            if score_components["atr_pct"] is not None
            else None
        ),
        **tradeplan,
    }


def _assign_relative_grades(ranked: list[dict]) -> list[dict]:
    if not ranked:
        return ranked

    total = len(ranked)
    top_a_count = max(1, round(total * 0.20))
    top_b_count = max(1, round(total * 0.50))

    for idx, setup in enumerate(ranked):
        primary = setup.get("primary_setup", "")
        raw_score = float(setup.get("raw_score", setup.get("score", 0)))
        rr = float(setup.get("rr", 0))
        trend_ok = bool(setup.get("trend_ok", False))
        momentum_ok = bool(setup.get("momentum_ok", False))
        regime_ok = bool(setup.get("regime_ok", False))

        rs_20d_pct = setup.get("rs_20d_pct")
        atr_pct = setup.get("atr_pct")

        rs_ok_for_a = rs_20d_pct is not None and rs_20d_pct >= 3.0
        atr_ok_for_a = atr_pct is not None and atr_pct >= 2.5

        allow_a = (
            regime_ok
            and trend_ok
            and momentum_ok
            and rr >= 2.0
            and rs_ok_for_a
            and atr_ok_for_a
        )

        if primary == "PULLBACK":
            allow_a = allow_a and raw_score >= 8.0
        elif primary == "BREAKOUT":
            allow_a = allow_a and raw_score >= 8.5
        elif primary == "VCP":
            allow_a = allow_a and raw_score >= 9.0

        if idx < top_a_count and allow_a:
            grade = "A"
        elif idx < top_b_count and raw_score >= 6.5:
            grade = "B"
        else:
            grade = "C"

        setup["grade"] = grade
        setup["score"] = round(raw_score, 2)

    return ranked


def rank_setups(setups: list[dict], top_n: int = 10) -> list[dict]:
    primary_priority = {"VCP": 3, "BREAKOUT": 2, "PULLBACK": 1}

    ranked = sorted(
        setups,
        key=lambda x: (
            primary_priority.get(x.get("primary_setup", ""), 0),
            x.get("raw_score", x.get("score", 0)),
            x.get("rr", 0),
            x.get("ticker", ""),
        ),
        reverse=True,
    )

    ranked = _assign_relative_grades(ranked)

    non_c_ranked = [s for s in ranked if s.get("grade") != "C"]
    if len(non_c_ranked) >= min(top_n, 3):
        ranked = non_c_ranked

    return ranked[:top_n]