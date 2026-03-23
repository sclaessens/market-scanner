import pandas as pd

from config.settings import (
    MIN_PRICE,
    MIN_AVG_VOLUME,
    TOP_SETUPS_PER_SECTION,
    VCP_LOOKBACK_DAYS,
    VCP_NEAR_HIGH_THRESHOLD,
    VCP_CONTRACTION_THRESHOLD,
)


def is_liquid_leader(df: pd.DataFrame) -> bool:
    if df.empty or len(df) < 60:
        return False

    latest = df.iloc[-1]

    close = latest.get("Close")
    avg_vol = latest.get("AVG_VOL_20")

    if pd.isna(close) or pd.isna(avg_vol):
        return False

    return close >= MIN_PRICE and avg_vol >= MIN_AVG_VOLUME


def detect_vcp(ticker: str, df: pd.DataFrame):
    if df.empty or len(df) < VCP_LOOKBACK_DAYS:
        return None

    recent = df.tail(VCP_LOOKBACK_DAYS).copy()
    latest = recent.iloc[-1]

    close = latest.get("Close")
    ma20 = latest.get("MA20")
    ma50 = latest.get("MA50")
    ma200 = latest.get("MA200")
    avg_vol = latest.get("AVG_VOL_20")

    if any(pd.isna(x) for x in [close, ma20, ma50, ma200, avg_vol]):
        return None

    if not is_liquid_leader(df):
        return None

    recent_high = recent["High"].max()
    recent_low = recent["Low"].min()

    if pd.isna(recent_high) or pd.isna(recent_low) or recent_high <= 0:
        return None

    near_high = close >= recent_high * VCP_NEAR_HIGH_THRESHOLD
    contraction = ((recent_high - recent_low) / recent_high) <= VCP_CONTRACTION_THRESHOLD
    trend_aligned = close > ma20 > ma50 > ma200

    if not (near_high and contraction and trend_aligned):
        return None

    score = score_vcp(df, recent_high, recent_low)

    return {
        "ticker": ticker,
        "setup": "VCP",
        "score": round(score, 2),
        "close": round(float(close), 2),
        "recent_high": round(float(recent_high), 2),
        "recent_low": round(float(recent_low), 2),
        "avg_vol": int(avg_vol),
        "summary": (
            f"{ticker} | score {score:.2f} | close {close:.2f} | "
            f"60d high {recent_high:.2f} | compact range"
        ),
    }


def score_vcp(df: pd.DataFrame, recent_high: float, recent_low: float) -> float:
    latest = df.iloc[-1]

    close = float(latest["Close"])
    ma20 = float(latest["MA20"])
    ma50 = float(latest["MA50"])
    ma200 = float(latest["MA200"])
    avg_vol = float(latest["AVG_VOL_20"])

    trend_score = 0
    if close > ma20:
        trend_score += 1
    if ma20 > ma50:
        trend_score += 1
    if ma50 > ma200:
        trend_score += 1

    distance_to_high = close / recent_high
    compression_score = 1 - ((recent_high - recent_low) / recent_high)

    volume_score = min(avg_vol / 5_000_000, 1.0)

    raw_score = (
        trend_score * 25
        + distance_to_high * 25
        + compression_score * 25
        + volume_score * 25
    )

    return raw_score


def rank_setups(setups: list[dict], top_n: int = TOP_SETUPS_PER_SECTION) -> list[dict]:
    sorted_setups = sorted(setups, key=lambda x: x["score"], reverse=True)
    return sorted_setups[:top_n]
