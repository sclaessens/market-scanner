def classify_market_regime(qqq_close: float, ma50: float, ma200: float) -> str:
    if qqq_close > ma50 and ma50 > ma200:
        return "BULLISH"
    if qqq_close < ma50 and ma50 < ma200:
        return "BEARISH"
    return "NEUTRAL"
