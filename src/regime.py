def classify_market_regime(close, ma50, ma200):

    if close > ma50 and ma50 > ma200:
        return "BULLISH"

    if close < ma50 and ma50 < ma200:
        return "BEARISH"

    return "NEUTRAL"
