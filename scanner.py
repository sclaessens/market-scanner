import math
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import yfinance as yf


# ---------- Universe (from tickers.txt) ----------
def load_tickers(path: str = "tickers.txt") -> List[str]:
    """
    Reads tickers from tickers.txt (one per line). Lines starting with # are ignored.
    Converts tickers like BRK.B -> BRK-B for Yahoo.
    """
    with open(path, "r", encoding="utf-8") as f:
        tickers = []
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            t = t.replace(".", "-")
            tickers.append(t)
    return sorted(set(tickers))


# ---------- Market regime (QQQ filter) ----------
def get_market_regime() -> dict:
    """
    Bullish regime if:
      QQQ close > MA50 and MA50 > MA200
    """
    qqq = yf.download("QQQ", period="2y", interval="1d", progress=False)
    if qqq is None or qqq.empty or len(qqq) < 210:
        # If we can't reliably compute MA200, fail-safe to "not bullish"
        return {"symbol": "QQQ", "close": float("nan"), "ma50": float("nan"), "ma200": float("nan"), "bullish": False}

    qqq["MA50"] = qqq["Close"].rolling(50).mean()
    qqq["MA200"] = qqq["Close"].rolling(200).mean()

    last = qqq.iloc[-1]
    close = float(last["Close"])
    ma50 = float(last["MA50"])
    ma200 = float(last["MA200"])

    bullish = (close > ma50) and (ma50 > ma200)

    return {
        "symbol": "QQQ",
        "close": round(close, 2),
        "ma50": round(ma50, 2),
        "ma200": round(ma200, 2),
        "bullish": bool(bullish),
    }


# ---------- Indicators ----------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


@dataclass
class SignalRow:
    ticker: str
    close: float
    ma20: float
    ma50: float
    atr14: float
    high20: float
    low20: float
    vol: float
    vol20: float
    note: str
    score: float


def safe_float(x) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


# ---------- Signal logic ----------
def classify_signals(
    ticker: str, df: pd.DataFrame
) -> Tuple[List[SignalRow], List[SignalRow], List[SignalRow]]:
    """
    Returns: (pullbacks, breakouts, failures)
    """
    if df is None or len(df) < 80:
        return [], [], []

    df = df.copy()
    df["MA20"] = sma(df["Close"], 20)
    df["MA50"] = sma(df["Close"], 50)
    df["ATR14"] = atr(df, 14)
    df["HIGH20"] = df["High"].rolling(20).max()
    df["LOW20"] = df["Low"].rolling(20).min()
    df["VOL20"] = df["Volume"].rolling(20).mean()

    last = df.iloc[-1]
    close = safe_float(last["Close"])
    ma20 = safe_float(last["MA20"])
    ma50 = safe_float(last["MA50"])
    atr14 = safe_float(last["ATR14"])
    high20 = safe_float(last["HIGH20"])
    low20 = safe_float(last["LOW20"])
    vol = safe_float(last["Volume"])
    vol20 = safe_float(last["VOL20"])

    if any(math.isnan(v) for v in [close, ma20, ma50, atr14, high20, low20, vol20]) or atr14 <= 0:
        return [], [], []

    # Trend filter: above MA50 and MA50 rising (simple slope proxy)
    ma50_prev = safe_float(df["MA50"].iloc[-6])  # ~1 week ago
    ma50_rising = (not math.isnan(ma50_prev)) and (ma50 > ma50_prev)

    # Pullback in uptrend:
    # - Close above MA50
    # - Close near MA20 (within ~1.25 ATR or within 1.5%)
    # - Not "collapsing" away from highs
    dist_ma20 = abs(close - ma20)
    near_ma20 = (dist_ma20 <= 1.25 * atr14) or (abs(close / ma20 - 1) <= 0.015)

    drawdown_from_high20 = (high20 - close) / high20 if high20 > 0 else 0.0
    not_collapsing = drawdown_from_high20 <= 0.12  # within 12% of 20d high

    pullbacks = []
    if close > ma50 and ma50_rising and near_ma20 and not_collapsing:
        room = max(high20 - close, 0.0)
        score = (room / atr14) - (dist_ma20 / atr14)
        note = f"Pullback: near MA20 | room~{room/atr14:.1f} ATR"
        pullbacks.append(
            SignalRow(ticker, close, ma20, ma50, atr14, high20, low20, vol, vol20, note, score)
        )

    # Breakout:
    # - Close at/above 20d high
    # - Volume confirmation
    breakouts = []
    vol_ok = vol >= 1.3 * vol20
    if close >= high20 * 0.999 and close > ma50 and vol_ok:
        score = (vol / vol20) + ((close - ma20) / atr14)
        note = f"Breakout: 20d high | vol {vol/vol20:.1f}x"
        breakouts.append(
            SignalRow(ticker, close, ma20, ma50, atr14, high20, low20, vol, vol20, note, score)
        )

    # Trend failure / weakening:
    # - Close below MA50 OR close below 20d low
    failures = []
    below_ma50 = close < ma50
    below_low20 = close <= low20 * 1.001
    if below_ma50 or below_low20:
        severity = (ma50 - close) / atr14 if below_ma50 else (low20 - close) / atr14
        score = severity + (0.5 if below_low20 else 0.0)
        note = "Failure: below MA50" + (" & near/below 20d low" if below_low20 else "")
        failures.append(
            SignalRow(ticker, close, ma20, ma50, atr14, high20, low20, vol, vol20, note, score)
        )

    return pullbacks, breakouts, failures


# ---------- Reporting ----------
def fmt_row(r: SignalRow) -> str:
    return (
        f"- **{r.ticker}** | close {r.close:.2f} | MA20 {r.ma20:.2f} | MA50 {r.ma50:.2f} | "
        f"ATR14 {r.atr14:.2f} | 20dH {r.high20:.2f} | 20dL {r.low20:.2f} | "
        f"Vol {r.vol/1e6:.2f}M (avg {r.vol20/1e6:.2f}M) — {r.note}"
    )


def main():
    # Load tickers
    universe = load_tickers()
    if not universe:
        raise RuntimeError("tickers.txt is empty (or missing). Add tickers, one per line.")

    # Market regime (QQQ)
    regime = get_market_regime()

    # Download in chunks
    start = (dt.date.today() - dt.timedelta(days=365)).isoformat()

    pullbacks_all: List[SignalRow] = []
    breakouts_all: List[SignalRow] = []
    failures_all: List[SignalRow] = []

    chunk = 120
    for i in range(0, len(universe), chunk):
        tickers = universe[i : i + chunk]

        data = yf.download(
            tickers=tickers,
            start=start,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )

        for t in tickers:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if t not in data.columns.get_level_values(0):
                        continue
                    df = data[t].dropna()
                else:
                    # single-ticker case
                    df = data.dropna()

                if df is None or df.empty:
                    continue

                pb, bo, fa = classify_signals(t, df)
                pullbacks_all.extend(pb)
                breakouts_all.extend(bo)
                failures_all.extend(fa)
            except Exception:
                continue

    pullbacks_all.sort(key=lambda r: r.score, reverse=True)
    breakouts_all.sort(key=lambda r: r.score, reverse=True)
    failures_all.sort(key=lambda r: r.score, reverse=True)

    today = dt.datetime.utcnow().strftime("%Y-%m-%d")

    # Apply regime filter:
    # If QQQ not bullish, disable LONG entries (pullbacks/breakouts),
    # but still report failures (useful for exit review).
    if regime["bullish"]:
        pullbacks_out = pullbacks_all[:15]
        breakouts_out = breakouts_all[:15]
        regime_line = "BULLISH ✅ (long setups enabled)"
    else:
        pullbacks_out = []
        breakouts_out = []
        regime_line = "BEARISH/NEUTRAL ⚠️ (long setups disabled)"

    failures_out = failures_all[:15]

    report_lines = []
    report_lines.append(f"# Market Scan — {today}")
    report_lines.append("")
    report_lines.append(f"Universe size: **{len(universe)}**")
    report_lines.append("")
    report_lines.append("## Market Regime (QQQ)")
    report_lines.append(
        f'QQQ close {regime["close"]} | MA50 {regime["ma50"]} | MA200 {regime["ma200"]} → {regime_line}'
    )
    report_lines.append("")
    report_lines.append("## Pullback setups (trend intact)")
    report_lines.extend([fmt_row(r) for r in pullbacks_out] or ["- (none)"])
    report_lines.append("")
    report_lines.append("## Breakouts (watch for pullback/retest)")
    report_lines.extend([fmt_row(r) for r in breakouts_out] or ["- (none)"])
    report_lines.append("")
    report_lines.append("## Trend failures / weakening (review exits)")
    report_lines.extend([fmt_row(r) for r in failures_out] or ["- (none)"])
    report_lines.append("")

    md = "\n".join(report_lines)

    with open("report.md", "w", encoding="utf-8") as f:
        f.write(md)

    print(md)


if __name__ == "__main__":
    main()
