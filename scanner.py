import math
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Tuple

import pandas as pd
import yfinance as yf


# ---------- Universe (S&P 500 + Nasdaq-100) ----------
def get_sp500_tickers() -> List[str]:
    # Wikipedia table "List of S&P 500 companies"
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df["Symbol"].astype(str).tolist()
    # Yahoo uses BRK-B and BF-B (not BRK.B)
    tickers = [t.replace(".", "-") for t in tickers]
    return sorted(set(tickers))


def get_nasdaq100_tickers() -> List[str]:
    # Wikipedia table "NASDAQ-100"
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)
    # The first table is often constituents; fallback to searching for "Ticker"
    cand = None
    for t in tables:
        cols = [c.lower() for c in t.columns.astype(str)]
        if any("ticker" in c for c in cols) or any("symbol" in c for c in cols):
            cand = t
            break
    if cand is None:
        return []
    # Try common column names
    for col in cand.columns:
        c = str(col).lower()
        if "ticker" in c or "symbol" in c:
            tickers = cand[col].astype(str).tolist()
            tickers = [t.replace(".", "-").strip() for t in tickers]
            # Some rows can include footnotes; keep only plausible tickers
            tickers = [t for t in tickers if 1 <= len(t) <= 6 and t.replace("-", "").isalnum()]
            return sorted(set(tickers))
    return []


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
def classify_signals(ticker: str, df: pd.DataFrame) -> Tuple[List[SignalRow], List[SignalRow], List[SignalRow]]:
    # Needs enough data
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
    # - No "trend break" (not below MA50)
    dist_ma20 = abs(close - ma20)
    near_ma20 = (dist_ma20 <= 1.25 * atr14) or (abs(close / ma20 - 1) <= 0.015)

    # Avoid buying extended: not too far below recent high (still strong)
    drawdown_from_high20 = (high20 - close) / high20 if high20 > 0 else 0.0
    not_collapsing = drawdown_from_high20 <= 0.12  # within 12% of 20d high

    pullbacks = []
    if close > ma50 and ma50_rising and near_ma20 and not_collapsing:
        # Score: closer to MA20 + stronger trend (distance from MA50) + some "room" to high20
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
        # Score: volume surprise + breakout strength
        score = (vol / vol20) + ((close - ma20) / atr14)
        note = f"Breakout: 20d high | vol {vol/vol20:.1f}x"
        breakouts.append(
            SignalRow(ticker, close, ma20, ma50, atr14, high20, low20, vol, vol20, note, score)
        )

    # Trend failure / weakening:
    # - Close below MA50 OR close below 20d low (harder fail)
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
    # Universe
    sp = get_sp500_tickers()
    ndx = get_nasdaq100_tickers()
    universe = sorted(set(sp + ndx))

    # Limit size a bit if you want faster runs (keep full if you prefer)
    # universe = universe[:650]

    # Download in chunks (Yahoo limits/robustness)
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
                    df = data[t].dropna()
                else:
                    # single ticker case
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
    report_lines = []
    report_lines.append(f"# Market Scan — {today} (S&P 500 + Nasdaq-100)")
    report_lines.append("")
    report_lines.append(f"Universe size: **{len(universe)}**")
    report_lines.append("")
    report_lines.append("## Pullback setups (trend intact)")
    report_lines.extend([fmt_row(r) for r in pullbacks_all[:15]] or ["- (none)"])
    report_lines.append("")
    report_lines.append("## Breakouts (watch for pullback/retest)")
    report_lines.extend([fmt_row(r) for r in breakouts_all[:15]] or ["- (none)"])
    report_lines.append("")
    report_lines.append("## Trend failures / weakening (review exits)")
    report_lines.extend([fmt_row(r) for r in failures_all[:15]] or ["- (none)"])
    report_lines.append("")

    md = "\n".join(report_lines)
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(md)

    print(md)


if __name__ == "__main__":
    main()
