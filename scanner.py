import math
import os
import uuid
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pandas as pd
import yfinance as yf

# How many rows to show per section
TOP_N_ACTIONABLE = 5
TOP_N_WATCHLIST = 5
TOP_N_FAILURES = 10

# Universe maintenance
MIN_PRICE = 10.0
MIN_AVG_DOLLAR_VOL = 25_000_000  # avg(close * volume) over last ~30 trading days


# ---------- Universe (from tickers.txt) ----------
def load_tickers(path: str = "tickers.txt") -> List[str]:
    """
    Reads tickers from tickers.txt (one per line).
    Lines starting with # are ignored.
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


def filter_liquid_universe(tickers: List[str]) -> List[str]:
    """
    Filters out illiquid / untradeable symbols using:
      - last close >= MIN_PRICE
      - average dollar volume >= MIN_AVG_DOLLAR_VOL
    """
    if not tickers:
        return []

    data = yf.download(
        tickers=tickers,
        period="2mo",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    kept = []
    for t in tickers:
        try:
            df = data[t].dropna() if isinstance(data.columns, pd.MultiIndex) else data.dropna()
            if df is None or df.empty or len(df) < 15:
                continue

            close_last = float(df["Close"].iloc[-1])
            dollar_vol = (df["Close"] * df["Volume"]).tail(30).mean()

            if close_last >= MIN_PRICE and float(dollar_vol) >= MIN_AVG_DOLLAR_VOL:
                kept.append(t)
        except Exception:
            continue

    return sorted(set(kept))


# ---------- Market regime (QQQ filter) ----------
def get_market_regime() -> dict:
    """
    Bullish regime if:
      QQQ close > MA50 and MA50 > MA200
    """
    qqq = yf.download("QQQ", period="2y", interval="1d", progress=False)
    if qqq is None or qqq.empty or len(qqq) < 210:
        return {
            "symbol": "QQQ",
            "close": float("nan"),
            "ma50": float("nan"),
            "ma200": float("nan"),
            "bullish": False,
        }

    qqq["MA50"] = qqq["Close"].rolling(50).mean()
    qqq["MA200"] = qqq["Close"].rolling(200).mean()

    last = qqq.iloc[-1]

    close = float(last["Close"].iloc[0] if hasattr(last["Close"], "iloc") else last["Close"])
    ma50 = float(last["MA50"].iloc[0] if hasattr(last["MA50"], "iloc") else last["MA50"])
    ma200 = float(last["MA200"].iloc[0] if hasattr(last["MA200"], "iloc") else last["MA200"])

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
    ma200: float
    atr14: float
    high20: float
    low20: float
    vol: float
    vol20: float
    note: str
    score: float
    entry: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    rr: Optional[float] = None


def safe_float(x) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


# ---------- Trade plan helpers ----------
def plan_pullback(ma20: float, atr14: float) -> Tuple[float, float, float, float]:
    """
    Pullback plan:
    - Entry: MA20
    - Stop: 1.6 ATR below entry
    - Target: 3.2 ATR above entry
    """
    entry = ma20
    stop = entry - 1.6 * atr14
    target = entry + 3.2 * atr14
    rr = (target - entry) / max(entry - stop, 1e-9)
    return entry, stop, target, rr


def plan_breakout(level: float, atr14: float) -> Tuple[float, float, float, float]:
    """
    Breakout plan:
    - Entry: breakout level (20d high)
    - Stop: 1.5 ATR below entry
    - Target: 3.0 ATR above entry
    """
    entry = level
    stop = entry - 1.5 * atr14
    target = entry + 3.0 * atr14
    rr = (target - entry) / max(entry - stop, 1e-9)
    return entry, stop, target, rr


def plan_vcp(high20: float, atr14: float) -> Tuple[float, float, float, float]:
    """
    VCP plan:
    - Entry: 20d high as proxy for breakout trigger
    - Stop: 1.6 ATR below entry
    - Target: 3.2 ATR above entry
    """
    entry = high20
    stop = entry - 1.6 * atr14
    target = entry + 3.2 * atr14
    rr = (target - entry) / max(entry - stop, 1e-9)
    return entry, stop, target, rr


# ---------- VCP detection ----------
def detect_vcp(df: pd.DataFrame) -> Tuple[bool, float, str]:
    """
    Simple VCP detector.

    Criteria:
    - Uptrend: close > MA50 and MA50 > MA200
    - ATR contraction: recent ATR < 0.8 * older ATR
    - Price near highs
    - Tight range
    """
    if df is None or len(df) < 260:
        return False, 0.0, ""

    recent40 = df.tail(40)

    if recent40.empty:
        return False, 0.0, ""

    close = float(recent40["Close"].iloc[-1])
    ma50 = float(df["MA50"].iloc[-1])
    ma200 = float(df["MA200"].iloc[-1])

    if not (close > ma50 and ma50 > ma200):
        return False, 0.0, ""

    high40 = float(recent40["High"].max())
    low40 = float(recent40["Low"].min())
    range_pct = (high40 - low40) / max(high40, 1e-9)

    atr_recent = float(df["ATR14"].tail(20).mean())
    atr_prev = float(df["ATR14"].iloc[-60:-40].mean())

    if math.isnan(atr_recent) or math.isnan(atr_prev) or atr_prev <= 0:
        return False, 0.0, ""

    contraction = atr_recent < 0.8 * atr_prev
    near_high = close >= 0.90 * high40
    tight = range_pct < 0.18

    if not (contraction and near_high and tight):
        return False, 0.0, ""

    contraction_strength = (atr_prev - atr_recent) / atr_prev
    closeness = close / high40
    score = 2.0 * contraction_strength + 0.5 * closeness - 1.0 * range_pct

    note = f"VCP: ATR↓ {contraction_strength:.2f}, range {range_pct:.2f}, close/high {closeness:.2f}"
    return True, float(score), note


# ---------- Signal logic ----------
def classify_signals(
    ticker: str, df: pd.DataFrame
) -> Tuple[List[SignalRow], List[SignalRow], List[SignalRow], List[SignalRow]]:
    """
    Returns: (pullbacks, breakouts, failures, vcps)
    """
    if df is None or len(df) < 260:
        return [], [], [], []

    df = df.copy()
    df["MA20"] = sma(df["Close"], 20)
    df["MA50"] = sma(df["Close"], 50)
    df["MA200"] = sma(df["Close"], 200)
    df["ATR14"] = atr(df, 14)
    df["HIGH20"] = df["High"].rolling(20).max()
    df["LOW20"] = df["Low"].rolling(20).min()
    df["VOL20"] = df["Volume"].rolling(20).mean()

    last = df.iloc[-1]
    close = safe_float(last["Close"])
    ma20 = safe_float(last["MA20"])
    ma50 = safe_float(last["MA50"])
    ma200 = safe_float(last["MA200"])
    atr14 = safe_float(last["ATR14"])
    high20 = safe_float(last["HIGH20"])
    low20 = safe_float(last["LOW20"])
    vol = safe_float(last["Volume"])
    vol20 = safe_float(last["VOL20"])

    if any(math.isnan(v) for v in [close, ma20, ma50, ma200, atr14, high20, low20, vol20]) or atr14 <= 0:
        return [], [], [], []

    ma50_prev = safe_float(df["MA50"].iloc[-6])
    ma50_rising = (not math.isnan(ma50_prev)) and (ma50 > ma50_prev)

    # Pullback
    dist_ma20 = abs(close - ma20)
    near_ma20 = (dist_ma20 <= 1.25 * atr14) or (abs(close / ma20 - 1) <= 0.015)
    drawdown_from_high20 = (high20 - close) / max(high20, 1e-9)
    not_collapsing = drawdown_from_high20 <= 0.12

    pullbacks: List[SignalRow] = []
    if close > ma50 and ma50_rising and near_ma20 and not_collapsing and ma50 > ma200:
        room = max(high20 - close, 0.0)
        score = (room / atr14) - (dist_ma20 / atr14)
        entry, stop, target, rr = plan_pullback(ma20, atr14)
        note = f"Pullback near MA20 | room~{room/atr14:.1f} ATR"
        pullbacks.append(
            SignalRow(
                ticker=ticker,
                close=close,
                ma20=ma20,
                ma50=ma50,
                ma200=ma200,
                atr14=atr14,
                high20=high20,
                low20=low20,
                vol=vol,
                vol20=vol20,
                note=note,
                score=score,
                entry=entry,
                stop=stop,
                target=target,
                rr=rr,
            )
        )

    # Breakout
    breakouts: List[SignalRow] = []
    vol_ok = vol >= 1.3 * vol20
    if close >= high20 * 0.999 and close > ma50 and vol_ok and ma50 > ma200:
        score = (vol / vol20) + ((close - ma20) / atr14)
        entry, stop, target, rr = plan_breakout(high20, atr14)
        note = f"Breakout 20dH | vol {vol/vol20:.1f}x (watch retest)"
        breakouts.append(
            SignalRow(
                ticker=ticker,
                close=close,
                ma20=ma20,
                ma50=ma50,
                ma200=ma200,
                atr14=atr14,
                high20=high20,
                low20=low20,
                vol=vol,
                vol20=vol20,
                note=note,
                score=score,
                entry=entry,
                stop=stop,
                target=target,
                rr=rr,
            )
        )

    # Failure
    failures: List[SignalRow] = []
    below_ma50 = close < ma50
    below_low20 = close <= low20 * 1.001
    if below_ma50 or below_low20:
        severity = (ma50 - close) / atr14 if below_ma50 else (low20 - close) / atr14
        score = severity + (0.5 if below_low20 else 0.0)
        note = "Below MA50" + (" & near/below 20d low" if below_low20 else "")
        failures.append(
            SignalRow(
                ticker=ticker,
                close=close,
                ma20=ma20,
                ma50=ma50,
                ma200=ma200,
                atr14=atr14,
                high20=high20,
                low20=low20,
                vol=vol,
                vol20=vol20,
                note=note,
                score=score,
            )
        )

    # VCP
    vcps: List[SignalRow] = []
    is_vcp, vcp_score, vcp_note = detect_vcp(df)
    if is_vcp:
        entry, stop, target, rr = plan_vcp(high20, atr14)
        vcps.append(
            SignalRow(
                ticker=ticker,
                close=close,
                ma20=ma20,
                ma50=ma50,
                ma200=ma200,
                atr14=atr14,
                high20=high20,
                low20=low20,
                vol=vol,
                vol20=vol20,
                note=vcp_note,
                score=vcp_score,
                entry=entry,
                stop=stop,
                target=target,
                rr=rr,
            )
        )

    return pullbacks, breakouts, failures, vcps


# ---------- Reporting ----------
def fmt_trade(r: SignalRow) -> str:
    if r.entry is None or r.stop is None or r.target is None or r.rr is None:
        return ""
    return f" | entry {r.entry:.2f} | stop {r.stop:.2f} | tgt {r.target:.2f} | R/R {r.rr:.2f}"


def fmt_row(r: SignalRow) -> str:
    vol_part = f"Vol {r.vol/1e6:.2f}M (avg {r.vol20/1e6:.2f}M)"
    base = (
        f"- **{r.ticker}** | close {r.close:.2f} | MA20 {r.ma20:.2f} | MA50 {r.ma50:.2f} | MA200 {r.ma200:.2f} | "
        f"ATR {r.atr14:.2f} | 20dH {r.high20:.2f} | 20dL {r.low20:.2f} | {vol_part} — {r.note}"
    )
    return base + fmt_trade(r)


# ---------- Scan logging ----------
def log_scans(rows: List[SignalRow], setup_type: str, status: str, regime: dict, run_id: str) -> None:
    """
    Logs scan results to data/scans_log.csv
    """
    if not rows:
        return

    os.makedirs("data", exist_ok=True)
    file_path = "data/scans_log.csv"

    records = []
    scan_date = dt.datetime.utcnow().strftime("%Y-%m-%d")

    for r in rows:
        setup_id = str(uuid.uuid4())

        records.append(
            {
                "run_id": run_id,
                "setup_id": setup_id,
                "scan_date": scan_date,
                "ticker": r.ticker,
                "setup_type": setup_type,
                "status": status,
                "close": r.close,
                "ma20": r.ma20,
                "ma50": r.ma50,
                "ma200": r.ma200,
                "atr14": r.atr14,
                "high20": r.high20,
                "low20": r.low20,
                "volume": r.vol,
                "volume_avg20": r.vol20,
                "score": r.score,
                "entry": r.entry,
                "stop": r.stop,
                "target": r.target,
                "rr": r.rr,
                "note": r.note,
                "market_symbol": regime["symbol"],
                "market_close": regime["close"],
                "market_ma50": regime["ma50"],
                "market_ma200": regime["ma200"],
                "market_bullish": regime["bullish"],
            }
        )

    df = pd.DataFrame(records)

    if os.path.exists(file_path):
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_path, mode="w", header=True, index=False)


def main():
    universe_raw = load_tickers()
    if not universe_raw:
        raise RuntimeError("tickers.txt is empty (or missing). Add tickers, one per line.")

    universe = filter_liquid_universe(universe_raw)
    regime = get_market_regime()

    start = (dt.date.today() - dt.timedelta(days=365)).isoformat()

    pullbacks_all: List[SignalRow] = []
    breakouts_all: List[SignalRow] = []
    failures_all: List[SignalRow] = []
    vcps_all: List[SignalRow] = []

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
                    df = data.dropna()

                if df is None or df.empty:
                    continue

                pb, bo, fa, vcp = classify_signals(t, df)
                pullbacks_all.extend(pb)
                breakouts_all.extend(bo)
                failures_all.extend(fa)
                vcps_all.extend(vcp)
            except Exception:
                continue

    pullbacks_all.sort(key=lambda r: r.score, reverse=True)
    breakouts_all.sort(key=lambda r: r.score, reverse=True)
    failures_all.sort(key=lambda r: r.score, reverse=True)
    vcps_all.sort(key=lambda r: r.score, reverse=True)

    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    run_id = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    if regime["bullish"]:
        regime_line = "BULLISH ✅ (long setups enabled)"

        vcp_out = vcps_all[:TOP_N_ACTIONABLE]
        pullbacks_out = pullbacks_all[:TOP_N_ACTIONABLE]
        breakouts_out = breakouts_all[:TOP_N_ACTIONABLE]

        vcp_watch = []
        pullbacks_watch = []
        breakouts_watch = []
    else:
        regime_line = "BEARISH/NEUTRAL ⚠️ (no new longs; watchlist only)"

        vcp_out = []
        pullbacks_out = []
        breakouts_out = []

        vcp_watch = vcps_all[:TOP_N_WATCHLIST]
        pullbacks_watch = pullbacks_all[:TOP_N_WATCHLIST]
        breakouts_watch = breakouts_all[:TOP_N_WATCHLIST]

    failures_out = failures_all[:TOP_N_FAILURES]

    report_lines = []
    report_lines.append(f"# Market Scan — {today}")
    report_lines.append("")
    report_lines.append(f"Universe size (raw): **{len(universe_raw)}**")
    report_lines.append(f"Universe size (liquid-filtered): **{len(universe)}**")
    report_lines.append("")
    report_lines.append("## Market Regime (QQQ)")
    report_lines.append(
        f'QQQ close {regime["close"]} | MA50 {regime["ma50"]} | MA200 {regime["ma200"]} → {regime_line}'
    )
    report_lines.append("")

    report_lines.append("## VCP setups (compression → potential breakout)")
    report_lines.extend([fmt_row(r) for r in vcp_out] or ["- (none)"])
    report_lines.append("")

    report_lines.append("## Pullback setups (actionable)")
    report_lines.extend([fmt_row(r) for r in pullbacks_out] or ["- (none)"])
    report_lines.append("")

    report_lines.append("## Breakouts (watch for pullback/retest)")
    report_lines.extend([fmt_row(r) for r in breakouts_out] or ["- (none)"])
    report_lines.append("")

    if not regime["bullish"]:
        report_lines.append("## WATCHLIST — VCP candidates (do not buy yet)")
        report_lines.extend([fmt_row(r) for r in vcp_watch] or ["- (none)"])
        report_lines.append("")

        report_lines.append("## WATCHLIST — Pullback candidates (do not buy yet)")
        report_lines.extend([fmt_row(r) for r in pullbacks_watch] or ["- (none)"])
        report_lines.append("")

        report_lines.append("## WATCHLIST — Breakout candidates (do not buy yet)")
        report_lines.extend([fmt_row(r) for r in breakouts_watch] or ["- (none)"])
        report_lines.append("")

    report_lines.append("## Trend failures / weakening (review exits)")
    report_lines.extend([fmt_row(r) for r in failures_out] or ["- (none)"])
    report_lines.append("")

    # ---------- LOGGING ----------
    if regime["bullish"]:
        log_scans(vcp_out, "vcp", "actionable", regime, run_id)
        log_scans(pullbacks_out, "pullback", "actionable", regime, run_id)
        log_scans(breakouts_out, "breakout", "actionable", regime, run_id)
    else:
        log_scans(vcp_watch, "vcp", "watchlist", regime, run_id)
        log_scans(pullbacks_watch, "pullback", "watchlist", regime, run_id)
        log_scans(breakouts_watch, "breakout", "watchlist", regime, run_id)

    log_scans(failures_out, "failure", "failure", regime, run_id)

    md = "\n".join(report_lines)

    with open("report.md", "w", encoding="utf-8") as f:
        f.write(md)

    print(md)


if __name__ == "__main__":
    main()
