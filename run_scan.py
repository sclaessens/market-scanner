from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


TICKERS_FILE = Path("tickers.txt")
REPORTS_DIR = Path("reports")
FAILED_TICKERS_FILE = Path("data/failed_tickers.csv")

LOOKBACK_PERIOD = "1y"
MIN_HISTORY_ROWS = 220
TOP_SETUPS_LIMIT = 5


def ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_TICKERS_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_tickers(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing tickers file: {path}")

    tickers: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ticker = line.strip().upper()
            if not ticker or ticker.startswith("#"):
                continue
            tickers.append(ticker)

    if not tickers:
        raise ValueError("No tickers found in tickers.txt")

    return tickers


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zorgt dat de dataframe standaard kolommen heeft:
    Open, High, Low, Close, Volume
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        if len(out.columns.levels) >= 2:
            if "Open" in out.columns.get_level_values(0):
                out.columns = out.columns.get_level_values(0)
            else:
                out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj close": "Adj Close",
        "volume": "Volume",
    }

    out.columns = [
        rename_map.get(str(col).strip().lower(), str(col).strip())
        for col in out.columns
    ]

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(out.columns)):
        return pd.DataFrame()

    out = out.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
    return out


def fetch_ticker_data(ticker: str, period: str = LOOKBACK_PERIOD) -> pd.DataFrame:
    """
    Robuuste single-ticker fetch.
    Eerst via download, daarna fallback via Ticker.history().
    """
    print(f"Fetching data for {ticker}...")

    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column",
        )
        df = normalize_columns(df)
        if not df.empty:
            return df
    except Exception as exc:
        print(f"Download error for {ticker}: {exc}")

    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, interval="1d", auto_adjust=False)
        df = normalize_columns(df)
        if not df.empty:
            return df
    except Exception as exc:
        print(f"History error for {ticker}: {exc}")

    print(f"Warning: no data returned for {ticker}")
    return pd.DataFrame()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["MA20"] = out["Close"].rolling(20).mean()
    out["MA50"] = out["Close"].rolling(50).mean()
    out["MA200"] = out["Close"].rolling(200).mean()

    prev_close = out["Close"].shift(1)
    tr1 = out["High"] - out["Low"]
    tr2 = (out["High"] - prev_close).abs()
    tr3 = (out["Low"] - prev_close).abs()
    out["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["ATR14"] = out["TR"].rolling(14).mean()

    out["20D_High"] = out["High"].rolling(20).max()
    out["20D_Low"] = out["Low"].rolling(20).min()
    out["AVG_VOL20"] = out["Volume"].rolling(20).mean()

    return out


def determine_market_regime(qqq_df: pd.DataFrame) -> str:
    if qqq_df.empty or len(qqq_df) < 200:
        raise ValueError("QQQ data is required to determine market regime")

    latest = qqq_df.iloc[-1]

    close = latest["Close"]
    ma50 = latest["MA50"]
    ma200 = latest["MA200"]

    if pd.isna(close) or pd.isna(ma50) or pd.isna(ma200):
        raise ValueError("QQQ regime cannot be determined because MA data is incomplete")

    if close > ma50 and ma50 > ma200:
        return "BULLISH"
    if close < ma50 and ma50 < ma200:
        return "BEARISH"
    return "NEUTRAL"


def validate_reference_index(ticker: str) -> pd.DataFrame:
    df = fetch_ticker_data(ticker)
    if df.empty:
        raise ValueError(f"{ticker} data is required but no valid data was returned")

    if len(df) < MIN_HISTORY_ROWS:
        raise ValueError(
            f"{ticker} data is incomplete: expected at least {MIN_HISTORY_ROWS} rows, got {len(df)}"
        )

    df = compute_indicators(df)
    if df[["MA50", "MA200"]].iloc[-1].isna().any():
        raise ValueError(f"{ticker} data is present but MA calculation is incomplete")

    return df


def scan_ticker(ticker: str, df: pd.DataFrame, regime: str) -> Optional[dict]:
    if df.empty or len(df) < MIN_HISTORY_ROWS:
        return None

    df = compute_indicators(df)
    latest = df.iloc[-1]

    if pd.isna(latest["MA20"]) or pd.isna(latest["MA50"]) or pd.isna(latest["MA200"]):
        return None

    close = latest["Close"]
    ma20 = latest["MA20"]
    ma50 = latest["MA50"]
    ma200 = latest["MA200"]
    high_20 = latest["20D_High"]

    if pd.isna(high_20):
        return None

    # Bearish regime = geen longs
    if regime == "BEARISH":
        return None

    # 1. Trend
    strong_trend = close > ma20 > ma50 > ma200

    # 2. Pullback quality
    distance_ma20 = abs((close - ma20) / ma20)
    distance_ma50 = abs((close - ma50) / ma50)
    healthy_pullback = distance_ma20 < 0.02 or distance_ma50 < 0.02

    # 3. Structuur intact
    not_broken = close > ma50

    if not (strong_trend and healthy_pullback and not_broken):
        return None

    # 4. Regime adaptation
    if regime == "NEUTRAL":
        momentum_threshold_5d = 0.01
        momentum_threshold_10d = 0.03
        setup_type = "B"
    else:
        momentum_threshold_5d = 0.05
        momentum_threshold_10d = 0.10
        setup_type = "A"

    # 5. Score
    score = 0

    # Relative strength
    distance_high = (high_20 - close) / high_20
    if distance_high < 0.03:
        score += 3
    elif distance_high < 0.06:
        score += 2
    elif distance_high < 0.10:
        score += 1
    else:
        return None

    # Trend quality
    trend_strength = (ma20 - ma50) / ma50
    if trend_strength > 0.05:
        score += 2
    elif trend_strength > 0.02:
        score += 1

    # Momentum via returns
    ret_5d = (df["Close"].iloc[-1] / df["Close"].iloc[-6]) - 1
    ret_10d = (df["Close"].iloc[-1] / df["Close"].iloc[-11]) - 1

    if ret_5d > momentum_threshold_5d:
        score += 2

    if ret_10d > momentum_threshold_10d:
        score += 2
    else:
        return None

    # Extra cleanup
    if close < 20:
        return None

    return {
        "ticker": ticker,
        "close": round(float(close), 2),
        "ma20": round(float(ma20), 2),
        "ma50": round(float(ma50), 2),
        "ma200": round(float(ma200), 2),
        "setup": f"{setup_type}_pullback",
        "score": score,
    }


def save_failed_tickers(rows: list[dict]) -> None:
    if not rows:
        return

    df = pd.DataFrame(rows)
    df.to_csv(FAILED_TICKERS_FILE, index=False)

def build_tradeplan(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]

    close = latest["Close"]
    ma20 = latest["MA20"]
    ma50 = latest["MA50"]
    atr = latest["ATR14"]

    # Entry = rond MA20 (pullback zone)
    entry = ma20

    # Stop = onder MA50 - buffer
    stop = ma50 - atr * 0.5

    # Risk
    risk = entry - stop

    if risk <= 0:
        return {}

    # Target = 2x risk
    target = entry + (2 * risk)

    rr = (target - entry) / (entry - stop)

    return {
        "entry": round(float(entry), 2),
        "stop": round(float(stop), 2),
        "target": round(float(target), 2),
        "rr": round(float(rr), 2),
    }

def write_report(
    total_tickers: int,
    successful_tickers: int,
    failed_tickers: int,
    regime: str,
    setups: list[dict],
) -> Path:
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    report_path = REPORTS_DIR / f"market_scan_{today}.md"

    lines: list[str] = []
    lines.append(f"# Market Scan — {today}")
    lines.append("")
    lines.append(f"Universe size (raw): **{total_tickers}**")
    lines.append(f"Universe size (valid data): **{successful_tickers}**")
    lines.append(f"Failed tickers: **{failed_tickers}**")
    lines.append("")
    lines.append("## Market Regime")
    lines.append(f"**{regime}**")
    lines.append("")
    lines.append("## Setups")

    if not setups:
        lines.append("- (none)")
        lines.append("")
        lines.append("## Interpretation")

        if regime == "NEUTRAL":
            lines.append("Market is NEUTRAL → low conviction environment.")
            lines.append("Scanner is intentionally selective → no A-quality setups.")
        elif regime == "BEARISH":
            lines.append("Market is BEARISH → long setups are filtered out.")
            lines.append("No long opportunities expected in this regime.")
        else:
            lines.append("No setups found despite bullish regime → check scan logic.")
    else:
        for setup in setups:
            lines.append(
                f"- {setup['ticker']} | close {setup['close']} | {setup['setup']} | "
                f"score {setup['score']} | entry {setup['entry']} | stop {setup['stop']} | "
                f"target {setup['target']} | RR {setup['rr']}"
            )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    ensure_dirs()

    print("Starting market scan...")

    tickers = load_tickers(TICKERS_FILE)

    qqq_df = validate_reference_index("QQQ")
    spy_df = validate_reference_index("SPY")

    regime = determine_market_regime(qqq_df)
    print(f"Market regime: {regime}")

    _ = spy_df  # later uitbreiden voor extra marktfilter

    setups: list[dict] = []
    failed_rows: list[dict] = []
    successful_count = 0

    seen_tickers: set[str] = set()

    for ticker in tickers:
        if ticker in {"QQQ", "SPY"}:
            continue

        if ticker in seen_tickers:
            continue
        seen_tickers.add(ticker)

        df = fetch_ticker_data(ticker)

        if df.empty:
            failed_rows.append({"ticker": ticker, "reason": "no_data_returned"})
            continue

        if len(df) < MIN_HISTORY_ROWS:
            failed_rows.append(
                {"ticker": ticker, "reason": f"insufficient_history_{len(df)}"}
            )
            continue

        successful_count += 1

        result = scan_ticker(ticker, df, regime)

        df_ind = compute_indicators(df)

        result = scan_ticker(ticker, df_ind, regime)
        
        if result is not None:
            tradeplan = build_tradeplan(df_ind)
        
            if tradeplan:
                result.update(tradeplan)
                setups.append(result)
        if result is not None:
            setups.append(result)

    # Sort & keep top setups
    setups = sorted(setups, key=lambda x: (x["score"], x["ticker"]), reverse=True)
    setups = setups[:TOP_SETUPS_LIMIT]

    save_failed_tickers(failed_rows)

    report_path = write_report(
        total_tickers=len(tickers),
        successful_tickers=successful_count,
        failed_tickers=len(failed_rows),
        regime=regime,
        setups=setups,
    )

    print(f"Report written to: {report_path}")
    print(f"Failed tickers log written to: {FAILED_TICKERS_FILE}")


if __name__ == "__main__":
    main()
