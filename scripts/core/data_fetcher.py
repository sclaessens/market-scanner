import pandas as pd
import yfinance as yf

from config.settings import TICKERS_FILE


def load_tickers():
    tickers = []

    with open(TICKERS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            ticker = line.strip().upper()

            if not ticker:
                continue

            if ticker.startswith("#"):
                continue

            tickers.append(ticker)

    return tickers


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        if len(out.columns.levels) >= 2:
            level0 = list(out.columns.get_level_values(0))
            if "Open" in level0:
                out.columns = out.columns.get_level_values(0)
            else:
                out.columns = [
                    c[1] if isinstance(c, tuple) and len(c) > 1 else c[0]
                    if isinstance(c, tuple)
                    else c
                    for c in out.columns
                ]

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


def fetch_ohlcv_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=False,
        )
        df = _normalize_columns(df)
        if not df.empty:
            return df
    except Exception:
        pass

    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(
            period=period,
            interval=interval,
            auto_adjust=False,
        )
        df = _normalize_columns(df)
        if not df.empty:
            return df
    except Exception:
        pass

    return pd.DataFrame()
