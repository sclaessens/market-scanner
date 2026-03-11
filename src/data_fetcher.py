import yfinance as yf
import pandas as pd

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


def fetch_ohlcv_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only expected OHLCV columns if present
    expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    available_cols = [col for col in expected_cols if col in df.columns]
    df = df[available_cols].copy()

    return df
