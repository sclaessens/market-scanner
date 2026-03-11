import yfinance as yf
import pandas as pd

from config.settings import TICKERS_FILE


def load_tickers():
    with open(TICKERS_FILE, "r", encoding="utf-8") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]
    return tickers


def fetch_ohlcv_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    return df
