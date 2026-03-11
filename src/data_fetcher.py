import yfinance as yf
import pandas as pd

from config.settings import TICKERS_FILE


def load_tickers():
    tickers = []

    with open(TICKERS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            ticker = line.strip().upper()

            # lege lijnen overslaan
            if not ticker:
                continue

            # commentaarlijnen overslaan
            if ticker.startswith("#"):
                continue

            # scheidingslijnen overslaan
            if "=" in ticker or "-" in ticker or "~" in ticker:
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
    )

    if df is None or df.empty:
        return pd.DataFrame()

    return df
