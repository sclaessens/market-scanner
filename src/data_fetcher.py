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


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        flattened = []

        for col in df.columns:
            parts = [str(x) for x in col if str(x) != ""]
            flattened.append("_".join(parts))

        df.columns = flattened

    return df


def _normalize_ohlcv_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = _flatten_columns(df)

    normalized = pd.DataFrame(index=df.index)

    column_map = {
        "Open": [f"Open_{ticker}", "Open"],
        "High": [f"High_{ticker}", "High"],
        "Low": [f"Low_{ticker}", "Low"],
        "Close": [f"Close_{ticker}", "Close"],
        "Adj Close": [f"Adj Close_{ticker}", "Adj Close"],
        "Volume": [f"Volume_{ticker}", "Volume"],
    }

    for target_col, candidates in column_map.items():
        for candidate in candidates:
            if candidate in df.columns:
                series = df[candidate]

                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]

                normalized[target_col] = pd.to_numeric(series, errors="coerce")
                break

    return normalized


def fetch_ohlcv_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=False,
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = _normalize_ohlcv_columns(df, ticker)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        return pd.DataFrame()

    df = df.dropna(subset=["Close"])
    return df
