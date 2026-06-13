from __future__ import annotations

import pandas as pd


MIN_REQUIRED_ROWS = 20  # minimale lengte voor indicators


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if len(df) < MIN_REQUIRED_ROWS:
        return pd.DataFrame()

    df = df.copy()

    required_cols = ["Open", "High", "Low", "Close", "Volume"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

        # Handle mogelijke multi-column issues
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    # =========================
    # Moving averages
    # =========================
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    # =========================
    # ATR (Average True Range)
    # =========================
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR14"] = true_range.rolling(14).mean()

    # =========================
    # Range + volume
    # =========================
    df["20D_HIGH"] = df["High"].rolling(20).max()
    df["20D_LOW"] = df["Low"].rolling(20).min()
    df["AVG_VOL_20"] = df["Volume"].rolling(20).mean()

    return df