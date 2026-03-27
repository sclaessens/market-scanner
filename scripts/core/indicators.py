import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR14"] = true_range.rolling(14).mean()

    df["20D_HIGH"] = df["High"].rolling(20).max()
    df["20D_LOW"] = df["Low"].rolling(20).min()
    df["AVG_VOL_20"] = df["Volume"].rolling(20).mean()

    return df
