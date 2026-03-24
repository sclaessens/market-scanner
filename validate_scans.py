from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


SCANS_LOG_FILE = Path("data/scans_log.csv")
OUTPUT_FILE = Path("data/scans_validation.csv")
LOOKAHEAD_DAYS = 10


def fetch_future_data(ticker: str, start_date: str) -> pd.DataFrame:
    start = pd.Timestamp(start_date)
    end = start + pd.Timedelta(days=LOOKAHEAD_DAYS + 10)

    try:
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column",
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = {"High", "Low", "Close"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    return df.dropna(subset=["High", "Low", "Close"]).copy()


def validate_trade(row: pd.Series) -> dict | None:
    ticker = row["ticker"]
    scan_date = row["scan_date"]
    entry = float(row["entry"])
    stop = float(row["stop"])
    target = float(row["target"])

    df = fetch_future_data(ticker, scan_date)
    if df.empty:
        return None

    df = df.iloc[1: LOOKAHEAD_DAYS + 1].copy()
    if df.empty:
        return None

    stop_hit = False
    target_hit = False
    first_hit = "none"

    for _, bar in df.iterrows():
        low = float(bar["Low"])
        high = float(bar["High"])

        if low <= stop and high >= target:
            first_hit = "both_same_day"
            stop_hit = True
            target_hit = True
            break

        if low <= stop:
            first_hit = "stop"
            stop_hit = True
            break

        if high >= target:
            first_hit = "target"
            target_hit = True
            break

    max_high = float(df["High"].max())
    min_low = float(df["Low"].min())
    final_close = float(df["Close"].iloc[-1])

    max_gain_pct = ((max_high / entry) - 1) * 100
    max_drawdown_pct = ((min_low / entry) - 1) * 100
    close_return_pct = ((final_close / entry) - 1) * 100

    return {
        "scan_date": scan_date,
        "ticker": ticker,
        "setup": row["setup"],
        "regime": row["regime"],
        "score": row["score"],
        "entry": entry,
        "stop": stop,
        "target": target,
        "rr": row["rr"],
        "first_hit": first_hit,
        "stop_hit": stop_hit,
        "target_hit": target_hit,
        "max_gain_pct": round(max_gain_pct, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "close_return_pct_10d": round(close_return_pct, 2),
    }


def main() -> None:
    if not SCANS_LOG_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {SCANS_LOG_FILE}")

    scans = pd.read_csv(SCANS_LOG_FILE)
    if scans.empty:
        raise ValueError("scans_log.csv exists but contains no rows")

    results: list[dict] = []

    for _, row in scans.iterrows():
        result = validate_trade(row)
        if result is not None:
            results.append(result)

    if not results:
        print("No validations could be completed.")
        return

    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values(["scan_date", "ticker"]).reset_index(drop=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_FILE, index=False)

    print(f"Validation written to: {OUTPUT_FILE}")
    print("")
    print("Summary:")
    print(f"Rows validated: {len(df_out)}")
    print(f"Target hit rate: {(df_out['target_hit'].mean() * 100):.2f}%")
    print(f"Stop hit rate: {(df_out['stop_hit'].mean() * 100):.2f}%")
    print(f"Average 10d close return: {df_out['close_return_pct_10d'].mean():.2f}%")


if __name__ == "__main__":
    main()
