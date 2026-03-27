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

    df = df.dropna(subset=["High", "Low", "Close"]).copy()
    if df.empty:
        return pd.DataFrame()

    return df


def _first_non_empty(row: pd.Series, candidates: list[str], default=""):
    for col in candidates:
        if col in row.index:
            value = row[col]
            if pd.notna(value) and str(value).strip() != "":
                return value
    return default


def validate_trade(row: pd.Series) -> dict | None:
    required = {"ticker", "scan_date", "entry", "stop", "target"}
    if not required.issubset(row.index):
        return None

    try:
        ticker = str(row["ticker"]).strip().upper()
        scan_date = str(row["scan_date"]).strip()
        entry = float(row["entry"])
        stop = float(row["stop"])
        target = float(row["target"])
    except Exception:
        return None

    if not ticker or not scan_date:
        return None

    if entry <= 0 or stop <= 0 or target <= 0:
        return None

    df = fetch_future_data(ticker, scan_date)
    if df.empty:
        return None

    df = df.iloc[1 : LOOKAHEAD_DAYS + 1].copy()
    if df.empty:
        return None

    stop_hit = False
    target_hit = False
    first_hit = "none"
    days_tracked = len(df)

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

    if first_hit == "target":
        outcome = "target_hit"
    elif first_hit == "stop":
        outcome = "stop_hit"
    elif first_hit == "both_same_day":
        outcome = "ambiguous_same_day"
    else:
        if close_return_pct > 0:
            outcome = "open_profit"
        elif close_return_pct < 0:
            outcome = "open_loss"
        else:
            outcome = "flat"

    return {
        "scan_date": scan_date,
        "ticker": ticker,
        "setup": _first_non_empty(row, ["setup"], ""),
        "primary_setup": _first_non_empty(row, ["primary_setup"], ""),
        "grade": _first_non_empty(row, ["grade"], ""),
        "regime": _first_non_empty(row, ["regime"], ""),
        "score": pd.to_numeric(_first_non_empty(row, ["score"], default=pd.NA), errors="coerce"),
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "rr": pd.to_numeric(_first_non_empty(row, ["rr"], default=pd.NA), errors="coerce"),
        "days_tracked": days_tracked,
        "first_hit": first_hit,
        "outcome": outcome,
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

    required_cols = {"ticker", "scan_date", "entry", "stop", "target"}
    missing = required_cols - set(scans.columns)
    if missing:
        raise ValueError(
            f"scans_log.csv is missing required columns: {', '.join(sorted(missing))}"
        )

    scans = scans.dropna(subset=["ticker", "scan_date", "entry", "stop", "target"]).copy()
    if scans.empty:
        raise ValueError("No valid scan rows remain after filtering missing required values")

    results: list[dict] = []

    for _, row in scans.iterrows():
        result = validate_trade(row)
        if result is not None:
            results.append(result)

    if not results:
        print("No validations could be completed.")
        return

    df_out = pd.DataFrame(results)

    sort_cols = [col for col in ["scan_date", "ticker", "primary_setup", "setup"] if col in df_out.columns]
    if sort_cols:
        df_out = df_out.sort_values(sort_cols).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_FILE, index=False)

    print(f"Validation written to: {OUTPUT_FILE}")
    print("")
    print("Summary:")
    print(f"Rows validated: {len(df_out)}")

    if "target_hit" in df_out.columns:
        print(f"Target hit rate: {(df_out['target_hit'].mean() * 100):.2f}%")
    if "stop_hit" in df_out.columns:
        print(f"Stop hit rate: {(df_out['stop_hit'].mean() * 100):.2f}%")
    if "close_return_pct_10d" in df_out.columns:
        print(f"Average 10d close return: {df_out['close_return_pct_10d'].mean():.2f}%")

    if "grade" in df_out.columns and df_out["grade"].notna().any():
        print("")
        print("By grade:")
        by_grade = (
            df_out.groupby("grade", dropna=False)
            .agg(
                signals=("ticker", "count"),
                target_hit_rate=("target_hit", "mean"),
                stop_hit_rate=("stop_hit", "mean"),
                avg_close_return_10d=("close_return_pct_10d", "mean"),
            )
            .reset_index()
        )
        for _, row in by_grade.iterrows():
            grade = row["grade"] if pd.notna(row["grade"]) and str(row["grade"]).strip() else "(blank)"
            print(
                f"{grade}: "
                f"signals={int(row['signals'])} | "
                f"target_hit_rate={row['target_hit_rate'] * 100:.2f}% | "
                f"stop_hit_rate={row['stop_hit_rate'] * 100:.2f}% | "
                f"avg_10d={row['avg_close_return_10d']:.2f}%"
            )

    if "primary_setup" in df_out.columns and df_out["primary_setup"].notna().any():
        print("")
        print("By primary setup:")
        by_setup = (
            df_out.groupby("primary_setup", dropna=False)
            .agg(
                signals=("ticker", "count"),
                target_hit_rate=("target_hit", "mean"),
                stop_hit_rate=("stop_hit", "mean"),
                avg_close_return_10d=("close_return_pct_10d", "mean"),
            )
            .reset_index()
        )
        for _, row in by_setup.iterrows():
            setup = (
                row["primary_setup"]
                if pd.notna(row["primary_setup"]) and str(row["primary_setup"]).strip()
                else "(blank)"
            )
            print(
                f"{setup}: "
                f"signals={int(row['signals'])} | "
                f"target_hit_rate={row['target_hit_rate'] * 100:.2f}% | "
                f"stop_hit_rate={row['stop_hit_rate'] * 100:.2f}% | "
                f"avg_10d={row['avg_close_return_10d']:.2f}%"
            )


if __name__ == "__main__":
    main()
