import os
import math
import datetime as dt
from typing import Optional, Dict, List

import pandas as pd
import yfinance as yf

# ---------- Settings ----------
INPUT_FILE = "data/scans_log.csv"
OUTPUT_FILE = "data/scans_outcomes.csv"

DEFAULT_HORIZON_DAYS = 20
MAX_FETCH_CALENDAR_DAYS = 90  # enough buffer for weekends/holidays
ONLY_UNVALIDATED = True       # skip rows already validated in OUTPUT_FILE


# ---------- Helpers ----------
def safe_float(x) -> float:
    try:
        if x is None:
            return float("nan")
        v = float(x)
        if math.isnan(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")


def normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper().replace(".", "-")


def ensure_data_dir() -> None:
    os.makedirs("data", exist_ok=True)


def load_scans(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)

    required_cols = [
        "run_id",
        "setup_id",
        "scan_date",
        "ticker",
        "setup_type",
        "status",
        "entry",
        "stop",
        "target",
        "rr",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"scans_log.csv is missing required columns: {missing}")

    df["ticker"] = df["ticker"].astype(str).map(normalize_ticker)
    df["scan_date"] = pd.to_datetime(df["scan_date"]).dt.date

    return df


def load_existing_outcomes(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "setup_id" in df.columns:
        df["setup_id"] = df["setup_id"].astype(str)
    return df


def fetch_price_data(ticker: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=(end_date + dt.timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    # yfinance sometimes returns Date or Datetime column name variants
    date_col = None
    for c in df.columns:
        if str(c).lower() in ("date", "datetime"):
            date_col = c
            break

    if date_col is None:
        raise ValueError(f"Could not find date column for ticker {ticker}")

    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    df = df.rename(columns={date_col: "Date"})

    needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
    existing = [c for c in needed if c in df.columns]
    df = df[existing].copy()

    return df.sort_values("Date").reset_index(drop=True)


def first_trading_day_on_or_after(df: pd.DataFrame, scan_date: dt.date) -> Optional[dt.date]:
    if df.empty:
        return None

    candidates = df[df["Date"] >= scan_date]
    if candidates.empty:
        return None

    return candidates.iloc[0]["Date"]


def evaluate_long_setup(
    price_df: pd.DataFrame,
    scan_date: dt.date,
    entry: float,
    stop: float,
    target: float,
    rr: float,
    horizon_days: int,
) -> Dict:
    """
    Assumptions:
    - setup is long-only
    - entry triggers when Low <= entry <= High
    - after trigger, stop/target are checked from the trigger day onward
    - if both stop and target are touched on the same day, we mark as ambiguous_same_day
    """
    result = {
        "evaluation_status": "invalid",
        "triggered": False,
        "trigger_date": None,
        "days_to_trigger": None,
        "outcome": "invalid",
        "exit_date": None,
        "days_in_trade": None,
        "max_gain_pct": None,
        "max_drawdown_pct": None,
        "max_gain_r": None,
        "max_drawdown_r": None,
        "exit_price_reference": None,
        "notes": "",
    }

    if price_df.empty:
        result["notes"] = "No price data"
        return result

    risk = entry - stop
    if risk <= 0:
        result["notes"] = "Invalid risk: entry <= stop"
        return result

    start_trade_date = first_trading_day_on_or_after(price_df, scan_date)
    if start_trade_date is None:
        result["notes"] = "No trading day on/after scan date"
        return result

    window_df = price_df[price_df["Date"] >= start_trade_date].head(horizon_days).copy()
    if window_df.empty:
        result["notes"] = "No data in evaluation window"
        return result

    trigger_idx = None
    trigger_date = None

    for i, row in window_df.iterrows():
        low = safe_float(row["Low"])
        high = safe_float(row["High"])
        if low <= entry <= high:
            trigger_idx = i
            trigger_date = row["Date"]
            break

    if trigger_idx is None:
        result["evaluation_status"] = "validated"
        result["outcome"] = "no_trigger"
        result["notes"] = "Entry not touched within horizon"
        return result

    triggered_df = window_df.loc[trigger_idx:].copy().reset_index(drop=True)

    result["triggered"] = True
    result["trigger_date"] = trigger_date.isoformat()
    result["days_to_trigger"] = (trigger_date - start_trade_date).days

    max_high = safe_float(triggered_df["High"].max())
    min_low = safe_float(triggered_df["Low"].min())

    result["max_gain_pct"] = ((max_high / entry) - 1.0) * 100.0
    result["max_drawdown_pct"] = ((min_low / entry) - 1.0) * 100.0
    result["max_gain_r"] = (max_high - entry) / risk
    result["max_drawdown_r"] = (min_low - entry) / risk

    exit_date = None
    exit_outcome = "open_after_window"
    exit_ref = None
    notes = ""

    for _, row in triggered_df.iterrows():
        low = safe_float(row["Low"])
        high = safe_float(row["High"])
        d = row["Date"]

        stop_hit = low <= stop
        target_hit = high >= target

        if stop_hit and target_hit:
            exit_date = d
            exit_outcome = "ambiguous_same_day"
            exit_ref = None
            notes = "Stop and target touched on same day"
            break
        if target_hit:
            exit_date = d
            exit_outcome = "target_hit"
            exit_ref = target
            break
        if stop_hit:
            exit_date = d
            exit_outcome = "stop_hit"
            exit_ref = stop
            break

    result["evaluation_status"] = "validated"
    result["outcome"] = exit_outcome
    result["exit_date"] = exit_date.isoformat() if exit_date else None
    result["days_in_trade"] = (exit_date - trigger_date).days if exit_date else (window_df.iloc[-1]["Date"] - trigger_date).days
    result["exit_price_reference"] = exit_ref
    result["notes"] = notes

    return result


def build_outcome_row(scan_row: pd.Series, eval_result: Dict, horizon_days: int) -> Dict:
    return {
        "run_id": scan_row["run_id"],
        "setup_id": scan_row["setup_id"],
        "scan_date": scan_row["scan_date"].isoformat(),
        "ticker": scan_row["ticker"],
        "setup_type": scan_row["setup_type"],
        "status": scan_row["status"],
        "entry": safe_float(scan_row["entry"]),
        "stop": safe_float(scan_row["stop"]),
        "target": safe_float(scan_row["target"]),
        "rr": safe_float(scan_row["rr"]),
        "horizon_days": horizon_days,
        "evaluation_status": eval_result["evaluation_status"],
        "triggered": eval_result["triggered"],
        "trigger_date": eval_result["trigger_date"],
        "days_to_trigger": eval_result["days_to_trigger"],
        "outcome": eval_result["outcome"],
        "exit_date": eval_result["exit_date"],
        "days_in_trade": eval_result["days_in_trade"],
        "max_gain_pct": eval_result["max_gain_pct"],
        "max_drawdown_pct": eval_result["max_drawdown_pct"],
        "max_gain_r": eval_result["max_gain_r"],
        "max_drawdown_r": eval_result["max_drawdown_r"],
        "exit_price_reference": eval_result["exit_price_reference"],
        "notes": eval_result["notes"],
        "validated_at_utc": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }


def summarize_results(df: pd.DataFrame) -> str:
    if df.empty:
        return "No outcomes written."

    total = len(df)
    validated = int((df["evaluation_status"] == "validated").sum()) if "evaluation_status" in df.columns else 0

    def count_outcome(name: str) -> int:
        if "outcome" not in df.columns:
            return 0
        return int((df["outcome"] == name).sum())

    no_trigger = count_outcome("no_trigger")
    target_hit = count_outcome("target_hit")
    stop_hit = count_outcome("stop_hit")
    open_after_window = count_outcome("open_after_window")
    ambiguous = count_outcome("ambiguous_same_day")

    return (
        f"Rows written: {total}\n"
        f"Validated: {validated}\n"
        f"Target hit: {target_hit}\n"
        f"Stop hit: {stop_hit}\n"
        f"No trigger: {no_trigger}\n"
        f"Open after window: {open_after_window}\n"
        f"Ambiguous same day: {ambiguous}"
    )


def main():
    ensure_data_dir()

    scans = load_scans(INPUT_FILE)
    existing_outcomes = load_existing_outcomes(OUTPUT_FILE)

    if ONLY_UNVALIDATED and not existing_outcomes.empty and "setup_id" in existing_outcomes.columns:
        done_ids = set(existing_outcomes["setup_id"].astype(str))
        scans = scans[~scans["setup_id"].astype(str).isin(done_ids)].copy()

    # We only validate setups that have a defined trade plan
    scans = scans[
        scans["setup_type"].isin(["pullback", "breakout", "vcp"])
    ].copy()

    scans["entry"] = scans["entry"].map(safe_float)
    scans["stop"] = scans["stop"].map(safe_float)
    scans["target"] = scans["target"].map(safe_float)
    scans["rr"] = scans["rr"].map(safe_float)

    scans = scans[
        (~scans["entry"].isna()) &
        (~scans["stop"].isna()) &
        (~scans["target"].isna())
    ].copy()

    if scans.empty:
        print("No scans to validate.")
        return

    all_rows: List[Dict] = []

    today = dt.date.today()

    grouped = scans.groupby("ticker", sort=True)

    for ticker, group in grouped:
        earliest_scan = min(group["scan_date"])
        latest_scan = max(group["scan_date"])

        fetch_start = earliest_scan - dt.timedelta(days=2)
        fetch_end = min(today, latest_scan + dt.timedelta(days=MAX_FETCH_CALENDAR_DAYS))

        try:
            price_df = fetch_price_data(ticker, fetch_start, fetch_end)
        except Exception as e:
            for _, scan_row in group.iterrows():
                failed = {
                    "evaluation_status": "invalid",
                    "triggered": False,
                    "trigger_date": None,
                    "days_to_trigger": None,
                    "outcome": "invalid",
                    "exit_date": None,
                    "days_in_trade": None,
                    "max_gain_pct": None,
                    "max_drawdown_pct": None,
                    "max_gain_r": None,
                    "max_drawdown_r": None,
                    "exit_price_reference": None,
                    "notes": f"Price fetch error: {e}",
                }
                all_rows.append(build_outcome_row(scan_row, failed, DEFAULT_HORIZON_DAYS))
            continue

        for _, scan_row in group.iterrows():
            scan_date = scan_row["scan_date"]

            # Do not validate very recent scans prematurely
            age_days = (today - scan_date).days
            if age_days < DEFAULT_HORIZON_DAYS:
                pending = {
                    "evaluation_status": "pending",
                    "triggered": False,
                    "trigger_date": None,
                    "days_to_trigger": None,
                    "outcome": "pending_window_not_finished",
                    "exit_date": None,
                    "days_in_trade": None,
                    "max_gain_pct": None,
                    "max_drawdown_pct": None,
                    "max_gain_r": None,
                    "max_drawdown_r": None,
                    "exit_price_reference": None,
                    "notes": f"Scan only {age_days} days old; waiting for full {DEFAULT_HORIZON_DAYS}-day window",
                }
                all_rows.append(build_outcome_row(scan_row, pending, DEFAULT_HORIZON_DAYS))
                continue

            entry = safe_float(scan_row["entry"])
            stop = safe_float(scan_row["stop"])
            target = safe_float(scan_row["target"])
            rr = safe_float(scan_row["rr"])

            result = evaluate_long_setup(
                price_df=price_df,
                scan_date=scan_date,
                entry=entry,
                stop=stop,
                target=target,
                rr=rr,
                horizon_days=DEFAULT_HORIZON_DAYS,
            )
            all_rows.append(build_outcome_row(scan_row, result, DEFAULT_HORIZON_DAYS))

    out_df = pd.DataFrame(all_rows)

    if os.path.exists(OUTPUT_FILE):
        out_df.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)
    else:
        out_df.to_csv(OUTPUT_FILE, mode="w", header=True, index=False)

    print(summarize_results(out_df))


if __name__ == "__main__":
    main()
