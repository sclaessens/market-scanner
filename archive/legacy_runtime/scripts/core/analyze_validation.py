from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR


VALIDATION_RESULTS_FILE = DATA_DIR / "processed" / "validation_results.csv"
VALIDATION_SUMMARY_FILE = DATA_DIR / "processed" / "validation_summary.csv"


VALID_OUTCOMES = {"TARGET", "STOP", "TIMEOUT", "STOP_FIRST_SAME_DAY"}
WIN_OUTCOMES = {"TARGET"}


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def clean_text(value, fallback: str = "UNKNOWN") -> str:
    if value is None:
        return fallback
    if isinstance(value, float) and pd.isna(value):
        return fallback
    text = str(value).strip()
    if not text or text.upper() == "NAN":
        return fallback
    return text.upper()


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []

    grouped = df.groupby(group_cols, dropna=False)

    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        total_signals = len(group)
        triggered = int(group["entry_hit"].sum())
        completed = group[group["outcome"].isin(VALID_OUTCOMES)].copy()

        target_hits = int((group["outcome"] == "TARGET").sum())
        stop_hits = int((group["outcome"] == "STOP").sum())
        timeouts = int((group["outcome"] == "TIMEOUT").sum())
        same_day_stop = int((group["outcome"] == "STOP_FIRST_SAME_DAY").sum())
        not_triggered = int((group["outcome"] == "NOT_TRIGGERED").sum())
        no_future_data = int((group["outcome"] == "NO_FUTURE_DATA").sum())
        no_price_data = int((group["outcome"] == "NO_PRICE_DATA").sum())

        winrate = None
        if len(completed) > 0:
            winrate = target_hits / len(completed) * 100

        trigger_rate = triggered / total_signals * 100 if total_signals > 0 else None

        avg_max_gain = group["max_gain_pct"].mean()
        avg_max_drawdown = group["max_drawdown_pct"].mean()
        avg_days_to_entry = group["days_to_entry"].mean()
        avg_days_to_outcome = group["days_to_outcome"].mean()

        row = {
            "group_type": "+".join(group_cols),
            "total_signals": total_signals,
            "triggered": triggered,
            "trigger_rate_pct": round(trigger_rate, 2) if trigger_rate is not None else None,
            "completed_signals": len(completed),
            "target_hits": target_hits,
            "stop_hits": stop_hits,
            "timeouts": timeouts,
            "same_day_stop": same_day_stop,
            "not_triggered": not_triggered,
            "no_future_data": no_future_data,
            "no_price_data": no_price_data,
            "winrate_pct": round(winrate, 2) if winrate is not None else None,
            "avg_max_gain_pct": round(avg_max_gain, 2) if not pd.isna(avg_max_gain) else None,
            "avg_max_drawdown_pct": round(avg_max_drawdown, 2) if not pd.isna(avg_max_drawdown) else None,
            "avg_days_to_entry": round(avg_days_to_entry, 2) if not pd.isna(avg_days_to_entry) else None,
            "avg_days_to_outcome": round(avg_days_to_outcome, 2) if not pd.isna(avg_days_to_outcome) else None,
        }

        for col, value in zip(group_cols, keys):
            row[col] = clean_text(value)

        rows.append(row)

    return pd.DataFrame(rows)


def analyze_validation() -> pd.DataFrame:
    df = read_csv_safe(VALIDATION_RESULTS_FILE)

    if df.empty:
        empty_df = pd.DataFrame()
        empty_df.to_csv(VALIDATION_SUMMARY_FILE, index=False)
        return empty_df

    df = df.copy()

    for col in ["setup_type", "grade", "regime", "outcome"]:
        if col not in df.columns:
            df[col] = "UNKNOWN"
        df[col] = df[col].apply(clean_text)

    for col in [
        "entry_hit",
        "max_gain_pct",
        "max_drawdown_pct",
        "days_to_entry",
        "days_to_outcome",
    ]:
        if col in df.columns:
            if col == "entry_hit":
                df[col] = df[col].astype(str).str.upper().isin(["TRUE", "1", "YES"])
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    summaries = []

    for group_cols in [
        ["setup_type"],
        ["grade"],
        ["regime"],
        ["setup_type", "grade"],
        ["setup_type", "regime"],
    ]:
        summary = summarize_group(df, group_cols)
        summaries.append(summary)

    result_df = pd.concat(summaries, ignore_index=True)

    sort_cols = ["group_type", "total_signals"]
    result_df = result_df.sort_values(sort_cols, ascending=[True, False]).reset_index(drop=True)

    VALIDATION_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(VALIDATION_SUMMARY_FILE, index=False)

    return result_df


def main() -> None:
    df = analyze_validation()
    print(f"Validation summary written to: {VALIDATION_SUMMARY_FILE}")
    print(f"Rows: {len(df)}")

    if not df.empty:
        cols = [
            "group_type",
            "setup_type",
            "grade",
            "regime",
            "total_signals",
            "trigger_rate_pct",
            "winrate_pct",
            "target_hits",
            "stop_hits",
            "timeouts",
        ]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()