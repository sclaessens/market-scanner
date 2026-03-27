from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/scans_validation.csv")
OUTPUT_FILE = Path("data/validation_summary.csv")


def safe_mean(series):
    if series.empty:
        return 0.0
    return float(series.mean())


def analyze_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby(group_col)

    rows = []

    for name, group in grouped:
        if group.empty:
            continue

        total = len(group)

        target_hits = group["target_hit"].sum()
        stop_hits = group["stop_hit"].sum()

        winrate = target_hits / total if total > 0 else 0
        stop_rate = stop_hits / total if total > 0 else 0

        avg_return = safe_mean(group["close_return_pct_10d"])
        avg_gain = safe_mean(group["max_gain_pct"])
        avg_drawdown = safe_mean(group["max_drawdown_pct"])

        avg_rr = safe_mean(group["rr"])

        rows.append({
            group_col: name if pd.notna(name) else "(blank)",
            "signals": total,
            "winrate": round(winrate * 100, 2),
            "stop_rate": round(stop_rate * 100, 2),
            "avg_return_10d": round(avg_return, 2),
            "avg_max_gain": round(avg_gain, 2),
            "avg_max_drawdown": round(avg_drawdown, 2),
            "avg_rr": round(avg_rr, 2),
        })

    df_out = pd.DataFrame(rows)

    if not df_out.empty:
        df_out = df_out.sort_values("winrate", ascending=False)

    return df_out


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError("Validation file is empty")

    print("🔍 Running validation analysis...\n")

    # -------------------------
    # OVERALL
    # -------------------------
    total = len(df)
    winrate = df["target_hit"].mean()
    stop_rate = df["stop_hit"].mean()
    avg_return = df["close_return_pct_10d"].mean()

    print("📊 OVERALL")
    print(f"Signals: {total}")
    print(f"Winrate: {winrate * 100:.2f}%")
    print(f"Stop rate: {stop_rate * 100:.2f}%")
    print(f"Avg return (10d): {avg_return:.2f}%")
    print("")

    # -------------------------
    # BY SETUP
    # -------------------------
    print("📈 BY PRIMARY SETUP")
    df_setup = analyze_by_group(df, "primary_setup")
    print(df_setup.to_string(index=False))
    print("")

    # -------------------------
    # BY GRADE
    # -------------------------
    print("🏆 BY GRADE")
    df_grade = analyze_by_group(df, "grade")
    print(df_grade.to_string(index=False))
    print("")

    # -------------------------
    # SAVE SUMMARY
    # -------------------------
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(OUTPUT_FILE.with_suffix(".xlsx")) as writer:
        df_setup.to_excel(writer, sheet_name="by_setup", index=False)
        df_grade.to_excel(writer, sheet_name="by_grade", index=False)

    print(f"✅ Summary saved to: {OUTPUT_FILE.with_suffix('.xlsx')}")


if __name__ == "__main__":
    main()
