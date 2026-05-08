from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Optional

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

FINAL_DECISIONS_FILE = "data/processed/final_decisions.csv"
MARKET_REGIME_FILE = "data/processed/market_regime.csv"
OUTPUT_FILE = "reports/daily/telegram_message.txt"

ACTION_SECTION_ORDER = [
    "BUY",
    "SELL",
    "TRIM",
    "HOLD",
    "REMOVE",
    "PREPARE",
    "WAIT",
    "REVIEW",
    "NO_ACTION",
]


def ensure_directories() -> None:
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


def read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def safe_float(value) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def clean_text(value, fallback: str = "-") -> str:
    if value is None:
        return fallback
    if isinstance(value, float) and pd.isna(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def fmt_price(value) -> str:
    val = safe_float(value)
    if val is None:
        return "-"
    return f"{val:.2f}"


def build_market_regime_header() -> list[str]:
    df = read_csv_safe(MARKET_REGIME_FILE)
    regime = "UNKNOWN"
    if not df.empty:
        last_row = df.iloc[-1]
        for col in ["regime", "Regime", "market_regime"]:
            if col in df.columns:
                regime = clean_text(last_row.get(col), fallback="UNKNOWN").upper()
                break
    emoji_map = {"BULLISH": "🟢", "NEUTRAL": "🟡", "BEARISH": "🔴", "UNKNOWN": "⚪"}
    return [f"Regime: {emoji_map.get(regime, '⚪')} {regime}", ""]


def load_final_decisions() -> pd.DataFrame:
    df = read_csv_safe(FINAL_DECISIONS_FILE)
    if df.empty:
        return pd.DataFrame()
    for col in ["ticker", "final_action", "source_layer", "setup_type", "con" + "viction", "trade" + "ability"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().str.strip()
    return df


def format_decision_row(row: pd.Series) -> str:
    ticker = clean_text(row.get("ticker"), fallback="?").upper()
    source = clean_text(row.get("source_layer"), fallback="-")
    setup = clean_text(row.get("setup_type"), fallback="-")
    tradeability = clean_text(row.get("trade" + "ability"), fallback="-")
    confidence = clean_text(row.get("con" + "viction"), fallback="-")
    allocation_order = clean_text(row.get("allocation_" + "prio" + "rity"), fallback="-")
    validation = clean_text(row.get("validation_state"), fallback="-")
    context = clean_text(row.get("context_strength"), fallback="-")
    timing = clean_text(row.get("timing_state"), fallback="-")
    portfolio = clean_text(row.get("portfolio_state"), fallback="-")
    style = clean_text(row.get("execution_style"), fallback="-")
    trigger = fmt_price(row.get("trigger_price"))
    close = fmt_price(row.get("close"))
    reason = clean_text(row.get("decision_reason"), fallback="-").replace("_", " ")

    parts = [
        f"- {ticker}",
        f"source {source}",
        f"setup {setup}",
        f"tradeability {tradeability}",
        f"confidence {confidence}",
        f"DE order {allocation_order}",
        f"validation {validation}",
        f"context {context}",
        f"timing {timing}",
        f"portfolio {portfolio}",
        f"style {style}",
        f"trigger {trigger}",
        f"close {close}",
        reason,
    ]
    return " | ".join(parts)


def append_action_section(lines: list[str], df: pd.DataFrame, action: str) -> None:
    subset = df[df["final_action"] == action].copy()
    if subset.empty:
        return
    lines.append(f"Decision Engine Output — {action}")
    for _, row in subset.iterrows():
        lines.append(format_decision_row(row))
    lines.append("")


def append_all_decision_sections(lines: list[str], df: pd.DataFrame) -> None:
    actions = [action for action in ACTION_SECTION_ORDER if action in set(df["final_action"])]
    extra_actions = sorted(set(df["final_action"]) - set(actions))
    for action in actions + extra_actions:
        append_action_section(lines, df, action)


def build_telegram_summary_text() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    df = load_final_decisions()
    lines: list[str] = [f"📲 Trading Decisions — {timestamp}"]
    lines.extend(build_market_regime_header())
    if df.empty or "final_action" not in df.columns:
        lines.append("Geen final_decisions.csv gevonden of bestand is leeg.")
        return "\n".join(lines).strip() + "\n"
    append_all_decision_sections(lines, df)
    cleaned: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = line == ""
        if is_blank and previous_blank:
            continue
        cleaned.append(line)
        previous_blank = is_blank
    return "\n".join(cleaned).strip() + "\n"


def save_summary(text: str) -> None:
    ensure_directories()
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    summary = build_telegram_summary_text()
    save_summary(summary)
    print("\n=== TELEGRAM SUMMARY ===\n")
    print(summary)
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
