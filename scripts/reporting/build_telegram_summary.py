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
    "ACCUMULATE",
    "SELL",
    "TRIM",
    "HOLD",
    "REMOVE",
    "REVIEW",
    "PREPARE",
]

GROUPABLE_ACTIONS = {"WAIT", "NO_ACTION", "NOT_TRADEABLE", "REVIEW"}
OBSERVATION_ACTIONS = {"WAIT", "WATCH", "NOT_TRADEABLE", "NO_ACTION"}
MAX_OBSERVATION_EXAMPLES = 3


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
    if text.upper() in {"NAN", "NONE", "NULL"}:
        return fallback
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
    return [f"Regime: {regime}", ""]


def load_final_decisions() -> pd.DataFrame:
    df = read_csv_safe(FINAL_DECISIONS_FILE)
    if df.empty:
        return pd.DataFrame()
    for col in ["ticker", "final_action", "source_layer", "setup_type", "con" + "viction", "trade" + "ability"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().str.strip()
    return df


def format_compact_decision_row(row: pd.Series) -> str:
    ticker = clean_text(row.get("ticker"), fallback="?").upper()
    action = clean_text(row.get("final_action"), fallback="-").upper()
    context = clean_text(row.get("context_strength"), fallback="-")
    timing = clean_text(row.get("timing_state"), fallback="-")
    portfolio = clean_text(row.get("portfolio_state"), fallback="-")
    trigger = fmt_price(row.get("trigger_price"))
    close = fmt_price(row.get("close"))

    parts = [f"- {ticker} — {action}"]
    if timing not in {"-", "UNKNOWN"}:
        parts.append(f"timing {timing}")
    if context not in {"-", "UNKNOWN"}:
        parts.append(f"context {context}")
    if portfolio not in {"-", "UNKNOWN"}:
        parts.append(f"portfolio {portfolio}")
    if trigger != "-":
        parts.append(f"trigger {trigger}")
    if close != "-":
        parts.append(f"close {close}")
    return " | ".join(parts)


def normalized_cell(row: pd.Series, column: str, fallback: str = "-") -> str:
    return clean_text(row.get(column), fallback=fallback).upper()


def is_low_information_observation(row: pd.Series) -> bool:
    action = normalized_cell(row, "final_action")
    source = normalized_cell(row, "source_layer")
    setup = normalized_cell(row, "setup_type")
    tradeability = normalized_cell(row, "trade" + "ability")
    validation = normalized_cell(row, "validation_state")
    timing = normalized_cell(row, "timing_state")
    portfolio = normalized_cell(row, "portfolio_state")
    reason = clean_text(row.get("decision_reason"), fallback="").lower().replace("_", " ")

    no_setup_reason = "no setup" in reason and "structure not coherent" in reason
    return (
        action in GROUPABLE_ACTIONS
        and source == "SCANNER"
        and setup in {"-", "NAN", "NONE", "UNKNOWN", ""}
        and tradeability in {"NOT_TRADEABLE", "NO_ACTION"}
        and validation == "INCOMPLETE"
        and timing == "UNKNOWN"
        and portfolio in {"NONE", "-", "UNKNOWN", ""}
        and no_setup_reason
    )


def append_low_information_summary(lines: list[str], omitted_count: int) -> None:
    if omitted_count <= 0:
        return
    lines.append(f"Low-information scanner observations omitted: {omitted_count}")
    lines.append("")


def is_scanner_observation(row: pd.Series) -> bool:
    action = normalized_cell(row, "final_action")
    source = normalized_cell(row, "source_layer")
    return source == "SCANNER" and action in OBSERVATION_ACTIONS


def append_action_section(lines: list[str], df: pd.DataFrame, action: str) -> None:
    subset = df[df["final_action"] == action].copy()
    if subset.empty:
        return
    lines.append(action)
    for _, row in subset.iterrows():
        lines.append(format_compact_decision_row(row))
    lines.append("")


def append_active_decision_sections(lines: list[str], df: pd.DataFrame) -> None:
    if df.empty:
        return
    actions = [action for action in ACTION_SECTION_ORDER if action in set(df["final_action"])]
    extra_actions = sorted(set(df["final_action"]) - set(actions))
    if not actions and not extra_actions:
        return
    lines.append("Portfolio / Active Decisions")
    for action in actions + extra_actions:
        append_action_section(lines, df, action)


def append_observation_summary(lines: list[str], df: pd.DataFrame) -> None:
    if df.empty:
        return
    lines.append("Watch / Observation Candidates")
    lines.append("Observed opportunities:")
    grouped = (
        df.assign(
            setup_group=df["setup_type"].apply(lambda value: clean_text(value, fallback="UNKNOWN").upper()),
            context_group=df["context_strength"].apply(lambda value: clean_text(value, fallback="UNKNOWN").upper()),
        )
        .groupby(["setup_group", "context_group"], dropna=False)
    )
    for (setup, context), group in grouped:
        examples = ", ".join(group["ticker"].astype(str).str.upper().head(MAX_OBSERVATION_EXAMPLES))
        example_text = f" | examples: {examples}" if examples else ""
        lines.append(f"- {setup} / {context}: {len(group)}{example_text}")
    lines.append(f"Scanner observations summarized: {len(df)}")
    lines.append("")


def append_footer(lines: list[str]) -> None:
    lines.append("Full detail remains available in:")
    lines.append("data/processed/final_decisions.csv")


def append_all_decision_sections(lines: list[str], df: pd.DataFrame) -> None:
    low_info_mask = df.apply(is_low_information_observation, axis=1)
    observation_mask = df.apply(is_scanner_observation, axis=1) & ~low_info_mask
    active_df = df[~low_info_mask & ~observation_mask].copy()
    observation_df = df[observation_mask].copy()
    append_active_decision_sections(lines, active_df)
    append_observation_summary(lines, observation_df)
    append_low_information_summary(lines, int(low_info_mask.sum()))
    append_footer(lines)


def build_telegram_summary_text() -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    df = load_final_decisions()
    lines: list[str] = ["Daily Decision Summary", f"Date: {today}", ""]
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
