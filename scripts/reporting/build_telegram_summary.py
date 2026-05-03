import os
import sys
from datetime import datetime
from typing import Optional

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

FINAL_DECISIONS_FILE = "data/processed/final_decisions.csv"
MARKET_REGIME_FILE = "data/processed/market_regime.csv"
OUTPUT_FILE = "reports/daily/telegram_message.txt"


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


def human_reason(reason: str) -> str:
    key = clean_text(reason, fallback="-").lower().strip()

    mapping = {
        "mixed_structure": "signalen zijn gemengd",
        "below_ma50_above_ma200": "onder MA50, maar lange trend houdt nog stand",
        "below_ma50_and_ma200": "middellange en lange trend zijn gebroken",
        "above_ma20_and_ma50": "trend blijft gezond",
        "missing_price_data": "prijsdata ontbreekt",
        "missing_moving_average_data": "trenddata ontbreekt",
    }

    return mapping.get(key, clean_text(reason, fallback="-").replace("_", " "))


def build_market_regime_header() -> list[str]:
    df = read_csv_safe(MARKET_REGIME_FILE)

    regime = "UNKNOWN"
    if not df.empty:
        last_row = df.iloc[-1]
        for col in ["regime", "Regime", "market_regime"]:
            if col in df.columns:
                regime = clean_text(last_row.get(col), fallback="UNKNOWN").upper()
                break

    emoji_map = {
        "BULLISH": "🟢",
        "NEUTRAL": "🟡",
        "BEARISH": "🔴",
        "UNKNOWN": "⚪",
    }

    readable_map = {
        "BULLISH": "Markt is sterk",
        "NEUTRAL": "Markt is gemengd",
        "BEARISH": "Markt is zwak",
        "UNKNOWN": "Marktstatus onbekend",
    }

    return [
        f"Regime: {emoji_map.get(regime, '⚪')} {readable_map.get(regime, 'Marktstatus onbekend')} ({regime})",
        "",
    ]


def load_final_decisions() -> pd.DataFrame:
    df = read_csv_safe(FINAL_DECISIONS_FILE)

    if df.empty:
        return pd.DataFrame()

    for col in ["ticker", "final_action", "source_layer", "setup_type", "status"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().str.strip()

    return df


def append_action_section(
    lines: list[str],
    df: pd.DataFrame,
    action: str,
    title: str,
    show_when_empty: bool = False,
    empty_text: str = "- geen",
) -> None:
    subset = df[df["final_action"] == action].copy()

    if subset.empty and not show_when_empty:
        return

    lines.append(title)

    if subset.empty:
        lines.append(empty_text)
        lines.append("")
        return

    for _, row in subset.iterrows():
        ticker = clean_text(row.get("ticker"), fallback="?").upper()
        trigger_price = row.get("trigger_price")
        reason = human_reason(row.get("reason"))

        if action in {"BUY NOW", "SET LIMIT BUY", "SET STOP BUY"}:
            lines.append(f"- {ticker} → {fmt_price(trigger_price)}")
        elif action == "REMOVE":
            lines.append(f"- {ticker}")
        else:
            lines.append(f"- {ticker} → {reason}")

    lines.append("")


def build_scanner_section(lines: list[str], df: pd.DataFrame) -> None:
    scanner_df = df[
        (df["source_layer"] == "SCANNER")
        & (df["final_action"] == "NONE")
    ].copy()

    if scanner_df.empty:
        return

    lines.append("🎯 IDEEËN (geen directe actie)")

    for _, row in scanner_df.head(6).iterrows():
        ticker = clean_text(row.get("ticker"), fallback="?").upper()
        setup = clean_text(row.get("setup_type"), fallback="-").title()
        entry = fmt_price(row.get("entry"))
        rr = fmt_price(row.get("rr"))
        lines.append(f"- {ticker} — {setup} | entry {entry} | RR {rr}")

    lines.append("")


def build_portfolio_section(lines: list[str], df: pd.DataFrame) -> None:
    portfolio_df = df[df["source_layer"] == "PORTFOLIO"].copy()

    if portfolio_df.empty:
        return

    lines.append("💼 PORTFOLIO")

    action_order = ["SELL", "TRIM", "REVIEW", "HOLD", "ADD"]

    for action in action_order:
        subset = portfolio_df[portfolio_df["final_action"] == action]
        if subset.empty:
            continue

        lines.append(action)

        for _, row in subset.iterrows():
            ticker = clean_text(row.get("ticker"), fallback="?").upper()
            close = fmt_price(row.get("close"))
            reason = human_reason(row.get("reason"))

            label_map = {
                "SELL": "verkopen",
                "TRIM": "gedeeltelijk verkopen",
                "REVIEW": "opvolgen",
                "HOLD": "houden",
                "ADD": "uitbreiden",
            }

            label = label_map.get(action, action.lower())
            lines.append(f"- {ticker} → {label} | close {close} | {reason}")

        lines.append("")


def build_telegram_summary_text() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    df = load_final_decisions()

    lines: list[str] = [f"📲 Trading Signals — {timestamp}"]
    lines.extend(build_market_regime_header())

    if df.empty or "final_action" not in df.columns:
        lines.append("Geen final_decisions.csv gevonden of bestand is leeg.")
        return "\n".join(lines).strip() + "\n"

    append_action_section(
        lines,
        df,
        "BUY NOW",
        "🔥 ACTIE NU",
        show_when_empty=True,
        empty_text="Geen directe BUY NOW setups.",
    )

    append_action_section(lines, df, "SET LIMIT BUY", "📌 SET LIMIT BUY")
    append_action_section(lines, df, "SET STOP BUY", "📌 SET STOP BUY")
    append_action_section(lines, df, "REMOVE", "❌ REMOVE")

    build_scanner_section(lines, df)
    build_portfolio_section(lines, df)

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