import os
import sys
from datetime import datetime
from typing import Optional

import pandas as pd

# Zorg dat project root in path zit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

MARKET_REGIME_FILE = "data/processed/market_regime.csv"
SCANNER_FILE = "data/processed/scanner_ranked.csv"
WATCHLIST_FILE = "data/watchlist/watchlist_status.csv"
PORTFOLIO_REVIEW_FILE = "data/portfolio/portfolio_review.csv"
OUTPUT_FILE = "reports/daily/telegram_message.txt"


# =========================
# HELPERS
# =========================

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


def fmt_price(value) -> str:
    val = safe_float(value)
    if val is None:
        return "-"
    return f"{val:.2f}"


def fmt_pct(value) -> str:
    val = safe_float(value)
    if val is None:
        return "-"
    return f"{val:.2f}%"


def fmt_qty(value) -> str:
    val = safe_float(value)
    if val is None:
        return "-"
    if val.is_integer():
        return str(int(val))
    return f"{val:.4f}"


def clean_text(value, fallback: str = "-") -> str:
    if value is None:
        return fallback
    if isinstance(value, float) and pd.isna(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def normalize_reason(reason: str) -> str:
    """
    Maak redenen leesbaarder in Telegram.
    """
    mapping = {
        "still_below_ma20": "nog onder MA20",
        "below_ma20": "nog onder MA20",
        "below_ma50": "onder MA50",
        "neutral_regime_wait_for_better_entry": "neutraal regime, wacht op betere entry",
        "no_reclaim_ma20": "nog geen reclaim van MA20",
        "still_no_reclaim_ma20": "nog geen reclaim van MA20",
        "below_ma50_above_ma200": "onder MA50 maar boven MA200",
        "below_ma50_and_ma200": "onder MA50 en MA200",
        "missing_price_data": "ontbrekende prijsdata",
        "missing_moving_average_data": "ontbrekende moving average data",
        "mixed_structure": "gemengde structuur",
    }
    key = clean_text(reason, fallback="-")
    return mapping.get(key, key.replace("_", " "))


def get_grouped_setup_order() -> list[str]:
    return ["BREAKOUT", "PULLBACK", "VCP"]


# =========================
# MARKET REGIME
# =========================

def build_market_regime_header() -> list[str]:
    df = read_csv_safe(MARKET_REGIME_FILE)

    regime = "UNKNOWN"
    if not df.empty:
        last_row = df.iloc[-1]
        for col in ["regime", "Regime"]:
            if col in df.columns:
                regime = clean_text(last_row[col], fallback="UNKNOWN").upper()
                break

    emoji_map = {
        "BULLISH": "🟢",
        "NEUTRAL": "🟡",
        "BEARISH": "🔴",
        "UNKNOWN": "⚪",
    }
    emoji = emoji_map.get(regime, "⚪")

    return [f"Regime: {emoji} {regime}", ""]


# =========================
# WATCHLIST / DECISION LAYER
# =========================

def format_watchlist_line(row: pd.Series) -> str:
    ticker = clean_text(row.get("ticker", "?")).upper()
    close = fmt_price(row.get("close"))
    ma20 = fmt_price(row.get("ma20"))
    ma50 = fmt_price(row.get("ma50"))
    reason = normalize_reason(row.get("reason"))

    parts = [ticker]

    if close != "-":
        parts.append(f"close {close}")
    if ma20 != "-":
        parts.append(f"MA20 {ma20}")
    if ma50 != "-":
        parts.append(f"MA50 {ma50}")
    if reason != "-":
        parts.append(reason)

    return "- " + " | ".join(parts)


def build_watchlist_action_sections() -> list[str]:
    df = read_csv_safe(WATCHLIST_FILE)

    if df.empty or "status" not in df.columns:
        return [
            "✅ No READY setups right now.",
            "",
            "👀 No watchlist data available.",
            "",
        ]

    df = df.copy()
    df["status"] = df["status"].astype(str).str.upper()

    if "setup_type" in df.columns:
        df["setup_type"] = df["setup_type"].astype(str).str.upper()
    else:
        df["setup_type"] = "UNKNOWN"

    lines: list[str] = []

    # ACTION = READY
    ready_df = df[df["status"] == "READY"].copy()
    if ready_df.empty:
        lines.append("✅ No READY setups right now.")
        lines.append("")
    else:
        lines.append(f"🚀 READY SETUPS ({len(ready_df)})")

        for setup_type in get_grouped_setup_order():
            subset = ready_df[ready_df["setup_type"] == setup_type].copy()
            if subset.empty:
                continue

            lines.append(setup_type)
            for _, row in subset.iterrows():
                lines.append(format_watchlist_line(row))
            lines.append("")

        remaining = ready_df[~ready_df["setup_type"].isin(get_grouped_setup_order())]
        if not remaining.empty:
            lines.append("OTHER")
            for _, row in remaining.iterrows():
                lines.append(format_watchlist_line(row))
            lines.append("")

    # PREPARE = WAIT
    wait_df = df[df["status"] == "WAIT"].copy()
    if wait_df.empty:
        lines.append("👀 No WAIT setups right now.")
        lines.append("")
    else:
        lines.append(f"👀 WAIT SETUPS ({len(wait_df)})")

        for setup_type in get_grouped_setup_order():
            subset = wait_df[wait_df["setup_type"] == setup_type].copy()
            if subset.empty:
                continue

            lines.append(setup_type)
            for _, row in subset.iterrows():
                lines.append(format_watchlist_line(row))
            lines.append("")

        remaining = wait_df[~wait_df["setup_type"].isin(get_grouped_setup_order())]
        if not remaining.empty:
            lines.append("OTHER")
            for _, row in remaining.iterrows():
                lines.append(format_watchlist_line(row))
            lines.append("")

    # REMOVED / REJECTED / EXPIRED
    removed_df = df[df["status"].isin(["REJECTED", "EXPIRED"])].copy()
    if not removed_df.empty:
        lines.append(f"❌ REMOVED FROM WATCHLIST ({len(removed_df)})")
        for _, row in removed_df.iterrows():
            ticker = clean_text(row.get("ticker", "?")).upper()
            lines.append(f"- {ticker}")
        lines.append("")

    return lines


# =========================
# SCANNER CONTEXT
# =========================

def build_scanner_context_section() -> list[str]:
    df = read_csv_safe(SCANNER_FILE)

    header = ["🎯 SCANNER IDEAS"]

    if df.empty:
        return header + ["- no scanner data", ""]

    if "setup_grade" in df.columns:
        df["setup_grade"] = df["setup_grade"].astype(str).str.upper()

    if "score_total" in df.columns:
        df["score_total"] = pd.to_numeric(df["score_total"], errors="coerce")
        df = df.sort_values(by="score_total", ascending=False)

    lines: list[str] = []
    grade_order = ["A", "B"]

    for grade in grade_order:
        subset_grade = df[df["setup_grade"] == grade].copy() if "setup_grade" in df.columns else pd.DataFrame()
        if subset_grade.empty:
            continue

        lines.append(f"{grade} setups")

        for setup_type in get_grouped_setup_order():
            subset = subset_grade[subset_grade["setup_type"].astype(str).str.upper() == setup_type].copy() \
                if "setup_type" in subset_grade.columns else pd.DataFrame()

            if subset.empty:
                continue

            lines.append(setup_type)

            for _, row in subset.head(5).iterrows():
                ticker = clean_text(row.get("ticker", "?")).upper()
                entry = fmt_price(row.get("entry"))
                stop = fmt_price(row.get("stop"))
                target = fmt_price(row.get("target"))
                rr = fmt_price(row.get("rr"))

                lines.append(
                    f"- {ticker} | entry {entry} | stop {stop} | target {target} | R:R {rr}"
                )

            lines.append("")

    if not lines:
        lines.append("- no A/B setups")
        lines.append("")

    return header + lines


# =========================
# PORTFOLIO SECTION
# =========================

def build_portfolio_section() -> list[str]:
    df = read_csv_safe(PORTFOLIO_REVIEW_FILE)

    header = ["💼 PORTFOLIO REVIEW"]

    if df.empty or "decision" not in df.columns:
        return header + ["- no portfolio review data", ""]

    df = df.copy()
    df["decision"] = df["decision"].astype(str).str.upper()

    lines: list[str] = []
    decision_order = ["SELL", "TRIM", "REVIEW", "HOLD"]

    for decision in decision_order:
        subset = df[df["decision"] == decision].copy()
        if subset.empty:
            continue

        lines.append(f"{decision} ({len(subset)})")

        for _, row in subset.iterrows():
            ticker = clean_text(row.get("ticker", "?")).upper()
            qty = fmt_qty(row.get("quantity"))
            last_price = fmt_price(row.get("last_price"))
            pnl_pct = fmt_pct(row.get("pnl_pct"))
            reason = normalize_reason(row.get("reason"))
            risk_flag = clean_text(row.get("risk_flag", "-")).upper()

            lines.append(
                f"- {ticker} | qty {qty} | price {last_price} | pnl {pnl_pct} | {reason} | risk {risk_flag}"
            )

        lines.append("")

    if not lines:
        lines.append("- no open portfolio decisions")
        lines.append("")

    return header + lines


# =========================
# MAIN BUILDER
# =========================

def build_telegram_summary_text() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [f"📲 Trading Signals — {timestamp}"]
    lines.extend(build_market_regime_header())
    lines.extend(build_watchlist_action_sections())
    lines.extend(build_scanner_context_section())
    lines.extend(build_portfolio_section())

    # Dubbele lege lijnen opruimen
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