import os
import sys
from datetime import datetime
from typing import Optional

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

MARKET_REGIME_FILE = "data/processed/market_regime.csv"
SCANNER_FILE = "data/processed/scanner_ranked.csv"
WATCHLIST_FILE = "data/watchlist/watchlist_status.csv"
PORTFOLIO_REVIEW_FILE = "data/portfolio/portfolio_review.csv"
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


def title_case_setup(setup_type: str) -> str:
    mapping = {
        "BREAKOUT": "Breakout",
        "PULLBACK": "Pullback",
        "VCP": "Rustige opbouw",
        "UNKNOWN": "Overig",
    }
    key = clean_text(setup_type, fallback="UNKNOWN").upper()
    return mapping.get(key, key.title())


def normalize_reason(reason: str) -> str:
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
        "above_ma20_and_ma50": "boven MA20 en MA50",
        "above_ma20_ma50": "boven MA20 en MA50",
        "extended_above_ma20": "ver boven MA20",
    }
    key = clean_text(reason, fallback="-")
    return mapping.get(key, key.replace("_", " "))


def urgency_icon(level: str) -> str:
    return {
        "low": "✅",
        "medium": "⚠️",
        "high": "🚨",
    }.get(level, "ℹ️")


def get_portfolio_tickers() -> set[str]:
    df = read_csv_safe(PORTFOLIO_REVIEW_FILE)
    if df.empty or "ticker" not in df.columns:
        return set()

    return set(df["ticker"].astype(str).str.upper().str.strip().tolist())


def get_watchlist_tickers() -> set[str]:
    df = read_csv_safe(WATCHLIST_FILE)
    if df.empty or "ticker" not in df.columns:
        return set()

    return set(df["ticker"].astype(str).str.upper().str.strip().tolist())


def get_scanner_close_map() -> dict[str, float]:
    df = read_csv_safe(SCANNER_FILE)
    if df.empty or "ticker" not in df.columns or "close" not in df.columns:
        return {}

    out: dict[str, float] = {}
    for _, row in df.iterrows():
        ticker = clean_text(row.get("ticker"), fallback="").upper()
        close = safe_float(row.get("close"))
        if ticker and close is not None:
            out[ticker] = close
    return out


def get_effective_watchlist_close(
    row: pd.Series,
    scanner_close_map: dict[str, float],
) -> tuple[Optional[float], str]:
    ticker = clean_text(row.get("ticker"), fallback="").upper()
    watchlist_close = safe_float(row.get("close"))
    scanner_close = scanner_close_map.get(ticker)

    if scanner_close is None:
        return watchlist_close, "watchlist"

    if watchlist_close is None or watchlist_close == 0:
        return scanner_close, "scanner"

    difference_pct = abs(scanner_close - watchlist_close) / abs(watchlist_close)

    if difference_pct > 0.02:
        return scanner_close, "scanner"

    return watchlist_close, "watchlist"


def exclude_tickers(df: pd.DataFrame, tickers: set[str]) -> pd.DataFrame:
    if df.empty or "ticker" not in df.columns or not tickers:
        return df.copy()

    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    return out[~out["ticker"].isin(tickers)].copy()


def build_watchlist_action_from_row(row: pd.Series) -> str:
    explicit = clean_text(row.get("action_now"), fallback="-")
    if explicit != "-":
        return explicit

    status = clean_text(row.get("status"), fallback="WAIT").upper()
    mapping = {
        "READY": "Koopcheck doen",
        "WAIT": "Nog niets doen",
        "REJECTED": "Niet meer volgen",
        "EXPIRED": "Van watchlist halen",
    }
    return mapping.get(status, "Opvolgen")


def get_watchlist_display_decision(
    row: pd.Series,
    scanner_close_map: dict[str, float],
) -> dict:
    action_now = build_watchlist_action_from_row(row)
    action_group = clean_text(action_now).upper()

    ticker = clean_text(row.get("ticker"), fallback="?").upper()
    setup_type = clean_text(row.get("setup_type"), fallback="-").upper()
    close_value, close_source = get_effective_watchlist_close(row, scanner_close_map)

    trigger_price = safe_float(row.get("trigger_price"))
    if trigger_price is None:
        if action_group == "SET LIMIT BUY":
            trigger_price = safe_float(row.get("ma20"))
        elif action_group == "SET STOP BUY":
            trigger_price = safe_float(row.get("high_20d"))

    display_price = trigger_price
    why = clean_text(row.get("why_now"), fallback="-")
    if why == "-":
        why = clean_text(
            row.get("reason_text"),
            fallback=clean_text(row.get("reason"), fallback="-"),
        )

    safety_note = ""

    if action_group == "SET STOP BUY" and trigger_price is not None and close_value is not None:
        if trigger_price <= close_value:
            action_group = "SET LIMIT BUY"
            action_now = "SET LIMIT BUY"
            display_price = safe_float(row.get("ma20"))
            why = (
                "Breakout-trigger ligt onder de actuele koers. "
                "Breakout niet najagen; wacht op pullback richting MA20."
            )
            safety_note = "⚠️ Stop-buy gecorrigeerd omdat trigger onder actuele koers ligt."

    context_parts = []

    if close_value is not None:
        label = "actuele close" if close_source == "scanner" else "close"
        context_parts.append(f"{label} {close_value:.2f}")

    if trigger_price is not None:
        context_parts.append(f"trigger {trigger_price:.2f}")

    ma20 = safe_float(row.get("ma20"))
    if ma20 is not None:
        context_parts.append(f"MA20 {ma20:.2f}")

    high_20d = safe_float(row.get("high_20d"))
    if high_20d is not None and setup_type == "BREAKOUT":
        context_parts.append(f"20D high {high_20d:.2f}")

    return {
        "ticker": ticker,
        "action_group": action_group,
        "action_now": action_now,
        "display_price": display_price,
        "why": why,
        "context": " | ".join(context_parts),
        "safety_note": safety_note,
    }


def append_compact_watchlist_decision(
    lines: list[str],
    row: pd.Series,
    scanner_close_map: dict[str, float],
) -> None:
    decision = get_watchlist_display_decision(row, scanner_close_map)
    price = fmt_price(decision["display_price"])
    ticker = decision["ticker"]

    if decision["safety_note"]:
        lines.append(f"- {ticker} → {price} (breakout gemist; wacht op pullback)")
    else:
        lines.append(f"- {ticker} → {price}")


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
    readable_map = {
        "BULLISH": "Markt is sterk",
        "NEUTRAL": "Markt is gemengd",
        "BEARISH": "Markt is zwak",
        "UNKNOWN": "Marktstatus onbekend",
    }

    emoji = emoji_map.get(regime, "⚪")
    readable = readable_map.get(regime, "Marktstatus onbekend")

    return [f"Regime: {emoji} {readable} ({regime})", ""]


def get_scanner_df() -> pd.DataFrame:
    df = read_csv_safe(SCANNER_FILE)

    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    if "setup_grade" in df.columns:
        df["setup_grade"] = df["setup_grade"].astype(str).str.upper().str.strip()
    else:
        df["setup_grade"] = ""

    if "setup_type" in df.columns:
        df["setup_type"] = df["setup_type"].astype(str).str.upper().str.strip()
    else:
        df["setup_type"] = "UNKNOWN"

    numeric_cols = [
        "score_total",
        "entry",
        "stop",
        "target",
        "rr",
        "close",
        "high_20d",
        "ma20",
        "ma50",
        "volume_ratio",
        "breakout_strength",
        "extension_atr",
        "rs_20d_pct",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "score_total" in df.columns:
        df = df.sort_values(by="score_total", ascending=False)

    return df.reset_index(drop=True)


def get_scanner_buy_now_candidates(max_items: int = 4) -> pd.DataFrame:
    df = get_scanner_df()

    if df.empty:
        return pd.DataFrame()

    portfolio_tickers = get_portfolio_tickers()
    watchlist_tickers = get_watchlist_tickers()
    blocked_tickers = portfolio_tickers | watchlist_tickers

    df = exclude_tickers(df, blocked_tickers)

    required_cols = {"setup_grade", "setup_type", "breakout_strength", "extension_atr"}
    if not required_cols.issubset(df.columns):
        return pd.DataFrame()

    df = df[
        (df["setup_grade"] == "A")
        & (df["setup_type"] == "BREAKOUT")
        & (df["breakout_strength"] >= 3.0)
        & (df["extension_atr"] <= 1.5)
    ].copy()

    if df.empty:
        return df

    return df.head(max_items).reset_index(drop=True)


def get_strong_extended_breakouts(max_items: int = 6) -> pd.DataFrame:
    df = get_scanner_df()

    if df.empty:
        return pd.DataFrame()

    portfolio_tickers = get_portfolio_tickers()
    watchlist_tickers = get_watchlist_tickers()
    scanner_buy_df = get_scanner_buy_now_candidates()

    scanner_buy_tickers: set[str] = set()
    if not scanner_buy_df.empty and "ticker" in scanner_buy_df.columns:
        scanner_buy_tickers = set(
            scanner_buy_df["ticker"].astype(str).str.upper().str.strip().tolist()
        )

    blocked_tickers = portfolio_tickers | watchlist_tickers | scanner_buy_tickers
    df = exclude_tickers(df, blocked_tickers)

    required_cols = {"setup_type", "breakout_strength", "extension_atr"}
    if not required_cols.issubset(df.columns):
        return pd.DataFrame()

    df = df[
        (df["setup_type"] == "BREAKOUT")
        & (df["breakout_strength"] >= 3.0)
        & (df["extension_atr"] > 1.5)
    ].copy()

    if df.empty:
        return df

    if "score_total" in df.columns:
        df = df.sort_values(by="score_total", ascending=False)

    return df.head(max_items).reset_index(drop=True)


def append_scanner_buy_now(lines: list[str], row: pd.Series) -> None:
    ticker = clean_text(row.get("ticker"), fallback="?").upper()
    entry = safe_float(row.get("entry"))
    close = safe_float(row.get("close"))
    high_20d = safe_float(row.get("high_20d"))
    price = entry if entry is not None else close

    buy_label = "BUY NOW"

    if close is not None and high_20d is not None and high_20d > 0:
        breakout_extension_pct = (close - high_20d) / high_20d
        if breakout_extension_pct > 0.03:
            buy_label = "BUY NOW (agressief)"
        else:
            buy_label = "BUY NOW (sterk)"

    lines.append(f"- {ticker} → {buy_label} rond {fmt_price(price)}")


def build_watchlist_action_sections() -> list[str]:
    df = read_csv_safe(WATCHLIST_FILE)
    portfolio_tickers = get_portfolio_tickers()
    scanner_close_map = get_scanner_close_map()
    scanner_buy_df = get_scanner_buy_now_candidates()

    lines: list[str] = []

    if df.empty or "status" not in df.columns:
        df = pd.DataFrame()
    else:
        df = df.copy()
        df["status"] = df["status"].astype(str).str.upper()
        df = exclude_tickers(df, portfolio_tickers)

        def get_action_group_for_row(row: pd.Series) -> str:
            return get_watchlist_display_decision(row, scanner_close_map)["action_group"]

        df["action_group"] = df.apply(get_action_group_for_row, axis=1)

    buy_df = df[df["action_group"] == "BUY NOW"] if not df.empty else pd.DataFrame()

    lines.append("🔥 ACTIE NU")
    if buy_df.empty and scanner_buy_df.empty:
        lines.append("Geen directe BUY NOW setups.")
    else:
        for _, row in buy_df.iterrows():
            append_compact_watchlist_decision(lines, row, scanner_close_map)
        for _, row in scanner_buy_df.iterrows():
            append_scanner_buy_now(lines, row)
    lines.append("")

    if df.empty:
        return lines

    limit_df = df[df["action_group"] == "SET LIMIT BUY"]
    if not limit_df.empty:
        lines.append("📌 SET LIMIT BUY")
        for _, row in limit_df.iterrows():
            append_compact_watchlist_decision(lines, row, scanner_close_map)
        lines.append("")

    stop_df = df[df["action_group"] == "SET STOP BUY"]
    if not stop_df.empty:
        lines.append("📌 SET STOP BUY")
        for _, row in stop_df.iterrows():
            append_compact_watchlist_decision(lines, row, scanner_close_map)
        lines.append("")

    remove_df = df[df["action_group"] == "REMOVE"]
    if not remove_df.empty:
        lines.append("❌ REMOVE")
        for _, row in remove_df.iterrows():
            ticker = clean_text(row.get("ticker"), fallback="?").upper()
            lines.append(f"- {ticker}")
        lines.append("")

    return lines


def build_scanner_context_section() -> list[str]:
    df = get_scanner_df()
    strong_extended_df = get_strong_extended_breakouts()

    portfolio_tickers = get_portfolio_tickers()
    watchlist_tickers = get_watchlist_tickers()

    scanner_buy_df = get_scanner_buy_now_candidates()
    scanner_buy_tickers: set[str] = set()
    if not scanner_buy_df.empty and "ticker" in scanner_buy_df.columns:
        scanner_buy_tickers = set(
            scanner_buy_df["ticker"].astype(str).str.upper().str.strip().tolist()
        )

    strong_extended_tickers: set[str] = set()
    if not strong_extended_df.empty and "ticker" in strong_extended_df.columns:
        strong_extended_tickers = set(
            strong_extended_df["ticker"].astype(str).str.upper().str.strip().tolist()
        )

    blocked_tickers = (
        portfolio_tickers
        | watchlist_tickers
        | scanner_buy_tickers
        | strong_extended_tickers
    )

    lines = ["🎯 NIEUWE IDEEËN"]

    if df.empty:
        return lines + ["- geen data", ""]

    if not strong_extended_df.empty:
        lines.append("Sterke breakouts, maar niet najagen")
        for _, row in strong_extended_df.iterrows():
            ticker = clean_text(row.get("ticker"), fallback="?").upper()
            ext = fmt_price(row.get("extension_atr"))
            strength = fmt_price(row.get("breakout_strength"))
            lines.append(
                f"- {ticker} → sterk momentum, maar wacht op betere instap "
                f"(ext {ext} ATR | strength {strength})"
            )
        lines.append("")

    df = exclude_tickers(df, blocked_tickers)

    if "setup_grade" not in df.columns:
        lines.append("- geen sterke setups")
        lines.append("")
        return lines

    df = df[df["setup_grade"] == "A"]

    if df.empty:
        if strong_extended_df.empty:
            lines.append("- geen sterke setups")
            lines.append("")
        return lines

    lines.append("Sterkste setups")

    for _, row in df.head(6).iterrows():
        ticker = clean_text(row.get("ticker"), fallback="?").upper()
        setup = title_case_setup(row.get("setup_type"))
        lines.append(f"- {ticker} — {setup}")

    lines.append("")
    return lines


def map_portfolio_decision(decision: str) -> dict:
    key = clean_text(decision, fallback="REVIEW").upper()
    mapping = {
        "SELL": {
            "action": "Volledig verkopen",
            "urgency": "high",
        },
        "TRIM": {
            "action": "Verkoop 25%",
            "urgency": "medium",
        },
        "REVIEW": {
            "action": "Nog niets doen",
            "urgency": "medium",
        },
        "HOLD": {
            "action": "Houden",
            "urgency": "low",
        },
    }
    return mapping.get(key, mapping["REVIEW"])


def explain_portfolio_reason(reason: str) -> str:
    text = clean_text(reason, fallback="-")
    key = text.lower().strip()

    direct_mapping = {
        "extended": "koers is sterk opgelopen tegenover de korte trend",
        "extended_above_ma20": "koers staat ver boven de korte trend",
        "above_ma20_and_ma50": "trend blijft gezond boven korte en middellange trend",
        "above_ma20_ma50": "trend blijft gezond boven korte en middellange trend",
        "below_ma50": "prijs is onder de middellange trend gezakt",
        "below_ma50_above_ma200": "middellange trend verzwakt, maar lange trend houdt nog stand",
        "below_ma50_and_ma200": "zowel middellange als lange trend zijn gebroken",
        "mixed_structure": "signalen spreken elkaar tegen",
        "missing_price_data": "prijsdata ontbreekt, beslissing is minder betrouwbaar",
        "missing_moving_average_data": "trenddata ontbreekt, beslissing is minder betrouwbaar",
    }

    if key in direct_mapping:
        return direct_mapping[key]

    normalized = normalize_reason(text)
    return normalized.capitalize()


def build_portfolio_section() -> list[str]:
    df = read_csv_safe(PORTFOLIO_REVIEW_FILE)

    lines = ["💼 PORTFOLIO"]

    if df.empty or "decision" not in df.columns:
        return lines + ["- geen data", ""]

    df = df.copy()
    df["decision"] = df["decision"].astype(str).str.upper()

    def fmt_line(row: pd.Series) -> str:
        ticker = clean_text(row.get("ticker"), fallback="?").upper()
        pnl = fmt_pct(row.get("pnl_pct"))
        decision = clean_text(row.get("decision"), fallback="REVIEW").upper()

        if decision == "TRIM":
            return f"- {ticker} → verkoop 25% | {pnl}"
        if decision == "SELL":
            return f"- {ticker} → verkopen | {pnl}"
        if decision == "HOLD":
            return f"- {ticker} → houden | {pnl}"
        if decision == "REVIEW":
            return f"- {ticker} → opvolgen | {pnl}"

        return f"- {ticker} → {decision} | {pnl}"

    order = ["SELL", "TRIM", "REVIEW", "HOLD"]

    for decision in order:
        subset = df[df["decision"] == decision]
        if subset.empty:
            continue

        lines.append(decision)

        for _, row in subset.iterrows():
            lines.append(fmt_line(row))

        lines.append("")

    return lines


def build_telegram_summary_text() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [f"📲 Trading Signals — {timestamp}"]
    lines.extend(build_market_regime_header())
    lines.extend(build_watchlist_action_sections())
    lines.extend(build_scanner_context_section())
    lines.extend(build_portfolio_section())

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