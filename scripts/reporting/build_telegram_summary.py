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


def title_case_setup(setup_type: str) -> str:
    mapping = {
        "BREAKOUT": "Breakout",
        "PULLBACK": "Pullback",
        "VCP": "Rustige opbouw",
        "UNKNOWN": "Overig",
    }
    key = clean_text(setup_type, fallback="UNKNOWN").upper()
    return mapping.get(key, key.title())


def get_grouped_setup_order() -> list[str]:
    return ["BREAKOUT", "PULLBACK", "VCP"]


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
    """
    Alle tickers die momenteel als open portfolio-actie in portfolio_review.csv staan.
    Deze tickers krijgen prioriteit boven watchlist en scanner.
    """
    df = read_csv_safe(PORTFOLIO_REVIEW_FILE)
    if df.empty or "ticker" not in df.columns:
        return set()

    return set(
        df["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
        .tolist()
    )

def get_watchlist_tickers() -> set[str]:
    """
    Alle tickers die momenteel op de watchlist staan.
    Deze tickers krijgen prioriteit boven scanner.
    """
    df = read_csv_safe(WATCHLIST_FILE)
    if df.empty or "ticker" not in df.columns:
        return set()

    return set(
        df["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
        .tolist()
    )

def exclude_portfolio_tickers(df: pd.DataFrame, portfolio_tickers: set[str]) -> pd.DataFrame:
    """
    Verwijder tickers uit scanner/watchlist als ze al in portfolio zitten.
    """
    if df.empty or "ticker" not in df.columns or not portfolio_tickers:
        return df.copy()

    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    return out[~out["ticker"].isin(portfolio_tickers)].copy()


# =========================
# PORTFOLIO TRANSLATION LAYER
# =========================

def map_portfolio_decision(decision: str) -> dict:
    key = clean_text(decision, fallback="REVIEW").upper()
    mapping = {
        "SELL": {
            "header": "🔴 Volledig verkopen",
            "action": "Volledig verkopen",
            "urgency": "high",
            "summary": "trend is gebroken → positie sluiten",
        },
        "TRIM": {
            "header": "🟠 Gedeeltelijk winst nemen",
            "action": "Verkoop 25%",
            "urgency": "medium",
            "summary": "koers is hard opgelopen → deel van winst vastzetten",
        },
        "REVIEW": {
            "header": "🟡 Extra aandacht nodig",
            "action": "Nog niets doen",
            "urgency": "medium",
            "summary": "beeld is gemengd → positie extra opvolgen",
        },
        "HOLD": {
            "header": "🟢 Gewoon aanhouden",
            "action": "Houden",
            "urgency": "low",
            "summary": "trend is gezond → niets doen",
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
    normalized_lower = normalized.lower()

    if "extended" in normalized_lower and "ma20" in normalized_lower:
        return "koers staat ver boven de korte trend"
    if "above ma20" in normalized_lower and "ma50" in normalized_lower:
        return "trend blijft gezond boven korte en middellange trend"
    if "below ma50 and ma200" in normalized_lower:
        return "zowel middellange als lange trend zijn gebroken"
    if "below ma50 but above ma200" in normalized_lower or "onder ma50 maar boven ma200" in normalized_lower:
        return "middellange trend verzwakt, maar lange trend houdt nog stand"
    if "below ma50" in normalized_lower or "onder ma50" in normalized_lower:
        return "prijs is onder de middellange trend gezakt"
    if "missing" in normalized_lower or "ontbrekende" in normalized_lower:
        return normalized.capitalize()

    return normalized.capitalize()


# =========================
# WATCHLIST TRANSLATION LAYER
# =========================

def choose_watchlist_setup_label(row: pd.Series) -> str:
    explicit = clean_text(row.get("setup_label"), fallback="-")
    if explicit != "-":
        return explicit
    return title_case_setup(row.get("setup_type", "UNKNOWN"))


def explain_watchlist_reason(reason: str, regime: str) -> str:
    normalized = normalize_reason(reason)
    key = normalized.lower()
    regime_key = clean_text(regime, fallback="UNKNOWN").upper()

    mapping = {
        "nog onder ma20": "wacht tot de koers opnieuw boven de korte trend komt",
        "onder ma50": "trend is te zwak om nu in te stappen",
        "nog geen reclaim van ma20": "er is nog geen bevestiging van hernieuwde sterkte",
        "onder ma50 maar boven ma200": "middellange trend is zwak, dus voorlopig afwachten",
        "onder ma50 en ma200": "trend is duidelijk kapot",
        "gemengde structuur": "de signalen zijn niet overtuigend genoeg",
        "ontbrekende prijsdata": "er is te weinig data om dit goed te beoordelen",
        "ontbrekende moving average data": "trenddata ontbreekt voor een goede evaluatie",
        "too extended above ma20": "de koers is al te ver opgelopen voor een mooie instap",
        "trend ok but not in buy zone": "trend is oké, maar de instapzone is nog niet ideaal",
        "not close enough to breakout": "de koers zit nog niet dicht genoeg bij een overtuigende uitbraak",
        "vcp not ready": "de opbouw ziet er goed uit, maar er is nog geen kooptrigger",
    }
    if key in mapping:
        return mapping[key]

    if "neutral regime" in key or "neutraal regime" in key:
        if regime_key == "BULLISH":
            return "setup is nog niet mooi genoeg voor een instap"
        if regime_key == "BEARISH":
            return "markt geeft tegenwind, dus liever nog niet instappen"
        return "markt is nog niet overtuigend genoeg voor een instap"

    if "bearish regime" in key:
        return "markt geeft tegenwind, dus liever nog niet instappen"
    if "near breakout trigger" in key:
        return "de koers zit dicht bij een mogelijke uitbraak"
    if "near pivot" in key:
        return "de opbouw is goed en de koers zit dicht bij een kooptrigger"
    if "pullback near ma20" in key:
        return "de koers zit in een interessante terugvalzone"
    if "expired_after_" in key:
        return "de setup staat al te lang op de watchlist zonder trigger"

    return normalized.capitalize()


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


def build_watchlist_trigger_text(row: pd.Series) -> str:
    explicit_action = clean_text(row.get("action_now"), fallback="-")
    trigger_type = clean_text(row.get("trigger_type"), fallback="-").lower()
    trigger_price = fmt_price(row.get("trigger_price"))
    entry_plan = clean_text(row.get("entry_plan"), fallback="-")
    reason = clean_text(row.get("reason"), fallback="-").lower()
    close = fmt_price(row.get("close"))
    ma20 = fmt_price(row.get("ma20"))
    high_20d = fmt_price(row.get("high_20d"))
    status = clean_text(row.get("status"), fallback="WAIT").upper()

    explicit_trigger = clean_text(row.get("trigger_text"), fallback="-")
    if explicit_trigger != "-":
        return explicit_trigger

    if trigger_type == "buy_now":
        if trigger_price != "-":
            return f"nu koopbaar rond {trigger_price}"
        if close != "-":
            return f"nu koopbaar rond {close}"
        return "nu koopbaar"

    if trigger_type == "buy_above":
        if trigger_price != "-":
            return f"koop pas boven {trigger_price}"
        if high_20d != "-":
            return f"koop pas boven {high_20d}"
        return "koop pas bij een duidelijke uitbraak"

    if trigger_type == "limit_buy":
        if trigger_price != "-":
            return f"limietorder rond {trigger_price}"
        if ma20 != "-":
            return f"limietorder rond {ma20}"
        return "koop alleen op een betere terugval"

    if trigger_type == "none" and status in {"REJECTED", "EXPIRED"}:
        return "geen nieuwe instap zoeken"

    if explicit_action != "-":
        action_lower = explicit_action.lower()
        if "stop order" in action_lower and trigger_price != "-":
            return f"zet een stop order boven {trigger_price}"
        if "limietorder" in action_lower and trigger_price != "-":
            return f"leg een limietorder rond {trigger_price}"

    if "below_ma20" in reason or "still_below_ma20" in reason:
        if ma20 != "-":
            return f"koop pas boven {ma20}"
        return "koop pas boven de korte trend"
    if "breakout" in reason or "vcp" in reason:
        if high_20d != "-":
            return f"koop pas boven {high_20d}"
        return "koop pas bij een duidelijke uitbraak"
    if status == "READY":
        if close != "-":
            return f"koopcheck rond {close}"
        return "nu verder opvolgen voor instap"

    return "-"


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
    readable_map = {
        "BULLISH": "Markt is sterk",
        "NEUTRAL": "Markt is gemengd",
        "BEARISH": "Markt is zwak",
        "UNKNOWN": "Marktstatus onbekend",
    }
    emoji = emoji_map.get(regime, "⚪")
    readable = readable_map.get(regime, "Marktstatus onbekend")

    return [f"Regime: {emoji} {readable} ({regime})", ""]


# =========================
# WATCHLIST / DECISION LAYER
# =========================

def format_watchlist_line(row: pd.Series, regime: str) -> list[str]:
    ticker = clean_text(row.get("ticker", "?")).upper()
    status = clean_text(row.get("status", "WAIT")).upper()
    action_now = build_watchlist_action_from_row(row)
    trigger_text = build_watchlist_trigger_text(row)
    why_now = clean_text(row.get("why_now"), fallback="-")
    if why_now == "-":
        why_now = explain_watchlist_reason(row.get("reason"), regime)

    close = fmt_price(row.get("close"))
    ma20 = fmt_price(row.get("ma20"))
    ma50 = fmt_price(row.get("ma50"))

    details = []
    if close != "-":
        details.append(f"close {close}")
    if ma20 != "-":
        details.append(f"MA20 {ma20}")
    if ma50 != "-":
        details.append(f"MA50 {ma50}")

    lines = [f"- {ticker}"]
    lines.append(f"  → Actie nu: {action_now}")
    if trigger_text != "-" and status not in {"REJECTED", "EXPIRED"}:
        lines.append(f"  → Trigger: {trigger_text}")
    lines.append(f"  → Waarom: {why_now}")
    if details:
        lines.append(f"  → Context: {' | '.join(details)}")
    return lines


def append_watchlist_group(lines: list[str], title: str, subset: pd.DataFrame, regime: str) -> None:
    if subset.empty:
        return
    lines.append(title)
    for _, row in subset.iterrows():
        lines.extend(format_watchlist_line(row, regime))
    lines.append("")


def build_watchlist_action_sections() -> list[str]:
    df = read_csv_safe(WATCHLIST_FILE)
    portfolio_tickers = get_portfolio_tickers()

    regime_df = read_csv_safe(MARKET_REGIME_FILE)
    regime = "UNKNOWN"
    if not regime_df.empty and "regime" in regime_df.columns:
        regime = clean_text(regime_df.iloc[-1].get("regime"), fallback="UNKNOWN").upper()

    if df.empty or "status" not in df.columns:
        return [
            "✅ Geen READY setups op dit moment.",
            "",
            "👀 Geen watchlist data beschikbaar.",
            "",
        ]

    df = df.copy()
    df["status"] = df["status"].astype(str).str.upper()
    if "setup_type" in df.columns:
        df["setup_type"] = df["setup_type"].astype(str).str.upper()
    else:
        df["setup_type"] = "UNKNOWN"

    # Portfolio heeft prioriteit boven watchlist
    df = exclude_portfolio_tickers(df, portfolio_tickers)

    lines: list[str] = []

    ready_df = df[df["status"] == "READY"].copy()
    if ready_df.empty:
        lines.append("✅ Geen READY setups op dit moment.")
        lines.append("")
    else:
        lines.append(f"🚀 KLAAR VOOR KOOPCHECK ({len(ready_df)})")
        lines.append("")
        for setup_type in get_grouped_setup_order():
            subset = ready_df[ready_df["setup_type"] == setup_type].copy()
            append_watchlist_group(lines, title_case_setup(setup_type), subset, regime)
        remaining = ready_df[~ready_df["setup_type"].isin(get_grouped_setup_order())]
        append_watchlist_group(lines, "Overig", remaining, regime)

    wait_df = df[df["status"] == "WAIT"].copy()
    if wait_df.empty:
        lines.append("👀 Geen aandelen om verder op te volgen.")
        lines.append("")
    else:
        lines.append(f"👀 OPVOLGEN VOOR LATER ({len(wait_df)})")
        lines.append("")
        for setup_type in get_grouped_setup_order():
            subset = wait_df[wait_df["setup_type"] == setup_type].copy()
            append_watchlist_group(lines, title_case_setup(setup_type), subset, regime)
        remaining = wait_df[~wait_df["setup_type"].isin(get_grouped_setup_order())]
        append_watchlist_group(lines, "Overig", remaining, regime)

    removed_df = df[df["status"].isin(["REJECTED", "EXPIRED"])].copy()
    if not removed_df.empty:
        lines.append(f"❌ VAN WATCHLIST HALEN ({len(removed_df)})")
        lines.append("")
        for _, row in removed_df.iterrows():
            lines.extend(format_watchlist_line(row, regime))
        lines.append("")

    return lines


# =========================
# SCANNER CONTEXT
# =========================

def format_scanner_line(row: pd.Series) -> str:
    ticker = clean_text(row.get("ticker", "?")).upper()
    entry = fmt_price(row.get("entry"))
    stop = fmt_price(row.get("stop"))
    target = fmt_price(row.get("target"))
    rr = fmt_price(row.get("rr"))
    return f"- {ticker} | instap {entry} | stop {stop} | doel {target} | R:R {rr}"


def build_scanner_context_section() -> list[str]:
    df = read_csv_safe(SCANNER_FILE)
    portfolio_tickers = get_portfolio_tickers()
    watchlist_tickers = get_watchlist_tickers()
    blocked_tickers = portfolio_tickers | watchlist_tickers

    header = ["🎯 SCANNER IDEEËN"]

    if df.empty:
        return header + ["- geen scanner data", ""]

    df = df.copy()
    if "setup_grade" in df.columns:
        df["setup_grade"] = df["setup_grade"].astype(str).str.upper()
    if "setup_type" in df.columns:
        df["setup_type"] = df["setup_type"].astype(str).str.upper()
    else:
        df["setup_type"] = "UNKNOWN"
    if "score_total" in df.columns:
        df["score_total"] = pd.to_numeric(df["score_total"], errors="coerce")
        df = df.sort_values(by="score_total", ascending=False)

    # Prioriteit: Portfolio > Watchlist > Scanner
    df = exclude_portfolio_tickers(df, blocked_tickers)

    lines: list[str] = []
    grade_order = ["A", "B"]

    for grade in grade_order:
        subset_grade = df[df["setup_grade"] == grade].copy() if "setup_grade" in df.columns else pd.DataFrame()
        if subset_grade.empty:
            continue

        label = "Sterkste setups" if grade == "A" else "Interessante kansen voor later"
        lines.append(label)

        for setup_type in get_grouped_setup_order():
            subset = subset_grade[subset_grade["setup_type"] == setup_type].copy()
            if subset.empty:
                continue
            lines.append(title_case_setup(setup_type))
            for _, row in subset.head(5).iterrows():
                lines.append(format_scanner_line(row))
            lines.append("")

        remaining = subset_grade[~subset_grade["setup_type"].isin(get_grouped_setup_order())]
        if not remaining.empty:
            lines.append("Overig")
            for _, row in remaining.head(5).iterrows():
                lines.append(format_scanner_line(row))
            lines.append("")

    if len(lines) == 0:
        lines.append("- geen A/B setups")
        lines.append("")

    return header + lines


# =========================
# PORTFOLIO SECTION
# =========================

def format_portfolio_position(row: pd.Series) -> list[str]:
    decision = clean_text(row.get("decision", "REVIEW")).upper()
    decision_info = map_portfolio_decision(decision)

    ticker = clean_text(row.get("ticker", "?")).upper()
    qty = fmt_qty(row.get("quantity"))
    last_price = fmt_price(row.get("last_price"))
    pnl_pct = fmt_pct(row.get("pnl_pct"))
    why = explain_portfolio_reason(row.get("reason"))
    urgency = urgency_icon(decision_info["urgency"])
    risk_flag = clean_text(row.get("risk_flag", "-")).upper()

    lines = [f"- {ticker}"]
    lines.append(f"  → Actie: {decision_info['action']}")
    lines.append(f"  → Waarom: {why}")
    lines.append(f"  → Positie: {qty} stuks | koers {last_price} | winst/verlies {pnl_pct}")
    if risk_flag != "-":
        lines.append(f"  → Risico: {risk_flag} {urgency}")
    else:
        lines.append(f"  → Urgentie: {urgency}")
    return lines


def build_portfolio_section() -> list[str]:
    df = read_csv_safe(PORTFOLIO_REVIEW_FILE)
    header = ["💼 PORTFOLIO — ACTIES"]

    if df.empty or "decision" not in df.columns:
        return header + ["- geen portfolio review data", ""]

    df = df.copy()
    df["decision"] = df["decision"].astype(str).str.upper()

    lines: list[str] = []
    decision_order = ["SELL", "TRIM", "REVIEW", "HOLD"]

    for decision in decision_order:
        subset = df[df["decision"] == decision].copy()
        if subset.empty:
            continue

        info = map_portfolio_decision(decision)
        lines.append(f"{info['header']} ({len(subset)})")
        lines.append(f"({info['summary']})")
        lines.append("")

        for _, row in subset.iterrows():
            lines.extend(format_portfolio_position(row))
        lines.append("")

    if not lines:
        lines.append("- geen open portfoliobeslissingen")
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