from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR


SCANNER_FILE = DATA_DIR / "processed" / "scanner_ranked.csv"
WATCHLIST_FILE = DATA_DIR / "watchlist" / "watchlist_status.csv"
PORTFOLIO_FILE = DATA_DIR / "portfolio" / "portfolio_review.csv"
MARKET_REGIME_FILE = DATA_DIR / "processed" / "market_regime.csv"
OUTPUT_FILE = DATA_DIR / "processed" / "final_decisions.csv"


MIN_EXECUTION_RR = 2.0
MAX_BUY_EXTENSION_ATR = 1.5
MAX_WAIT_EXTENSION_ATR = 1.2


OUTPUT_COLUMNS = [
    "ticker",
    "source_layer",
    "setup_type",
    "status",
    "final_action",
    "reason",
    "entry",
    "stop",
    "target",
    "rr",
    "trigger_price",
    "urgency",
    "regime",
    "close",
    "ma20",
    "ma50",
    "high_20d",
]


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def clean_text(value, fallback: str = "") -> str:
    if value is None:
        return fallback
    if isinstance(value, float) and pd.isna(value):
        return fallback

    text = str(value).strip()
    return text if text else fallback


def safe_float(value) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def load_regime() -> str:
    df = read_csv_safe(MARKET_REGIME_FILE)

    if df.empty:
        return "UNKNOWN"

    last = df.iloc[-1]

    for col in ["regime", "Regime", "market_regime"]:
        if col in df.columns:
            value = clean_text(last.get(col), fallback="UNKNOWN").upper()
            return value if value else "UNKNOWN"

    return "UNKNOWN"


def normalize_ticker_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ticker" not in df.columns:
        return df

    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return df


def portfolio_decisions(regime: str) -> pd.DataFrame:
    df = normalize_ticker_column(read_csv_safe(PORTFOLIO_FILE))

    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    rows = []

    for _, row in df.iterrows():
        ticker = clean_text(row.get("ticker"), fallback="?").upper()
        decision = clean_text(row.get("decision"), fallback="REVIEW").upper()
        reason = clean_text(row.get("reason"), fallback="-")

        if decision not in {"HOLD", "TRIM", "SELL", "REVIEW", "ADD"}:
            final_action = "REVIEW"
        else:
            final_action = decision

        rows.append(
            {
                "ticker": ticker,
                "source_layer": "PORTFOLIO",
                "setup_type": "",
                "status": decision,
                "final_action": final_action,
                "reason": reason,
                "entry": None,
                "stop": None,
                "target": None,
                "rr": None,
                "trigger_price": None,
                "urgency": (
                    "high"
                    if final_action == "SELL"
                    else "medium"
                    if final_action in {"TRIM", "REVIEW"}
                    else "low"
                ),
                "regime": regime,
                "close": safe_float(row.get("last_price")),
                "ma20": safe_float(row.get("ma20")),
                "ma50": safe_float(row.get("ma50")),
                "high_20d": None,
            }
        )

    return pd.DataFrame(rows)


def resolve_watchlist_action(
    status: str,
    setup_type: str,
    action_hint: str,
    regime: str,
    rr: Optional[float],
    extension_atr: Optional[float],
) -> tuple[str, str]:
    """
    Performance-filter op watchlist-acties.

    Belangrijk:
    - BULLISH wordt op basis van validation geblokkeerd voor nieuwe entries.
    - VCP wordt voorlopig volledig geblokkeerd als koopsetup.
    - Alleen A-context zou koopbaar mogen zijn, maar watchlist_status bevat meestal geen grade.
      Daarom worden agressieve BUY-acties hier extra streng behandeld.
    """

    if status in {"REJECTED", "EXPIRED"}:
        return "REMOVE", "Setup is niet langer geldig."

    if setup_type == "VCP":
        return "WAIT", "VCP wordt voorlopig niet als koopsetup gebruikt op basis van validation."

    if regime == "BEARISH":
        return "WAIT", "Bearish regime: geen nieuwe long-entry."

    if regime == "BULLISH":
        return "WAIT", "Bullish regime presteert zwak in validation. Geen nieuwe entry."

    if rr is not None and rr < MIN_EXECUTION_RR:
        return "WAIT", "RR onder minimum. Geen koopactie."

    if extension_atr is not None:
        if extension_atr >= MAX_BUY_EXTENSION_ATR:
            return "WAIT", "Te ver extended tegenover MA20. Geen koopactie."
        if extension_atr >= MAX_WAIT_EXTENSION_ATR and action_hint == "BUY NOW":
            return "SET LIMIT BUY", "Koers is licht extended. Wacht op pullback richting MA20."

    if status == "MISSED":
        return "SET LIMIT BUY", "Setup is gemist. Niet najagen; wacht op pullback."

    if status == "READY":
        if setup_type == "BREAKOUT":
            if action_hint in {"BUY NOW", "SET STOP BUY"}:
                return action_hint, "Agressieve breakout-entry enkel toegestaan na watchlist-confirmatie."
            return "WAIT", "Breakout is nog niet concreet uitvoerbaar."

        if setup_type == "PULLBACK":
            if action_hint == "BUY NOW":
                return "BUY NOW", "Pullback is bevestigd in koopzone."
            if action_hint == "SET LIMIT BUY":
                return "SET LIMIT BUY", "Wacht op pullback richting MA20."
            return "WAIT", "Pullback is nog niet concreet uitvoerbaar."

    if action_hint == "SET LIMIT BUY":
        return "SET LIMIT BUY", "Wacht op betere instap via limit order."

    if action_hint == "SET STOP BUY" and setup_type == "BREAKOUT":
        return "SET STOP BUY", "Koop pas bij bevestiging boven breakout-trigger."

    return "WAIT", "Geen uitvoerbare watchlist-actie."


def watchlist_decisions(regime: str, blocked_tickers: set[str]) -> pd.DataFrame:
    df = normalize_ticker_column(read_csv_safe(WATCHLIST_FILE))

    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    if blocked_tickers:
        df = df[~df["ticker"].isin(blocked_tickers)].copy()

    rows = []

    for _, row in df.iterrows():
        ticker = clean_text(row.get("ticker"), fallback="?").upper()
        status = clean_text(row.get("status"), fallback="WAIT").upper()
        setup_type = clean_text(row.get("setup_type"), fallback="").upper()
        action_hint = clean_text(row.get("action_now"), fallback="WAIT").upper()

        rr = safe_float(row.get("rr"))
        extension_atr = safe_float(row.get("extension_atr"))

        original_reason = clean_text(
            row.get("why_now"),
            fallback=clean_text(
                row.get("reason_text"),
                fallback=clean_text(row.get("reason"), fallback="-"),
            ),
        )

        final_action, filter_reason = resolve_watchlist_action(
            status=status,
            setup_type=setup_type,
            action_hint=action_hint,
            regime=regime,
            rr=rr,
            extension_atr=extension_atr,
        )

        reason = filter_reason if filter_reason else original_reason

        rows.append(
            {
                "ticker": ticker,
                "source_layer": "WATCHLIST",
                "setup_type": setup_type,
                "status": status,
                "final_action": final_action,
                "reason": reason,
                "entry": None,
                "stop": None,
                "target": None,
                "rr": rr,
                "trigger_price": safe_float(row.get("trigger_price")),
                "urgency": clean_text(row.get("urgency"), fallback="low"),
                "regime": regime,
                "close": safe_float(row.get("close")),
                "ma20": safe_float(row.get("ma20")),
                "ma50": safe_float(row.get("ma50")),
                "high_20d": safe_float(row.get("high_20d")),
            }
        )

    return pd.DataFrame(rows)


def scanner_decisions(regime: str, blocked_tickers: set[str]) -> pd.DataFrame:
    df = normalize_ticker_column(read_csv_safe(SCANNER_FILE))

    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    if blocked_tickers:
        df = df[~df["ticker"].isin(blocked_tickers)].copy()

    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    numeric_cols = [
        "score_total",
        "score",
        "raw_score",
        "entry",
        "stop",
        "target",
        "rr",
        "close",
        "high_20d",
        "ma20",
        "ma50",
        "breakout_strength",
        "extension_atr",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    rows = []

    for _, row in df.iterrows():
        ticker = clean_text(row.get("ticker"), fallback="?").upper()
        setup_type = clean_text(
            row.get("primary_setup", row.get("setup_type")),
            fallback="",
        ).upper()
        grade = clean_text(
            row.get("setup_grade", row.get("grade")),
            fallback="",
        ).upper()

        breakout_strength = safe_float(row.get("breakout_strength"))
        extension_atr = safe_float(row.get("extension_atr"))
        rr = safe_float(row.get("rr"))

        final_action = "NONE"
        reason = "Scanneridee. Nog geen actie."

        if setup_type == "VCP":
            final_action = "NONE"
            reason = "VCP heeft 0% winrate in validation. Geen koopactie."

        elif grade != "A":
            final_action = "NONE"
            reason = "Niet A-grade. B/C setups worden niet uitgevoerd."

        elif regime == "BEARISH":
            final_action = "NONE"
            reason = "Bearish regime: geen nieuwe long-entry."

        elif regime == "BULLISH":
            final_action = "NONE"
            reason = "Bullish regime presteert zwak in validation. Geen scanner-entry."

        elif rr is None or rr < MIN_EXECUTION_RR:
            final_action = "NONE"
            reason = "RR onder minimum. Geen koopactie."

        elif extension_atr is not None and extension_atr >= MAX_BUY_EXTENSION_ATR:
            final_action = "NONE"
            reason = "Te ver extended tegenover MA20. Geen koopactie."

        elif setup_type == "BREAKOUT":
            if (
                regime == "NEUTRAL"
                and breakout_strength is not None
                and breakout_strength >= 3.0
                and (extension_atr is None or extension_atr <= MAX_BUY_EXTENSION_ATR)
            ):
                final_action = "BUY NOW"
                reason = "A-grade breakout in NEUTRAL regime met bewezen edge."
            else:
                final_action = "NONE"
                reason = "Breakout voldoet niet aan performance-filters."

        elif setup_type == "PULLBACK":
            if regime == "NEUTRAL":
                final_action = "NONE"
                reason = "A-grade pullback is watchlistmateriaal; timing gebeurt via watchlist."
            else:
                final_action = "NONE"
                reason = "Pullback buiten NEUTRAL regime wordt geblokkeerd."

        else:
            final_action = "NONE"
            reason = "Setup-type heeft geen bewezen execution-edge."

        rows.append(
            {
                "ticker": ticker,
                "source_layer": "SCANNER",
                "setup_type": setup_type,
                "status": grade,
                "final_action": final_action,
                "reason": reason,
                "entry": safe_float(row.get("entry")),
                "stop": safe_float(row.get("stop")),
                "target": safe_float(row.get("target")),
                "rr": rr,
                "trigger_price": safe_float(row.get("entry")),
                "urgency": "high" if final_action == "BUY NOW" else "low",
                "regime": regime,
                "close": safe_float(row.get("close")),
                "ma20": safe_float(row.get("ma20")),
                "ma50": safe_float(row.get("ma50")),
                "high_20d": safe_float(row.get("high_20d")),
            }
        )

    return pd.DataFrame(rows)


def build_final_decisions() -> pd.DataFrame:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    regime = load_regime()

    portfolio_df = portfolio_decisions(regime)
    portfolio_tickers = set(portfolio_df["ticker"].tolist()) if not portfolio_df.empty else set()

    watchlist_df = watchlist_decisions(regime, blocked_tickers=portfolio_tickers)
    watchlist_tickers = set(watchlist_df["ticker"].tolist()) if not watchlist_df.empty else set()

    blocked_for_scanner = portfolio_tickers | watchlist_tickers
    scanner_df = scanner_decisions(regime, blocked_tickers=blocked_for_scanner)

    final_df = pd.concat(
        [portfolio_df, watchlist_df, scanner_df],
        ignore_index=True,
    )

    if final_df.empty:
        final_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        for col in OUTPUT_COLUMNS:
            if col not in final_df.columns:
                final_df[col] = None

        final_df = final_df[OUTPUT_COLUMNS]

        action_order = {
            "SELL": 0,
            "TRIM": 1,
            "BUY NOW": 2,
            "SET STOP BUY": 3,
            "SET LIMIT BUY": 4,
            "ADD": 5,
            "REVIEW": 6,
            "HOLD": 7,
            "WAIT": 8,
            "REMOVE": 9,
            "NONE": 10,
        }

        final_df["action_order"] = final_df["final_action"].map(action_order).fillna(99)
        final_df = final_df.sort_values(["action_order", "ticker"]).drop(columns=["action_order"])
        final_df = final_df.reset_index(drop=True)

    final_df.to_csv(OUTPUT_FILE, index=False)
    return final_df


def main() -> None:
    df = build_final_decisions()

    print(f"Final decisions written to: {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")

    if not df.empty:
        print(df[["ticker", "source_layer", "final_action", "reason"]].to_string(index=False))


if __name__ == "__main__":
    main()