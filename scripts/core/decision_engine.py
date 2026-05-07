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
VALIDATION_FILE = DATA_DIR / "processed" / "validation_layer.csv"
CONTEXT_FILE = DATA_DIR / "processed" / "context_strength.csv"
WATCHLIST_FILE = DATA_DIR / "watchlist" / "watchlist_status.csv"
PORTFOLIO_FILE = DATA_DIR / "portfolio" / "portfolio_review.csv"
MARKET_REGIME_FILE = DATA_DIR / "processed" / "market_regime.csv"
OUTPUT_FILE = DATA_DIR / "processed" / "final_decisions.csv"

ACTION_BUY = "BUY"
ACTION_SELL = "SELL"
ACTION_HOLD = "HOLD"
ACTION_WAIT = "WAIT"
ACTION_TRIM = "TRIM"
ACTION_REMOVE = "REMOVE"
ACTION_REVIEW = "REVIEW"
ACTION_NO_ACTION = "NO_ACTION"
ACTION_PREPARE = "PREPARE"

OUTPUT_COLUMNS = [
    "ticker", "date", "source_layer", "setup_type", "final_action", "tradeability",
    "conviction", "allocation_priority", "validation_state", "context_strength",
    "leadership_state", "timing_state", "portfolio_state", "execution_style",
    "decision_reason", "entry", "stop", "target", "rr", "trigger_price", "regime",
    "close", "ma20", "ma50", "high_20d",
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
            return clean_text(last.get(col), fallback="UNKNOWN").upper()
    return "UNKNOWN"


def normalize_ticker_date(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ticker" not in df.columns:
        return df
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def _latest_date(*frames: pd.DataFrame) -> str:
    dates: list[str] = []
    for frame in frames:
        if not frame.empty and "date" in frame.columns:
            dates.extend(frame["date"].dropna().astype(str).tolist())
    return max(dates) if dates else pd.Timestamp.today().strftime("%Y-%m-%d")


def _base_row(ticker: str, date: str, regime: str) -> dict:
    return {
        "ticker": ticker, "date": date, "source_layer": "", "setup_type": "",
        "final_action": ACTION_NO_ACTION, "tradeability": "NOT_ASSESSED",
        "conviction": "LOW", "allocation_priority": 0, "validation_state": "UNKNOWN",
        "context_strength": "UNKNOWN", "leadership_state": "UNKNOWN", "timing_state": "UNKNOWN",
        "portfolio_state": "NONE", "execution_style": "NONE", "decision_reason": "no_decision_inputs",
        "entry": None, "stop": None, "target": None, "rr": None, "trigger_price": None,
        "regime": regime, "close": None, "ma20": None, "ma50": None, "high_20d": None,
    }


def _portfolio_action(risk_state: str) -> tuple[str, str, str, int, str]:
    if risk_state == "STRUCTURE_BROKEN":
        return ACTION_SELL, "TRADEABLE", "HIGH", 90, "portfolio_structure_broken"
    if risk_state == "EXTENDED_PROFIT":
        return ACTION_TRIM, "TRADEABLE", "MEDIUM", 70, "portfolio_extended_profit"
    if risk_state in {"STRUCTURE_WEAKENING", "DATA_GAP"}:
        return ACTION_REVIEW, "REVIEW_REQUIRED", "MEDIUM", 60, "portfolio_requires_review"
    return ACTION_HOLD, "HELD", "MEDIUM", 40, "portfolio_state_normal"


def portfolio_rows(portfolio_df: pd.DataFrame, regime: str, date: str) -> list[dict]:
    rows: list[dict] = []
    if portfolio_df.empty or "ticker" not in portfolio_df.columns:
        return rows
    for _, row in portfolio_df.iterrows():
        ticker = clean_text(row.get("ticker"), fallback="?").upper()
        risk_state = clean_text(row.get("risk_state"), fallback="UNKNOWN").upper()
        final_action, tradeability, conviction, priority, reason = _portfolio_action(risk_state)
        out = _base_row(ticker, date, regime)
        out.update({
            "source_layer": "PORTFOLIO", "final_action": final_action,
            "tradeability": tradeability, "conviction": conviction,
            "allocation_priority": priority, "portfolio_state": risk_state,
            "decision_reason": reason, "close": safe_float(row.get("last_price")),
            "ma20": safe_float(row.get("ma20")), "ma50": safe_float(row.get("ma50")),
        })
        rows.append(out)
    return rows


def _timing_state_from_watchlist(row: pd.Series) -> str:
    for column in ["timing_state", "status", "state"]:
        value = clean_text(row.get(column), fallback="").upper()
        if value:
            return value
    return "UNKNOWN"


def _scanner_timing_state(row: pd.Series) -> str:
    extension = safe_float(row.get("extension_atr"))
    setup_type = clean_text(row.get("primary_setup"), fallback="").upper()
    if extension is not None and extension >= 2.0:
        return "EXTENDED"
    if setup_type == "BREAKOUT":
        return "BREAKOUT_PENDING"
    if setup_type == "PULLBACK":
        return "PULLBACK"
    return "EARLY"


def _conviction_from_context(context_strength: str, validation_state: str, timing_state: str, regime: str) -> tuple[str, int]:
    score = 0
    if validation_state == "COHERENT":
        score += 2
    if context_strength == "LEADING":
        score += 3
    elif context_strength == "STRONG":
        score += 2
    elif context_strength == "NEUTRAL":
        score += 1
    if timing_state in {"READY", "CONFIRMED", "PULLBACK", "BREAKOUT_PENDING"}:
        score += 1
    if regime == "BULLISH":
        score += 1
    elif regime == "BEARISH":
        score -= 1
    if score >= 6:
        return "VERY_HIGH", 90
    if score >= 5:
        return "HIGH", 75
    if score >= 3:
        return "MEDIUM", 50
    if score >= 1:
        return "LOW", 25
    return "VERY_LOW", 10


def _allocation_action(validation_state: str, context_strength: str, timing_state: str, portfolio_block: bool, regime: str) -> tuple[str, str, str]:
    if portfolio_block:
        return ACTION_WAIT, "NOT_TRADEABLE", "existing_portfolio_position_controls_allocation"
    if validation_state != "COHERENT":
        return ACTION_REVIEW, "NOT_TRADEABLE", "structure_not_coherent"
    if context_strength in {"LEADING", "STRONG"} and timing_state in {"READY", "CONFIRMED", "PULLBACK"} and regime != "BEARISH":
        return ACTION_BUY, "TRADEABLE", "capital_allocation_ready"
    if timing_state in {"BREAKOUT_PENDING", "EARLY"}:
        return ACTION_PREPARE, "WATCH", "opportunity_recognized_timing_pending"
    if timing_state == "STALE":
        return ACTION_REMOVE, "NOT_TRADEABLE", "opportunity_no_longer_relevant"
    return ACTION_WAIT, "WATCH", "classification_recognized_allocation_not_ready"


def opportunity_rows(scanner_df: pd.DataFrame, validation_df: pd.DataFrame, context_df: pd.DataFrame, watchlist_df: pd.DataFrame, portfolio_tickers: set[str], regime: str, date: str) -> list[dict]:
    rows: list[dict] = []
    if scanner_df.empty and watchlist_df.empty:
        return rows

    scanner = normalize_ticker_date(scanner_df)
    validation = normalize_ticker_date(validation_df)
    context = normalize_ticker_date(context_df)
    watchlist = normalize_ticker_date(watchlist_df)

    if not scanner.empty:
        base = scanner.copy()
    else:
        base = watchlist[["ticker"]].drop_duplicates().copy()
        base["date"] = date

    if not validation.empty:
        validation_cols = [
            column
            for column in ["ticker", "date", "structure_state", "structure_reason", "validation_reason"]
            if column in validation.columns
        ]
        base = base.merge(validation[validation_cols], on=["ticker", "date"], how="left")
    if not context.empty:
        base = base.merge(context[["ticker", "date", "context_strength", "leadership_state"]], on=["ticker", "date"], how="left")
    if not watchlist.empty:
        watch_cols = [column for column in ["ticker", "status", "timing_state", "trigger_price"] if column in watchlist.columns]
        base = base.merge(watchlist[watch_cols].drop_duplicates(subset=["ticker"]), on="ticker", how="left")

    for _, row in base.iterrows():
        ticker = clean_text(row.get("ticker"), fallback="?").upper()
        validation_state = clean_text(row.get("structure_state"), fallback="UNKNOWN").upper()
        structure_reason = clean_text(row.get("structure_reason", row.get("validation_reason")), fallback="").lower()
        context_strength = clean_text(row.get("context_strength"), fallback="UNKNOWN").upper()
        leadership_state = clean_text(row.get("leadership_state"), fallback=context_strength).upper()
        timing_state = _timing_state_from_watchlist(row) if "status" in row.index or "timing_state" in row.index else _scanner_timing_state(row)
        conviction, priority = _conviction_from_context(context_strength, validation_state, timing_state, regime)
        final_action, tradeability, reason = _allocation_action(validation_state, context_strength, timing_state, ticker in portfolio_tickers, regime)
        out = _base_row(ticker, date, regime)
        out.update({
            "source_layer": "WATCHLIST" if timing_state not in {"UNKNOWN", "EARLY"} else "SCANNER",
            "setup_type": clean_text(row.get("primary_setup", row.get("setup_type")), fallback="").upper(),
            "final_action": final_action, "tradeability": tradeability, "conviction": conviction,
            "allocation_priority": priority, "validation_state": validation_state,
            "context_strength": context_strength, "leadership_state": leadership_state,
            "timing_state": timing_state, "portfolio_state": "EXISTING" if ticker in portfolio_tickers else "NONE",
            "execution_style": "AGGRESSIVE" if final_action == ACTION_BUY and conviction in {"VERY_HIGH", "HIGH"} else "PASSIVE",
            "decision_reason": reason if not structure_reason else f"{reason}:{structure_reason}", "entry": safe_float(row.get("entry")),
            "stop": safe_float(row.get("stop")), "target": safe_float(row.get("target")),
            "rr": safe_float(row.get("rr")), "trigger_price": safe_float(row.get("trigger_price", row.get("entry"))),
            "close": safe_float(row.get("close")), "ma20": safe_float(row.get("ma20")),
            "ma50": safe_float(row.get("ma50")), "high_20d": safe_float(row.get("high_20d")),
        })
        rows.append(out)
    return rows


def build_final_decisions() -> pd.DataFrame:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    regime = load_regime()
    scanner_df = read_csv_safe(SCANNER_FILE)
    validation_df = read_csv_safe(VALIDATION_FILE)
    context_df = read_csv_safe(CONTEXT_FILE)
    watchlist_df = read_csv_safe(WATCHLIST_FILE)
    portfolio_df = read_csv_safe(PORTFOLIO_FILE)
    date = _latest_date(scanner_df, validation_df, context_df)

    portfolio_df = normalize_ticker_date(portfolio_df)
    portfolio_tickers = set(portfolio_df["ticker"].tolist()) if not portfolio_df.empty and "ticker" in portfolio_df.columns else set()

    rows = []
    rows.extend(portfolio_rows(portfolio_df, regime, date))
    rows.extend(opportunity_rows(scanner_df, validation_df, context_df, watchlist_df, portfolio_tickers, regime, date))

    final_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if final_df.empty:
        final_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        final_df = final_df.sort_values(["allocation_priority", "ticker"], ascending=[False, True]).drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
        final_df = final_df[OUTPUT_COLUMNS]
    final_df.to_csv(OUTPUT_FILE, index=False)
    return final_df


def main() -> None:
    df = build_final_decisions()
    print(f"Final decisions written to: {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")
    if not df.empty:
        print(df[["ticker", "source_layer", "final_action", "decision_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
