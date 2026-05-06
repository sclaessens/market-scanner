from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.settings import (
    DATA_DIR,
    REPORTS_DIR,
    SCANS_LOG_FILE,
    TOP_SETUPS_PER_SECTION,
)
from scripts.core.data_fetcher import fetch_ohlcv_data, load_tickers
from scripts.core.indicators import add_indicators
from scripts.core.regime import classify_market_regime
from scripts.core.scanner import rank_setups, scan_ticker
from scripts.core.build_validation_layer import build_validation_layer
from scripts.core.build_context_layer import build_context_layer
from scripts.portfolio.build_portfolio import build_portfolio
from scripts.portfolio.evaluate_positions import evaluate_positions
from scripts.reporting.build_telegram_summary import (
    build_telegram_summary_text,
    save_summary,
)
from scripts.reporting.send_telegram import send_daily_summary
from scripts.core.decision_engine import build_final_decisions


FAILED_TICKERS_FILE = DATA_DIR / "logs" / "failed_tickers.csv"
TELEGRAM_MESSAGE_FILE = REPORTS_DIR / "daily" / "telegram_message.txt"
MARKET_REGIME_FILE = DATA_DIR / "processed" / "market_regime.csv"
SCANNER_RANKED_FILE = DATA_DIR / "processed" / "scanner_ranked.csv"
MIN_HISTORY_ROWS = 220


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_TICKERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TELEGRAM_MESSAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCANS_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    MARKET_REGIME_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCANNER_RANKED_FILE.parent.mkdir(parents=True, exist_ok=True)


def prepare_scanner_output(setups: list[dict]) -> pd.DataFrame:
    scanner_df = pd.DataFrame(setups)

    if scanner_df.empty:
        raise ValueError("scanner output is empty; scanner_ranked.csv would be empty")

    if "ticker" not in scanner_df.columns:
        raise ValueError("scanner output is missing required column: ticker")

    if "date" not in scanner_df.columns:
        scanner_df["date"] = pd.Timestamp.today().strftime("%Y-%m-%d")

    scanner_df["ticker"] = scanner_df["ticker"].astype(str).str.strip().str.upper()
    scanner_df["date"] = pd.to_datetime(
        scanner_df["date"],
        errors="raise",
    ).dt.strftime("%Y-%m-%d")

    sort_columns = ["ticker", "date"]
    sort_ascending = [True, True]

    if "score_total" in scanner_df.columns:
        sort_columns.append("score_total")
        sort_ascending.append(False)

    if "rr" in scanner_df.columns:
        sort_columns.append("rr")
        sort_ascending.append(False)

    scanner_df = scanner_df.sort_values(
        by=sort_columns,
        ascending=sort_ascending,
    )

    duplicate_mask = scanner_df.duplicated(
        subset=["ticker", "date"],
        keep="first",
    )

    if duplicate_mask.any():
        debug_columns = [
            column
            for column in [
                "ticker",
                "date",
                "primary_setup",
                "setup_type",
                "score_total",
                "rr",
            ]
            if column in scanner_df.columns
        ]

        duplicate_rows = scanner_df.loc[duplicate_mask, debug_columns]

        print("Warning: duplicate scanner rows removed before validation:")
        print(duplicate_rows.to_string(index=False))

    scanner_df = scanner_df.drop_duplicates(
        subset=["ticker", "date"],
        keep="first",
    ).reset_index(drop=True)

    return scanner_df


def main() -> None:
    ensure_dirs()

    print("Starting market scan...")

    tickers = load_tickers()

    qqq_df = fetch_ohlcv_data("QQQ")
    spy_df = fetch_ohlcv_data("SPY")

    qqq_df = add_indicators(qqq_df)
    spy_df = add_indicators(spy_df)

    qqq_latest = qqq_df.iloc[-1]

    regime = classify_market_regime(
        qqq_close=float(qqq_latest["Close"]),
        ma50=float(qqq_latest["MA50"]),
        ma200=float(qqq_latest["MA200"]),
    )

    print(f"Market regime: {regime}")

    setups = []
    failed_rows = []

    for ticker in tickers:
        df = fetch_ohlcv_data(ticker)

        if df.empty or len(df) < MIN_HISTORY_ROWS:
            failed_rows.append({"ticker": ticker, "reason": "invalid_data"})
            continue

        df = add_indicators(df)

        result = scan_ticker(
            ticker=ticker,
            df=df,
            regime=regime,
            qqq_return_20d=0,
        )

        if result:
            setups.append(result)

    setups = rank_setups(setups, top_n=TOP_SETUPS_PER_SECTION)

    scanner_df = prepare_scanner_output(setups)

    print("Saving scanner ranked output...")
    scanner_df.to_csv(SCANNER_RANKED_FILE, index=False)

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(FAILED_TICKERS_FILE, index=False)

    print("Building validation layer...")
    build_validation_layer()

    print("Building context strength layer...")
    build_context_layer()

    print("Building portfolio...")
    build_portfolio()

    print("Evaluating positions...")
    evaluate_positions()

    print("Building final decisions...")
    build_final_decisions()

    print("Building telegram summary...")
    telegram_text = build_telegram_summary_text()
    save_summary(telegram_text)

    try:
        send_daily_summary()
        print("Telegram summary succesvol verstuurd.")
    except Exception as exc:
        print(f"Waarschuwing: Telegram verzending mislukt: {exc}")


if __name__ == "__main__":
    main()