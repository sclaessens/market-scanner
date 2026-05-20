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
from scripts.core.build_fundamental_layer import build_fundamental_layer
from scripts.core.build_timing_state_layer import build_timing_state_layer
from scripts.core.build_portfolio_intelligence import build_portfolio_intelligence
from scripts.portfolio.build_portfolio import build_portfolio
from scripts.portfolio.evaluate_positions import evaluate_positions
from scripts.reporting.build_reporting_layer import (
    REPORTING_DASHBOARD_FILE,
    REPORTING_LOG_FILE,
    TELEGRAM_MESSAGE_FILE as REPORTING_TELEGRAM_MESSAGE_FILE,
    build_reporting_layer,
    write_outputs as write_reporting_outputs,
)
from scripts.reporting.send_telegram import send_daily_summary
from scripts.core.decision_engine import build_final_decisions


FAILED_TICKERS_FILE = DATA_DIR / "logs" / "failed_tickers.csv"
TELEGRAM_MESSAGE_FILE = REPORTS_DIR / "daily" / "telegram_message.txt"
MARKET_REGIME_FILE = DATA_DIR / "processed" / "market_regime.csv"
SCANNER_RANKED_FILE = DATA_DIR / "processed" / "scanner_ranked.csv"
VALIDATION_LAYER_FILE = DATA_DIR / "processed" / "validation_layer.csv"
CONTEXT_LAYER_FILE = DATA_DIR / "processed" / "context_strength.csv"
FUNDAMENTAL_QUALITY_FILE = DATA_DIR / "processed" / "fundamental_quality.csv"
TIMING_STATE_LAYER_FILE = DATA_DIR / "processed" / "timing_state_layer.csv"
PORTFOLIO_POSITIONS_FILE = DATA_DIR / "portfolio" / "portfolio_positions.csv"
PORTFOLIO_REVIEW_FILE = DATA_DIR / "portfolio" / "portfolio_review.csv"
PORTFOLIO_INTELLIGENCE_FILE = DATA_DIR / "processed" / "portfolio_intelligence.csv"
FINAL_DECISIONS_FILE = DATA_DIR / "processed" / "final_decisions.csv"
MIN_HISTORY_ROWS = 220
SCAN_PROGRESS_INTERVAL = 25


def format_artifact_message(path: Path, row_count: int | None = None) -> str:
    if row_count is None:
        return f"Artifact written: {path}"
    return f"Artifact written: {path} rows={row_count}"


def print_step_started(name: str) -> None:
    print(f"Pipeline step started: {name}")


def print_step_completed(name: str, row_count: int | None = None) -> None:
    if row_count is None:
        print(f"Pipeline step completed: {name}")
        return
    print(f"Pipeline step completed: {name} rows={row_count}")


def print_artifact_written(path: Path, row_count: int | None = None) -> None:
    print(format_artifact_message(path, row_count=row_count))


def print_scan_progress(
    processed_count: int,
    total_count: int,
    setup_count: int,
    failed_count: int,
) -> None:
    print(
        "Scanner progress: "
        f"processed={processed_count}/{total_count} "
        f"setup_rows_collected={setup_count} "
        f"failed_rows={failed_count}"
    )


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

    print("Pipeline run started: market scan")

    print_step_started("Load ticker universe")
    tickers = load_tickers()
    print_step_completed("Load ticker universe", row_count=len(tickers))

    print_step_started("Classify market regime")
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

    print(f"Market regime classified: {regime}")
    print_step_completed("Classify market regime")

    setups = []
    failed_rows = []

    print_step_started("Scan ticker universe")
    for index, ticker in enumerate(tickers, start=1):
        df = fetch_ohlcv_data(ticker)

        if df.empty or len(df) < MIN_HISTORY_ROWS:
            failed_rows.append({"ticker": ticker, "reason": "invalid_data"})
            if index % SCAN_PROGRESS_INTERVAL == 0 or index == len(tickers):
                print_scan_progress(index, len(tickers), len(setups), len(failed_rows))
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

        if index % SCAN_PROGRESS_INTERVAL == 0 or index == len(tickers):
            print_scan_progress(index, len(tickers), len(setups), len(failed_rows))

    print_step_completed("Scan ticker universe", row_count=len(setups))

    print_step_started("Prepare scanner output")
    setups = rank_setups(setups, top_n=TOP_SETUPS_PER_SECTION)

    scanner_df = prepare_scanner_output(setups)
    print_step_completed("Prepare scanner output", row_count=len(scanner_df))

    scanner_df.to_csv(SCANNER_RANKED_FILE, index=False)
    print_artifact_written(SCANNER_RANKED_FILE, row_count=len(scanner_df))

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(FAILED_TICKERS_FILE, index=False)
        print_artifact_written(FAILED_TICKERS_FILE, row_count=len(failed_rows))
    else:
        print("Skipped failed ticker artifact: failed_rows=0")

    print_step_started("Build validation layer")
    validation_df = build_validation_layer()
    print_artifact_written(VALIDATION_LAYER_FILE, row_count=len(validation_df))
    print_step_completed("Build validation layer", row_count=len(validation_df))

    print_step_started("Build context layer")
    context_df = build_context_layer()
    print_artifact_written(CONTEXT_LAYER_FILE, row_count=len(context_df))
    print_step_completed("Build context layer", row_count=len(context_df))

    print_step_started("Build fundamental layer")
    fundamental_df = build_fundamental_layer()
    print_artifact_written(FUNDAMENTAL_QUALITY_FILE, row_count=len(fundamental_df))
    print_step_completed("Build fundamental layer", row_count=len(fundamental_df))

    print_step_started("Build timing state layer")
    timing_df = build_timing_state_layer()
    print_artifact_written(TIMING_STATE_LAYER_FILE, row_count=len(timing_df))
    print_step_completed("Build timing state layer", row_count=len(timing_df))

    print_step_started("Build portfolio state")
    portfolio_df = build_portfolio()
    print_artifact_written(PORTFOLIO_POSITIONS_FILE, row_count=len(portfolio_df))
    print_step_completed("Build portfolio state", row_count=len(portfolio_df))

    print_step_started("Evaluate portfolio positions")
    portfolio_review_df = evaluate_positions()
    print_artifact_written(PORTFOLIO_REVIEW_FILE, row_count=len(portfolio_review_df))
    print_step_completed("Evaluate portfolio positions", row_count=len(portfolio_review_df))

    print_step_started("Build portfolio intelligence")
    portfolio_intelligence_df = build_portfolio_intelligence()
    print_artifact_written(PORTFOLIO_INTELLIGENCE_FILE, row_count=len(portfolio_intelligence_df))
    print_step_completed("Build portfolio intelligence", row_count=len(portfolio_intelligence_df))

    print_step_started("Build final decisions")
    final_decisions_df = build_final_decisions()
    print_artifact_written(FINAL_DECISIONS_FILE, row_count=len(final_decisions_df))
    print_step_completed("Build final decisions", row_count=len(final_decisions_df))

    print_step_started("Build reporting layer")
    reporting_dashboard_df, reporting_log_row, telegram_text = build_reporting_layer()
    write_reporting_outputs(reporting_dashboard_df, reporting_log_row, telegram_text)
    print_artifact_written(REPORTING_DASHBOARD_FILE, row_count=len(reporting_dashboard_df))
    print_artifact_written(REPORTING_LOG_FILE, row_count=1)
    print_artifact_written(REPORTING_TELEGRAM_MESSAGE_FILE)
    print_step_completed("Build reporting layer", row_count=len(reporting_dashboard_df))

    print_step_started("Send telegram summary")
    try:
        send_daily_summary()
        print_step_completed("Send telegram summary")
    except Exception as exc:
        print(f"Pipeline step failed: Send telegram summary")
        print(f"Failure context: {exc}")

    print("Pipeline run completed: market scan")


if __name__ == "__main__":
    main()
