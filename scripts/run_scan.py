from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (  # noqa: E402
    DATA_DIR,
    REPORTS_DIR,
    SCANS_LOG_FILE,
    TOP_SETUPS_PER_SECTION,
)
from scripts.core.data_fetcher import fetch_ohlcv_data, load_tickers  # noqa: E402
from scripts.core.indicators import add_indicators  # noqa: E402
from scripts.core.regime import classify_market_regime  # noqa: E402
from scripts.core.scanner import rank_setups, scan_ticker  # noqa: E402
from scripts.reporting.reporter import build_report  # noqa: E402


FAILED_TICKERS_FILE = DATA_DIR / "logs" / "failed_tickers.csv"
TELEGRAM_MESSAGE_FILE = REPORTS_DIR / "daily" / "telegram_message.txt"
MIN_HISTORY_ROWS = 220


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_TICKERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TELEGRAM_MESSAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCANS_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def validate_reference_index(ticker: str) -> pd.DataFrame:
    df = fetch_ohlcv_data(ticker)

    if df.empty:
        raise ValueError(f"{ticker} data is required but no valid data was returned")

    if len(df) < MIN_HISTORY_ROWS:
        raise ValueError(
            f"{ticker} data is incomplete: expected at least {MIN_HISTORY_ROWS} rows, got {len(df)}"
        )

    df = add_indicators(df)

    latest = df.iloc[-1]
    if pd.isna(latest.get("MA50")) or pd.isna(latest.get("MA200")):
        raise ValueError(f"{ticker} MA calculation is incomplete")

    return df


def append_scan_log(scan_date: str, regime: str, setups: list[dict]) -> None:
    if not setups:
        return

    rows: list[dict] = []
    for setup in setups:
        rows.append(
            {
                "scan_date": scan_date,
                "ticker": setup["ticker"],
                "regime": regime,
                "setup": setup["setup"],
                "primary_setup": setup.get("primary_setup", ""),
                "grade": setup.get("grade", ""),
                "score": setup["score"],
                "close": setup["close"],
                "entry": setup["entry"],
                "stop": setup["stop"],
                "target": setup["target"],
                "rr": setup["rr"],
            }
        )

    df_new = pd.DataFrame(rows)

    if SCANS_LOG_FILE.exists():
        df_old = pd.read_csv(SCANS_LOG_FILE)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        subset_cols = ["scan_date", "ticker", "setup", "entry"]

        existing_subset_cols = [col for col in subset_cols if col in df_all.columns]
        if existing_subset_cols:
            df_all = df_all.drop_duplicates(subset=existing_subset_cols, keep="last")
    else:
        df_all = df_new

    SCANS_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(SCANS_LOG_FILE, index=False)


def save_failed_tickers(rows: list[dict]) -> None:
    FAILED_TICKERS_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        pd.DataFrame(columns=["ticker", "reason"]).to_csv(FAILED_TICKERS_FILE, index=False)
        return

    pd.DataFrame(rows).to_csv(FAILED_TICKERS_FILE, index=False)


def _format_setup_line(setup: dict) -> str:
    grade = setup.get("grade", "C")
    primary_setup = setup.get("primary_setup", setup.get("setup", ""))
    return (
        f"{setup['ticker']} | {primary_setup} | grade {grade} | "
        f"score {setup['score']} | entry {setup['entry']} | stop {setup['stop']} | "
        f"target {setup['target']} | RR {setup['rr']}"
    )


def write_report(
    total_tickers: int,
    successful_tickers: int,
    failed_tickers: int,
    regime: str,
    qqq_df: pd.DataFrame,
    setups: list[dict],
) -> Path:
    report_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    report_path = REPORTS_DIR / "daily" / f"market_scan_{report_date}.md"

    latest_qqq = qqq_df.iloc[-1]
    qqq_regime_line = (
        f"QQQ close {latest_qqq['Close']:.2f} | "
        f"MA50 {latest_qqq['MA50']:.2f} | "
        f"MA200 {latest_qqq['MA200']:.2f} → {regime}"
    )

    vcp_lines = [_format_setup_line(s) for s in setups if s.get("primary_setup") == "VCP"]
    pullback_lines = [_format_setup_line(s) for s in setups if s.get("primary_setup") == "PULLBACK"]
    breakout_lines = [_format_setup_line(s) for s in setups if s.get("primary_setup") == "BREAKOUT"]

    weakening_lines = []
    if failed_tickers > 0:
        weakening_lines.append(f"Failed tickers during fetch/validation: {failed_tickers}")

    report_text = build_report(
        universe_size=total_tickers,
        liquid_universe_size=successful_tickers,
        regime=qqq_regime_line,
        vcp=vcp_lines,
        pullbacks=pullback_lines,
        breakouts=breakout_lines,
        weakening=weakening_lines,
    )

    if setups:
        a_setups = [s for s in setups if s.get("grade") == "A"]
        b_setups = [s for s in setups if s.get("grade") == "B"]

        extra_lines: list[str] = []
        extra_lines.append("")
        extra_lines.append("## Ranked setups")
        for setup in setups:
            extra_lines.append(f"- {_format_setup_line(setup)}")

        extra_lines.append("")
        extra_lines.append("## Grade summary")
        extra_lines.append(f"- A setups: {len(a_setups)}")
        extra_lines.append(f"- B setups: {len(b_setups)}")
        extra_lines.append(f"- Total ranked setups: {len(setups)}")

        report_text = report_text.rstrip() + "\n" + "\n".join(extra_lines) + "\n"

    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def write_telegram_message(
    report_date: str,
    regime: str,
    total_tickers: int,
    successful_tickers: int,
    failed_tickers: int,
    setups: list[dict],
) -> Path:
    lines: list[str] = []
    now = pd.Timestamp.now().strftime("%H:%M")

    if regime == "BULLISH":
        regime_icon = "🟢"
    elif regime == "BEARISH":
        regime_icon = "🔴"
    else:
        regime_icon = "🟡"

    a_count = len([s for s in setups if s.get("grade") == "A"])
    b_count = len([s for s in setups if s.get("grade") == "B"])

    lines.append(f"📊 Market Scan — {report_date} {now}")
    lines.append(f"Regime: {regime_icon} {regime}")
    lines.append(
        f"Universe: {successful_tickers}/{total_tickers} valid | Failed: {failed_tickers}"
    )

    if not setups:
        lines.append("Setups: none")
        if regime == "NEUTRAL":
            lines.append("Context: low conviction environment.")
        elif regime == "BEARISH":
            lines.append("Context: long setups filtered out.")
    else:
        lines.append(f"Ranked setups: {len(setups)} | A: {a_count} | B: {b_count}")
        for setup in setups:
            lines.append(
                f"{setup['ticker']} | {setup.get('primary_setup', setup['setup'])} | "
                f"{setup.get('grade', 'C')} | score {setup['score']} | "
                f"RS20 {setup.get('rs_20d_pct', 'n/a')}% | "
                f"entry {setup['entry']} | stop {setup['stop']} | "
                f"target {setup['target']} | RR {setup['rr']}"
            )

    TELEGRAM_MESSAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    TELEGRAM_MESSAGE_FILE.write_text("\n".join(lines), encoding="utf-8")
    return TELEGRAM_MESSAGE_FILE


def compute_return_20d(df: pd.DataFrame) -> float:
    if df.empty or len(df) < 21:
        return float("nan")

    latest_close = float(df["Close"].iloc[-1])
    ref_close = float(df["Close"].iloc[-21])

    if ref_close == 0:
        return float("nan")

    return (latest_close / ref_close) - 1.0


def main() -> None:
    ensure_dirs()

    print("Starting market scan...")

    tickers = load_tickers()

    qqq_df = validate_reference_index("QQQ")
    spy_df = validate_reference_index("SPY")

    qqq_return_20d = compute_return_20d(qqq_df)

    qqq_latest = qqq_df.iloc[-1]
    regime = classify_market_regime(
        qqq_close=float(qqq_latest["Close"]),
        ma50=float(qqq_latest["MA50"]),
        ma200=float(qqq_latest["MA200"]),
    )
    print(f"Market regime: {regime}")

    _ = spy_df

    setups: list[dict] = []
    failed_rows: list[dict] = []
    successful_count = 0
    seen_tickers: set[str] = set()

    for ticker in tickers:
        ticker = ticker.upper().strip()

        if ticker in {"QQQ", "SPY"}:
            continue

        if ticker in seen_tickers:
            continue
        seen_tickers.add(ticker)

        print(f"Scanning {ticker}...")

        df = fetch_ohlcv_data(ticker)

        if df.empty:
            failed_rows.append({"ticker": ticker, "reason": "no_data_returned"})
            continue

        if len(df) < MIN_HISTORY_ROWS:
            failed_rows.append(
                {"ticker": ticker, "reason": f"insufficient_history_{len(df)}"}
            )
            continue

        try:
            df_ind = add_indicators(df)
        except Exception as exc:
            failed_rows.append({"ticker": ticker, "reason": f"indicator_error: {exc}"})
            continue

        successful_count += 1

        result = scan_ticker(ticker, df_ind, regime, qqq_return_20d)
        if result is None:
            continue

        setups.append(result)

    setups = rank_setups(setups, top_n=TOP_SETUPS_PER_SECTION)

    save_failed_tickers(failed_rows)

    report_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    append_scan_log(report_date, regime, setups)

    report_path = write_report(
        total_tickers=len(tickers),
        successful_tickers=successful_count,
        failed_tickers=len(failed_rows),
        regime=regime,
        qqq_df=qqq_df,
        setups=setups,
    )

    telegram_path = write_telegram_message(
        report_date=report_date,
        regime=regime,
        total_tickers=len(tickers),
        successful_tickers=successful_count,
        failed_tickers=len(failed_rows),
        setups=setups,
    )

    print(f"Report written to: {report_path}")
    print(f"Telegram message written to: {telegram_path}")
    print(f"Failed tickers log written to: {FAILED_TICKERS_FILE}")
    print(f"Scan log updated: {SCANS_LOG_FILE}")


if __name__ == "__main__":
    main()
