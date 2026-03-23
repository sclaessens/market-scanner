import pandas as pd

from datetime import datetime
from pathlib import Path

from config.settings import INDEX_TICKERS, REPORTS_DIR
from src.validator import validate_inputs
from src.data_fetcher import load_tickers, fetch_ohlcv_data
from src.indicators import add_indicators
from src.regime import classify_market_regime
from src.reporter import build_report
from src.utils import save_text_file, save_csv_file
from src.scanner import detect_vcp, rank_setups


def main():
    print("Starting market scan...")

    # 1. Validate project structure and required inputs
    validate_inputs()

    # 2. Load user tickers
    tickers = load_tickers()
    if not tickers:
        raise ValueError("No tickers found in tickers.txt")

    # 3. Add index tickers needed for regime analysis
    combined_tickers = list(dict.fromkeys(tickers + INDEX_TICKERS))

    # 4. Fetch raw price data
    price_data = {}

    for ticker in combined_tickers:
        print(f"Fetching data for {ticker}...")
        df = fetch_ohlcv_data(ticker)

        if df is None or df.empty:
            print(f"Warning: no data returned for {ticker}")
            continue

        price_data[ticker] = df

    if "QQQ" not in price_data:
        raise ValueError("QQQ data is required to determine market regime")

    # 5. Enrich with indicators and save feature files
    feature_data = {}

    for ticker, df in price_data.items():
        enriched = add_indicators(df)
        feature_data[ticker] = enriched

        save_csv_file(enriched, Path("data/features") / f"{ticker}.csv")

    # 6. Build liquid universe count
    liquid_tickers = []

    for ticker in tickers:
        if ticker not in feature_data:
            continue

        df = feature_data[ticker]
        if df.empty:
            continue

        latest = df.iloc[-1]
        close = latest.get("Close")
        avg_vol = latest.get("AVG_VOL_20")

        if pd.notna(close) and pd.notna(avg_vol):
            if close >= 10 and avg_vol >= 1_000_000:
                liquid_tickers.append(ticker)

    # 7. Determine market regime from QQQ
    if "QQQ" not in feature_data or feature_data["QQQ"].empty:
        raise ValueError("QQQ data is missing after fetch/indicator processing")

    qqq_df = feature_data["QQQ"]
    latest_qqq = qqq_df.iloc[-1]

    if (
        pd.isna(latest_qqq["Close"])
        or pd.isna(latest_qqq["MA50"])
        or pd.isna(latest_qqq["MA200"])
    ):
        raise ValueError("QQQ regime data contains NaN values")

    qqq_close = float(latest_qqq["Close"])
    qqq_ma50 = float(latest_qqq["MA50"])
    qqq_ma200 = float(latest_qqq["MA200"])

    regime_label = classify_market_regime(
        qqq_close=qqq_close,
        ma50=qqq_ma50,
        ma200=qqq_ma200,
    )

    regime_text = (
        f"QQQ close {qqq_close:.2f} | "
        f"MA50 {qqq_ma50:.2f} | "
        f"MA200 {qqq_ma200:.2f} "
        f"→ {regime_label}"
    )

    # 8. Sprint 2: detect and rank VCP setups
    all_vcp_setups = []

    for ticker in tickers:
        if ticker not in feature_data:
            continue

        df = feature_data[ticker]
        vcp_setup = detect_vcp(ticker, df)

        if vcp_setup:
            all_vcp_setups.append(vcp_setup)

    ranked_vcp_setups = rank_setups(all_vcp_setups)
    vcp_setups = [item["summary"] for item in ranked_vcp_setups]

    # Placeholder sections for later Sprint 2 additions
    pullback_setups = []
    breakout_setups = []
    weakening_setups = []

    # 9. Build report
    report_text = build_report(
        universe_size=len(tickers),
        liquid_universe_size=len(liquid_tickers),
        regime=regime_text,
        vcp=vcp_setups,
        pullbacks=pullback_setups,
        breakouts=breakout_setups,
        weakening=weakening_setups,
    )

    # 10. Save report
    today = datetime.now().strftime("%Y-%m-%d")
    report_path = REPORTS_DIR / f"market_scan_{today}.md"
    save_text_file(report_text, report_path)

    print(f"Report saved to: {report_path}")
    print("Market scan completed successfully.")


if __name__ == "__main__":
    main()
