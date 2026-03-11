import pandas as pd

from datetime import datetime
from pathlib import Path

from config.settings import (
    INDEX_TICKERS,
    REPORTS_DIR,
)
from src.validator import validate_inputs
from src.data_fetcher import load_tickers, fetch_ohlcv_data
from src.indicators import add_indicators
from src.regime import classify_market_regime
from src.reporter import build_report
from src.utils import save_text_file, save_csv_file


def main():
    print("Starting market scan...")

    # 1. Validatie van structuur en inputbestanden
    validate_inputs()

    # 2. Tickers laden
    tickers = load_tickers()
    if not tickers:
        raise ValueError("No tickers found in tickers.txt")

    # 3. Indexen toevoegen indien nog niet aanwezig
    combined_tickers = list(dict.fromkeys(tickers + INDEX_TICKERS))

    # 4. Data ophalen
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

    # 5. Indicatoren berekenen
    feature_data = {}
    for ticker, df in price_data.items():
        enriched = add_indicators(df)
        feature_data[ticker] = enriched

        # optioneel: features opslaan per ticker
        save_csv_file(enriched, Path("data/features") / f"{ticker}.csv")

    # 6. Marktregime bepalen op basis van QQQ
    if "QQQ" not in feature_data or feature_data["QQQ"].empty:
        raise ValueError("QQQ data is missing after fetch/indicator processing")

    qqq_df = feature_data["QQQ"]
    latest_qqq = qqq_df.iloc[-1]
    
    if pd.isna(latest_qqq["Close"]) or pd.isna(latest_qqq["MA50"]) or pd.isna(latest_qqq["MA200"]):
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

    # 7. Sprint 1: nog geen echte scanlogica, dus lege secties
    vcp_setups = []
    pullback_setups = []
    breakout_setups = []
    weakening_setups = []

    # 8. Rapport bouwen
    report_text = build_report(
        universe_size=len(tickers),
        liquid_universe_size=len(tickers),
        regime=regime_text,
        vcp=vcp_setups,
        pullbacks=pullback_setups,
        breakouts=breakout_setups,
        weakening=weakening_setups,
    )

    # 9. Rapport opslaan
    today = datetime.now().strftime("%Y-%m-%d")
    report_path = REPORTS_DIR / f"market_scan_{today}.md"
    save_text_file(report_text, report_path)

    print(f"Report saved to: {report_path}")
    print("Market scan completed successfully.")


if __name__ == "__main__":
    main()
