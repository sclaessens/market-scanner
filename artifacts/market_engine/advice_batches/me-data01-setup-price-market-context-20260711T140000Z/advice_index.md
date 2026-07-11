# Market Engine Advice Index

Run ID: `me-data01-setup-price-market-context-20260711T140000Z`
Generated at: `2026-07-11T14:00:00Z`
Input ticker status index: `artifacts/market_engine/batch_status/me-gh02-sample-status-index-20260711T120000Z/ticker_status_index.json`

## Summary

| Advice | Count |
|---|---:|
| buy_candidate | 4 |
| wait_for_price | 2 |
| watchlist | 5 |
| avoid_for_now | 1 |
| hold_existing | 0 |
| take_loss_review | 0 |
| unable_to_advise | 0 |

## Advice Table

| Ticker | Advice | Confidence | Readiness | Setup | Trend | Price position | Risk | Primary reason | Missing for buy candidate | Next action |
|---|---|---|---|---|---|---|---|---|---|---|
| AMD | wait_for_price | medium | partial | breakout_candidate | uptrend | above_preferred_entry | normal | Setup/price/market context is constructive, but price is above the preferred entry zone. | market_context, portfolio_context | Wait for price to return closer to the preferred entry zone. |
| ASML | buy_candidate | medium | partial | pullback_watch | uptrend | near_entry_zone | normal | Setup/price/market context is constructive and price is near a reasonable entry zone. | market_context, portfolio_context | Review manually as a buy candidate; no order is created. |
| AVGO | wait_for_price | medium | partial | breakout_candidate | uptrend | above_preferred_entry | normal | Setup/price/market context is constructive, but price is above the preferred entry zone. | market_context, portfolio_context | Wait for price to return closer to the preferred entry zone. |
| CLS | watchlist | low | partial | unknown | unknown | unknown | unknown | Valid non-stale artifact exists, but setup/price and portfolio context are missing. | local_price_history, portfolio_context, price_level_context, setup_detection, setup_price_market_context | Collect setup/price context before considering buy_candidate. |
| COST | buy_candidate | medium | partial | pullback_watch | uptrend | near_entry_zone | normal | Setup/price/market context is constructive and price is near a reasonable entry zone. | market_context, portfolio_context | Review manually as a buy candidate; no order is created. |
| CRDO | watchlist | low | partial | unknown | unknown | unknown | unknown | Valid non-stale artifact exists, but setup/price and portfolio context are missing. | local_price_history, portfolio_context, price_level_context, setup_detection, setup_price_market_context | Collect setup/price context before considering buy_candidate. |
| IREN | watchlist | low | partial | unknown | unknown | unknown | unknown | Valid non-stale artifact exists, but setup/price and portfolio context are missing. | local_price_history, portfolio_context, price_level_context, setup_detection, setup_price_market_context | Collect setup/price context before considering buy_candidate. |
| META | avoid_for_now | medium | partial | weak_setup | downtrend | below_support_or_breakdown | normal | Setup/price/market context indicates a weak, downtrend, breakdown, or elevated-risk setup. | market_context, portfolio_context | Avoid for now until setup, price position, or risk improves. |
| MSFT | watchlist | low | partial | no_clear_setup | sideways | near_entry_zone | normal | Setup/price/market context is partial or inconclusive. | market_context, portfolio_context | Keep on watchlist while improving setup/price/market evidence. |
| NVDA | buy_candidate | medium | partial | pullback_watch | uptrend | near_entry_zone | normal | Setup/price/market context is constructive and price is near a reasonable entry zone. | market_context, portfolio_context | Review manually as a buy candidate; no order is created. |
| TSM | buy_candidate | medium | partial | pullback_watch | uptrend | fair_zone | normal | Setup/price/market context is constructive and price is near a reasonable entry zone. | market_context, portfolio_context | Review manually as a buy candidate; no order is created. |
| VRT | watchlist | low | partial | unknown | unknown | unknown | unknown | Valid non-stale artifact exists, but setup/price and portfolio context are missing. | local_price_history, portfolio_context, price_level_context, setup_detection, setup_price_market_context | Collect setup/price context before considering buy_candidate. |
