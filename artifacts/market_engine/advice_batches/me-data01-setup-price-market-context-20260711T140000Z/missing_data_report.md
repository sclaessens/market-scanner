# Missing Data Report

## Top Missing Inputs For Buy Candidate

| Missing input | Count |
|---|---:|
| local_price_history | 4 |
| market_context | 8 |
| portfolio_context | 12 |
| price_level_context | 4 |
| setup_detection | 4 |
| setup_price_market_context | 4 |

## Missing Inputs By Ticker

| Ticker | Advice | Setup | Trend | Price position | Risk | Missing for buy candidate | Blockers |
|---|---|---|---|---|---|---|---|
| AMD | wait_for_price | breakout_candidate | uptrend | above_preferred_entry | normal | market_context, portfolio_context | Stage preserves an upstream blocked state., missing_setup_or_price_context, portfolio_review |
| ASML | buy_candidate | pullback_watch | uptrend | near_entry_zone | normal | market_context, portfolio_context | Stage preserves an upstream blocked state., missing_setup_or_price_context, portfolio_review |
| AVGO | wait_for_price | breakout_candidate | uptrend | above_preferred_entry | normal | market_context, portfolio_context | Stage preserves an upstream blocked state., missing_setup_or_price_context, portfolio_review |
| CLS | watchlist | unknown | unknown | unknown | unknown | local_price_history, portfolio_context, price_level_context, setup_detection, setup_price_market_context | Stage preserves an upstream blocked state., missing_setup_or_price_context, portfolio_review, local_price_history_not_found |
| COST | buy_candidate | pullback_watch | uptrend | near_entry_zone | normal | market_context, portfolio_context | Stage preserves an upstream blocked state., missing_setup_or_price_context, portfolio_review |
| CRDO | watchlist | unknown | unknown | unknown | unknown | local_price_history, portfolio_context, price_level_context, setup_detection, setup_price_market_context | Stage preserves an upstream blocked state., missing_setup_or_price_context, portfolio_review, local_price_history_not_found |
| IREN | watchlist | unknown | unknown | unknown | unknown | local_price_history, portfolio_context, price_level_context, setup_detection, setup_price_market_context | Stage preserves an upstream blocked state., missing_setup_or_price_context, portfolio_review, local_price_history_not_found |
| META | avoid_for_now | weak_setup | downtrend | below_support_or_breakdown | normal | market_context, portfolio_context | Stage preserves an upstream blocked state., missing_setup_or_price_context, portfolio_review |
| MSFT | watchlist | no_clear_setup | sideways | near_entry_zone | normal | market_context, portfolio_context | Stage preserves an upstream blocked state., missing_setup_or_price_context, portfolio_review |
| NVDA | buy_candidate | pullback_watch | uptrend | near_entry_zone | normal | market_context, portfolio_context | Stage preserves an upstream blocked state., missing_setup_or_price_context, portfolio_review |
| TSM | buy_candidate | pullback_watch | uptrend | fair_zone | normal | market_context, portfolio_context | Stage preserves an upstream blocked state., missing_setup_or_price_context, portfolio_review |
| VRT | watchlist | unknown | unknown | unknown | unknown | local_price_history, portfolio_context, price_level_context, setup_detection, setup_price_market_context | Stage preserves an upstream blocked state., missing_setup_or_price_context, portfolio_review, local_price_history_not_found |
