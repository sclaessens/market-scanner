# Market Engine Advice Batch Coverage Report

Run ID: `me-adv02-advice-batch-20260711T130000Z`
Generated at: `2026-07-11T13:00:00Z`
Input ticker status index: `artifacts/market_engine/batch_status/me-gh02-sample-status-index-20260711T120000Z/ticker_status_index.json`
Target universe: `not provided`
Target size: `500`

## Coverage

| Metric | Value |
|---|---:|
| Target tickers | 500 |
| Tickers in status index | 12 |
| Tickers with advice labels | 12 |
| Tickers missing artifact/status | 488 |
| Coverage percentage | 2.40% |

## Advice Distribution

| Advice | Count |
|---|---:|
| buy_candidate | 0 |
| wait_for_price | 0 |
| watchlist | 12 |
| avoid_for_now | 0 |
| hold_existing | 0 |
| take_loss_review | 0 |
| unable_to_advise | 0 |

## Interpretation

- What worked: 12 ticker(s) received deterministic advice labels from the available status index.
- What is still missing: 488 target ticker(s) lack status/advice coverage, and top missing inputs are portfolio_context: 12, setup_price_market_context: 12.
- Whether this is ready for evaluation: Only watchlist labels were produced; price/setup or portfolio context is needed before outcome tracking can evaluate advice quality.
- Recommended next sprint: ME-DATA01 - Close highest-impact advice data coverage gaps
