# Market Engine Advice Batch Coverage Report

Run ID: `me-data01-setup-price-market-context-20260711T140000Z`
Generated at: `2026-07-11T14:00:00Z`
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
| buy_candidate | 4 |
| wait_for_price | 2 |
| watchlist | 5 |
| avoid_for_now | 1 |
| hold_existing | 0 |
| take_loss_review | 0 |
| unable_to_advise | 0 |

## Setup/Price/Market Context

| Context status | Count |
|---|---:|
| available | 0 |
| partial | 8 |
| missing | 4 |
| invalid | 0 |

## Interpretation

- What worked: 12 ticker(s) received deterministic advice labels from the available status index.
- What is still missing: 488 target ticker(s) lack status/advice coverage, and top missing inputs are portfolio_context: 12, market_context: 8, local_price_history: 4.
- Whether this is ready for evaluation: At least one outcome-trackable advice label was produced.
- Recommended next sprint: ME-EVAL01 - Advice outcome tracking and feedback loop
