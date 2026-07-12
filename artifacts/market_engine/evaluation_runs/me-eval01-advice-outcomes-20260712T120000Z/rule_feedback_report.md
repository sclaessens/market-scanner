# Advice Rule Feedback Report

Run ID: me-eval01-advice-outcomes-20260712T120000Z
Input advice batch: artifacts/market_engine/advice_batches/me-data01-setup-price-market-context-20260711T140000Z/advice_index.json
Generated at: 2026-07-12T12:00:00Z

## Evaluation readiness

| Metric | Value |
|---|---:|
| Tickers evaluated | 12 |
| Resolved 1w outcomes | 0 |
| Resolved 1m outcomes | 0 |
| Resolved 3m outcomes | 0 |
| Unresolved outcomes | 12 |

## Label Feedback

### avoid_for_now

- Count: 1
- Resolved: 0
- Preliminary supportive: 0
- Preliminary adverse: 0
- Interpretation: weak_or_down
- Suggested rule feedback: Need future/local price history before rule quality can be judged.

### buy_candidate

- Count: 4
- Resolved: 0
- Preliminary supportive: 0
- Preliminary adverse: 0
- Interpretation: up_or_constructive
- Suggested rule feedback: Need future/local price history before rule quality can be judged.

### wait_for_price

- Count: 2
- Resolved: 0
- Preliminary supportive: 0
- Preliminary adverse: 0
- Interpretation: avoid_overpaying
- Suggested rule feedback: Need future/local price history before rule quality can be judged.

### watchlist

- Count: 5
- Resolved: 0
- Preliminary supportive: 0
- Preliminary adverse: 0
- Interpretation: neutral_observe
- Suggested rule feedback: Need future/local price history before rule quality can be judged.

## Limitations

- No live data was acquired.
- Outcomes are unresolved where local price history does not extend beyond advice date.
- This is not a full backtest.
