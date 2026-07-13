# Advice Outcome Refresh Report

Run ID: me-data05-idempotency-refresh-20260713T141000Z-evaluation-after
Previous evaluation: artifacts/market_engine/evaluation_runs/me-eval01-advice-outcomes-20260712T120000Z/advice_outcome_index.json
Price history root: data/processed
Generated at: 2026-07-12T12:00:00Z

## Summary

| Metric | Value |
|---|---:|
| Selected outcomes | 12 |
| Resolved | 0 |
| Still unresolved | 12 |
| Insufficient forward data | 12 |
| Missing price history | 0 |
| Other blockers | 0 |

## Missing Price History

None

## Refresh Outcomes

| Ticker | Advice | Previous status | Previous blocker | Snapshot | Last price date | New status | New blocker | Explanation |
|---|---|---|---|---|---|---|---|---|
| AMD | wait_for_price | unresolved | insufficient_forward_data | data/processed/AMD.csv | 2026-07-10 | unresolved | insufficient_forward_data | Local price history exists but still does not extend far enough beyond the advice date. |
| ASML | buy_candidate | unresolved | insufficient_forward_data | data/processed/ASML.csv | 2026-07-10 | unresolved | insufficient_forward_data | Local price history exists but still does not extend far enough beyond the advice date. |
| AVGO | wait_for_price | unresolved | insufficient_forward_data | data/processed/AVGO.csv | 2026-07-10 | unresolved | insufficient_forward_data | Local price history exists but still does not extend far enough beyond the advice date. |
| CLS | watchlist | unresolved | missing_price_history | data/processed/CLS.csv | 2026-07-10 | unresolved | insufficient_forward_data | Local price history exists but still does not extend far enough beyond the advice date. |
| COST | buy_candidate | unresolved | insufficient_forward_data | data/processed/COST.csv | 2026-07-10 | unresolved | insufficient_forward_data | Local price history exists but still does not extend far enough beyond the advice date. |
| CRDO | watchlist | unresolved | missing_price_history | data/processed/CRDO.csv | 2026-07-10 | unresolved | insufficient_forward_data | Local price history exists but still does not extend far enough beyond the advice date. |
| IREN | watchlist | unresolved | missing_price_history | data/processed/IREN.csv | 2026-07-10 | unresolved | insufficient_forward_data | Local price history exists but still does not extend far enough beyond the advice date. |
| META | avoid_for_now | unresolved | insufficient_forward_data | data/processed/META.csv | 2026-07-10 | unresolved | insufficient_forward_data | Local price history exists but still does not extend far enough beyond the advice date. |
| MSFT | watchlist | unresolved | insufficient_forward_data | data/processed/MSFT.csv | 2026-07-10 | unresolved | insufficient_forward_data | Local price history exists but still does not extend far enough beyond the advice date. |
| NVDA | buy_candidate | unresolved | insufficient_forward_data | data/processed/NVDA.csv | 2026-07-10 | unresolved | insufficient_forward_data | Local price history exists but still does not extend far enough beyond the advice date. |
| TSM | buy_candidate | unresolved | insufficient_forward_data | data/processed/TSM.csv | 2026-07-10 | unresolved | insufficient_forward_data | Local price history exists but still does not extend far enough beyond the advice date. |
| VRT | watchlist | unresolved | missing_price_history | data/processed/VRT.csv | 2026-07-10 | unresolved | insufficient_forward_data | Local price history exists but still does not extend far enough beyond the advice date. |
