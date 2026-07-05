# ME-GV06 - Governor Explanation Backlog Entry

Sprint ID: ME-GV06
Status: COMPLETED BY ME-GV06
Job family: ME-GV / The Governor
Date: 2026-07-05

## Result

ME-GV06 implements deterministic explanation from approved recommendation,
price/setup, risk, data-confidence, invalidation, and position-context
evidence.

It provides conditional pullback, breakout, acceptable-zone, extension, and
invalidation context plus held-position hold/add/reduce/exit review context.
Every numeric level is copied from explicit approved evidence.

## Authority Boundary

Explanation states do not authorize execution, orders, stops, quantities,
allocation, portfolio/watchlist mutation, broker actions, or Decision Engine
decisions. All authority booleans remain false, and all aggregation fields
remain null.

## Next Backlog Item

```text
ME-DS01 - Define Dispatch Station output contract for Governor reports
```
