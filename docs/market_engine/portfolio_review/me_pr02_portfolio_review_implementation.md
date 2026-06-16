# ME-PR02 — Portfolio Review implementation

## Status

COMPLETED BY ME-PR02

## Sprint

ME-PR02 — Implement Portfolio Review

## Job Family

ME-PR — Portfolio Review jobs

## Purpose

ME-PR02 implements Portfolio Review as the downstream consumer of approved Recommendation Review output and explicitly supplied portfolio context.

Portfolio Review produces non-actionable portfolio-context review output suitable for later Decision Engine handoff contract work.

Portfolio Review does not execute, allocate, size positions, rank candidates, send Telegram messages, generate reports, mutate portfolio state, or create BUY / SELL / HOLD action semantics.

## Files Changed

Runtime:

* `src/market_engine/portfolio_review/__init__.py`
* `src/market_engine/portfolio_review/sec_companyfacts_portfolio_review.py`

Tests:

* `tests/market_engine/portfolio_review/test_sec_companyfacts_portfolio_review.py`

Documentation:

* `docs/market_engine/portfolio_review/me_pr02_portfolio_review_implementation.md`
* `docs/market_engine/audits/me_pr02_portfolio_review_implementation_audit.md`
* `docs/market_engine/backlog/market_engine_backlog.md`
* `docs/market_engine/roadmap/market_engine_roadmap.md`

## Contract Behavior Implemented

Approved Recommendation Review input contract:

* `sec-companyfacts-recommendation-review-v1`

Approved portfolio context input family:

* `market-engine-portfolio-context-v1`

Portfolio Review output contract:

* `sec-companyfacts-portfolio-review-v1`

The implementation consumes only in-memory objects supplied by the caller.

It does not read portfolio files, broker exports, watchlists, reports, generated data folders, or provider data.

## Portfolio Context Behavior

ME-PR02 introduces a structured `MarketEnginePortfolioContext` object for explicitly supplied portfolio context.

The context preserves:

* portfolio context format version;
* portfolio context run ID;
* portfolio snapshot timestamp;
* base currency;
* reviewed ticker;
* current position state;
* current quantity;
* current market value;
* total portfolio value;
* current ticker exposure percentage;
* exposure buckets when supplied;
* concentration thresholds when supplied;
* policy constraints when supplied;
* missing portfolio-context fields;
* stale portfolio-context fields;
* context provenance.

## Review Behavior

ME-PR02 emits review items for:

* position context;
* exposure context;
* concentration context;
* portfolio fit context;
* portfolio data limitations;
* downstream handoff readiness.

The output remains review-only.

## Missing, Stale, And Invalid Data

Missing portfolio context produces `blocked_by_missing_portfolio_context`.

Unsupported portfolio context contracts produce `blocked_by_invalid_input`.

Stale portfolio context produces `portfolio_context_stale`.

Partial portfolio context produces `portfolio_context_partial`.

Invalid portfolio context produces `portfolio_context_invalid` or `blocked_by_invalid_input`.

Missing numeric values are not converted to zero.

## Numeric-Zero Behavior

Numeric zero remains valid when explicitly supplied.

Examples:

* `current_quantity = 0`;
* `current_market_value = 0.0`;
* `portfolio_total_value = 0`;
* `current_ticker_exposure_pct = 0`.

Zero exposure can support `exposure_known` and `position_not_held` when explicitly supplied by approved portfolio context.

## Provenance Behavior

Portfolio Review preserves:

* Recommendation Review provenance;
* Recommendation Review item references;
* Setup Detection-aware provenance when present upstream;
* portfolio-context provenance;
* missing portfolio-context markers;
* stale portfolio-context markers.

## Persistence Behavior

ME-PR02 implements JSON persistence under:

```text
data/market_engine/portfolio_reviews/<portfolio_review_run_id>/<ticker>/portfolio_review.json
```

The helper refuses overwrite by default.

Tests use temporary directories only.

No production data writes are introduced.

## Backward Compatibility

ME-PR02 does not change Analysis Review or Recommendation Review contracts.

Recommendation Review remains the only approved upstream input to Portfolio Review.

## Authority Boundaries

ME-PR02 does not introduce:

* provider calls;
* broker calls;
* portfolio writes;
* watchlist writes;
* Telegram behavior;
* reporting delivery;
* Decision Engine behavior;
* BUY / SELL / HOLD action advice;
* allocation advice;
* target weights;
* position sizing advice;
* execution advice;
* ranking;
* scoring;
* conviction;
* urgency;
* tradeability decisions.

## Follow-Up Candidate

Possible future Portfolio Review follow-up candidate:

```text
ME-PR03 — Define approved portfolio context source and persistence contract
```

This candidate is not inserted ahead of ME-DE01 because ME-PR02 did not uncover a blocker.

## Next Sprint

Recommended next sprint:

```text
ME-DE01 — Define Decision Engine handoff contract
```
