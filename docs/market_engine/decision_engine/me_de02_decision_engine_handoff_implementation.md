# ME-DE02 - Controlled Decision Engine handoff implementation

## Status

COMPLETED BY ME-DE02

## Sprint

ME-DE02 - Implement controlled Decision Engine handoff

## Job family

ME-DE - Decision Engine handoff jobs

## Purpose

ME-DE02 implements the ME-DE01 handoff contract as deterministic Python behavior.

The implementation converts approved `sec-companyfacts-portfolio-review-v1` Portfolio Review output into `market-engine-decision-engine-handoff-v1` readiness payloads.

The output is a handoff-readiness and provenance artifact only. It does not create Decision Engine decisions, trade instructions, allocation output, ranking, scoring, delivery, reporting, provider calls, broker calls, or portfolio mutation.

## Files changed

Runtime:

* `src/market_engine/decision_engine_handoff/__init__.py`
* `src/market_engine/decision_engine_handoff/sec_companyfacts_handoff.py`

Tests:

* `tests/market_engine/decision_engine_handoff/test_sec_companyfacts_handoff.py`

Documentation:

* `docs/market_engine/decision_engine/me_de02_decision_engine_handoff_implementation.md`
* `docs/market_engine/audits/me_de02_decision_engine_handoff_implementation_audit.md`
* `docs/market_engine/backlog/market_engine_backlog.md`
* `docs/market_engine/roadmap/market_engine_roadmap.md`

## Contract implemented

Implemented contract:

* `docs/market_engine/decision_engine/me_de01_decision_engine_handoff_contract.md`

Approved upstream input:

* `sec-companyfacts-portfolio-review-v1`

Output contract:

* `market-engine-decision-engine-handoff-v1`

## Implementation summary

ME-DE02 adds:

* `MarketEngineDecisionEngineHandoffReadinessState`
* `MarketEngineDecisionEngineHandoff`
* `build_market_engine_decision_engine_handoff(...)`

The builder accepts a Portfolio Review object or `None`.

It returns a deterministic handoff payload in all cases:

* eligible Portfolio Review input produces `ready_for_decision_engine_review`;
* missing Portfolio Review input produces `blocked_missing_portfolio_review`;
* invalid Portfolio Review contract produces `blocked_invalid_portfolio_review_contract`;
* unapproved or invalid Portfolio Review state produces `blocked_unapproved_portfolio_review`;
* missing portfolio context produces `blocked_missing_portfolio_context`;
* stale portfolio context produces `blocked_stale_portfolio_context`;
* incomplete provenance produces `blocked_incomplete_provenance`;
* ticker mismatch produces `blocked_ticker_mismatch`;
* missing handoff-readiness evidence produces `blocked_insufficient_evidence`.

## Eligibility behavior

A handoff is ready only when Portfolio Review:

* declares `sec-companyfacts-portfolio-review-v1`;
* has verified ticker identity;
* includes approved `market-engine-portfolio-context-v1` context;
* is not stale, missing, invalid, or blocked;
* preserves Recommendation Review provenance;
* preserves portfolio-context provenance;
* includes a `downstream_handoff_readiness_review` item with `ready_for_decision_engine_handoff_review`.

All blocked cases preserve deterministic blocked reasons.

## Provenance behavior

The handoff payload preserves:

* Portfolio Review reference;
* Portfolio Review item references;
* portfolio-context provenance;
* Recommendation Review provenance;
* Analysis Review references when available through Recommendation Review provenance;
* Setup Detection-aware provenance when available;
* source context and source refresh references when available;
* missing-data markers;
* stale-data markers.

The builder does not invent provenance.

## Missing, stale, and invalid data behavior

Missing Portfolio Review input produces a blocked payload.

Missing or unsupported portfolio context produces a blocked payload.

Stale portfolio-context evidence produces a blocked payload.

Invalid Portfolio Review state or category values produce a blocked payload.

Missing values remain explicit and are not coerced into valid evidence.

## Numeric-zero behavior

Numeric zero values are preserved as valid values.

Tests cover explicit zero quantity, zero market value, zero total value, zero exposure, and zero concentration threshold.

The handoff builder does not treat valid numeric zero as missing.

## Persistence behavior

ME-DE02 does not implement persistence.

The approved future persistence path remains:

```text
data/market_engine/decision_engine_handoffs/<run_id>/<ticker>/decision_engine_handoff.json
```

Any future persistence implementation must be explicitly scoped, write JSON only, refuse overwrite by default, and avoid old data/report/portfolio/watchlist paths.

## Tests

ME-DE02 tests cover:

* eligible Portfolio Review handoff;
* missing Portfolio Review input;
* wrong Portfolio Review contract;
* blocked Portfolio Review state;
* missing portfolio context;
* stale portfolio context;
* missing handoff-readiness evidence;
* invalid state values;
* ticker mismatch;
* numeric-zero preservation;
* provenance and lineage preservation;
* deterministic blocked reasons;
* output contract identity;
* absence of action-authority guidance text;
* no legacy `scripts` or old `market_scanner` imports.

## Authority boundaries

ME-DE02 does not introduce:

* provider calls;
* broker calls;
* live data calls;
* portfolio writes;
* watchlist writes;
* portfolio mutation;
* Telegram behavior;
* reporting or delivery behavior;
* Decision Engine runtime decisions;
* trade instructions;
* allocation;
* target weights;
* position sizing;
* order generation;
* ranking;
* scoring;
* urgency;
* conviction;
* tradeability;
* execution advice.

## Next sprint

Recommended next sprint:

```text
ME-DL01 - Define Delivery / Reporting contract
```
