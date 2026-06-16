# ME-DO01 — Derived Cash Generation Observations Audit

## Sprint

ME-DO01 — Add first derived cash-generation observation layer

## Job family

ME-DO — Derived Observation jobs

## Status

COMPLETED BY ME-DO01

## Branch

me-do01-derived-cash-generation-observation

## Scope audited

This audit covers the ME-DO01 implementation sprint.

## Files changed

Created:

* src/market_engine/derived_observations/**init**.py
* src/market_engine/derived_observations/sec_companyfacts_cash_generation.py
* tests/market_engine/derived_observations/test_sec_companyfacts_cash_generation.py
* docs/market_engine/derived_observations/README.md
* docs/market_engine/derived_observations/me_do01_derived_cash_generation_observations.md
* docs/market_engine/audits/me_do01_derived_cash_generation_observations_audit.md

Updated:

* docs/market_engine/backlog/market_engine_backlog.md

## Implementation audit

Python code changed: YES

Python code scope:

* src/market_engine/derived_observations/

Tests changed: YES

Test scope:

* tests/market_engine/derived_observations/

Documentation changed: YES

Data files changed: NO

Generated artifacts committed: NO

Runtime behavior outside ME-DO changed: NO

Provider calls introduced: NO

Live provider calls made: NO

## Boundary audit

ME-DO01 implements Derived Observations only.

It does not implement:

* raw SEC fetching;
* cached raw snapshot loading as a primary input;
* Source Refresh behavior changes;
* Source Context behavior changes;
* Fundamental Observation behavior changes;
* Analysis Review behavior;
* Recommendation Review behavior;
* Portfolio Review behavior;
* Delivery behavior;
* Telegram behavior;
* Decision Engine behavior;
* BUY / SELL / HOLD behavior;
* allocation;
* ranking;
* score;
* conviction;
* urgency;
* tradeability;
* position sizing;
* execution advice.

## Contract compliance

ME-DO01 adds one approved derived cash-generation calculation:

```text
free_cash_flow = operating_cash_flow - capital_expenditures
```

The implementation:

* consumes ME-FO02 Fundamental Observations;
* preserves upstream source observation references;
* preserves upstream source refresh metadata;
* preserves source context metadata;
* keeps missingness explicit;
* treats numeric zero as present;
* emits non-decision derived observation output;
* persists JSON under the approved Derived Observation path.

## Test coverage

ME-DO01 tests prove:

* positive free cash flow derivation;
* negative free cash flow derivation;
* zero free cash flow derivation;
* zero operating cash flow remains present;
* missing operating cash flow limits derivation;
* missing capital expenditures limits derivation;
* upstream source observation references are preserved;
* persistence writes JSON to the approved path;
* persistence refuses overwrite;
* analysis, recommendation, score, ranking, portfolio, delivery, Telegram, and Decision Engine authority are not emitted;
* legacy runtime modules are not imported.

## Tests run

Focused ME-DO01 tests:

* PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/derived_observations -q
* Result: 10 passed

Targeted Market Engine regression:

* PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/derived_observations tests/market_engine/fundamental_observations tests/market_engine/source_context tests/market_engine/source_refresh tests/market_engine/source_intake tests/market_engine/fundamentals -q
* Result: 120 passed

Full tests/market_engine regression:

* PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine -q
* Result: 120 passed

## Follow-up

Possible next sprint:

* ME-DO02 — Add another strictly derived non-decision metric

or:

* ME-AR01 — Define Analysis Review contract from Fundamental and Derived Observations

No recommendation, portfolio, delivery, Telegram, or Decision Engine authority should be introduced without an explicit future sprint.
