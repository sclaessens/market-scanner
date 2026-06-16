# ME-DO01 — Add first derived cash-generation observation layer

## Status

COMPLETED BY ME-DO01

## Job family

ME-DO — Derived Observation jobs

## Purpose

ME-DO01 adds the first Derived Observation layer on top of ME-FO02 Fundamental Observations.

The sprint implements one source-grounded, non-decision derived calculation:

```text
free_cash_flow = operating_cash_flow - capital_expenditures
```

ME-DO01 does not introduce analysis review, recommendations, portfolio review, delivery, Telegram behavior, or Decision Engine authority.

## Background

ME-FO02 emits non-decision Fundamental Observations from approved SEC CompanyFacts Source Context.

ME-DO01 consumes the ME-FO02 Fundamental Observation output and computes the first derived cash-generation observation. It does not consume raw SEC payloads, cached raw snapshots, or Source Context directly.

## Files added

Implementation:

* src/market_engine/derived_observations/**init**.py
* src/market_engine/derived_observations/sec_companyfacts_cash_generation.py

Tests:

* tests/market_engine/derived_observations/test_sec_companyfacts_cash_generation.py

Documentation and audit:

* docs/market_engine/derived_observations/README.md
* docs/market_engine/derived_observations/me_do01_derived_cash_generation_observations.md
* docs/market_engine/audits/me_do01_derived_cash_generation_observations_audit.md

Backlog:

* docs/market_engine/backlog/market_engine_backlog.md

## Implemented output format

ME-DO01 introduces this output format version:

* sec-companyfacts-derived-cash-generation-observations-v1

The implementation emits a SecCompanyFactsDerivedCashGenerationObservationSet containing:

* ticker;
* CIK;
* provider name;
* derived observation format version;
* fundamental observation format version;
* source context format version;
* source context state;
* source refresh snapshot metadata;
* derived cash-generation observations;
* explicit non-decision boundary marker.

Each derived observation contains:

* ticker;
* CIK;
* provider name;
* category;
* state;
* message;
* formula;
* derived values;
* required source fields;
* missing source fields;
* source observation references;
* derived observation format version;
* explicit non-decision boundary marker.

## Implemented categories

ME-DO01 implements these categories:

* FREE_CASH_FLOW_DERIVATION;
* CASH_GENERATION_DERIVATION_LIMITATION.

## Implemented states

ME-DO01 implements these states:

* DERIVED_POSITIVE_SOURCE_VALUE;
* DERIVED_NEGATIVE_SOURCE_VALUE;
* DERIVED_ZERO_SOURCE_VALUE;
* MISSING_SOURCE_DATA;
* NOT_ASSESSED;
* SOURCE_LIMITED.

## Input contract

ME-DO01 consumes:

* SecCompanyFactsFundamentalObservationSet

The required upstream observations are:

* OPERATING_CASH_FLOW_SOURCE_VALUE;
* CAPEX_SOURCE_PRESENCE.

The required source fields are:

* operating_cash_flow;
* capital_expenditures.

## Derivation behavior

When both required source fields are present:

```text
free_cash_flow = operating_cash_flow - capital_expenditures
```

The result is emitted as:

* DERIVED_POSITIVE_SOURCE_VALUE when free cash flow is greater than zero;
* DERIVED_NEGATIVE_SOURCE_VALUE when free cash flow is less than zero;
* DERIVED_ZERO_SOURCE_VALUE when free cash flow equals zero.

When one or more required source fields are missing:

* free_cash_flow is emitted as null;
* state is MISSING_SOURCE_DATA;
* missing source fields are explicit;
* a CASH_GENERATION_DERIVATION_LIMITATION observation is emitted.

## Zero-value handling

Numeric zero remains present.

A zero operating cash flow source value is not treated as missing.

A zero derived free cash flow value is emitted as DERIVED_ZERO_SOURCE_VALUE.

## Provenance handling

ME-DO01 preserves references to the upstream Fundamental Observations used for derivation.

Source observation references include:

* upstream category;
* upstream state;
* canonical fields;
* source values;
* source references;
* missing source fields.

The derived observation set also preserves upstream metadata:

* fundamental observation format version;
* source context format version;
* source context state;
* source refresh snapshot ID;
* source refresh fetched timestamp;
* source refresh payload format version.

## Persistence

ME-DO01 adds persistence for Derived Cash Generation output.

Approved path shape:

* data/market_engine/derived_observations/cash_generation/<derived_observation_run_id>/<ticker>/derived_cash_generation_observations.json

The persistence function refuses to overwrite existing output.

## Explicit non-scope

ME-DO01 does not implement or authorize:

* raw SEC CompanyFacts fetching;
* cached raw snapshot loading as a primary input;
* Source Refresh behavior changes;
* Source Context behavior changes;
* Fundamental Observation behavior changes;
* FCF yield;
* margins;
* growth;
* ratios;
* valuation metrics;
* peer comparison;
* trend analysis;
* scoring;
* ranking;
* BUY / SELL / HOLD;
* recommendation review;
* portfolio review;
* delivery;
* Telegram;
* reporting;
* Decision Engine behavior;
* position sizing;
* execution advice.

Interpretive analysis remains reserved for ME-AR.

Recommendation review remains reserved for ME-RR.

Portfolio review remains reserved for ME-PR.

Delivery remains reserved for ME-DL.

## Tests

ME-DO01 adds tests proving:

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

## Test results

Focused ME-DO01 tests:

* PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/derived_observations -q
* Result: 10 passed

Targeted Market Engine regression:

* PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/derived_observations tests/market_engine/fundamental_observations tests/market_engine/source_context tests/market_engine/source_refresh tests/market_engine/source_intake tests/market_engine/fundamentals -q
* Result: 120 passed

Full tests/market_engine regression:

* PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine -q
* Result: 120 passed

## Governance status

ME-DO01 completes the first Derived Observation implementation.

Recommended next sprint:

* ME-DO02 or ME-AR01, depending on whether the next step should add another strictly derived metric or start an explicit analysis-review contract.

No analysis or recommendation authority is created by ME-DO01.
