# ME-FO02 — Implement Fundamental Observations from SEC CompanyFacts Source Context

## Status

COMPLETED BY ME-FO02

## Job family

ME-FO — Fundamental Observation jobs

## Purpose

ME-FO02 implements non-decision Fundamental Observations from approved SEC CompanyFacts Source Context.

The implementation consumes ME-SC02 Source Context objects and emits source-grounded observation output that preserves source values, missingness, source context state, and provenance.

ME-FO02 does not introduce derived calculations, analysis review, recommendations, portfolio review, delivery, Telegram behavior, or Decision Engine authority.

## Background

ME-FO01 defined the Fundamental Observation contract from SEC CompanyFacts Source Context.

ME-FO02 implements that contract in a job-scoped module under the Fundamental Observation job family.

The implementation depends on the approved Source Context output from ME-SC02 and does not consume raw SEC CompanyFacts payloads directly.

## Files added

Implementation:

* src/market_engine/fundamental_observations/**init**.py
* src/market_engine/fundamental_observations/sec_companyfacts_observations.py

Tests:

* tests/market_engine/fundamental_observations/test_sec_companyfacts_observations.py

Documentation and audit:

* docs/market_engine/fundamental_observations/me_fo02_fundamental_observations_implementation.md
* docs/market_engine/audits/me_fo02_fundamental_observations_implementation_audit.md

Backlog:

* docs/market_engine/backlog/market_engine_backlog.md

## Implemented output format

ME-FO02 introduces this output format version:

* sec-companyfacts-fundamental-observations-v1

The implementation emits a SecCompanyFactsFundamentalObservationSet containing:

* ticker;
* CIK;
* provider name;
* observation format version;
* source context format version;
* source context state;
* source context reference;
* source refresh snapshot metadata;
* observations;
* explicit non-decision boundary marker.

Each observation contains:

* ticker;
* CIK;
* provider name;
* category;
* state;
* message;
* source context state;
* canonical fields;
* source values;
* source references;
* missing source fields;
* observation format version;
* explicit non-decision boundary marker.

## Implemented observation categories

ME-FO02 implements the approved ME-FO01 categories:

* SOURCE_CONTEXT_AVAILABILITY;
* REVENUE_SOURCE_PRESENCE;
* NET_INCOME_SOURCE_VALUE;
* OPERATING_CASH_FLOW_SOURCE_VALUE;
* CAPEX_SOURCE_PRESENCE;
* CASH_GENERATION_SOURCE_COMPLETENESS;
* DATA_LIMITATION.

## Implemented observation states

ME-FO02 implements the approved ME-FO01 states:

* PRESENT;
* MISSING_DATA;
* POSITIVE_SOURCE_VALUE;
* NEGATIVE_SOURCE_VALUE;
* ZERO_SOURCE_VALUE;
* NOT_ASSESSED;
* SOURCE_LIMITED.

## Source Context handling

The implementation consumes SecCompanyFactsSourceContext objects.

For AVAILABLE Source Context:

* source-context availability is PRESENT;
* present field observations are emitted;
* positive, negative, and zero values are represented as source-value states;
* no DATA_LIMITATION observation is emitted.

For PARTIAL Source Context:

* present field observations are emitted;
* missing field observations are emitted as MISSING_DATA;
* cash-generation completeness is SOURCE_LIMITED when operating cash flow or capital expenditures is missing;
* DATA_LIMITATION is emitted.

For MISSING Source Context:

* source-context availability is NOT_ASSESSED;
* field observations are MISSING_DATA;
* DATA_LIMITATION is emitted.

The implementation also supports reserved Source Context states from ME-SC01 and ME-SC02 by emitting NOT_ASSESSED-style limitation behavior.

## Zero-value handling

Numeric zero remains present.

ME-FO02 does not convert zero to missing.

A zero source value produces ZERO_SOURCE_VALUE where the observation category is a source-value observation.

## Provenance handling

Each observation that uses a present source field preserves Source Context provenance.

Source references include, when available:

* selected SEC tag;
* provider name;
* taxonomy namespace;
* unit;
* fiscal year;
* fiscal period;
* filing form;
* filing date;
* period start date;
* period end date;
* accession number;
* frame;
* selection reason;
* fallback alias used.

The observation set also preserves source refresh metadata from Source Context:

* source refresh snapshot ID;
* source refresh fetched timestamp;
* source refresh payload format version.

## Persistence

ME-FO02 adds persistence for Fundamental Observation output.

Approved path shape:

* data/market_engine/fundamental_observations/<fundamental_observation_run_id>/<ticker>/fundamental_observations.json

The persistence function refuses to overwrite existing output.

## Explicit non-scope

ME-FO02 does not implement or authorize:

* raw SEC CompanyFacts fetching;
* cached raw snapshot loading as a primary input;
* Source Refresh behavior changes;
* Source Context behavior changes;
* derived calculations;
* free cash flow;
* growth;
* margins;
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

Derived calculations remain reserved for ME-DO.

Interpretive analysis remains reserved for ME-AR.

Recommendation review remains reserved for ME-RR.

Portfolio review remains reserved for ME-PR.

Delivery remains reserved for ME-DL.

## Tests

ME-FO02 adds tests proving:

* available Source Context produces approved observations;
* partial Source Context preserves missingness;
* missing Source Context produces NOT_ASSESSED and MISSING_DATA observations;
* positive, negative, zero, and missing source values are handled correctly;
* numeric zero remains present;
* source values and provenance are preserved;
* derived calculations are not emitted;
* recommendation, score, ranking, portfolio, delivery, Telegram, and Decision Engine authority are not emitted;
* persistence writes JSON to the approved path;
* persistence refuses overwrite;
* legacy runtime modules are not imported.

## Test results

Focused ME-FO02 tests:

* PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/fundamental_observations -q
* Result: 9 passed

Targeted Market Engine regression:

* PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/fundamental_observations tests/market_engine/source_context tests/market_engine/source_refresh tests/market_engine/source_intake tests/market_engine/fundamentals -q
* Result: 110 passed

Full tests/market_engine regression:

* PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine -q
* Result: 110 passed

## Governance status

ME-FO02 completes the initial Fundamental Observation implementation from SEC CompanyFacts Source Context.

Recommended next sprint:

* ME-DO01 — Add first derived cash-generation observation layer

ME-DO01 must remain non-decision and must not introduce recommendation, portfolio, delivery, Telegram, or Decision Engine authority.
