# ME-AR02 — Implement Analysis Review from Fundamental and Derived Observations

## Status

COMPLETED BY ME-AR02

## Job family

ME-AR — Analysis Review jobs

## Purpose

ME-AR02 implements the Analysis Review contract defined by ME-AR01.

The implementation consumes approved upstream observation sets:

* ME-FO02 Fundamental Observations;
* ME-DO01 Derived Cash Generation Observations.

It emits non-recommendation Analysis Review output.

ME-AR02 does not introduce Recommendation Review, Portfolio Review, Delivery, Telegram behavior, reporting, or Decision Engine authority.

## Files added

Implementation:

* src/market_engine/analysis_review/**init**.py
* src/market_engine/analysis_review/sec_companyfacts_analysis_review.py

Tests:

* tests/market_engine/analysis_review/test_sec_companyfacts_analysis_review.py

Documentation and audit:

* docs/market_engine/analysis_review/me_ar02_analysis_review_implementation.md
* docs/market_engine/audits/me_ar02_analysis_review_implementation_audit.md

Backlog:

* docs/market_engine/backlog/market_engine_backlog.md

## Implemented output format

ME-AR02 introduces this output format version:

* sec-companyfacts-analysis-review-v1

The implementation emits a `SecCompanyFactsAnalysisReview` object containing:

* ticker;
* CIK;
* provider name;
* analysis review format version;
* fundamental observation format version;
* derived observation format version;
* source context format version;
* source context state;
* source refresh snapshot metadata;
* review items;
* explicit non-recommendation boundary marker.

Each review item contains:

* category;
* state;
* message;
* input observation families;
* required observations;
* missing observations;
* source observation references;
* derived observation references;
* explicit non-recommendation boundary marker.

## Implemented categories

ME-AR02 implements the ME-AR01-approved Analysis Review categories:

* SOURCE_AVAILABILITY_REVIEW;
* FUNDAMENTAL_OBSERVATION_COMPLETENESS_REVIEW;
* CASH_GENERATION_REVIEW;
* FREE_CASH_FLOW_REVIEW;
* DATA_LIMITATION_REVIEW;
* HUMAN_REVIEW_REQUIREMENT.

## Implemented states

ME-AR02 implements the ME-AR01-approved Analysis Review states:

* SOURCE_HEALTHY;
* SOURCE_LIMITED;
* OBSERVATIONS_COMPLETE;
* OBSERVATIONS_LIMITED;
* CASH_GENERATION_POSITIVE;
* CASH_GENERATION_NEGATIVE;
* CASH_GENERATION_NEUTRAL;
* DATA_LIMITED;
* REQUIRES_HUMAN_REVIEW;
* NOT_ASSESSED.

## Input behavior

ME-AR02 consumes:

* `SecCompanyFactsFundamentalObservationSet`;
* `SecCompanyFactsDerivedCashGenerationObservationSet`.

The implementation validates that the upstream observation sets align on:

* ticker;
* CIK;
* provider name;
* source context format version;
* source context state;
* source refresh snapshot ID.

If upstream observation sets do not align, ME-AR02 fails safely with a `ValueError`.

## Review behavior

For complete and available upstream observations:

* source availability review emits SOURCE_HEALTHY;
* fundamental observation completeness review emits OBSERVATIONS_COMPLETE;
* cash-generation review emits the matching cash-generation state;
* free cash flow review emits the matching cash-generation state;
* no data limitation review is emitted;
* no human review requirement is emitted.

For limited or incomplete upstream observations:

* source availability may emit SOURCE_LIMITED;
* fundamental completeness may emit OBSERVATIONS_LIMITED;
* cash-generation and free cash flow review may emit DATA_LIMITED;
* data limitation review is emitted;
* human review requirement is emitted.

## Provenance handling

ME-AR02 preserves references to upstream observations.

Source observation references include:

* upstream category;
* upstream state;
* upstream message;
* canonical fields;
* source values;
* source references;
* missing source fields.

Derived observation references include:

* upstream category;
* upstream state;
* upstream message;
* formula;
* derived values;
* required source fields;
* missing source fields;
* source observation references.

The analysis review object also preserves:

* fundamental observation format version;
* derived observation format version;
* source context format version;
* source context state;
* source refresh snapshot ID;
* source refresh fetched timestamp;
* source refresh payload format version.

## Persistence

ME-AR02 adds persistence for Analysis Review output.

Approved path shape:

* data/market_engine/analysis_reviews/<analysis_review_run_id>/<ticker>/analysis_review.json

The persistence function refuses to overwrite existing output.

## Explicit non-scope

ME-AR02 does not implement or authorize:

* raw SEC CompanyFacts fetching;
* provider calls;
* Source Refresh behavior changes;
* Source Context behavior changes;
* Fundamental Observation behavior changes;
* Derived Observation behavior changes;
* Recommendation Review behavior;
* Portfolio Review behavior;
* Delivery behavior;
* Telegram behavior;
* reporting;
* Decision Engine behavior;
* BUY / SELL / HOLD;
* target price;
* score;
* ranking;
* rating;
* conviction;
* urgency;
* tradeability;
* allocation;
* position sizing;
* execution advice;
* watchlist mutation;
* portfolio mutation.

## Tests

ME-AR02 adds tests proving:

* complete positive upstream observations produce non-recommendation Analysis Review;
* negative cash generation is reviewed without recommendation authority;
* neutral cash generation is reviewed without recommendation authority;
* missing capital expenditures emits limitation and human review requirement;
* upstream source and derived observation references are preserved;
* upstream observation set mismatch fails safely;
* persistence writes JSON to the approved path;
* persistence refuses overwrite;
* recommendation, score, ranking, portfolio, delivery, Telegram, and Decision Engine authority are not emitted;
* legacy runtime modules are not imported.

## Test results

Focused ME-AR02 tests:

* PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/analysis_review -q
* Result: 9 passed

Targeted Market Engine regression:

* PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/analysis_review tests/market_engine/derived_observations tests/market_engine/fundamental_observations tests/market_engine/source_context tests/market_engine/source_refresh tests/market_engine/source_intake tests/market_engine/fundamentals -q
* Result: 129 passed

Full tests/market_engine regression:

* PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine -q
* Result: 129 passed

## Governance status

ME-AR02 completes the first non-recommendation Analysis Review implementation.

Recommended next sprint:

* ME-RR01 — Define Recommendation Review contract from Analysis Review

ME-RR01 must be documentation-only first. It must define the boundary between analysis interpretation and recommendation review before any recommendation implementation is attempted.
