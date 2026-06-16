# ME-SD01 — Setup Detection contract audit

## Status

COMPLETED BY ME-SD01

## Sprint

ME-SD01 — Define Setup Detection contract

## Branch

me-sd01-define-setup-detection-contract

## Objective

Define the first Setup Detection contract for Market Engine.

The sprint defines how approved Fundamental Observations and Derived Observations can be transformed into a structured, non-actionable Setup Detection output.

The sprint does not implement runtime code.

## Files added

* `docs/market_engine/setup_detection/me_sd01_setup_detection_contract.md`
* `docs/market_engine/audits/me_sd01_setup_detection_contract_audit.md`

## Files to update before completion

* `docs/market_engine/backlog/market_engine_backlog.md`
* `docs/market_engine/roadmap/market_engine_roadmap.md`

## Contract defined

### Input contracts

* `sec-companyfacts-fundamental-observations-v1`
* `sec-companyfacts-derived-cash-generation-observations-v1`

### Output contract

* `sec-companyfacts-setup-detection-v1`

### Future output root

* `data/market_engine/setup_detections`

### Future output filename

* `setup_detection.json`

## Job-family boundary

ME-SD01 defines Setup Detection as a separate Market Engine job family.

Setup Detection sits between:

* Derived Observations
* Analysis Review

Target chain:

* Source Refresh / raw snapshots
* Source Context
* Fundamental Observations
* Derived Observations
* Setup Detection
* Analysis Review
* Recommendation Review
* Portfolio Review
* Decision Engine handoff / action authority
* Delivery / reporting

## Setup Detection purpose

Setup Detection identifies structured, non-actionable evidence patterns from approved observation inputs.

A setup can describe:

* what evidence pattern is present;
* what evidence supports the pattern;
* what evidence is missing;
* whether the setup is detected, partially detected, not detected, conflicted, blocked, or not assessed;
* which upstream observations support the setup;
* why the setup is not actionable by itself.

A setup must not describe:

* whether to buy;
* whether to sell;
* whether to hold;
* allocation;
* position sizing;
* execution;
* ranking;
* scoring;
* portfolio mutation;
* Decision Engine action;
* delivery or reporting output.

## Initial setup families approved

ME-SD01 approves these initial setup families for future ME-SD02 implementation:

* Cash generation setup
* Fundamental availability setup
* Profitability evidence setup
* Revenue evidence setup
* Balance-sheet evidence setup

These setup families are limited to the evidence available from the currently approved Fundamental Observations and Derived Cash Generation Observations.

## Initial setup categories approved

* `cash_generation_setup`
* `fundamental_availability_setup`
* `profitability_evidence_setup`
* `revenue_evidence_setup`
* `balance_sheet_evidence_setup`
* `data_limitation_setup`
* `not_assessed_setup`

## Initial setup states approved

* `setup_detected`
* `setup_partially_detected`
* `setup_not_detected`
* `setup_conflicted`
* `setup_blocked_by_missing_data`
* `setup_not_assessed`

## Required evidence model

ME-SD01 defines that each future setup item must preserve:

* setup category;
* setup state;
* message;
* input observation families;
* required observations;
* missing observations;
* source observation references;
* derived observation references;
* setup evidence;
* setup limitations;
* non-actionable boundary.

## Required provenance model

ME-SD01 defines that future Setup Detection output must preserve:

* ticker;
* CIK;
* provider name;
* source refresh snapshot ID;
* source refresh fetched timestamp;
* raw source payload format version;
* source context format version;
* fundamental observation format version;
* derived observation format version;
* setup detection format version;
* source context state;
* input observation references;
* derived observation references;
* missing observations.

## Missing-data policy

The Setup Detection contract preserves the project-wide missing-data rules:

* Missing observations must be explicit.
* Missing observations must not be converted to zero.
* Numeric zero must not be treated as missing.
* Missing upstream evidence must result in blocked or partial setup state where applicable.
* Setup Detection must not invent replacement values.
* Setup Detection must not hide missing data in generic warning text only.

## Evidence policy

Allowed evidence sources for the first Setup Detection contract:

* `sec-companyfacts-fundamental-observations-v1`
* `sec-companyfacts-derived-cash-generation-observations-v1`

Disallowed evidence sources in ME-SD01:

* live provider responses;
* raw provider calls;
* portfolio holdings;
* watchlists;
* manually typed opinions;
* analyst estimates;
* price action;
* technical indicators;
* sentiment;
* news;
* generated assumptions;
* Decision Engine outputs.

Future evidence sources require explicit future sprint approval.

## Future ME-SD02 implementation requirements defined

ME-SD01 defines that ME-SD02 should implement a builder equivalent to:

* `build_sec_companyfacts_setup_detection(...)`

The future builder should:

* consume approved Fundamental Observations and Derived Observations;
* validate input contract versions;
* produce `sec-companyfacts-setup-detection-v1`;
* emit one or more setup items;
* preserve source and derived observation references;
* preserve missing observations;
* preserve numeric zero values;
* fail closed on unsupported contracts;
* avoid live providers;
* avoid portfolio state;
* avoid Decision Engine behavior;
* avoid delivery/reporting behavior.

## Future persistence requirements defined

ME-SD01 defines that ME-SD02 may implement a persistence helper equivalent to:

* `persist_sec_companyfacts_setup_detection(...)`

Future persistence should write:

* `<root>/<run_id>/<ticker>/setup_detection.json`

Future persistence must:

* refuse overwrite;
* write deterministic JSON;
* preserve provenance;
* preserve missing-data state;
* avoid production writes unless explicitly scoped through local test roots.

## Future ME-SD02 test requirements defined

ME-SD01 defines that ME-SD02 must use local synthetic tests only.

Future tests should cover:

* setup detected from complete input evidence;
* setup partially detected from incomplete evidence;
* setup blocked by missing required observations;
* conflicted setup evidence;
* unsupported input contracts fail closed;
* numeric zero is preserved and not treated as missing;
* persistence writes JSON under a temporary root;
* persistence refuses overwrite;
* no forbidden action-authority terms are emitted in normal setup text;
* no legacy `scripts` or `market_scanner` imports are introduced.

## Relationship to downstream layers

ME-SD01 preserves the planned future sequence:

1. ME-SD02 — Implement first Setup Detection layer
2. ME-AR03 — Extend Analysis Review contract for Setup Detection input
3. ME-AR04 — Implement Analysis Review consumption of Setup Detection
4. ME-RR03 — Extend Recommendation Review contract for Setup Detection-aware Analysis Review
5. ME-RR04 — Implement Setup Detection-aware Recommendation Review behavior
6. ME-PR01 — Define Portfolio Review contract from Recommendation Review
7. ME-PR02 — Implement Portfolio Review
8. ME-DE01 — Define Decision Engine handoff contract
9. ME-DE02 — Implement controlled Decision Engine handoff
10. ME-DL01 — Define Delivery / Reporting contract
11. ME-DL02 — Implement controlled Delivery / Reporting output

## Boundaries preserved

ME-SD01 does not introduce:

* Python runtime code
* tests
* provider calls
* live SEC calls
* live EDGAR calls
* yfinance calls
* production data writes
* portfolio mutation
* watchlist mutation
* Telegram delivery
* reporting output
* Decision Engine behavior
* BUY / SELL / HOLD action semantics
* allocation
* position sizing
* execution advice
* ranking
* scoring
* conviction scoring
* urgency scoring
* tradeability scoring

## Legacy boundary

ME-SD01 does not introduce imports from:

* `scripts`
* `market_scanner`

No runtime files are changed in this sprint.

## Validation to perform before commit

Required commands:

* `git diff --check`
* `grep -n "ME-SD01\|ME-SD02\|ME-AR03\|ME-RR03\|ME-PR01\|ME-DE01\|ME-DL01\|Status: RECOMMENDED NEXT" docs/market_engine/backlog/market_engine_backlog.md`
* `grep -n "ME-SD01\|ME-SD02\|ME-AR03\|ME-RR03\|ME-PR01\|ME-DE01\|ME-DL01" docs/market_engine/roadmap/market_engine_roadmap.md`
* `git status --short`

Expected validation result:

* documentation-only changes;
* no Python files changed;
* no test files changed;
* no provider files changed;
* no data files changed;
* no portfolio files changed;
* no Telegram/reporting files changed;
* no Decision Engine files changed;
* ME-SD01 remains the only `Status: RECOMMENDED NEXT` sprint until ME-SD01 is marked completed;
* ME-SD02 becomes the next recommended sprint after ME-SD01 completion.

## Audit conclusion

ME-SD01 defines the missing Setup Detection contract needed before Portfolio Review.

The contract establishes a non-actionable setup-detection layer from approved Fundamental Observations and Derived Observations, preserves source grounding and missing-data semantics, defines initial setup families, categories, states, provenance, evidence requirements, future implementation requirements, and future test requirements.

The sprint keeps all runtime, provider, portfolio, delivery, and Decision Engine authority out of scope.
