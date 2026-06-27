# ME-SA07 - Company Profile Analysis Review Descriptive Context Audit

Sprint ID: ME-SA07
Status: COMPLETED BY ME-SA07
Job family: ME-SA / Analysis Review
Date: 2026-06-27
Branch: `me-sa07-company-profile-analysis-review-descriptive-context`

## Objective

ME-SA07 allows consumed Company Profile Fundamental Observations to reach
Analysis Review as deterministic descriptive context only.

The implementation follows:

* ME-SA06 Company Profile Fundamental Observations;
* ME-RUN27 cross-ticker evidence and controlled Derived Observations stop;
* ME-RM04 ticker-agnostic and fast full-output guardrails.

## Implementation

ME-SA07 introduces three explicit contracts:

```text
market-engine-company-profile-derived-context-bridge-v1
market-engine-company-profile-setup-not-applicable-v1
market-engine-company-profile-analysis-context-v1
```

The Derived Observations bridge copies approved observation references without
computing financial values.

The Setup Detection boundary records that setup processing is not applicable to
descriptive profile context and emits no setup items.

The Analysis Review context maps approved company-profile observations into
source-linked descriptive context items.

## Descriptive Analysis Context

Available context types are:

```text
company_identity_context
symbol_context
exchange_context
sector_context
industry_context
country_context
currency_context
description_availability_context
website_context
provenance_context
as_of_context
```

Each item preserves:

* the source observation code;
* the source field;
* the source-derived value.

The context also preserves ticker, symbol, provider, source formats, snapshot
identity, provenance, and as-of metadata.

## Pipeline Behavior

Before ME-SA07, valid profile-only runs completed Source Context and Fundamental
Observations and stopped at Derived Observations.

After ME-SA07, valid profile-only runs complete:

```text
Source Context
Fundamental Observations
Derived Observations descriptive bridge
Setup Detection not-applicable boundary
Analysis Review descriptive context
```

They then stop at Recommendation Review with:

```text
company_profile_descriptive_analysis_context_has_no_recommendation_input
```

This is a controlled authority boundary. ME-SA07 does not fabricate
recommendation input to force downstream progression.

## Consumption States

`absent_optional`:

* no Company Profile Analysis Review context is produced;
* the existing SEC CompanyFacts path remains unchanged.

`blocked`:

* no profile body or profile observations are trusted;
* no Company Profile Analysis Review context is produced;
* upstream gate and blocker metadata remain traceable.

`consumed`:

* the existing ME-SA06 observation set is bridged;
* descriptive Analysis Review context is produced;
* no raw profile payload is reread by Analysis Review.

## Combined SEC and Company Profile Behavior

The pure additive helper
`attach_company_profile_context_to_analysis_review` adds a validated profile
context under `company_profile_context` while preserving every existing SEC
Analysis Review field.

The current cached-source command still accepts one snapshot path. ME-SA07 does
not add hidden multi-source discovery. The helper defines safe additive behavior
for a future explicit combined-source orchestrator.

## Ticker-Agnostic Evidence

Runtime modules contain no `NVDA`, `AMD`, or `ASML` branches.

Tests use a synthetic ticker and an ASML/NL profile variant. Both pass through
the same bridge, setup boundary, and Analysis Review builder.

No provider, exchange, country, or US-specific runtime workaround was added.

## Non-Advisory Guarantee

The new context contains no recommendation, target, rank, conviction, urgency,
score, setup item, trade instruction, broker instruction, portfolio action, or
Decision Engine authority.

The bridge records `financial_derivations_performed=false`. The setup boundary
contains an empty `setup_items` tuple.

ME-SA07 does not change SEC Analysis Review, Recommendation Review, Portfolio
Review, Decision Engine, handoff, or delivery builders.

## Files

Runtime:

```text
src/market_engine/derived_observations/company_profile_context_bridge.py
src/market_engine/setup_detection/company_profile_not_applicable.py
src/market_engine/analysis_review/company_profile_analysis_context.py
src/market_engine/run/cached_source_execution.py
src/market_engine/run/end_to_end_dry_run.py
```

Tests:

```text
tests/market_engine/analysis_review/test_company_profile_analysis_context.py
tests/market_engine/run/test_me_run10_cached_source_local_execution.py
tests/market_engine/run/test_me_run27_company_profile_cross_ticker_dry_run.py
```

The ME-RUN27 runner expectation was updated to recognize Analysis Review
completion and the new controlled Recommendation Review stop.

## Tests

Validation:

```text
7 passed - Company Profile Analysis Context tests
21 passed - cached-source local execution tests
114 passed - tests/market_engine/run
518 passed - tests/market_engine
1185 passed - full pytest
```

## Safety and Non-Goals

ME-SA07 adds no provider calls, network access, yfinance, SEC/EDGAR access,
Telegram sending, production writes, broker actions, portfolio/watchlist
mutation, source refresh, or delivery side effects.

ME-SA07 adds no financial derivation, setup signal, investment evaluation,
recommendation, target, ranking, urgency, conviction, score, allocation, or
Decision Engine authority.

## Final Status

```text
PASS
```

## Follow-Up

ME-SA07 exposes a controlled stop before reportability:

```text
ME-SA08 - Define safe descriptive Analysis Review continuation beyond the Recommendation Review boundary
```

ME-SA08 must not create recommendation semantics. ME-DL03 remains next after a
safe continuation contract exists.
