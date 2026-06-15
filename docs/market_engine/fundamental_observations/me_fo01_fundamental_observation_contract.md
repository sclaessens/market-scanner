# ME-FO01 — Define Fundamental Observation contract from SEC CompanyFacts Source Context

## Status

COMPLETED BY ME-FO01

## Job family

ME-FO — Fundamental Observation jobs

## Purpose

ME-FO01 defines the Fundamental Observation contract for SEC CompanyFacts Source Context.

The goal is to define how Market Engine may convert approved Source Context output into non-decision Fundamental Observations without creating Analysis Review, Recommendation Review, Portfolio Review, Delivery output, Telegram behavior, or Decision Engine authority.

This sprint is a contract/design sprint only. It does not authorize implementation.

## Background

ME-GOV01 introduced the job-scoped sprint naming convention.

ME-SR01 implemented raw SEC CompanyFacts source snapshot persistence and cached source loading.

ME-SC01 defined the SEC CompanyFacts Source Context contract from cached raw snapshots.

ME-SC02 implemented SEC CompanyFacts Source Context from cached raw snapshots.

ME-FO01 defines the next job-family boundary: Fundamental Observation.

The Fundamental Observation job consumes approved Source Context and emits explicit non-decision observations.

It must not re-fetch source data, reinterpret raw SEC payloads directly, produce recommendations, calculate portfolio actions, send reports, or influence the Decision Engine.

## Decision

Fundamental Observation jobs may consume approved Source Context output and produce source-grounded, non-decision observations.

Fundamental Observation output may describe what source-backed facts or source limitations are present.

It must not say what action the user should take.

It must not emit:

- BUY;
- SELL;
- HOLD;
- recommendation;
- ranking;
- score;
- conviction;
- urgency;
- tradeability;
- allocation;
- position sizing;
- execution advice;
- portfolio action;
- delivery instruction;
- Telegram/reporting output;
- Decision Engine authority.

Fundamental Observations are allowed to describe fundamental source facts and data availability, but not to judge investment attractiveness.

## Naming

Sprint name:

- ME-FO01 — Define Fundamental Observation contract from SEC CompanyFacts Source Context

Recommended next implementation sprint:

- ME-FO02 — Implement Fundamental Observations from SEC CompanyFacts Source Context

ME-FO01 must not be renamed to a generic ME14 or mixed with Source Context, Analysis Review, Recommendation Review, Portfolio Review, or Delivery.

## Input Contract

The input to the Fundamental Observation job is approved SEC CompanyFacts Source Context output from ME-SC02.

Canonical source-context input path shape:

- data/market_engine/source_contexts/fundamentals/<source_context_run_id>/<ticker>/source_context.json

Approved input format version:

- sec-companyfacts-source-context-v1

Required input properties:

- ticker;
- CIK;
- source name;
- provider name;
- source context format version;
- source context state;
- source refresh snapshot metadata;
- canonical fields;
- field-level states;
- field-level provenance;
- missing canonical fields.

The Fundamental Observation job must not consume raw SEC CompanyFacts payloads directly when approved Source Context exists.

The Fundamental Observation job must not call SEC, providers, or Source Refresh.

## Output Contract

The Fundamental Observation job emits a deterministic observation object or collection.

Recommended output path shape:

- data/market_engine/fundamental_observations/<fundamental_observation_run_id>/<ticker>/fundamental_observations.json

Recommended output format version:

- sec-companyfacts-fundamental-observations-v1

The output must include:

- ticker;
- CIK;
- provider name;
- source context reference;
- source context format version;
- observation format version;
- observation run ID;
- observation categories;
- observation states;
- observation messages;
- canonical fields used;
- source values used;
- source references/provenance;
- missing source fields;
- explicit non-decision boundary marker.

## Observation Object Contract

Each observation should contain:

- ticker;
- cik;
- provider_name;
- category;
- state;
- message;
- source_context_state;
- canonical_fields;
- source_values;
- source_references;
- missing_source_fields;
- observation_format_version;
- non_decision_boundary.

Recommended field meanings:

| Field | Meaning |
|---|---|
| `ticker` | Market ticker from Source Context |
| `cik` | SEC CIK from Source Context |
| `provider_name` | Source provider name, normally `SEC_COMPANYFACTS` |
| `category` | Observation category |
| `state` | Observation state |
| `message` | Human-readable non-decision message |
| `source_context_state` | Input Source Context state |
| `canonical_fields` | Canonical fields used by the observation |
| `source_values` | Source values used by the observation |
| `source_references` | SEC provenance and period metadata |
| `missing_source_fields` | Missing fields relevant to the observation |
| `observation_format_version` | Output contract version |
| `non_decision_boundary` | Explicit marker that the output is not recommendation or action authority |

## Approved Observation Categories

ME-FO01 approves the following initial categories:

- SOURCE_CONTEXT_AVAILABILITY;
- REVENUE_SOURCE_PRESENCE;
- NET_INCOME_SOURCE_VALUE;
- OPERATING_CASH_FLOW_SOURCE_VALUE;
- CAPEX_SOURCE_PRESENCE;
- CASH_GENERATION_SOURCE_COMPLETENESS;
- DATA_LIMITATION.

### SOURCE_CONTEXT_AVAILABILITY

Purpose: describe whether the input Source Context is available, partial, missing, invalid, provider-error, or unsupported.

Allowed message examples:

- All required SEC CompanyFacts Source Context fields are available.
- One or more required SEC CompanyFacts Source Context fields are missing.
- SEC CompanyFacts Source Context is missing for this ticker.

Forbidden message examples:

- This company is attractive.
- This company is unattractive.
- Buy this stock.
- Avoid this stock.

### REVENUE_SOURCE_PRESENCE

Purpose: describe whether the approved revenue source field is present.

Allowed message examples:

- Revenue source field is present for the selected SEC period.
- Revenue source field is missing for the selected SEC period.

This observation may expose the source value and provenance, but it must not interpret growth, valuation, or business quality.

### NET_INCOME_SOURCE_VALUE

Purpose: describe whether the approved net income source field is present and whether the selected source value is positive, negative, zero, or missing.

Allowed message examples:

- Net income source value is positive for the selected SEC period.
- Net income source value is negative for the selected SEC period.
- Net income source value is zero for the selected SEC period.
- Net income source field is missing for the selected SEC period.

This is still an observation, not an investment conclusion.

### OPERATING_CASH_FLOW_SOURCE_VALUE

Purpose: describe whether the approved operating cash flow source field is present and whether the selected source value is positive, negative, zero, or missing.

Allowed message examples:

- Operating cash flow source value is positive for the selected SEC period.
- Operating cash flow source value is negative for the selected SEC period.
- Operating cash flow source value is zero for the selected SEC period.
- Operating cash flow source field is missing for the selected SEC period.

This must not calculate free cash flow unless a later ME-DO Derived Observation sprint explicitly authorizes derived calculations.

### CAPEX_SOURCE_PRESENCE

Purpose: describe whether the approved capital expenditures source field is present.

Allowed message examples:

- Capital expenditures source field is present for the selected SEC period.
- Capital expenditures source field is missing for the selected SEC period.

This must not derive free cash flow or capital intensity.

### CASH_GENERATION_SOURCE_COMPLETENESS

Purpose: describe whether the Source Context contains both operating cash flow and capital expenditures source fields.

Allowed message examples:

- Operating cash flow and capital expenditures source fields are both present.
- One or more cash-generation source fields are missing.

This category checks source completeness only.

It must not calculate free cash flow.

### DATA_LIMITATION

Purpose: expose relevant missingness, partial context, unsupported source context, invalid source context, or provider/source limitation.

Allowed message examples:

- Fundamental observation is limited because required source fields are missing.
- Fundamental observation is not assessed because Source Context is unavailable.

This category may explain data limitations but must not produce recommendation language.

## Approved Observation States

ME-FO01 approves these observation states:

- PRESENT;
- MISSING_DATA;
- POSITIVE_SOURCE_VALUE;
- NEGATIVE_SOURCE_VALUE;
- ZERO_SOURCE_VALUE;
- NOT_ASSESSED;
- SOURCE_LIMITED.

State meanings:

| State | Meaning |
|---|---|
| `PRESENT` | Required source field or source condition is present |
| `MISSING_DATA` | Required source field is missing |
| `POSITIVE_SOURCE_VALUE` | Source value is present and greater than zero |
| `NEGATIVE_SOURCE_VALUE` | Source value is present and less than zero |
| `ZERO_SOURCE_VALUE` | Source value is present and equal to zero |
| `NOT_ASSESSED` | Observation cannot be assessed because input context is unavailable, invalid, unsupported, or provider-error |
| `SOURCE_LIMITED` | Observation is possible but constrained by partial or missing source context |

These states are descriptive, not prescriptive.

They must not be used as recommendation proxies.

## Source Context State Handling

The Fundamental Observation job must handle Source Context states as follows:

| Source Context state | Observation behavior |
|---|---|
| `AVAILABLE` | Emit all approved observations from present fields |
| `PARTIAL` | Emit observations for present fields and explicit missing-data observations for missing fields |
| `MISSING` | Emit `NOT_ASSESSED` or `MISSING_DATA` observations with explicit missingness |
| `INVALID` | Emit `NOT_ASSESSED` with invalid context limitation |
| `PROVIDER_ERROR` | Emit `NOT_ASSESSED` with provider/source limitation |
| `UNSUPPORTED` | Emit `NOT_ASSESSED` with unsupported context limitation |

The job must not silently drop missing or failed Source Context.

The job must not convert missing values to zero.

A numeric zero must be treated as present and should produce `ZERO_SOURCE_VALUE`.

## Canonical Fields

The first approved canonical fields are inherited from ME-SC02:

- revenue;
- net_income;
- operating_cash_flow;
- capital_expenditures.

ME-FO01 does not approve additional canonical fields.

Additional fields require a future ME-SC or ME-FO contract update, depending on whether the change is source-context or observation-level.

## Provenance Requirements

Every observation that uses a source value must preserve source references from Source Context.

Source references should include, where available:

- selected SEC tag;
- provider name;
- taxonomy namespace;
- unit;
- fiscal year;
- fiscal period;
- filing form;
- filing date;
- period start date;
- period end date;
- accession number;
- frame;
- selection reason;
- fallback alias used;
- source refresh snapshot ID;
- source refresh fetched timestamp;
- source refresh payload format version;
- source context path or source context reference.

The observation must remain traceable back to Source Context and ultimately to the raw Source Refresh snapshot.

## Explicit Non-Scope

ME-FO01 does not authorize:

- raw SEC CompanyFacts fetching;
- cached raw snapshot loading as a primary input;
- Source Refresh changes;
- Source Context changes;
- derived calculations;
- free cash flow;
- growth;
- margins;
- ratios;
- valuation metrics;
- peer comparison;
- trend analysis;
- scoring;
- ranking;
- BUY / SELL / HOLD;
- recommendation review;
- portfolio review;
- delivery;
- Telegram;
- reporting;
- Decision Engine behavior;
- position sizing;
- execution advice.

Derived calculations belong to ME-DO.

Interpretive analysis belongs to ME-AR.

Recommendation review belongs to ME-RR.

Portfolio review belongs to ME-PR.

Delivery belongs to ME-DL.

## Forbidden Output Fields and Terms

Fundamental Observation output must not contain these fields or equivalent semantics:

- recommendation;
- buy;
- sell;
- hold;
- rating;
- score;
- rank;
- ranking;
- conviction;
- urgency;
- tradeability;
- allocation;
- position_size;
- position_sizing;
- execution;
- target_price;
- portfolio_action;
- decision;
- telegram;
- delivery;
- report_instruction.

The implementation must include boundary tests proving these terms do not appear as output keys.

Human-readable messages must also avoid action language.

## Persistence Contract

Recommended persistence path:

- data/market_engine/fundamental_observations/<fundamental_observation_run_id>/<ticker>/fundamental_observations.json

Recommended top-level output shape:

- ticker;
- cik;
- provider_name;
- observation_format_version;
- source_context_format_version;
- source_context_state;
- observations;
- non_decision_boundary.

The output path is contractually approved by ME-FO01, but implementation must occur in ME-FO02.

## Testing Contract for ME-FO02

The implementation sprint must test:

1. available Source Context produces approved observations;
2. partial Source Context preserves missingness;
3. missing Source Context produces `NOT_ASSESSED` or `MISSING_DATA`;
4. positive, negative, zero, and missing source values are handled correctly;
5. numeric zero is present, not missing;
6. source references are preserved;
7. no derived calculations are emitted;
8. no recommendation, score, ranking, portfolio, delivery, Telegram, or Decision Engine authority is emitted;
9. tests do not use live SEC/provider calls;
10. tests do not import legacy runtime modules;
11. output can be serialized and persisted to the approved path.

Recommended test location:

- tests/market_engine/fundamental_observations/

Recommended implementation location:

- src/market_engine/fundamental_observations/

Do not implement ME-FO02 inside `source_context`.

Do not modify Source Refresh or Source Context unless the implementation discovers a contract defect that requires a separate sprint.

## Relationship to Existing Code

Earlier foundation work already contains preliminary fundamental/source-context and observation-like logic under the older `fundamentals` area.

ME-FO01 does not delete or rewrite that code.

ME-FO02 may reuse proven mapping concepts where appropriate, but it must implement the job-scoped ME-FO boundary cleanly.

The new Fundamental Observation job should not silently become Analysis Review.

## Backlog Rule

The backlog should mark:

- ME-FO01 — Define Fundamental Observation contract from SEC CompanyFacts Source Context
- Status: COMPLETED BY ME-FO01

The next recommended sprint should be:

- ME-FO02 — Implement Fundamental Observations from SEC CompanyFacts Source Context
- Status: RECOMMENDED NEXT

ME-FO02 must remain an implementation sprint inside the Fundamental Observation job family.

## Audit Rule

The ME-FO01 audit must confirm:

- documentation/contract only;
- no Python code changed;
- no tests changed;
- no data files changed;
- no generated artifacts changed;
- no provider calls introduced;
- no runtime behavior changed;
- no Source Refresh behavior changed;
- no Source Context behavior changed;
- no observations implemented;
- no analysis review introduced;
- no recommendation review introduced;
- no portfolio review introduced;
- no delivery or Telegram behavior introduced;
- no Decision Engine behavior introduced.

## Acceptance Criteria

ME-FO01 is complete when:

- the Fundamental Observation job boundary is defined;
- the Source Context input contract is defined;
- the Fundamental Observation output contract is defined;
- approved observation categories are defined;
- approved observation states are defined;
- Source Context state handling is defined;
- provenance requirements are defined;
- forbidden authority semantics are defined;
- persistence path is recommended;
- ME-FO02 implementation scope is clear;
- the sprint remains documentation/contract only.

## Governance Status

Status: Approved as ME-FO01 contract/design sprint.

Effective immediately after ME-SC02.

Implementation is deferred to:

- ME-FO02 — Implement Fundamental Observations from SEC CompanyFacts Source Context
