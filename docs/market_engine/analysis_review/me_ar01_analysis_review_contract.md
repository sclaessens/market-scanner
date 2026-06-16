# ME-AR01 — Define Analysis Review contract from Fundamental and Derived Observations

## Status

COMPLETED BY ME-AR01

## Job family

ME-AR — Analysis Review jobs

## Purpose

ME-AR01 defines the Analysis Review contract for consuming approved Fundamental Observations and Derived Observations.

The contract allows a future Analysis Review job to produce non-recommendation analytical review output from:

- ME-FO02 Fundamental Observations;
- ME-DO01 Derived Cash Generation Observations.

ME-AR01 is documentation-only.

It does not implement Python code, tests, runtime behavior, provider calls, data writes, recommendation review, portfolio review, delivery, Telegram behavior, or Decision Engine authority.

## Background

ME-FO02 produces source-grounded Fundamental Observations from SEC CompanyFacts Source Context.

ME-DO01 produces the first Derived Observation layer by calculating free cash flow from operating cash flow and capital expenditures.

ME-AR01 defines the next authority boundary: Analysis Review.

Analysis Review is allowed to interpret the observed state of source availability, fundamental observation quality, and derived cash-generation behavior.

Analysis Review is not allowed to recommend an action, rank securities, score securities, allocate capital, size positions, trigger delivery, or mutate portfolio/watchlist state.

## Input contract

A future ME-AR implementation may consume:

- `SecCompanyFactsFundamentalObservationSet`
- `SecCompanyFactsDerivedCashGenerationObservationSet`

Approved input families:

- ME-FO — Fundamental Observations;
- ME-DO — Derived Observations.

Approved initial input formats:

- `sec-companyfacts-fundamental-observations-v1`
- `sec-companyfacts-derived-cash-generation-observations-v1`

Approved initial input paths:

```text
data/market_engine/fundamental_observations/<fundamental_observation_run_id>/<ticker>/fundamental_observations.json
data/market_engine/derived_observations/cash_generation/<derived_observation_run_id>/<ticker>/derived_cash_generation_observations.json

A future implementation may accept in-memory objects or persisted JSON, but the input must remain explicitly tied to approved upstream observation formats.

Output contract

A future ME-AR implementation should emit an Analysis Review output object.

Recommended output format version:

sec-companyfacts-analysis-review-v1

Recommended output path:

data/market_engine/analysis_reviews/<analysis_review_run_id>/<ticker>/analysis_review.json

Recommended top-level fields:

ticker;
cik;
provider_name;
analysis_review_format_version;
fundamental_observation_format_version;
derived_observation_format_version;
source_context_format_version;
source_context_state;
source_refresh_snapshot_id;
source_refresh_fetched_at;
source_refresh_payload_format_version;
review_items;
source_observation_references;
derived_observation_references;
non_recommendation_boundary;
warnings.
Approved review categories

ME-AR01 approves these initial Analysis Review categories:

SOURCE_AVAILABILITY_REVIEW;
FUNDAMENTAL_OBSERVATION_COMPLETENESS_REVIEW;
CASH_GENERATION_REVIEW;
FREE_CASH_FLOW_REVIEW;
DATA_LIMITATION_REVIEW;
HUMAN_REVIEW_REQUIREMENT.

These categories are interpretive review categories only.

They must not imply recommendation, ranking, scoring, allocation, execution, or delivery authority.

Approved review states

ME-AR01 approves these initial Analysis Review states:

SOURCE_HEALTHY;
SOURCE_LIMITED;
OBSERVATIONS_COMPLETE;
OBSERVATIONS_LIMITED;
CASH_GENERATION_POSITIVE;
CASH_GENERATION_NEGATIVE;
CASH_GENERATION_NEUTRAL;
DATA_LIMITED;
REQUIRES_HUMAN_REVIEW;
NOT_ASSESSED.
State semantics
SOURCE_HEALTHY

The required source and observation inputs are available enough for an Analysis Review item.

This state must not mean the company is attractive.

SOURCE_LIMITED

One or more required upstream source or observation inputs are missing or limited.

This state must not mean the company is unattractive.

OBSERVATIONS_COMPLETE

The required Fundamental Observation and Derived Observation families are present for the reviewed category.

This state must not imply quality, score, ranking, or recommendation.

OBSERVATIONS_LIMITED

One or more required observations are missing, incomplete, or limited.

This state must not imply sell, avoid, or downgrade behavior.

CASH_GENERATION_POSITIVE

The reviewed cash-generation derived observation is positive.

This state may describe the observed cash-generation direction only.

It must not imply buy, attractiveness, high quality, conviction, score, ranking, or portfolio action.

CASH_GENERATION_NEGATIVE

The reviewed cash-generation derived observation is negative.

This state may describe the observed cash-generation direction only.

It must not imply sell, unattractiveness, low quality, downgrade, score, ranking, or portfolio action.

CASH_GENERATION_NEUTRAL

The reviewed cash-generation derived observation is zero or neutral.

This state may describe the observed cash-generation direction only.

It must not imply hold, neutral recommendation, score, ranking, or portfolio action.

DATA_LIMITED

The Analysis Review item is limited by missing or incomplete upstream observations.

REQUIRES_HUMAN_REVIEW

The Analysis Review item should be reviewed by a human operator before downstream use.

This state is allowed because it routes uncertainty to the operator without creating recommendation authority.

NOT_ASSESSED

The Analysis Review item cannot be assessed from the available approved upstream observations.

Recommended review item structure

A future implementation should emit review items with fields similar to:

category;
state;
message;
input_observation_families;
required_observations;
missing_observations;
source_observation_references;
derived_observation_references;
non_recommendation_boundary.

Messages must remain descriptive and non-actionable.

Approved message style

Allowed message style:

“Source observations are complete for the reviewed cash-generation category.”
“Free cash flow derived source value is positive.”
“Cash-generation review is limited because capital expenditures are missing.”
“Human review is required because upstream observations are incomplete.”

Forbidden message style:

“The company is attractive.”
“The company is unattractive.”
“This is a buy.”
“This is a sell.”
“This should be held.”
“High-conviction opportunity.”
“Low-quality stock.”
“Undervalued.”
“Overvalued.”
“Add to portfolio.”
“Reduce position.”
“Send alert.”
“Trigger report.”
Provenance requirements

A future ME-AR implementation must preserve upstream references to:

Fundamental Observations used;
Derived Observations used;
Source Context state;
Source Refresh metadata;
source values used by upstream observations;
missingness from upstream observations;
limitation states from upstream observations.

Analysis Review must not hide missingness.

Analysis Review must not convert missing data into zero.

Analysis Review must not silently drop upstream warnings or limitation states.

Persistence contract

Recommended persistence path:

data/market_engine/analysis_reviews/<analysis_review_run_id>/<ticker>/analysis_review.json

A future persistence implementation should:

create one ticker-scoped output file per ticker;
refuse overwrite by default;
preserve upstream format versions;
preserve upstream run references where available;
avoid old data/report paths;
avoid Telegram/reporting side effects.
Explicit non-scope

ME-AR01 does not authorize:

Python implementation;
tests;
runtime behavior;
provider calls;
data writes;
generated artifacts;
raw SEC CompanyFacts fetching;
Source Refresh changes;
Source Context changes;
Fundamental Observation changes;
Derived Observation changes;
Recommendation Review behavior;
Portfolio Review behavior;
Delivery behavior;
Telegram behavior;
reporting;
Decision Engine behavior;
BUY / SELL / HOLD;
target price;
score;
ranking;
rating;
conviction;
urgency;
tradeability;
allocation;
position sizing;
execution advice;
watchlist mutation;
portfolio mutation.
Future implementation requirements

A future ME-AR02 implementation must prove through tests that:

it consumes only approved upstream observation outputs;
it preserves upstream provenance;
it preserves missingness;
it handles positive, negative, zero, missing, and limited derived observations;
it emits only approved Analysis Review categories;
it emits only approved Analysis Review states;
it does not emit recommendation, score, ranking, portfolio, delivery, Telegram, or Decision Engine authority;
automated tests do not call live providers;
legacy runtime modules are not imported.
Governance status

ME-AR01 defines the Analysis Review contract.

Recommended next sprint:

ME-AR02 — Implement Analysis Review from Fundamental and Derived Observations

ME-AR02 must remain non-recommendation and must not introduce Recommendation Review, Portfolio Review, Delivery, Telegram, reporting, or Decision Engine authority.