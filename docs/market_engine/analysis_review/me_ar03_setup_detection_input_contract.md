# ME-AR03 — Extend Analysis Review contract for Setup Detection input

## Status

COMPLETED BY ME-AR03

## Sprint

ME-AR03 — Extend Analysis Review contract for Setup Detection input

## Job family

ME-AR — Analysis Review jobs

## Purpose

ME-AR03 extends the Analysis Review contract so a future Analysis Review implementation can consume Setup Detection output.

The existing Analysis Review layer already consumes:

* `sec-companyfacts-fundamental-observations-v1`;
* `sec-companyfacts-derived-cash-generation-observations-v1`.

ME-AR03 adds the contract rules for consuming:

* `sec-companyfacts-setup-detection-v1`.

This sprint is documentation-only.

It does not implement Python code, tests, runtime behavior, provider calls, data writes, Recommendation Review behavior, Portfolio Review behavior, Delivery behavior, Telegram behavior, reporting behavior, or Decision Engine authority.

## Background

ME-AR01 defined the original non-recommendation Analysis Review contract from Fundamental Observations and Derived Observations.

ME-AR02 implemented that Analysis Review contract.

ME-SD01 defined Setup Detection as a non-actionable pattern/setup layer between Derived Observations and Analysis Review.

ME-SD02 implemented the first Setup Detection runtime layer.

ME-AR03 updates the Analysis Review contract so ME-AR04 can later implement Analysis Review consumption of Setup Detection output without changing authority boundaries.

## Architectural position

The approved architectural chain after ME-SD02 is:

```text
Source Refresh / raw snapshots
→ Source Context
→ Fundamental Observations
→ Derived Observations
→ Setup Detection
→ Analysis Review
→ Recommendation Review
→ Portfolio Review
→ Decision Engine handoff / action authority
→ Delivery / reporting
```

ME-AR03 does not move Analysis Review.

ME-AR03 only defines how Analysis Review may consume Setup Detection output as an additional approved upstream input.

## Existing Analysis Review contract

The existing Analysis Review output contract remains:

* `sec-companyfacts-analysis-review-v1`.

The existing Analysis Review output path remains:

```text
data/market_engine/analysis_reviews/<analysis_review_run_id>/<ticker>/analysis_review.json
```

The existing Analysis Review job family remains non-recommendation and non-actionable.

ME-AR03 does not create a new Analysis Review job family.

ME-AR03 extends the existing Analysis Review contract with Setup Detection input requirements.

## Approved input families after ME-AR03

Analysis Review may consume the following approved input families:

* ME-FO — Fundamental Observations;
* ME-DO — Derived Observations;
* ME-SD — Setup Detection.

## Approved input contracts after ME-AR03

Analysis Review may consume these approved input contract versions:

* `sec-companyfacts-fundamental-observations-v1`;
* `sec-companyfacts-derived-cash-generation-observations-v1`;
* `sec-companyfacts-setup-detection-v1`.

The Setup Detection input must be derived from the same upstream observation family as the Analysis Review input set.

The Setup Detection input must not be treated as a standalone recommendation signal.

## Required input alignment

A future ME-AR04 implementation must validate alignment between Fundamental Observations, Derived Observations, and Setup Detection before producing Setup Detection-aware Analysis Review.

At minimum, the inputs must align on:

* ticker;
* CIK where available;
* provider name where available;
* source context format version where available;
* source context state where available;
* source refresh snapshot ID where available;
* upstream Fundamental Observation format version;
* upstream Derived Observation format version;
* Setup Detection format version.

If the inputs do not align, the implementation must fail closed or emit a controlled non-actionable Analysis Review limitation item.

The implementation must not silently merge mismatched evidence.

## Setup Detection input contract

The approved Setup Detection input contract is:

```text
sec-companyfacts-setup-detection-v1
```

The Setup Detection input may be consumed as an in-memory object or as persisted JSON.

Recommended future persisted input path:

```text
data/market_engine/setup_detections/<setup_detection_run_id>/<ticker>/setup_detection.json
```

A future ME-AR04 implementation may accept caller-provided in-memory Setup Detection objects if they conform to the approved contract.

## Setup Detection input fields

A future ME-AR04 implementation should preserve and interpret at least these Setup Detection fields where available:

* ticker;
* CIK;
* provider name;
* setup detection format version;
* fundamental observation format version;
* derived observation format version;
* source context format version;
* source context state;
* source refresh snapshot metadata;
* setup items;
* setup categories;
* setup states;
* setup messages;
* setup evidence;
* setup limitations;
* missing observations;
* source observation references;
* derived observation references;
* non-actionable boundary marker.

ME-AR04 must not require fields that are not present in the approved ME-SD02 implementation unless a later contract sprint extends Setup Detection first.

## Analysis Review output contract after ME-AR03

The output contract remains:

```text
sec-companyfacts-analysis-review-v1
```

ME-AR03 does not introduce `sec-companyfacts-analysis-review-v2`.

A future implementation may extend the existing `sec-companyfacts-analysis-review-v1` output with Setup Detection-aware review items while preserving backward compatibility where possible.

The top-level Analysis Review output should preserve existing fields and may add Setup Detection metadata fields such as:

* setup_detection_format_version;
* setup_detection_references;
* setup_detection_run_id where available;
* setup_detection_input_state where available;
* setup_detection_limitations;
* setup_detection_non_actionable_boundary.

If the current implementation style favors review-item-level references rather than top-level references, ME-AR04 may preserve Setup Detection provenance through review items instead.

## New Analysis Review input family marker

A future ME-AR04 implementation should include Setup Detection in review item input families where used.

Recommended input family value:

```text
ME-SD
```

Recommended human-readable family label:

```text
Setup Detection
```

This family marker indicates that the review item used Setup Detection output as evidence.

It must not imply recommendation authority.

## Approved Setup Detection-aware review categories

ME-AR03 approves the following additional Analysis Review categories for future implementation:

* `SETUP_DETECTION_REVIEW`;
* `SETUP_EVIDENCE_COMPLETENESS_REVIEW`;
* `SETUP_LIMITATION_REVIEW`;
* `SETUP_HUMAN_REVIEW_REQUIREMENT`.

These categories extend the existing ME-AR01 categories.

They do not replace the existing categories:

* `SOURCE_AVAILABILITY_REVIEW`;
* `FUNDAMENTAL_OBSERVATION_COMPLETENESS_REVIEW`;
* `CASH_GENERATION_REVIEW`;
* `FREE_CASH_FLOW_REVIEW`;
* `DATA_LIMITATION_REVIEW`;
* `HUMAN_REVIEW_REQUIREMENT`.

## Approved Setup Detection-aware review states

ME-AR03 approves the following additional Analysis Review states for future implementation:

* `SETUP_DETECTED`;
* `SETUP_PARTIALLY_DETECTED`;
* `SETUP_NOT_DETECTED`;
* `SETUP_CONFLICTED`;
* `SETUP_BLOCKED_BY_MISSING_DATA`;
* `SETUP_NOT_ASSESSED`;
* `SETUP_REQUIRES_HUMAN_REVIEW`.

These states are interpretive Analysis Review states only.

They must not imply:

* buy;
* sell;
* hold;
* attractiveness;
* quality ranking;
* score;
* conviction;
* urgency;
* tradeability;
* allocation;
* execution;
* delivery.

## State mapping from Setup Detection to Analysis Review

A future ME-AR04 implementation may map Setup Detection states to Analysis Review states as follows.

| Setup Detection state           | Analysis Review state           |
| ------------------------------- | ------------------------------- |
| `setup_detected`                | `SETUP_DETECTED`                |
| `setup_partially_detected`      | `SETUP_PARTIALLY_DETECTED`      |
| `setup_not_detected`            | `SETUP_NOT_DETECTED`            |
| `setup_conflicted`              | `SETUP_CONFLICTED`              |
| `setup_blocked_by_missing_data` | `SETUP_BLOCKED_BY_MISSING_DATA` |
| `setup_not_assessed`            | `SETUP_NOT_ASSESSED`            |

When Setup Detection emits conflicted, blocked, partial, or not-assessed states, Analysis Review should consider emitting `SETUP_REQUIRES_HUMAN_REVIEW`.

This is a human-review routing state only.

It must not become a recommendation state.

## Category mapping from Setup Detection to Analysis Review

A future ME-AR04 implementation may map Setup Detection categories to Analysis Review categories as follows.

| Setup Detection category         | Analysis Review category             |
| -------------------------------- | ------------------------------------ |
| `cash_generation_setup`          | `SETUP_DETECTION_REVIEW`             |
| `fundamental_availability_setup` | `SETUP_EVIDENCE_COMPLETENESS_REVIEW` |
| `profitability_evidence_setup`   | `SETUP_DETECTION_REVIEW`             |
| `revenue_evidence_setup`         | `SETUP_DETECTION_REVIEW`             |
| `balance_sheet_evidence_setup`   | `SETUP_DETECTION_REVIEW`             |
| `data_limitation_setup`          | `SETUP_LIMITATION_REVIEW`            |
| `not_assessed_setup`             | `SETUP_HUMAN_REVIEW_REQUIREMENT`     |

The mapping may be implemented deterministically in ME-AR04.

ME-AR04 must preserve the original Setup Detection category and state in references or evidence fields.

## Required Setup Detection-aware review item structure

A future ME-AR04 implementation should extend review items with Setup Detection references where used.

Recommended review item fields:

* category;
* state;
* message;
* input_observation_families;
* required_observations;
* missing_observations;
* source_observation_references;
* derived_observation_references;
* setup_detection_references;
* setup_categories;
* setup_states;
* setup_evidence;
* setup_limitations;
* non_recommendation_boundary.

The existing review item fields must remain valid.

ME-AR04 must not remove existing Fundamental Observation or Derived Observation references.

## Setup Detection references

Setup Detection references should preserve:

* setup category;
* setup state;
* setup message;
* setup evidence;
* setup limitations;
* missing observations;
* source observation references;
* derived observation references;
* non-actionable boundary marker.

Setup Detection references must remain traceable to upstream Fundamental Observations and Derived Observations where available.

Analysis Review must not flatten Setup Detection in a way that loses provenance.

## Missing-data handling

Analysis Review must preserve missingness from Setup Detection.

If Setup Detection reports missing observations, Analysis Review must not:

* convert missing values to zero;
* omit the missing-observation list;
* treat missing setup evidence as negative evidence;
* treat missing setup evidence as a recommendation reason;
* silently proceed as if setup evidence were complete.

If Setup Detection emits `setup_blocked_by_missing_data`, Analysis Review should emit:

* `SETUP_BLOCKED_BY_MISSING_DATA`; and/or
* `SETUP_REQUIRES_HUMAN_REVIEW`.

The output message must explain that Analysis Review is limited by missing Setup Detection evidence.

## Numeric-zero handling

Numeric zero remains a valid observed or derived value.

Analysis Review must not treat numeric zero as missing.

When Setup Detection preserves numeric zero in setup evidence, Analysis Review must preserve that value or reference.

If a setup is neutral because of a zero derived value, the review message may describe the setup evidence as neutral or zero, but must not use `HOLD` or imply a neutral recommendation.

## Conflicted evidence handling

If Setup Detection emits `setup_conflicted`, Analysis Review must preserve that conflict.

Analysis Review should emit a Setup Detection-aware review item with:

* category: `SETUP_DETECTION_REVIEW` or `SETUP_LIMITATION_REVIEW`;
* state: `SETUP_CONFLICTED`;
* setup references;
* conflict explanation;
* human review requirement where appropriate.

A conflict must not be converted into a positive or negative recommendation.

## Partial setup handling

If Setup Detection emits `setup_partially_detected`, Analysis Review must preserve partiality.

Analysis Review may describe the setup as partially supported by available evidence.

It must also preserve missing or incomplete observations that prevent full setup detection.

A partial setup must not be treated as a complete investment thesis.

## Not-assessed handling

If Setup Detection emits `setup_not_assessed`, Analysis Review must preserve the not-assessed state.

Analysis Review may emit:

* `SETUP_NOT_ASSESSED`;
* `SETUP_REQUIRES_HUMAN_REVIEW`.

The message should explain that Setup Detection could not assess the pattern from approved inputs.

## Human review routing

Setup Detection-aware Analysis Review may route items to human review.

Human review routing is allowed because it keeps uncertainty with the operator.

Human review routing must not:

* recommend a trade;
* imply a portfolio action;
* trigger delivery;
* trigger Telegram;
* call the Decision Engine;
* create urgency or conviction scoring.

## Approved message style

Allowed Setup Detection-aware message examples:

* “Setup Detection reports positive cash-generation setup evidence from approved observations.”
* “Setup Detection is partially supported because some required observations are missing.”
* “Setup Detection is blocked because required upstream observations are missing.”
* “Setup Detection reports conflicted evidence and requires human review.”
* “Setup Detection could not assess the setup from the approved inputs.”

These messages are descriptive and non-actionable.

## Forbidden message style

ME-AR03 does not allow messages such as:

* “This setup is a buy.”
* “This is a sell signal.”
* “Hold this position.”
* “High-conviction setup.”
* “Best opportunity.”
* “Top-ranked candidate.”
* “Tradeable setup.”
* “Urgent opportunity.”
* “Increase allocation.”
* “Reduce position.”
* “Enter position.”
* “Exit position.”
* “Send Telegram alert.”
* “Publish report.”
* “Trigger Decision Engine action.”

Forbidden terms and semantics remain forbidden even when Setup Detection evidence is strong.

## Provenance requirements

A future ME-AR04 implementation must preserve:

* Fundamental Observation references;
* Derived Observation references;
* Setup Detection references;
* source values used by upstream observations;
* derived values used by Setup Detection;
* missingness from upstream observations;
* limitation states from upstream observations;
* setup evidence;
* setup limitations;
* source context state;
* source refresh metadata;
* upstream format versions.

Analysis Review must not hide Setup Detection limitations.

Analysis Review must not silently drop Setup Detection warnings or boundary markers.

## Persistence contract

The Analysis Review persistence path remains:

```text
data/market_engine/analysis_reviews/<analysis_review_run_id>/<ticker>/analysis_review.json
```

ME-AR03 does not create a new persistence root.

A future ME-AR04 persistence implementation should:

* preserve Setup Detection input format version;
* preserve Setup Detection references;
* preserve upstream observation references;
* refuse overwrite by default;
* avoid old data/report paths;
* avoid Telegram/reporting side effects;
* avoid production data writes in tests.

## Backward compatibility

ME-AR04 should preserve existing ME-AR02 behavior for Analysis Review built without Setup Detection input unless ME-AR04 explicitly scopes a required Setup Detection input.

Allowed implementation strategies for ME-AR04:

1. require Setup Detection input for the new Setup Detection-aware builder only;
2. add an optional Setup Detection argument to the existing builder while preserving existing behavior when omitted;
3. create a separate Setup Detection-aware builder if that better preserves contract clarity.

ME-AR04 must choose the strategy that best fits existing active Market Engine patterns.

ME-AR04 must not break existing ME-AR02 tests unless the test update is explicitly justified by the new contract.

## ME-AR04 implementation requirements

ME-AR04 must implement Analysis Review consumption of Setup Detection according to this contract.

ME-AR04 must:

* consume `sec-companyfacts-setup-detection-v1`;
* preserve existing Fundamental Observation and Derived Observation behavior;
* validate input alignment;
* emit Setup Detection-aware review items;
* preserve setup categories and states;
* preserve setup evidence;
* preserve setup limitations;
* preserve missing observations;
* preserve source and derived references;
* preserve numeric-zero semantics;
* preserve non-recommendation and non-actionable boundary markers;
* fail closed or emit controlled limitation output for unsupported Setup Detection input contracts;
* add local synthetic tests only;
* avoid live provider calls;
* avoid production data writes.

ME-AR04 must test:

* complete setup detection input creates Setup Detection-aware Analysis Review;
* partial setup input creates partial Setup Detection-aware review;
* missing setup evidence creates blocked or data-limited review;
* conflicted setup input creates conflicted review and human-review routing;
* not-assessed setup input remains not assessed;
* unsupported Setup Detection input contract fails closed;
* numeric zero remains present and is not treated as missing;
* Setup Detection references are preserved;
* Fundamental Observation and Derived Observation references remain preserved;
* existing Analysis Review behavior is not broken;
* forbidden action-authority terms are not emitted;
* persistence preserves Setup Detection references if persistence is touched;
* no legacy `scripts` or old `market_scanner` imports are introduced.

## Explicit non-scope

ME-AR03 does not authorize:

* Python implementation;
* tests;
* runtime behavior;
* provider calls;
* live SEC calls;
* EDGAR calls;
* yfinance calls;
* data writes;
* generated artifacts;
* Source Refresh changes;
* Source Context changes;
* Fundamental Observation changes;
* Derived Observation changes;
* Setup Detection runtime changes;
* Recommendation Review behavior;
* Portfolio Review behavior;
* Delivery behavior;
* Telegram behavior;
* reporting behavior;
* Decision Engine behavior;
* BUY / SELL / HOLD;
* target price;
* rating;
* score;
* ranking;
* conviction;
* urgency;
* tradeability;
* allocation;
* position sizing;
* execution advice;
* order generation;
* watchlist mutation;
* portfolio mutation.

## Acceptance criteria

ME-AR03 is complete when:

* the Setup Detection input contract for Analysis Review is documented;
* approved input families after ME-AR03 are documented;
* approved input contracts after ME-AR03 are documented;
* input alignment requirements are documented;
* Setup Detection-aware Analysis Review categories are documented;
* Setup Detection-aware Analysis Review states are documented;
* state mapping from Setup Detection to Analysis Review is documented;
* category mapping from Setup Detection to Analysis Review is documented;
* review item structure is documented;
* missing-data handling is documented;
* numeric-zero handling is documented;
* conflict handling is documented;
* human-review routing is documented;
* message style boundaries are documented;
* provenance requirements are documented;
* persistence expectations are documented;
* backward compatibility expectations are documented;
* ME-AR04 implementation requirements are documented;
* backlog and roadmap mark ME-AR03 completed;
* backlog and roadmap mark ME-AR04 as the only recommended next sprint;
* no runtime, test, provider, data, recommendation, portfolio, delivery, Telegram, reporting, or Decision Engine behavior is changed.

## Completion outcome

ME-AR03 extends the Analysis Review contract so the future ME-AR04 implementation can consume `sec-companyfacts-setup-detection-v1`.

The contract keeps Analysis Review descriptive, source-grounded, provenance-preserving, missing-data-aware, numeric-zero-safe, non-recommendation, and non-actionable.
