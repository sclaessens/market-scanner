# ME-SD01 — Setup Detection contract

## Status

COMPLETED BY ME-SD01

## Sprint

ME-SD01 — Define Setup Detection contract

## Job family

Setup Detection

## Purpose

Setup Detection defines a non-actionable contract for identifying market-engine setups from approved observation inputs.

The first Setup Detection contract consumes SEC CompanyFacts Fundamental Observations and Derived Cash Generation Observations.

It emits a structured setup-detection output that can later be consumed by Analysis Review.

Setup Detection does not create recommendation authority, portfolio authority, delivery authority, or Decision Engine authority.

## Contract summary

### Input contracts

ME-SD01 defines the first Setup Detection contract from these approved inputs:

* `sec-companyfacts-fundamental-observations-v1`
* `sec-companyfacts-derived-cash-generation-observations-v1`

### Output contract

ME-SD01 defines the future output contract:

* `sec-companyfacts-setup-detection-v1`

### Future output root

Future implementation may persist output under:

* `data/market_engine/setup_detections`

### Future output filename

Future implementation may persist one setup-detection result per ticker as:

* `setup_detection.json`

## Architectural position

Setup Detection sits after Fundamental Observations and Derived Observations, and before Analysis Review.

Target chain:

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

Analysis Review and Recommendation Review already exist before Setup Detection was formally added.

That completed work remains valid.

Future Analysis Review and Recommendation Review sprints may extend the existing layers to consume Setup Detection output.

## Non-actionable boundary

Setup Detection only detects and describes setups.

Setup Detection must not emit:

* BUY
* SELL
* HOLD
* strong buy
* strong sell
* accumulate
* trim
* exit
* enter position
* increase position
* reduce position
* take profit
* stop loss
* price target
* target allocation
* position size
* portfolio weight
* conviction score
* urgency score
* tradeability score
* ranking
* top pick
* best candidate
* execute
* order
* rebalance
* send alert
* send Telegram
* publish report

Setup Detection must not mutate:

* portfolio files
* watchlists
* production data
* delivery state
* Decision Engine state

Setup Detection must not call:

* live providers
* SEC
* EDGAR
* yfinance
* Telegram
* reporting pipelines
* Decision Engine actions

## Definition of a setup

A setup is a structured, non-actionable pattern description derived from approved observations.

A setup answers:

* what pattern appears to be present;
* what evidence supports the pattern;
* what evidence is missing;
* whether the setup is complete, partial, weak, conflicted, or blocked;
* which upstream observations were used;
* why the setup is not actionable by itself.

A setup does not answer:

* whether to buy;
* whether to sell;
* whether to hold;
* how much to allocate;
* whether to rebalance;
* whether to execute;
* whether to send a signal or report.

## Initial setup families

ME-SD01 approves the following initial setup families for future implementation.

These families are intentionally fundamental and cash-generation focused because the current approved input layers are Fundamental Observations and Derived Cash Generation Observations.

### Cash generation setup

Detects whether the company appears to generate positive, neutral, negative, incomplete, or conflicting free-cash-flow evidence from approved derived observations.

Purpose:

* identify cash-generative business evidence;
* preserve missing operating-cash-flow or capital-expenditure evidence;
* distinguish positive cash generation from incomplete or blocked evidence.

Non-actionable examples:

* cash generation appears positive;
* cash generation appears neutral;
* cash generation appears negative;
* cash generation evidence is incomplete;
* cash generation cannot be assessed due to missing observations.

### Fundamental availability setup

Detects whether enough fundamental observation families are present to support later review.

Purpose:

* prevent downstream review layers from over-interpreting thin source evidence;
* preserve explicit missingness;
* make data limitations visible before Analysis Review.

Non-actionable examples:

* required observations are available;
* required observations are partially available;
* required observations are blocked by missing data;
* source evidence is insufficient for setup detection.

### Profitability evidence setup

Detects whether approved fundamental observations provide evidence of profitability.

Purpose:

* describe presence, absence, or limitation of profitability evidence;
* preserve references to relevant upstream observations;
* avoid turning profitability evidence into recommendation authority.

Non-actionable examples:

* profitability evidence appears present;
* profitability evidence appears absent;
* profitability evidence is mixed;
* profitability evidence is blocked by missing observations.

### Revenue evidence setup

Detects whether approved fundamental observations provide evidence of revenue availability and interpretable revenue presence.

Purpose:

* describe whether revenue evidence exists;
* preserve missing or unavailable revenue observations;
* provide a setup input for later Analysis Review.

Non-actionable examples:

* revenue evidence is present;
* revenue evidence is missing;
* revenue evidence is not assessable;
* revenue evidence is available but not sufficient for action.

### Balance-sheet evidence setup

Detects whether approved fundamental observations provide enough balance-sheet evidence for later review.

Purpose:

* identify whether balance-sheet observations are available;
* preserve missing debt, cash, equity, or asset observations when available in the observation contract;
* avoid creating leverage recommendations.

Non-actionable examples:

* balance-sheet evidence is available;
* balance-sheet evidence is partial;
* balance-sheet evidence is missing;
* balance-sheet evidence is not assessed.

## Initial setup categories

Future implementation should use explicit setup categories.

Approved initial categories:

* `cash_generation_setup`
* `fundamental_availability_setup`
* `profitability_evidence_setup`
* `revenue_evidence_setup`
* `balance_sheet_evidence_setup`
* `data_limitation_setup`
* `not_assessed_setup`

These categories describe evidence patterns only.

They do not describe actions.

## Initial setup states

Future implementation should use explicit setup states.

Approved initial states:

* `setup_detected`
* `setup_partially_detected`
* `setup_not_detected`
* `setup_conflicted`
* `setup_blocked_by_missing_data`
* `setup_not_assessed`

State semantics:

### setup_detected

The required evidence for the setup is present and internally consistent enough to describe the setup.

This state is not actionable.

### setup_partially_detected

Some evidence supports the setup, but required evidence is incomplete.

This state is not actionable.

### setup_not_detected

The approved observations do not support the setup.

This state is not actionable.

### setup_conflicted

The approved observations contain conflicting evidence.

This state is not actionable.

### setup_blocked_by_missing_data

The setup cannot be assessed because required observations are missing.

This state is not actionable.

### setup_not_assessed

The setup was not assessed by the current implementation scope.

This state is not actionable.

## Required evidence model

Every setup item must define:

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

Future implementation must preserve upstream references.

Future implementation must not flatten missing data into zero.

Future implementation must not treat numeric zero as missing.

## Recommended future dataclass model

Future ME-SD02 implementation may define dataclasses equivalent to:

### SecCompanyFactsSetupDetectionItem

Required fields:

* `category`
* `state`
* `message`
* `input_observation_families`
* `required_observations`
* `missing_observations`
* `source_observation_references`
* `derived_observation_references`
* `setup_evidence`
* `setup_limitations`
* `non_actionable_boundary`

### SecCompanyFactsSetupDetection

Required fields:

* `ticker`
* `cik`
* `provider_name`
* `setup_detection_format_version`
* `fundamental_observation_format_version`
* `derived_observation_format_version`
* `source_context_format_version`
* `source_context_state`
* `source_refresh_snapshot_id`
* `source_refresh_fetched_at`
* `source_refresh_payload_format_version`
* `setup_detection_run_id`
* `setup_items`
* `warnings`
* `non_actionable_boundary`

## Required provenance

Future Setup Detection output must preserve:

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

Setup Detection must preserve missing-data semantics.

Rules:

* Missing observations must be explicit.
* Missing observations must not be converted to zero.
* Numeric zero must not be treated as missing.
* Missing upstream evidence must result in `setup_blocked_by_missing_data` or `setup_partially_detected`, depending on available evidence.
* Setup Detection must not invent values for missing observations.
* Setup Detection must not hide missing data in generic warning text only.

## Evidence policy

Setup Detection must remain source-grounded.

Every setup item must be traceable to approved input observations.

Allowed evidence sources:

* Fundamental Observations from `sec-companyfacts-fundamental-observations-v1`;
* Derived Cash Generation Observations from `sec-companyfacts-derived-cash-generation-observations-v1`.

Not allowed as evidence sources in ME-SD01:

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

Those may be considered in future contracts only if explicitly approved in future sprints.

## Future builder requirements for ME-SD02

ME-SD02 should implement a builder equivalent to:

* `build_sec_companyfacts_setup_detection(...)`

The builder should:

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

## Future persistence requirements for ME-SD02

ME-SD02 may implement a persistence helper equivalent to:

* `persist_sec_companyfacts_setup_detection(...)`

Persistence should write:

* `<root>/<run_id>/<ticker>/setup_detection.json`

Persistence must:

* refuse overwrite;
* write deterministic JSON;
* preserve provenance;
* preserve missing-data state;
* avoid production writes unless explicitly scoped through local test roots.

## Expected ME-SD02 test requirements

ME-SD02 must use local synthetic tests only.

Tests should cover:

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

## Relationship to Analysis Review

Setup Detection output is intended as a future input to Analysis Review.

ME-AR03 should define how Analysis Review consumes:

* Fundamental Observations;
* Derived Observations;
* Setup Detection output.

ME-AR04 should implement that behavior.

Setup Detection itself does not replace Analysis Review.

Setup Detection identifies evidence patterns.

Analysis Review evaluates whether the detected setups are sufficiently supported, limited, conflicted, or blocked for downstream review.

## Relationship to Recommendation Review

Recommendation Review should not consume Setup Detection directly in the first future extension unless explicitly approved.

Preferred sequence:

* ME-SD01 defines Setup Detection contract;
* ME-SD02 implements Setup Detection;
* ME-AR03 extends Analysis Review contract for Setup Detection input;
* ME-AR04 implements Analysis Review consumption of Setup Detection;
* ME-RR03 extends Recommendation Review contract for Setup Detection-aware Analysis Review;
* ME-RR04 implements Setup Detection-aware Recommendation Review behavior.

This preserves the authority chain.

## Relationship to Portfolio Review

Portfolio Review must remain downstream of Recommendation Review.

Portfolio Review should only be defined after Setup Detection-aware Analysis Review and Recommendation Review are defined.

ME-PR01 must not be moved ahead of ME-SD01, ME-SD02, ME-AR03, ME-AR04, ME-RR03, and ME-RR04 unless the roadmap and backlog document a governance-approved insertion or deferral reason.

## Relationship to Decision Engine

Setup Detection has no Decision Engine authority.

Setup Detection must not:

* call the Decision Engine;
* simulate Decision Engine outcomes;
* produce action decisions;
* produce allocation decisions;
* produce execution instructions;
* trigger downstream delivery.

Decision Engine authority remains separate and must be defined later through ME-DE01 and ME-DE02.

## Relationship to Delivery and Reporting

Setup Detection must not produce Telegram output, reports, alerts, dashboards, or published delivery payloads.

Delivery and reporting remain later job families.

ME-DL01 and ME-DL02 must define delivery/reporting behavior only after upstream authority boundaries are defined.

## Naming decision

The approved job family prefix is:

* `ME-SD`

Meaning:

* Setup Detection

Do not use:

* `ME-PS`

Rationale:

Setup Detection is the clearer job-family name because the layer detects structured setups built from observed patterns.

## Approved future sprint sequence after ME-SD01

The preserved future sequence remains:

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

## Insertion rule

Future sprints identified as logical next steps must be preserved in the roadmap and backlog.

A sprint may be inserted ahead of the planned sequence only when a real problem, blocker, architectural gap, governance risk, test gap, data-quality issue, or newly discovered dependency requires it.

If a sprint is inserted, the roadmap and backlog must document:

* inserted sprint name;
* insertion position;
* insertion reason;
* affected planned sprint;
* whether the existing sequence remains valid.

## ME-SD01 acceptance criteria

ME-SD01 is complete when:

* the Setup Detection job-family boundary is defined;
* the input contracts are defined;
* the output contract is defined;
* setup categories are defined;
* setup states are defined;
* missing-data policy is defined;
* evidence requirements are defined;
* provenance requirements are defined;
* future ME-SD02 implementation requirements are defined;
* future ME-SD02 test requirements are defined;
* relationships to Analysis Review, Recommendation Review, Portfolio Review, Decision Engine, and Delivery are defined;
* the backlog is updated;
* the roadmap is updated;
* no runtime code is changed;
* no tests are changed;
* no providers, data files, portfolio files, Telegram/reporting files, or Decision Engine files are changed.
