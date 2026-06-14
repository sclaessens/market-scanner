# Market Engine Financial, Scanner, and Fundamental Logic

Owner role: Financial Analyst / Functional Analyst / Data Steward

Status: ME03 SPECIFICATION DRAFT

## Purpose

This document defines the Market Engine financial, scanner, and fundamental logic specification.

It translates the Market Engine functional flow into concrete analysis domains, data/source principles, scanner-context rules, fundamental-context rules, missing-data rules, and implementation/testing implications.

This document does not authorize implementation. It prepares ME04 technical architecture and ME05 all-ticker source intake smoke.

## Scope Of ME03

ME03 defines what Market Engine must understand before implementation begins.

In scope:

* financial-analysis concepts;
* scanner-context concepts;
* fundamental-context concepts;
* provider/source-readiness lessons;
* missing-data and quality-state rules;
* ticker failure handling;
* data coverage principles;
* allowed pre-decision outputs;
* prohibited outputs and side effects;
* keep / reject / defer decisions from old logic;
* implications for ME04 architecture;
* implications for ME05 all-ticker source intake smoke.

Out of scope:

* Python implementation;
* test implementation;
* live provider calls;
* yfinance calls;
* SEC / EDGAR calls;
* runtime execution;
* report generation;
* Telegram delivery;
* portfolio mutation;
* watchlist mutation;
* Decision Engine changes;
* BUY / SELL / HOLD logic;
* allocation;
* urgency;
* conviction;
* tradeability;
* ranking as hidden recommendation;
* price targets as action triggers;
* portfolio transaction advice.

## Relationship To ME02 Functional Flow

ME03 uses the ME02 functional flow as its baseline.

The relevant ME02 stages are:

```text
operator intent
-> ticker universe / watchlist selection
-> source intake request
-> provider/source access
-> source coverage validation
-> raw source result preservation
-> normalized data view
-> missing-data and quality-state handling
-> scanner context preparation
-> fundamental context preparation
-> first analysis pass
-> risk and quality flags
-> local operator review output
-> later optional decision/reporting/notification layers
```

ME03 focuses on the middle part of this flow:

```text
source coverage validation
-> raw source result preservation
-> normalized data view
-> missing-data and quality-state handling
-> scanner context preparation
-> fundamental context preparation
-> first analysis pass preparation
```

ME03 does not define final decisions. It defines descriptive, auditable, review-oriented inputs for later analysis and operator review.

## Core Doctrine

Market Engine must preserve the following doctrine:

* Classification upstream.
* Allocation downstream.
* Source readiness is not investment quality.
* Scanner context is not recommendation logic.
* Fundamental context is not recommendation logic.
* Analysis preparation is not Decision Engine behavior.
* Missing data remains missing.
* Raw evidence, normalized data, generated output, reporting output, and local-only output are separate roles.
* Ticker-level failures must not stop the full batch.
* Early layers must be side-effect-free unless a later architecture explicitly authorizes a bounded action.
* Decision Engine remains the only authority for final action, allocation, execution semantics, urgency, conviction, tradeability, and arbitration.
* Reporting and Telegram are downstream communication layers only.

## Product-Level Financial Intent

Market Engine exists to support disciplined investment review, not uncontrolled recommendations.

Financial logic should help the operator understand:

* what kind of business a company is;
* whether financial evidence is available and usable;
* whether growth, profitability, balance sheet, and cash generation are directionally understandable;
* whether source evidence is complete, partial, stale, missing, invalid, or review-required;
* which uncertainties must remain visible;
* which companies require further review because of evidence gaps;
* which descriptive observations can later feed an approved analysis or Decision Engine boundary.

Financial logic must not decide whether to buy, sell, hold, increase, reduce, avoid, rotate, size, prioritize, or execute.

## Financial Concepts To Support

Market Engine should eventually support the following financial concept groups.

### 1. Business Understanding

Purpose:
Describe what the company does and which economic drivers matter.

Possible descriptive fields:

* business model summary;
* sector;
* industry;
* revenue drivers;
* geographic exposure;
* cyclicality;
* regulatory exposure;
* technology exposure;
* commodity exposure;
* customer concentration when available;
* supplier concentration when available;
* business-model clarity state.

Allowed descriptive states:

* `AVAILABLE`
* `PARTIAL`
* `MISSING`
* `REVIEW_REQUIRED`
* `INSUFFICIENT_DATA`

Not allowed:

* business quality score as final investment approval;
* moat score as hidden recommendation;
* buy/sell/hold interpretation;
* allocation priority.

ME decision:
Keep business understanding as descriptive context. Defer exact field set and provider source to ME04/ME05.

### 2. Growth Profile

Purpose:
Describe whether available evidence shows growth, stagnation, decline, volatility, or insufficient data.

Possible descriptive fields:

* revenue trend;
* earnings trend;
* operating income trend;
* free cash flow trend;
* margin trend;
* segment growth when available;
* backlog or order growth when source evidence supports it;
* growth evidence period count;
* missing growth periods;
* growth evidence state.

Allowed descriptive states:

* `IMPROVING`
* `DETERIORATING`
* `STABLE`
* `VOLATILE`
* `PARTIAL`
* `INSUFFICIENT_DATA`
* `REVIEW_REQUIRED`

Not allowed:

* growth score as recommendation;
* growth ranking as hidden opportunity priority;
* buy because growth is positive;
* sell because growth is negative.

ME decision:
Keep growth profile as descriptive financial evidence. Reject automatic action triggers. Defer exact trend formulas.

### 3. Profitability Profile

Purpose:
Describe whether a company appears profitable, pressured, improving, deteriorating, volatile, or insufficiently documented.

Possible descriptive fields:

* gross margin;
* operating margin;
* net margin;
* return on equity;
* return on invested capital;
* free cash flow margin;
* profitability consistency;
* one-off or exceptional effect notes when available;
* profitability evidence state.

Allowed descriptive states:

* `STRUCTURALLY_PROFITABLE`
* `TEMPORARILY_PRESSURED`
* `IMPROVING`
* `DETERIORATING`
* `VOLATILE`
* `PARTIAL`
* `INSUFFICIENT_DATA`
* `REVIEW_REQUIRED`

Not allowed:

* quality score as final action;
* profitability threshold producing buy/sell/hold;
* hidden ranking by margin.

ME decision:
Keep profitability concepts as review evidence. Defer exact metric thresholds and classification formulas.

### 4. Balance Sheet And Financial Risk

Purpose:
Describe resilience, fragility, leverage, liquidity, and financial-risk evidence.

Possible descriptive fields:

* cash and equivalents;
* total debt;
* net debt or net cash when derivable;
* leverage evidence;
* interest coverage when available;
* liquidity position;
* working capital stress indicators;
* dilution or share issuance evidence;
* refinancing risk notes when explicitly supported by source evidence;
* balance sheet risk state.

Allowed descriptive states:

* `RESILIENT`
* `LEVERAGED`
* `FRAGILE`
* `IMPROVING`
* `DETERIORATING`
* `PARTIAL`
* `INSUFFICIENT_DATA`
* `REVIEW_REQUIRED`

Not allowed:

* automatic exclusion;
* final risk score as recommendation;
* portfolio action based on leverage alone.

ME decision:
Keep financial risk as explicit context. Reject opaque single-score risk behavior. Defer exact calculations.

### 5. Cash Generation

Purpose:
Distinguish accounting earnings from actual cash generation.

Possible descriptive fields:

* operating cash flow;
* capital expenditures;
* free cash flow;
* free cash flow derivation status;
* free cash flow conversion;
* working capital impact;
* reinvestment intensity;
* cash generation stability;
* cash generation evidence state.

Mandatory rule:
Free cash flow must only be derived when the required components are available and valid. Missing capex or operating cash flow must remain missing, partial, or review-required. Missing values must not be converted to zero.

Allowed descriptive states:

* `AVAILABLE`
* `DERIVED`
* `PARTIAL`
* `MISSING_COMPONENTS`
* `STALE`
* `INSUFFICIENT_DATA`
* `REVIEW_REQUIRED`

Not allowed:

* deriving free cash flow from incomplete data;
* treating missing capex as zero;
* treating unavailable cash flow as neutral.

ME decision:
Keep cash generation as a core fundamental area. Preserve strict missing-data rules. Prioritize this area for ME05/ME06 source-readiness observation.

### 6. Valuation Context

Purpose:
Provide context for how the market prices the company, without turning valuation into an action rule.

Possible descriptive fields:

* price-to-earnings context;
* enterprise value to EBITDA context;
* price-to-sales context;
* free cash flow yield context;
* valuation versus own history when available;
* valuation versus peer group when approved peer data exists;
* valuation versus growth context;
* valuation evidence state.

Allowed descriptive states:

* `AVAILABLE`
* `PARTIAL`
* `STALE`
* `EXPENSIVE_CONTEXT`
* `CHEAP_CONTEXT`
* `MIXED_CONTEXT`
* `INSUFFICIENT_DATA`
* `REVIEW_REQUIRED`

Important:
Cheap is not automatically attractive. Expensive is not automatically unacceptable. Valuation context requires interpretation and later Decision Engine governance before any action can exist.

Not allowed:

* buy because cheap;
* sell because expensive;
* target price as action authority;
* valuation rank as hidden recommendation.

ME decision:
Keep valuation as context only. Defer valuation thresholds, peer logic, and target-price behavior.

### 7. Capital Allocation And Shareholder Returns

Purpose:
Describe how a company uses capital.

Possible descriptive fields:

* dividends;
* buybacks;
* reinvestment;
* acquisitions;
* debt reduction;
* dilution;
* capital intensity;
* shareholder return evidence state.

Allowed descriptive states:

* `AVAILABLE`
* `PARTIAL`
* `MISSING`
* `REVIEW_REQUIRED`

Not allowed:

* portfolio allocation decisions;
* buyback or dividend behavior creating recommendation;
* shareholder return score as final action.

ME decision:
Keep as informational context. Defer depth until reliable source coverage exists.

### 8. Portfolio Relevance

Purpose:
Market Engine may later show whether a company is relevant to the operator’s current portfolio or watchlist, but only as read-only context.

Possible descriptive fields:

* held/not held;
* watchlist presence;
* portfolio-adjacent review reason;
* sector exposure context;
* position visibility context;
* source reference to approved portfolio input.

Mandatory boundary:
Portfolio records are source or normalized input only. Generated review, reporting input, Telegram text, and local operator output must never become portfolio source truth.

Not allowed:

* portfolio mutation;
* watchlist mutation;
* allocation instructions;
* execution instructions;
* urgency;
* conviction;
* tradeability;
* ranking;
* recommendation text;
* reporting-only authority;
* Telegram message text.

ME decision:
Keep portfolio relevance as possible read-only context. Defer exact portfolio/watchlist input boundary to ME04 or later.

## Scanner Concepts To Support

Scanner logic in Market Engine must describe market/setup context. It must not become source access, scoring, ranking, recommendation, or trade planning unless a later approved boundary explicitly owns that behavior.

### Scanner Context Purpose

Scanner context should help answer:

* Why is this ticker under review?
* What setup/context does the ticker appear to have?
* Is there available market/context evidence?
* Is scanner evidence complete, partial, stale, missing, or review-required?
* Which descriptive context should be visible to the operator before analysis?

### Scanner Concepts To Keep

Market Engine may keep these concepts as descriptive context:

* universe membership;
* selection reason;
* discovery reason;
* candidate identity;
* setup category as descriptive label;
* liquidity context;
* trend context;
* momentum context;
* relative strength context;
* volatility context;
* timing context as non-actionable evidence;
* market regime context when approved;
* source readiness for scanner evidence;
* missing scanner data flags;
* review-required scanner flags.

These concepts must remain descriptive and traceable.

### Scanner Concepts To Reject

Market Engine must reject old scanner behavior that mixes context with authority.

Reject:

* implicit yfinance/provider calls inside scanner logic;
* scanner-triggered network access;
* scanner-triggered file writes;
* hidden filtering that removes rows without explanation;
* ranking as allocation priority;
* grading as investment recommendation;
* tradeability fields in scanner output;
* urgency fields in scanner output;
* conviction fields in scanner output;
* entry price as recommendation;
* stop loss as recommendation;
* price target as action authority;
* risk/reward as trade instruction;
* BUY / SELL / HOLD in scanner output;
* Telegram/reporting output from scanner layer;
* portfolio/watchlist mutation from scanner layer;
* Decision Engine invocation from scanner layer.

### Scanner Concepts To Defer

Defer:

* exact setup taxonomy;
* exact trend formula;
* exact momentum formula;
* exact relative strength calculation;
* exact liquidity thresholds;
* exact volatility thresholds;
* exact timing-state model;
* exact market-regime dependency;
* any ranking or prioritization model;
* any trade-plan-shaped fields.

These may be revisited only after ME04 defines module boundaries and ME05/ME06 establish source coverage realities.

### Scanner Output Rules

Allowed scanner-context outputs:

* ticker identity;
* universe membership;
* selection reason;
* descriptive setup context;
* descriptive trend/momentum/liquidity context;
* scanner evidence state;
* scanner missing-data notes;
* scanner source references;
* scanner review flags.

Forbidden scanner-context outputs:

* buy;
* sell;
* hold;
* recommendation;
* allocation;
* execution;
* urgency;
* conviction;
* tradeability;
* hidden rank;
* final action;
* Telegram text;
* reporting text;
* portfolio mutation;
* watchlist mutation.

ME decision:
Keep descriptive scanner context. Reject mixed scanner/provider/recommendation behavior. Defer exact scanner semantics until architecture and source availability are clearer.

## Fundamental Concepts To Support

Fundamental logic in Market Engine must describe company and financial evidence using governed source evidence and normalized data.

### Fundamental Context Purpose

Fundamental context should help answer:

* Is fundamental data available for this ticker?
* Which fields are available, partial, stale, missing, invalid, or review-required?
* Which periods are covered?
* Which source produced the evidence?
* Which metrics can be derived safely?
* Which metrics cannot be derived because components are missing?
* Which financial observations can be shown without becoming final actions?

### Fundamental Concepts To Keep

Keep:

* raw evidence preservation;
* normalized fundamentals records;
* source provenance;
* period metadata;
* source readiness records;
* missing field names;
* data freshness;
* data completeness;
* income statement evidence;
* balance sheet evidence;
* cash flow evidence;
* shares outstanding when available;
* market price context when approved;
* revenue evidence;
* income evidence;
* cash flow evidence;
* capex evidence;
* free cash flow derivation status;
* growth evidence;
* profitability evidence;
* balance sheet risk evidence;
* valuation context when source coverage allows;
* review-required flags.

### Fundamental Concepts To Reject

Reject:

* missing numeric values converted to zero;
* derived metrics from incomplete components;
* source readiness interpreted as company quality;
* normalized fundamentals interpreted as recommendation;
* final action fields in fundamental records;
* allocation fields in fundamental records;
* urgency fields in fundamental records;
* conviction fields in fundamental records;
* tradeability fields in fundamental records;
* score/ranking fields as hidden recommendation;
* target price as action authority;
* threshold fields that imply action;
* report message fields;
* Telegram message fields;
* provider calls hidden inside analysis or tests.

### Fundamental Concepts To Defer

Defer:

* exact full fundamental schema;
* exact SEC CompanyFacts field aliases;
* exact yfinance field mapping;
* exact period alignment rules;
* exact stale-data thresholds;
* exact peer comparison rules;
* exact valuation metrics for first analysis pass;
* exact formula library for all derived metrics;
* exact source priority model;
* exact fallback-provider strategy.

ME decision:
Keep fundamental context as descriptive source-backed evidence. Reject authority fields and incomplete derivations. Defer exact provider mapping and formula implementation to ME04/ME05.

## Provider And Source-Readiness Principles

Market Engine must treat provider/source access as a governed boundary.

### Source Roles

Market Engine must keep separate:

* raw source evidence;
* normalized data input;
* generated analysis output;
* reporting output;
* Telegram output;
* local operator review output;
* portfolio source records;
* watchlist input records.

Generated output must not silently become source truth.

### Source Readiness Is Not Investment Quality

Source readiness describes whether evidence can be used. It does not describe whether the company is attractive.

Source readiness may describe:

* source available;
* source unavailable;
* provider error;
* unsupported ticker;
* partial response;
* stale response;
* invalid response;
* missing required fields;
* review required;
* insufficient data.

A company with poor source readiness is not automatically bad. A company with good source readiness is not automatically good.

### Provider Access Boundary

Provider access must be:

* explicit;
* bounded;
* reviewable;
* separated from analysis;
* separated from decisions;
* separated from tests;
* separated from imports;
* separated from reporting and Telegram;
* separated from portfolio/watchlist mutation.

Normal automated tests must not execute live provider calls.

Manual smoke harnesses may execute provider calls only when explicitly authorized by architecture and sprint scope.

## Data Coverage Principles

Market Engine must preserve coverage evidence per ticker.

For each ticker, future source intake should be able to record:

* ticker identity;
* provider/source requested;
* source response status;
* coverage status;
* missing required fields;
* missing optional fields;
* stale fields;
* invalid fields;
* unsupported ticker status;
* provider error status;
* raw evidence reference;
* normalized record status;
* review-required status.

Coverage results must be batch-safe. A single ticker failure must not stop the whole batch.

Coverage evidence may support ME06 triage, but it must not become investment ranking.

## Missing-Data Rules

Missing data must remain explicit.

Mandatory rules:

* missing numeric values must not become zero;
* missing text values must not become empty truth;
* missing periods must not be silently interpolated;
* missing components must block unsafe derived metrics;
* unavailable metrics must be labelled unavailable, partial, stale, missing, invalid, insufficient, or review-required;
* uncertainty must be communicated;
* row identity must be preserved even when fields are missing;
* generated output must not hide source gaps;
* missing-data status must travel downstream as evidence.

Examples:

* Missing capex means free cash flow cannot be safely derived.
* Missing operating cash flow means free cash flow cannot be safely derived.
* Missing revenue period means growth trend must be partial or insufficient.
* Missing current price means valuation context must be unavailable or review-required.
* Missing portfolio display value must remain unavailable, not zero.

## Quality-State Rules

Market Engine may use quality-state language only when the meaning is explicit.

Allowed quality-state meanings:

* source quality;
* data quality;
* evidence completeness;
* freshness;
* validity;
* review readiness;
* analysis readiness.

Forbidden quality-state meanings in early layers:

* investment quality;
* buy quality;
* sell quality;
* portfolio priority;
* tradeability;
* conviction;
* urgency;
* allocation eligibility.

Suggested source/data states:

* `AVAILABLE`
* `MISSING`
* `PARTIAL`
* `STALE`
* `INVALID`
* `UNSUPPORTED`
* `PROVIDER_ERROR`
* `INSUFFICIENT_DATA`
* `REVIEW_REQUIRED`

Suggested trend/context states:

* `IMPROVING`
* `DETERIORATING`
* `STABLE`
* `VOLATILE`
* `MIXED`
* `INSUFFICIENT_DATA`
* `REVIEW_REQUIRED`

All states must be descriptive. They must not imply final action.

## Ticker Failure Handling

Market Engine must be resilient to ticker-level failure.

Rules:

* one ticker failure must not stop the full batch;
* failures must be recorded per ticker;
* failure status must preserve ticker identity;
* failure reason must be explicit where possible;
* provider errors must be distinguishable from missing data;
* unsupported tickers must be distinguishable from temporary provider failure;
* parse failures must not become zero values;
* failed tickers must remain visible in operator review output;
* failures must support ME06 triage.

Suggested failure categories:

* `TICKER_UNSUPPORTED`
* `PROVIDER_UNAVAILABLE`
* `PROVIDER_ERROR`
* `SOURCE_EMPTY`
* `SOURCE_PARTIAL`
* `SOURCE_STALE`
* `SOURCE_INVALID`
* `REQUIRED_FIELD_MISSING`
* `NORMALIZATION_FAILED`
* `DERIVATION_BLOCKED`
* `REVIEW_REQUIRED`

## Source Intake Versus Analysis Boundary

Source intake may produce:

* source request plan;
* provider/source response status;
* raw evidence;
* provenance;
* freshness;
* coverage state;
* missing-field state;
* normalized data view;
* source readiness state.

Source intake must not produce:

* financial conclusion;
* scanner conclusion;
* final action;
* recommendation;
* allocation;
* urgency;
* conviction;
* tradeability;
* ranking;
* report message;
* Telegram message;
* portfolio/watchlist mutation.

Analysis may begin only after governed evidence and normalized/context records exist.

## Analysis Versus Decision Boundary

First analysis pass may produce:

* descriptive financial observations;
* descriptive scanner observations;
* evidence summaries;
* limitation flags;
* review-required flags;
* missingness summaries;
* source-readiness summaries;
* analysis readiness notes;
* local operator review inputs.

First analysis pass must not produce:

* BUY;
* SELL;
* HOLD;
* final action;
* allocation;
* execution instruction;
* position sizing;
* urgency;
* conviction;
* tradeability;
* hidden ranking;
* target-price action;
* portfolio transaction advice.

Decision Engine remains the only future authority for final action behavior.

## Allowed Pre-Decision Outputs

Allowed early Market Engine outputs:

* source coverage summary;
* source readiness state;
* raw evidence reference;
* normalized data view;
* missing-data summary;
* scanner context record;
* fundamental context record;
* business context note;
* growth context note;
* profitability context note;
* balance sheet context note;
* cash generation context note;
* valuation context note;
* portfolio relevance note as read-only context when approved;
* risk/limitation flag;
* review-required flag;
* local operator review input.

These outputs are review support only.

## Prohibited Outputs And Side Effects

Prohibited in source/data/scanner/fundamental/first-analysis layers:

* BUY / SELL / HOLD;
* recommendation;
* final action;
* allocation;
* position sizing;
* execution instruction;
* urgency;
* conviction;
* tradeability;
* hidden ranking;
* target-price action trigger;
* portfolio transaction advice;
* Telegram text;
* report generation;
* production writes;
* portfolio mutation;
* watchlist mutation;
* Decision Engine invocation;
* provider calls from imports;
* live provider calls in automated tests;
* generated output becoming source truth.

## Keep / Reject / Defer Decisions

### Keep

* operator visibility;
* preserved ticker identity;
* preserved source evidence;
* source readiness separate from investment quality;
* raw evidence and normalized data separation;
* generated output and source truth separation;
* explicit missingness;
* batch-safe ticker failure handling;
* scanner context as descriptive evidence;
* fundamental context as descriptive evidence;
* financial analysis as upstream classification;
* Decision Engine authority protection;
* reporting and Telegram as downstream communication only;
* portfolio/watchlist as read-only context unless later authorized;
* fake/synthetic provider responses for automated tests.

### Reject

* blind copying of old script-era code;
* old quick scripts as canonical runtime;
* implicit provider access inside scanner or tests;
* yfinance calls from scanner context without explicit source boundary;
* SEC / EDGAR calls from tests or hidden runtime paths;
* missing values converted to zero;
* generated outputs as source truth;
* scanner ranking as hidden allocation;
* trade-plan-shaped scanner output as early-layer authority;
* BUY / SELL / HOLD leakage;
* recommendation leakage;
* reporting/Telegram/portfolio/watchlist/Decision Engine side effects in lower layers.

### Defer

* exact ticker universe source for ME05;
* exact first approved provider/source family for all-ticker smoke;
* exact scanner setup taxonomy;
* exact fundamental schema;
* exact provider field mappings;
* exact SEC CompanyFacts aliases;
* exact stale-data thresholds;
* exact formula library;
* exact valuation methodology;
* exact peer comparison logic;
* exact local operator review format;
* optional downstream decision/reporting/notification integration.

## Implementation Implications For ME04

ME04 must translate this specification into technical architecture.

ME04 must define:

* module ownership boundaries;
* source/provider access ownership;
* raw evidence model ownership;
* normalization ownership;
* scanner context ownership;
* fundamental context ownership;
* analysis ownership;
* local operator review ownership;
* forbidden field policy;
* side-effect policy;
* file strategy;
* test-family placement;
* manual smoke harness rules;
* separation between source intake, analysis, Decision Engine, reporting, Telegram, portfolio, and watchlist.

ME04 must explicitly prevent the creation of new Python files for every step unless a new ownership boundary justifies it.

ME04 must not authorize hidden provider access, production writes, Telegram delivery, portfolio/watchlist mutation, or Decision Engine behavior from lower layers.

## Source-Intake Implications For ME05

ME05 should build only an explicit all-ticker source intake smoke after ME04 authorizes the architecture.

ME05 should focus on:

* source availability;
* provider response status;
* coverage by ticker;
* failure by ticker;
* raw evidence preservation;
* normalized data feasibility;
* missing required fields;
* derivation-blocked metrics;
* source readiness;
* review-required states.

ME05 must not include:

* recommendation logic;
* BUY / SELL / HOLD;
* allocation;
* urgency;
* conviction;
* tradeability;
* portfolio/watchlist mutation;
* reporting;
* Telegram;
* Decision Engine behavior;
* production pipeline integration;
* normal automated tests with live calls.

ME05 output should feed ME06 triage without becoming source truth by default.

## Testing Implications

Future tests must prove:

* missing numeric values remain missing;
* missing values do not become zero;
* missing components block derived metrics;
* ticker-level failures do not stop the full batch;
* provider errors are recorded per ticker;
* unsupported tickers are distinguishable from provider errors;
* raw evidence remains separate from normalized data;
* normalized data remains separate from generated analysis output;
* generated output does not become source truth;
* scanner context contains no BUY / SELL / HOLD;
* scanner context contains no allocation, urgency, conviction, or tradeability;
* fundamental context contains no authority fields;
* analysis output contains no final action or capital-action semantics;
* lower layers do not mutate portfolio/watchlist files;
* lower layers do not generate reports;
* lower layers do not send Telegram;
* normal automated tests use fake/synthetic provider responses;
* live provider access is limited to explicit manual smoke harnesses.

## Open Questions

* Which ticker universe should ME05 use first?
* Which provider/source family should be the first all-ticker smoke target?
* Which fundamental fields are mandatory for the first smoke?
* Which scanner fields are safe enough for first descriptive context?
* Should ME05 preserve raw evidence files, summary records, or both?
* What exact source-readiness state enum should ME04 standardize?
* What exact local operator review format should ME08 produce?
* When, if ever, should portfolio/watchlist read-only context enter Market Engine?

## Readiness Criteria For ME04

ME04 is ready when this document is accepted as the financial/scanner/fundamental logic baseline.

ME04 must use this document to define:

* architecture boundaries;
* module ownership;
* file strategy;
* provider access separation;
* source evidence model;
* normalization boundaries;
* scanner and fundamental context boundaries;
* forbidden field policy;
* test-family strategy;
* manual smoke harness policy.

## Readiness Criteria For ME05

ME05 is ready only after ME04 defines the architecture.

ME05 must use this document to limit the first all-ticker smoke to source intake, source coverage, failure observation, raw evidence, normalized data feasibility, missingness, and readiness states.

ME05 must not implement analysis, recommendation, reporting, Telegram, portfolio mutation, watchlist mutation, or Decision Engine behavior.
