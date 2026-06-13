# Market Engine Functional Flow

Owner role: Functional Analyst / Product Owner

Status: ME02 FUNCTIONAL SPECIFICATION

## Scope Of ME02

ME02 defines the functional flow for Market Engine from operator intent through local operator review output.

This document uses existing repository documentation, code, tests, audits, and backlog records as reference material only. It extracts useful functional lessons and boundaries. It does not authorize implementation.

## Non-Goals

ME02 does not:

- modify Python code;
- modify tests;
- execute provider calls;
- execute yfinance, SEC, EDGAR, scanner, reporting, Telegram, portfolio, watchlist, or runtime commands;
- generate reports;
- send Telegram messages;
- mutate portfolio or watchlist data;
- change Decision Engine behavior;
- migrate old script-era code;
- make old runtime scripts the Market Engine foundation.

## Product Definition

Market Engine is a local operator decision-support product that turns an approved ticker universe and approved source evidence into auditable, review-oriented market context.

Functionally, Market Engine must help the operator:

- choose or confirm the universe to inspect;
- request source intake in a bounded and reviewable way;
- understand source coverage, freshness, missingness, and provenance;
- preserve raw source evidence before interpretation;
- view normalized data without confusing it with raw evidence;
- see scanner and fundamental context without hidden filtering;
- see analysis limitations before any decision layer;
- receive local operator review output that communicates evidence and gaps.

Market Engine is not a broker, trading bot, production automation shortcut, Telegram sender, portfolio mutator, or recommendation engine in the early flow.

## Operator Workflow

The intended operator workflow is:

1. The operator starts with an intent, such as reviewing a configured universe, a watchlist, a portfolio-adjacent set, or a manually supplied ticker list.
2. Market Engine resolves the intended universe under an approved input contract.
3. Market Engine requests source intake for each ticker through an explicit provider/source boundary.
4. Source intake captures source coverage and raw evidence, including failures, without stopping the whole batch.
5. Market Engine creates a normalized data view only from source evidence that can be mapped under an approved contract.
6. Market Engine marks missing, partial, stale, invalid, unavailable, or review-required data explicitly.
7. Market Engine prepares scanner context and fundamental context as descriptive evidence.
8. Market Engine performs a first analysis pass only after source and context boundaries are clear.
9. Market Engine exposes risk, quality, readiness, and limitation flags for local review.
10. Market Engine produces local operator review output that is communication only.
11. Later optional layers may route approved analysis into decision, reporting, and notification boundaries after later sprints authorize those flows.

## Functional Input Types

Market Engine functional inputs may include:

- operator intent;
- configured ticker universe references;
- manual ticker lists;
- watchlist references as read-only input when approved;
- portfolio references as read-only context when approved;
- governed source/provider responses;
- raw source evidence;
- normalized fundamentals records;
- scanner candidate input records;
- source readiness records;
- synthetic fixtures for tests;
- local-only smoke evidence when explicitly approved.

Generated outputs, reports, Telegram text, old processed CSVs, old logs, and old script-era artifacts are reference material only unless a future contract explicitly reclassifies them.

## Functional Output Types

Early Market Engine outputs may include:

- universe resolution summary;
- source intake coverage summary;
- raw source evidence references;
- normalized data view;
- source readiness states;
- missing-data and quality-state notes;
- scanner context records;
- fundamental context records;
- first analysis pass records;
- risk, readiness, and limitation flags;
- local operator review output.

Early Market Engine outputs must not include BUY / SELL / HOLD recommendations, allocation instructions, execution urgency, conviction, tradeability, hidden ranking, Telegram delivery, production reports, or portfolio/watchlist mutations.

## High-Level Flow

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

## Required Stages

| Stage | Functional purpose | Owner role | Output | Boundary |
|---|---|---|---|---|
| 1. Operator intent | Capture what the operator wants reviewed. | Operator / User, Product Owner | Intent record or run request concept | No provider access, no runtime side effects. |
| 2. Ticker universe / watchlist selection | Resolve the ticker set to inspect. | Product Owner, Functional Analyst, Data Steward | Universe selection summary | Watchlist may be read-only input only when approved. |
| 3. Source intake request | Define which source evidence is requested for each ticker. | Data Steward, Functional Analyst | Intake request plan | No recommendation semantics. |
| 4. Provider/source access | Reach approved sources only when a later sprint authorizes it. | Data Steward, Development Lead | Provider/source response | ME02 documents only; no provider calls. |
| 5. Source coverage validation | Classify source availability and failures. | Data Steward, QA / Test Lead | Coverage and readiness state | Source readiness is not investment quality. |
| 6. Raw source result preservation | Preserve raw evidence and provenance before interpretation. | Data Steward | Raw evidence reference | Raw evidence is not program-ready input by itself. |
| 7. Normalized data view | Map approved evidence into program-ready fields. | Data Steward, Technical Architect | Normalized data records | Missing values remain missing. |
| 8. Missing-data and quality-state handling | Make gaps and review needs explicit. | Data Steward, Financial Analyst | Missingness and review states | Do not convert missing values to zero. |
| 9. Scanner context preparation | Prepare descriptive market/setup context. | Functional Analyst, Technical Architect | Scanner context records | No hidden filtering or allocation. |
| 10. Fundamental context preparation | Prepare descriptive company/fundamental context. | Financial Analyst, Data Steward | Fundamental context records | No investment recommendation. |
| 11. First analysis pass | Combine governed evidence into review-oriented analysis. | Financial Analyst, Development Lead | Analysis records with limitations | No final actions. |
| 12. Risk and quality flags | Surface limitations, source risks, and context risks. | Functional Analyst, QA / Test Lead | Review flags | Flags do not allocate. |
| 13. Local operator review output | Communicate evidence and limitations locally. | Product Owner, Operator / User | Local review output | Communication only. |
| 14. Later optional layers | Future decision, reporting, and notification. | Technical Architect, Governance Auditor | Deferred outputs | Requires later authorization. |

## Stage Descriptions

### 1. Operator Intent

The operator intent stage identifies the review purpose and requested universe. It should be simple and explicit: review all approved tickers, review a watchlist, review portfolio-adjacent names, or review a manual ticker set.

Implementation implication: ME04 must define the entrypoint and input contract before implementation.

Testing implication: future tests should prove that intent parsing or selection does not trigger provider calls, file writes, reports, Telegram, or portfolio/watchlist mutation.

### 2. Ticker Universe / Watchlist Selection

Universe selection determines which ticker identities move forward. It may use configured universe references, manual ticker lists, or watchlist references when approved.

Market Engine should preserve ticker identity and selection reason. Watchlist data must not be mutated in this stage.

Implementation implication: ME03 should identify useful scanner/universe semantics, and ME04 should define ownership for universe input contracts.

Testing implication: tests must prove ticker failures or missing metadata do not silently drop the whole batch.

### 3. Source Intake Request

Source intake request defines which evidence Market Engine attempts to collect or review for each ticker. It is a request plan, not a recommendation.

This is where source intake starts. It should state provider/source family, ticker identity, requested evidence type, and allowed scope.

Implementation implication: ME05 may build a bounded all-ticker source intake smoke only after ME03 and ME04 define source and architecture rules.

Testing implication: tests must use fake or synthetic provider responses unless a manual smoke harness is explicitly approved.

### 4. Provider / Source Access

Provider/source access is the only stage that may eventually contact external sources, and only after a later sprint authorizes it.

This stage must be explicit, bounded, and reviewable. It must not run implicitly from imports, tests, scanner logic, analysis, reporting, or Telegram.

Implementation implication: ME04 must keep provider access separated from analysis and decision logic.

Testing implication: normal automated tests must prove no live provider calls occur.

### 5. Source Coverage Validation

Source coverage validation classifies whether requested evidence is available, partial, stale, invalid, unavailable, or blocked by provider error.

Source coverage is not company quality and not investment quality.

Implementation implication: ME03 must define source readiness states and failure categories that can support all-ticker coverage triage.

Testing implication: tests must prove source failures are recorded per ticker and do not stop the whole batch.

### 6. Raw Source Result Preservation

Raw source evidence must be preserved before normalization. Raw evidence includes provenance, source reference, timestamps, provider status, raw fields, missing-field evidence, and capture metadata when available.

Raw evidence is evidence, not a program-ready analytical input and not a decision artifact.

Implementation implication: ME03 must define what raw evidence fields matter for Market Engine source intake.

Testing implication: tests must prove raw evidence remains distinguishable from normalized input and generated output.

### 7. Normalized Data View

The normalized data view maps approved source evidence into program-ready fields with traceability.

Normalization may make data easier to consume, but it must not improve, score, rank, recommend, or allocate.

Implementation implication: ME03 must define normalized financial, scanner, and fundamental fields separately from raw evidence and analysis output.

Testing implication: tests must prove missing values remain missing and parse failures do not become zero.

### 8. Missing-Data And Quality-State Handling

Market Engine must represent missing, partial, stale, invalid, unavailable, and review-required data explicitly.

Quality-state language must be careful. In early Market Engine layers, quality means evidence quality, data quality, or source readiness, not investment recommendation.

Implementation implication: ME03 must separate source readiness, data quality, fundamental context, and investment quality language.

Testing implication: tests must guard against missing-to-zero conversion and forbidden authority fields.

### 9. Scanner Context Preparation

Scanner context preparation describes market/setup context for each ticker. Old scanner concepts such as setup classification, discovery reason, liquidity state, trend, momentum, position, relative strength, and trade-plan-shaped fields may be useful reference material.

The old mixed scanner runtime shape must not be copied. Provider access, yfinance sector lookup, scoring, ranking, trade-plan semantics, and final operator interpretation require explicit contracts.

Implementation implication: ME03 must decide which scanner concepts are kept, rejected, or deferred. ME04 must ensure scanner context has no provider side effects unless a source boundary provides the data.

Testing implication: tests must prove scanner context does not emit BUY / SELL / HOLD, recommendation, allocation, urgency, conviction, or tradeability.

### 10. Fundamental Context Preparation

Fundamental context preparation describes company and financial evidence using governed source and normalized data.

Fundamental context may include revenue, income, cash flow, free cash flow derivation status, growth evidence, balance-sheet fields, period metadata, and source readiness when approved.

Implementation implication: ME03 must define financial and fundamental context as descriptive evidence and must keep source intake separate from recommendation logic.

Testing implication: tests must prove fundamental context does not contain final action, allocation, score, ranking, target price, threshold, conviction, urgency, tradeability, or recommendation fields unless a later approved boundary owns them.

### 11. First Analysis Pass

The first analysis pass starts after source intake, coverage validation, raw preservation, normalization, and context preparation have created governed evidence.

Analysis may summarize evidence, identify review limitations, and produce descriptive profiles. It must not become the Decision Engine.

Implementation implication: ME07 will need this boundary: analysis consumes governed evidence and outputs review-oriented analysis only.

Testing implication: tests must prove analysis is side-effect-free and does not emit final actions or capital-action outputs.

### 12. Risk And Quality Flags

Risk and quality flags identify review limitations, source gaps, freshness problems, incomplete context, conflicting evidence, or operator attention areas.

These flags help the operator review evidence. They must not rank opportunities as hidden allocation priority.

Implementation implication: ME03 should identify source and financial risk flags; ME04 should define where flags live and how they avoid Decision Engine leakage.

Testing implication: tests must prove flags do not change row preservation or final decision semantics.

### 13. Local Operator Review Output

Local operator review output communicates the evidence, limitations, and status to the operator without sending messages or writing production reports unless later approved.

This output should be local, explicit, and audit-friendly. It should show what was reviewed, what was missing, what failed, and what remains deferred.

Implementation implication: ME08 should use this flow to produce local operator review output as communication only.

Testing implication: tests must prove local output does not mutate portfolio/watchlist data, send Telegram, or change decisions.

### 14. Later Optional Decision / Reporting / Notification Layers

Decision, reporting, and notification layers are deferred from early Market Engine source/scanner/fundamental layers.

The Decision Engine remains the only authority for allocation, final action, execution semantics, and arbitration. Reporting communicates only. Telegram delivery is a transport surface only.

Implementation implication: ME04 must document architecture gates before any optional downstream layer is connected.

Testing implication: tests must guard all early layers from downstream side effects.

## Boundary Map

| Boundary | Must stop before | May pass forward |
|---|---|---|
| Source intake | analysis, recommendation, allocation, reporting, Telegram | source status, raw evidence, provider failure, provenance |
| Validation | hidden filtering, allocation eligibility, urgency | structure issues, missing fields, row identity, review-required state |
| Scanner context | provider calls, recommendation, allocation, hidden ranking | descriptive setup/context fields when governed |
| Fundamental context | investment conclusion, score, rank, final action | financial evidence, source readiness, missingness, derivation status |
| Analysis | Decision Engine final action, reporting writes, Telegram delivery | review-oriented analysis, limitations, evidence flags |
| Decision preparation | upstream mutation, reporting dependency | approved descriptive evidence for future Decision Engine review |
| Operator review | Telegram delivery, portfolio/watchlist mutation, production reporting | local communication of evidence and limitations |

## Recommendation Leakage Rule

Early Market Engine layers must not emit BUY / SELL / HOLD, recommendations, allocation instructions, execution urgency, conviction, tradeability, hidden ranking, or final action semantics.

Terms such as buy now, buy on pullback, and buy on breakout may appear only as downstream communication shapes when an approved decision contract supplies those states. Early source, scanner, fundamental, and analysis layers must not create them.

## Side-Effect Exclusions

Early Market Engine layers must exclude:

- Telegram delivery;
- report generation;
- production data writes;
- portfolio mutation;
- watchlist mutation;
- Decision Engine behavior changes;
- hidden provider calls;
- import-time side effects;
- live provider calls in automated tests;
- generated output becoming source truth without explicit approval.

## Functional Guardrails

- Classification remains upstream.
- Allocation remains downstream.
- Decision Engine is the only final-action authority.
- Reporting is communication only.
- Source readiness is not investment quality.
- Missing data remains missing.
- Raw evidence, normalized input, generated output, reporting output, and local-only data remain separate.
- Ticker-level source failures must not stop the whole batch.
- Old code and tests are reference sources, not implementation foundation.
- Manual smoke harnesses must be explicit, bounded, and non-canonical unless promoted by architecture.

## Implementation Implications For ME03

ME03 must use this functional flow to extract:

- scanner context concepts worth keeping, rejecting, or deferring;
- financial and fundamental fields required for source and normalized data;
- source readiness states and failure categories;
- missing-data and source-quality rules;
- boundaries preventing source intake from becoming recommendation logic;
- data/source implications for all-ticker source intake.

ME03 should not define BUY / SELL / HOLD behavior, portfolio mutation, Telegram behavior, or reporting behavior.

## Implementation Implications For ME04

ME04 must use this functional flow to define:

- module ownership boundaries for intent, universe, source intake, normalization, scanner context, fundamental context, analysis, and local operator review;
- side-effect controls for imports and normal execution;
- provider/data/analysis/decision separation;
- test-family placement;
- manual smoke harness rules;
- forbidden authority fields and leakage checks.

ME04 should not implement runtime behavior unless a later sprint explicitly authorizes it.

## Implementation Implications For ME05

ME05 must use this functional flow to build only an explicit all-ticker source intake smoke after ME03 and ME04 define the source and architecture boundaries.

ME05 must:

- keep source intake separate from analysis and recommendations;
- capture per-ticker source coverage and failures;
- preserve missingness;
- avoid normal automated tests with live calls;
- avoid production writes, reports, Telegram, portfolio/watchlist mutation, and Decision Engine behavior.

## Testing Implications

Future tests must prove:

- operator intent and universe selection are side-effect-free;
- provider/source access does not occur in normal automated tests;
- source failures are per-ticker and do not stop the batch;
- raw evidence, normalized input, generated output, reporting output, and local-only data remain separate;
- missing values remain missing;
- forbidden authority fields are rejected before early-layer output;
- scanner and fundamental contexts do not emit recommendation language;
- analysis does not emit final actions or capital-action outputs;
- reporting and Telegram cannot alter decision semantics;
- portfolio and watchlist files are not mutated by source, scanner, fundamental, or analysis layers.

## Open Questions

- Which ticker universe source should ME05 use first: configured universe, manual list, read-only watchlist reference, or a synthetic fixture-expanded universe?
- Which provider/source family is the first approved all-ticker smoke target?
- Which scanner context fields are descriptive enough to keep without creating hidden ranking or trade-plan authority?
- Which fundamental metrics are required for the first analysis pass versus deferred to later financial depth?
- What exact local operator review format should ME08 produce?

## Readiness Criteria For ME03

ME03 is ready when it can use this functional flow to extract financial, scanner, fundamental, and source-readiness specifications without reopening ME02 scope.

ME03 should start from:

- source intake stops before analysis and recommendations;
- analysis begins only after governed evidence, normalized data, missingness, and context exist;
- scanner and fundamentals remain descriptive until later authority is approved;
- early layers exclude Telegram, reporting, portfolio mutation, watchlist mutation, and Decision Engine behavior;
- keep / reject / defer decisions from ME02 are recorded in the extraction document.

