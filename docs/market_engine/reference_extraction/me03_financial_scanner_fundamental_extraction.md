# ME03 Financial, Scanner, and Fundamental Extraction

Owner role: Financial Analyst / Data Steward / Governance Auditor

Status: ME03 EXTRACTION DRAFT

## Purpose

This document records the source areas used to create the Market Engine financial, scanner, and fundamental logic specification.

The extraction is intentionally bounded. It is not an exhaustive archive index and does not attempt to document every old implementation detail.

ME03 extracts only the logic, lessons, risks, and keep / reject / defer decisions needed to steer:

* ME04 technical, coding, and testing architecture;
* ME05 all-ticker source intake smoke;
* later financial/scanner/fundamental implementation.

## Extraction Scope

In scope:

* Market Engine functional flow;
* Market Engine backlog requirements;
* financial-analysis intent;
* scanner-context lessons;
* fundamental-context lessons;
* source/provider-readiness lessons;
* missing-data and quality-state rules;
* ticker failure handling;
* portfolio/watchlist boundary lessons;
* Decision Engine authority protection;
* reporting and Telegram side-effect boundaries;
* testing implications.

Out of scope:

* Python implementation;
* test implementation;
* live provider calls;
* yfinance calls;
* SEC / EDGAR calls;
* runtime execution;
* exhaustive old-code migration;
* old script-era cleanup;
* BUY / SELL / HOLD behavior;
* allocation;
* urgency;
* conviction;
* tradeability;
* reporting;
* Telegram;
* portfolio mutation;
* watchlist mutation;
* Decision Engine behavior changes.

## Extraction Records

### 1. Market Engine Functional Flow

Reference source: Market Engine functional flow

Repository path: `docs/market_engine/analysis/functional_flow.md`

Owner role: Functional Analyst / Product Owner

Observed logic:
Market Engine is defined as a local operator decision-support product. Its flow starts with operator intent and ticker universe selection, then moves through source intake, provider/source access, coverage validation, raw evidence preservation, normalized data, missing-data handling, scanner context, fundamental context, first analysis pass, risk/quality flags, and local operator review output.

Useful lesson:
ME03 must focus on the middle of this flow: source coverage, raw evidence, normalization, missingness, scanner context, fundamental context, and first-analysis preparation.

Known risk / failure mode:
Financial, scanner, or fundamental logic could accidentally become recommendation logic if the boundaries are not explicit.

Market Engine decision:
Keep.

Implementation implication:
ME04 must translate the ME02 flow and ME03 logic into module ownership boundaries.

Testing implication:
Future tests must prove that source/scanner/fundamental layers do not emit BUY / SELL / HOLD, allocation, urgency, conviction, tradeability, or final-action semantics.

Data/source implication:
Raw evidence, normalized data, generated analysis output, reporting output, and local-only output must remain separate.

Extraction status:
Complete.

### 2. ME02 Functional Flow Extraction

Reference source: ME02 extraction records

Repository path: `docs/market_engine/reference_extraction/me02_functional_flow_extraction.md`

Owner role: Functional Analyst / Governance Auditor

Observed logic:
ME02 preserved operator visibility, source readiness separated from investment quality, raw/normalized/generated/report/local-only data separation, explicit missingness, Decision Engine authority protection, side-effect-free early boundaries, and fake/synthetic provider responses for automated tests.

Useful lesson:
ME03 should not reopen ME02. It should apply the already extracted functional boundaries to the financial, scanner, and fundamental domains.

Known risk / failure mode:
Old scanner and provider concepts may be useful, but they also contain dangerous mixed semantics such as implicit provider calls, ranking, scoring, trade-plan-shaped fields, and hidden recommendation authority.

Market Engine decision:
Keep / reject / defer.

Keep:

* operator visibility;
* preserved evidence;
* explicit missingness;
* source readiness separate from investment quality;
* scanner/fundamental context as descriptive evidence;
* Decision Engine authority protection.

Reject:

* blind copying of old script-era code;
* implicit provider calls;
* missing-to-zero conversion;
* BUY / SELL / HOLD leakage;
* reporting, Telegram, portfolio, watchlist, or Decision Engine side effects in lower layers.

Defer:

* exact scanner taxonomy;
* exact provider source family;
* exact fundamental schema;
* exact local operator review format.

Implementation implication:
ME04 must define forbidden fields, source boundaries, side-effect controls, and test placement.

Testing implication:
Tests must use fake/synthetic provider responses and must guard against recommendation leakage.

Data/source implication:
ME05 must capture coverage and failures without making source success equivalent to investment quality.

Extraction status:
Complete.

### 3. Market Engine Backlog

Reference source: Market Engine backlog

Repository path: `docs/market_engine/backlog/market_engine_backlog.md`

Owner role: Scrum Master / PM / Product Owner

Observed logic:
ME03 is explicitly scoped to financial logic, scanner classification lessons, fundamental data lessons, provider/source readiness, data implications, missing-data rules, and failure modes. It excludes BUY / SELL / HOLD, allocation, urgency, conviction, tradeability, portfolio/watchlist mutation, Telegram, reporting behavior, provider calls, and implementation.

Useful lesson:
ME03 should produce a specification that is good enough to steer ME04 and ME05, not a theoretical financial model.

Known risk / failure mode:
ME03 could become too broad and delay implementation if it tries to finalize every metric, formula, provider mapping, and scanner model.

Market Engine decision:
Keep.

Implementation implication:
ME04 should use ME03 as a bounded specification, not as an excuse to design the whole application forever.

Testing implication:
ME04 testing strategy must use ME03 to define leakage tests, missing-data tests, source-failure tests, and fake-provider tests.

Data/source implication:
ME05 must focus on all-ticker source intake, source coverage, failure capture, and missingness preservation.

Extraction status:
Complete.

### 4. Financial Analysis Intent

Reference source: financial analysis document

Repository path: `docs/active/project/financial_analysis.md` or equivalent uploaded financial analysis source

Owner role: Financial Analyst

Observed logic:
Financial analysis is upstream classification. It may describe business understanding, growth profile, profitability profile, balance sheet and financial risk, cash generation, valuation context, capital allocation, shareholder returns, source-data readiness, and portfolio relevance. It must not produce buy actions, sell actions, hold actions, position sizing, execution instructions, tradeability states, urgency rankings, conviction scores, hidden filtering decisions, or portfolio transaction advice.

Useful lesson:
Market Engine financial logic should be descriptive, traceable, uncertainty-preserving, and source-aware.

Known risk / failure mode:
Financial concepts such as growth, profitability, valuation, quality, or risk can easily become hidden recommendation proxies if not bounded.

Market Engine decision:
Keep / defer.

Keep:

* business understanding;
* growth profile;
* profitability profile;
* balance sheet risk;
* cash generation;
* valuation context;
* capital allocation context;
* source-data readiness;
* portfolio relevance as read-only context.

Defer:

* exact formulas;
* exact thresholds;
* exact valuation methodology;
* exact peer comparison;
* exact first-pass financial schema.

Implementation implication:
ME04 must define where financial concepts live and how they remain descriptive before any Decision Engine boundary.

Testing implication:
Tests must prove financial outputs contain no final action, allocation, urgency, conviction, tradeability, recommendation, or hidden ranking.

Data/source implication:
Financial metrics must preserve provenance, period coverage, missingness, and derivation status.

Extraction status:
Complete.

### 5. Missing Data And Source-Data Readiness

Reference source: financial analysis, ME02 functional flow, ME02 extraction

Repository path:

* `docs/market_engine/analysis/functional_flow.md`
* `docs/market_engine/reference_extraction/me02_functional_flow_extraction.md`
* financial analysis source

Owner role: Data Steward / Financial Analyst

Observed logic:
Missing financial data must remain explicit. Missing numeric values must not be converted to zero. Missing periods must not be silently interpolated. Source-data gaps must remain visible downstream. Source readiness describes whether data is usable for analysis, not whether a company is attractive.

Useful lesson:
ME03 must make missing-data and source-readiness rules non-negotiable.

Known risk / failure mode:
Missing capex, missing operating cash flow, missing revenue periods, or missing price context could be treated as zero or neutral, creating false precision.

Market Engine decision:
Keep.

Implementation implication:
ME04 must define missing-data representations, derivation-blocking rules, and source-readiness states.

Testing implication:
Tests must prove missing values stay missing and missing components block unsafe derived metrics.

Data/source implication:
ME05 should record per-ticker missing fields, partial data, stale data, invalid data, provider errors, and review-required states.

Extraction status:
Complete.

### 6. Cash Generation And Free Cash Flow Derivation

Reference source: financial analysis document

Repository path: `docs/active/project/financial_analysis.md` or equivalent uploaded financial analysis source

Owner role: Financial Analyst / Data Steward

Observed logic:
Cash generation is a core financial domain. Operating cash flow, capital expenditures, and free cash flow are important concepts. Free cash flow must only be derived when the required source components are available and valid.

Useful lesson:
Cash generation is a high-priority area for source-readiness observation because incomplete data can easily create misleading derived metrics.

Known risk / failure mode:
Treating missing capex as zero or deriving free cash flow from incomplete source data would produce false analytical confidence.

Market Engine decision:
Keep.

Implementation implication:
ME04 should define derivation status separately from derived metric value.

Testing implication:
Tests must prove derivations are blocked when required components are missing.

Data/source implication:
ME05 should capture whether operating cash flow and capex are available per ticker and whether free cash flow derivation is possible, blocked, partial, or review-required.

Extraction status:
Complete.

### 7. Scanner Context Lessons

Reference source: ME02 extraction and scanner-boundary references summarized in ME02

Repository path:

* `docs/market_engine/reference_extraction/me02_functional_flow_extraction.md`
* representative scanner documentation/code/tests referenced by ME02

Owner role: Functional Analyst / Technical Architect

Observed logic:
Old scanner material contains useful concepts such as universe selection, candidate construction, setup classification, liquidity state, trend, momentum, relative strength, trade-plan-shaped fields, ranking, and grading. It also contains riskier behavior such as provider access, yfinance sector lookup, mixed scoring semantics, and trade-plan-shaped outputs.

Useful lesson:
Market Engine should keep scanner context as descriptive setup/market evidence only.

Known risk / failure mode:
Scanner output can leak recommendation authority through ranking, grading, tradeability, urgency, entry, stop, target, risk/reward, or setup labels that imply action.

Market Engine decision:
Keep / reject / defer.

Keep:

* universe membership;
* selection reason;
* discovery reason;
* descriptive setup context;
* liquidity context;
* trend context;
* momentum context;
* relative strength context;
* scanner evidence state;
* missing scanner data flags.

Reject:

* implicit provider access;
* scanner-triggered network calls;
* hidden filtering;
* ranking as allocation priority;
* grading as recommendation;
* trade-plan-shaped authority;
* BUY / SELL / HOLD in scanner output.

Defer:

* exact setup taxonomy;
* exact trend/momentum formulas;
* exact liquidity thresholds;
* exact timing-state model;
* any ranking or prioritization model.

Implementation implication:
ME04 must ensure scanner context cannot own provider access or final-action semantics.

Testing implication:
Tests must prove scanner context has no provider side effects and no recommendation fields.

Data/source implication:
Scanner context must consume approved source/normalized data rather than fetching data implicitly.

Extraction status:
Complete.

### 8. Fundamental Context Lessons

Reference source: ME02 extraction, financial analysis, fundamentals/provider contract lessons summarized by ME02

Repository path:

* `docs/market_engine/reference_extraction/me02_functional_flow_extraction.md`
* financial analysis source
* representative fundamentals provider contracts/tests referenced by ME02

Owner role: Financial Analyst / Data Steward

Observed logic:
Fundamental context should preserve raw evidence, normalized fundamentals, source provenance, readiness records, period metadata, growth evidence, and provider categories. Authority fields such as final action, allocation, urgency, conviction, tradeability, score, ranking, target price, threshold, recommendation, report message, and Telegram message are forbidden in provider/fundamental records.

Useful lesson:
Fundamental context should describe company and financial evidence without becoming investment conclusion.

Known risk / failure mode:
Normalized fundamentals may be mistaken for investment conclusions if naming and fields are not strict.

Market Engine decision:
Keep / reject / defer.

Keep:

* raw evidence;
* normalized fundamentals;
* source provenance;
* period metadata;
* source readiness;
* missing field names;
* data freshness;
* revenue, income, cash flow, capex, free cash flow derivation status;
* growth, profitability, balance sheet, valuation, and review-required evidence.

Reject:

* authority fields in fundamental records;
* missing-to-zero conversion;
* incomplete derived metrics;
* source readiness as company quality;
* provider calls hidden inside tests or analysis.

Defer:

* exact fundamental schema;
* exact provider mappings;
* exact SEC CompanyFacts aliases;
* exact stale-data thresholds;
* exact fallback-provider strategy.

Implementation implication:
ME04 must define the fundamental record boundary and forbidden fields.

Testing implication:
Tests must prove fundamental records reject authority fields and preserve missingness.

Data/source implication:
ME05 should collect enough source-readiness evidence to decide which fundamental fields are realistically available across the ticker universe.

Extraction status:
Complete.

### 9. Portfolio Source-Of-Truth Boundary

Reference source: portfolio source-of-truth contract

Repository path: `docs/active/portfolio/portfolio_source_of_truth.md` or equivalent uploaded portfolio source-of-truth source

Owner role: Data Steward / Governance Auditor

Observed logic:
Manual portfolio source records are the approved source-of-truth role. Generated portfolio review, generated portfolio intelligence, reporting display input, and Telegram output are not source truth. Portfolio source records must not contain allocation instructions, execution instructions, urgency, conviction, tradeability, ranking, score, recommendation text, reporting text, or Telegram message text.

Useful lesson:
Market Engine may later use portfolio/watchlist concepts as read-only context, but source/scanner/fundamental/analysis layers must not mutate or reinterpret them.

Known risk / failure mode:
Portfolio display fields or generated review output could be mistaken for source truth or decision authority.

Market Engine decision:
Keep / defer.

Keep:

* manual portfolio source-of-truth principle;
* generated output is not source truth;
* missing portfolio values remain explicit;
* portfolio records cannot contain recommendation authority.

Defer:

* exact Market Engine portfolio/watchlist read-only input contract;
* exact point at which portfolio context enters Market Engine.

Implementation implication:
ME04 must define read-only portfolio/watchlist boundaries before any implementation uses them.

Testing implication:
Tests must prove lower layers do not mutate portfolio/watchlist data.

Data/source implication:
Portfolio relevance may be context only and must not drive allocation before Decision Engine authority.

Extraction status:
Complete.

### 10. Functional Analysis And Operator Goals

Reference source: functional analysis

Repository path: `docs/active/project/functional_analysis.md` or equivalent uploaded functional analysis source

Owner role: Product Owner / Functional Analyst

Observed logic:
The operator needs deterministic market scans, preserved opportunities, structural validation, context, fundamentals readiness, timing state, portfolio state, one final decision authority, reporting that explains without changing decisions, and research outputs that do not create allocation authority.

Useful lesson:
ME03 must preserve visibility, traceability, row preservation, source-data readiness, and single-authority decision doctrine.

Known risk / failure mode:
Without clear functional boundaries, scanner/fundamental logic may silently filter rows or create hidden decision behavior.

Market Engine decision:
Keep.

Implementation implication:
ME04 must preserve row identity and explain what each layer adds and does not decide.

Testing implication:
Tests must prove rows are not silently removed and generated outputs are clearly labelled.

Data/source implication:
Source-data insufficiency must be visible in later local operator review output.

Extraction status:
Complete.

## Summary Of Keep / Reject / Defer Decisions

### Keep

* operator visibility;
* preserved ticker identity;
* preserved opportunities where contract requires row identity;
* raw evidence preservation;
* normalized data as separate from raw evidence;
* generated output as separate from source truth;
* source readiness separate from investment quality;
* explicit missingness;
* missing numeric values remaining missing;
* ticker-level failure handling;
* scanner context as descriptive evidence;
* fundamental context as descriptive evidence;
* financial analysis as upstream classification;
* Decision Engine as final-action authority;
* reporting and Telegram as downstream communication only;
* portfolio/watchlist as read-only context unless later authorized;
* fake/synthetic provider responses in normal automated tests.

### Reject

* blind copying of old script-era code;
* old quick scripts as canonical runtime;
* implicit provider access inside scanner, analysis, or tests;
* hidden yfinance, SEC, or EDGAR calls;
* missing values converted to zero;
* generated outputs as source truth;
* hidden filtering without traceability;
* scanner ranking as hidden allocation;
* scanner grading as recommendation;
* trade-plan-shaped authority in early layers;
* BUY / SELL / HOLD leakage;
* recommendation leakage;
* reporting, Telegram, portfolio, watchlist, or Decision Engine side effects in source/scanner/fundamental/analysis layers.

### Defer

* exact ticker universe source for ME05;
* first approved provider/source family for all-ticker smoke;
* exact scanner taxonomy;
* exact trend/momentum/liquidity formulas;
* exact fundamental schema;
* exact SEC CompanyFacts aliases;
* exact yfinance mapping;
* exact stale-data thresholds;
* exact valuation methodology;
* exact peer comparison logic;
* exact local operator review format;
* downstream decision/reporting/notification integration.

## Implementation Implications For ME04

ME04 must convert these extracted decisions into technical architecture.

ME04 must define:

* module ownership boundaries;
* source/provider access boundary;
* raw evidence model;
* normalized data model;
* scanner context model;
* fundamental context model;
* first-analysis boundary;
* local operator review boundary;
* forbidden fields;
* side-effect controls;
* file strategy;
* test-family placement;
* manual smoke harness rules.

ME04 must explicitly prevent lower layers from:

* calling providers implicitly;
* writing production files;
* mutating portfolio/watchlist data;
* generating reports;
* sending Telegram;
* invoking the Decision Engine;
* emitting recommendation semantics.

## Source-Intake Implications For ME05

ME05 must be limited to all-ticker source intake smoke.

ME05 should observe and capture:

* provider/source availability;
* per-ticker response status;
* raw evidence feasibility;
* normalization feasibility;
* missing required fields;
* source freshness;
* invalid or partial responses;
* unsupported tickers;
* provider errors;
* derivation-blocked metrics;
* review-required states.

ME05 must not produce:

* financial conclusions;
* scanner recommendations;
* BUY / SELL / HOLD;
* allocation;
* urgency;
* conviction;
* tradeability;
* reporting;
* Telegram;
* portfolio/watchlist mutation;
* Decision Engine behavior.

## Testing Implications

Future tests must prove:

* no live provider calls in automated tests;
* fake/synthetic provider responses are used;
* missing numeric values do not become zero;
* missing components block derived metrics;
* ticker failures do not stop the batch;
* provider errors and unsupported tickers are distinguishable;
* scanner context contains no recommendation authority;
* fundamental context contains no forbidden authority fields;
* analysis output contains no final action semantics;
* lower layers do not mutate portfolio/watchlist files;
* lower layers do not generate reports;
* lower layers do not send Telegram;
* lower layers do not invoke the Decision Engine.

## Open Questions

* Which ticker universe should ME05 use first?
* Which provider/source family should ME05 target first?
* Which source-readiness enum should ME04 standardize?
* Which fundamental fields are mandatory for the first smoke?
* Which scanner context fields are safe enough for the first pass?
* Should raw evidence be persisted during ME05 or only summarized?
* What exact local operator review format should ME08 produce?
* When should portfolio/watchlist read-only context enter Market Engine?

## Extraction Status

ME03 extraction is complete enough to steer ME04 and ME05.

The specification deliberately defers formulas, provider mappings, scanner taxonomies, stale-data thresholds, valuation methodology, and local output formatting until architecture and source coverage realities are clearer.
