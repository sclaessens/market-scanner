# SEC EDGAR Fundamentals Source Strategy

Status: ACTIVE STRATEGY DRAFT
Backlog driver: BL-0015
Sprint: SEC-1A — SEC EDGAR Source Strategy Document
Date: 2026-05-30

## 1. Purpose

This document defines the first active strategy draft for using SEC EDGAR as the primary candidate source for real fundamentals source data.

SEC-1A is documentation-only. It does not authorize implementation, code changes, test changes, data changes, CSV commits, generated-output commits, provider/API calls, scraping, downloads, pipeline execution, Decision Engine changes, Reporting changes, Telegram changes, portfolio changes, ticker-category runtime logic, or runtime behavior changes.

The purpose of SEC-1A is to replace the prior manual-five-ticker-first direction with a broader, repeatable SEC EDGAR source-data strategy before any Codex implementation sprint begins.

## 2. Product Owner Direction

The Product Owner direction is:

- do not start by manually filling fundamentals data for only five tickers;
- investigate and adopt SEC EDGAR as the primary fundamentals source candidate;
- prefer SEC EDGAR Company Facts bulk data as the initial source architecture;
- retrieve data broadly rather than only for a small selected sample;
- build a repeatable refresh and update workflow before relying on real data operationally;
- map available SEC XBRL data into the internal fundamentals model;
- rationalize analysis assumptions to reliable available SEC data rather than making the application unnecessarily complex;
- use NVDA, TWLO, ON, AMD, and KEYS as review/sample tickers, not as the full source-data strategy.

## 3. Current Project Status

The fundamentals platform is technically implemented, organized, wired, and synthetically validated.

Current implemented fundamentals chain:

```text
raw fundamentals history
-> calculated fundamental metrics
-> fundamental quality compatibility
-> fundamental analysis classification
```

Current runtime area:

```text
scripts/fundamentals/
  build_history_intake.py
  build_metrics.py
  build_quality.py
  build_analysis.py
```

Compatibility wrappers remain available under:

```text
scripts/core/
```

The protected downstream artifact remains:

```text
data/processed/fundamental_quality.csv
```

`fundamental_analysis.csv` remains descriptive and is not a downstream-required dependency.

BL-0015 remains active because the platform still needs an approved real source-data operating workflow that is proven with source-supported data.

## 4. Why Manual Five-Ticker Entry Is Not The Preferred First Step

Manual entry for only NVDA, TWLO, ON, AMD, and KEYS would be useful as a review exercise, but it is no longer the preferred first source-data step.

Reasons:

- it does not prove that the source architecture can scale beyond a small sample;
- it risks creating a one-off data patch instead of a repeatable workflow;
- it may bias the internal model toward five companies before SEC coverage is understood;
- it does not solve ticker-to-CIK matching;
- it does not solve XBRL tag mapping;
- it does not define refresh/update behavior;
- it does not show which internal assumptions are supported by reliable available SEC data.

The five selected tickers should remain a controlled review set after broad SEC data intake, CIK matching, and XBRL mapping are understood.

## 5. SEC EDGAR As Primary Source Candidate

SEC EDGAR is the preferred primary candidate for real fundamentals source data because it provides official company filings and structured XBRL data for US-listed reporting companies.

This document does not yet approve SEC EDGAR implementation. It only defines SEC EDGAR as the primary candidate to be finalized in SEC-1.

A future implementation sprint may use SEC EDGAR only after SEC-1 finalizes:

- approved access method;
- fair-access and user-agent requirements;
- local cache policy;
- generated data commit policy;
- ticker-to-CIK matching strategy;
- bulk versus per-company access decision;
- failure behavior;
- validation requirements;
- fixture-based test approach.

## 6. SEC Company Facts Bulk Data Preference

The preferred initial architecture is SEC EDGAR Company Facts bulk data.

Rationale:

- it supports broad-data-first intake;
- it reduces the risk of designing around a few tickers;
- it allows repeatable local cache and refresh workflows;
- it supports later ticker/CIK coverage analysis;
- it supports XBRL tag mapping investigation across a wider universe;
- it can be reviewed before transformation into internal fundamentals history.

SEC-1A does not download the bulk data and does not create a runtime intake process.

## 7. Broad-Data-First Principle

The project should first determine how SEC facts can be retrieved, cached, indexed, mapped, and refreshed broadly.

Broad-data-first means:

1. understand the available SEC source structure;
2. define storage/cache boundaries;
3. define ticker/CIK matching;
4. determine coverage for the scanner and portfolio universe;
5. map XBRL tags to internal fields;
6. rationalize analysis to reliable available data;
7. only then use selected tickers for controlled analyst review.

Broad-data-first does not mean committing generated SEC datasets to the repository.

## 8. Role Responsibilities

### Product Owner

Confirms the direction away from manual five-ticker-first entry and prioritizes broad SEC EDGAR data intake, repeatable refresh, and the use of the five selected tickers as review examples.

### PM / Scrum Master

Splits the work into smaller sprint-safe steps, prevents implementation from being pulled into SEC-1A, and keeps SEC-1 as the recommended next sprint.

### Functional Analyst

Defines the business workflow needs for source-supported fundamentals review, including what the analyst must be able to inspect after SEC data becomes available.

### Data Steward

Owns source evidence, local source-data handling expectations, provenance requirements, and generated data commit boundaries.

### Financial Analyst

Defines which financial fields and metrics are useful, which are optional, and which must be reconsidered if SEC coverage or XBRL consistency is insufficient.

### Technical Analyst / Architect

Defines SEC source architecture questions, storage/cache boundaries, ticker/CIK matching needs, XBRL mapping investigation needs, and future implementation boundaries.

### Governance Auditor

Checks that no Decision Engine authority, allocation semantics, hidden filtering, ranking, scoring, or reporting decision semantics are introduced.

### Developer / Codex

Does not implement in SEC-1A. Future Codex work may begin only after SEC-1 finalizes developer-ready implementation scope.

## 9. Proposed Sprint Sequence

### Sprint SEC-1 — SEC EDGAR Source Architecture Finalization

Type: documentation/specification only.

Goal:

- finalize SEC EDGAR as the primary source candidate;
- define local storage/cache policy;
- define commit policy;
- define CIK/ticker matching requirements;
- define bulk versus per-company access decision;
- define fair-access/user-agent requirements;
- define what implementation work Codex may do later.

### Sprint SEC-2 — SEC Bulk Intake Implementation

Type: Codex implementation.

Goal:

- implement controlled download/cache of SEC Company Facts bulk data;
- do not commit generated SEC data;
- use metadata/logging only if approved;
- use fixture-based tests;
- make no analysis logic changes.

### Sprint SEC-3 — SEC Ticker/CIK Index and Coverage Report

Type: Codex implementation plus documentation.

Goal:

- build ticker-to-CIK mapping;
- determine coverage for scanner universe and portfolio universe;
- identify missing tickers;
- produce a non-committed or documentation-safe coverage summary.

### Sprint SEC-4 — SEC XBRL Mapping Investigation

Type: documentation plus optional fixture tests.

Goal:

- map SEC XBRL tags to internal fundamentals fields;
- identify alternative tags;
- identify missing or inconsistent tags;
- decide which fields are reliable.

### Sprint SEC-5 — Fundamental Analysis Rationalization

Type: documentation-first.

Goal:

- compare current analysis assumptions with reliable SEC data availability;
- mark metrics as core, optional, or deprecated;
- simplify analysis where data is not reliably available;
- update calculation registry if needed.

### Sprint SEC-6 — SEC-to-Fundamentals Transformation

Type: Codex implementation.

Goal:

- transform SEC facts into the internal fundamentals history structure;
- preserve source references;
- handle fiscal periods, units, duplicate facts, and amendments;
- generate outputs locally;
- do not commit generated outputs.

### Sprint SEC-7 — Controlled Real Fundamentals Analysis

Type: Codex execution plus ChatGPT analyst review.

Goal:

- run the new source-data pipeline for a controlled universe;
- include NVDA, TWLO, ON, AMD, and KEYS as review tickers;
- review metrics, quality, and analysis outputs;
- determine whether real analyst review can restart.

## 10. SEC-1A Boundary

SEC-1A creates only this strategy document.

SEC-1A does not:

- update the backlog;
- update the source-data operating workflow;
- update the role matrix;
- update the calculation registry;
- update code;
- update tests;
- update data;
- update reports;
- update generated files;
- update GitHub Actions;
- run the pipeline;
- call SEC or any provider;
- scrape websites;
- download SEC data;
- implement ticker/CIK matching;
- implement XBRL mapping;
- implement source-data automation.

Those actions belong to later sprints.

## 11. Storage and Commit Policy Direction

The working direction is:

- real SEC source-data downloads should remain local, ignored, cached, or otherwise non-committed unless repository policy explicitly changes;
- generated SEC raw caches should not be committed;
- generated fundamentals outputs should not be committed unless explicitly approved;
- documentation-safe summaries may be committed only after governance review;
- fixture data for tests must be synthetic or minimized and must not contain uncontrolled generated operational data.

SEC-1 must finalize this policy before SEC-2 implementation.

## 12. Refresh and Update Strategy Direction

A future SEC workflow should support repeatable refresh behavior.

SEC-1 should define:

- manual refresh versus controlled command execution;
- cache location;
- cache invalidation rules;
- metadata to record extraction date and source freshness date;
- fair-access constraints;
- failure behavior for unavailable or stale source data;
- whether refresh metadata may be committed.

No refresh implementation is approved in SEC-1A.

## 13. CIK/Ticker Mapping Strategy Direction

SEC source data is organized around CIK identity, while the project works primarily with tickers.

SEC-1 should define:

- authoritative mapping source;
- ticker normalization rules;
- handling of class shares and ticker changes;
- handling of delisted or non-US tickers;
- coverage reporting for scanner and portfolio universes;
- review workflow for missing or ambiguous mappings.

No ticker/CIK mapping implementation is approved in SEC-1A.

## 14. XBRL Tag Mapping Strategy Direction

The internal fundamentals model currently expects fields such as revenue, gross profit, operating income, net income, diluted EPS, total debt, total equity, and free cash flow.

SEC XBRL tags may not map one-to-one to those fields across all companies.

SEC-4 should investigate:

- primary XBRL tags for each internal field;
- accepted alternative tags;
- unit conventions;
- fiscal period handling;
- amended facts and duplicate facts;
- missing fields;
- company-specific reporting differences;
- which internal fields are reliable enough to be core.

SEC-1A does not define final XBRL mappings.

## 15. Analysis Rationalization Principle

The project should adapt analysis to reliable available data.

If SEC facts do not reliably support an existing metric or assumption, the project should prefer simplification over complex workaround logic.

Future rationalization may classify metrics as:

- core;
- optional;
- review-required;
- deprecated;
- unsupported by SEC facts;
- requires non-SEC source approval.

No metric is changed by SEC-1A.

## 16. Review Ticker Set

The selected review tickers are:

```text
NVDA
TWLO
ON
AMD
KEYS
```

These tickers should be used as a review/sample set after SEC data intake, CIK coverage, and XBRL mapping questions are understood.

They are not the full source-data strategy.

## 17. Backlog Impact Assessment

Backlog impact assessment:

- No backlog changes are made in SEC-1A.

Rationale:

- SEC-1A is intentionally limited to one new active strategy document.
- BL-0015 remains the primary backlog driver for approved fundamentals source data and quality classification work.
- BL-0017 remains relevant for governed automated ingestion strategy.
- Backlog alignment should be handled in SEC-1B or SEC-1, not in SEC-1A.

## 18. Recommended Next Sprint

Recommended next sprint:

```text
Sprint SEC-1 — SEC EDGAR Source Architecture Finalization
```

Sprint SEC-1 must remain documentation/specification only.

It should finalize the technical and governance architecture before Codex begins implementation in SEC-2.

## 19. No-Runtime-Change Confirmation

SEC-1A confirms:

- no scripts changed;
- no tests changed;
- no data changed;
- no reports changed;
- no CSV files changed;
- no generated files changed;
- no workflow files changed;
- no runtime behavior changed;
- no provider/API calls performed;
- no scraping performed;
- no SEC data downloaded;
- no Decision Engine behavior changed;
- no Reporting behavior changed;
- no Telegram behavior changed;
- no portfolio behavior changed;
- no ticker-category runtime logic implemented.