# Operational Sprint Automated Source Data Steward Protocol

## 1. Status and Scope

This document is a documentation-only governance and data-steward protocol.

It defines the Automated Source Data Steward role and the governed source-data provisioning flow for the `market-scanner` repository.

This document does not implement:

- code;
- tests;
- CSV edits;
- provider integrations;
- scraping;
- API calls;
- credentials;
- runtime behavior;
- Decision Engine changes;
- Reporting changes;
- Telegram changes;
- scanner changes;
- Fundamental Layer changes;
- Portfolio Intelligence changes.

No source-data values are approved or changed by this document.

No sprint is closed or certified complete by this document.

## 2. Background

The project now has working source-data contracts for fundamentals and portfolio metadata. These contracts allow the pipeline to distinguish governed source inputs from generated outputs while preserving the certified architecture doctrine.

Manual row-by-row approval is too slow for expanding metadata and fundamental-data coverage. The system needs a repeatable approval protocol that allows routine source-data rows to be evaluated consistently, safely, and audibly.

The Automated Source Data Steward role exists to approve routine source-data rows only under strict rules. Any exception, ambiguity, source-method change, provider/API introduction, credential use, or governance boundary question must remain human-reviewed.

Current source-data artifacts include:

- `data/portfolio/portfolio_metadata.csv` as tracked source data;
- `data/raw/fundamentals.csv` as local ignored source data;
- generated processed files as non-source outputs.

Relevant backlog items:

- `BL-0015 — Define and implement approved Fundamental data source and quality classification contract`;
- `BL-0016 — Define approved Portfolio Metadata and Sector Exposure contract`;
- `BL-0017 — Define governed automated data ingestion strategy for fundamentals and portfolio metadata`;
- related `BL-0011 — Define and repair authoritative active portfolio source`.

## 3. Repository Doctrine Boundary

All source-data provisioning must preserve:

- classification upstream;
- allocation downstream;
- Decision Engine = ONLY allocation authority;
- upstream layers classify only;
- portfolio metadata is descriptive only;
- fundamentals are descriptive/classification-only;
- Reporting communicates only;
- Telegram communicates only;
- no upstream tradeability;
- no hidden filtering;
- no hidden allocation semantics outside Decision Engine;
- no decision semantics outside Decision Engine;
- no ranking authority outside Decision Engine;
- no scoring authority outside Decision Engine;
- English-only repository content.

Data provisioning must never create buy/sell advice.

## 4. Automated Source Data Steward Role Definition

The Automated Source Data Steward is a governed execution role that evaluates source-data candidate rows against approved source methods, required schemas, validation rules, and escalation criteria.

The role may:

- inspect approved input universes;
- identify missing metadata or fundamentals coverage;
- create preview tables;
- apply approved source methods;
- classify rows as `APPROVED`, `REVIEW_REQUIRED`, or `REJECTED`;
- prepare approved CSV updates only when the protocol allows it;
- run validation checks;
- report distributions and blockers.

The role may not:

- invent values;
- choose new sources without governance;
- approve ambiguous values;
- approve conflicting source values;
- add buy/sell semantics;
- alter Decision Engine behavior;
- alter Reporting or Telegram semantics;
- use credentials without governance;
- call paid or restricted APIs without approval;
- scrape prohibited sources;
- commit ignored local source artifacts;
- commit generated outputs unless explicitly approved.

## 5. Approval States

### APPROVED

A row may be approved only when:

- the ticker is in the approved batch;
- the approved source method was used;
- required fields are present;
- values are non-empty;
- values are internally consistent;
- source/provenance fields are present;
- freshness dates are valid and not in the future;
- no credential/secret markers are present;
- no forbidden decision semantics are present.

### REVIEW_REQUIRED

A row must be marked review-required when:

- a required field is missing;
- the ticker mapping is ambiguous;
- the source returns unclear sector/industry/metric values;
- two approved sources conflict;
- metric definitions are unclear;
- freshness date is missing or questionable;
- currency is unclear;
- the row would require manual judgment.

### REJECTED

A row must be rejected when:

- the ticker is outside approved scope;
- source data is unavailable;
- required source method is not allowed;
- source values are contradictory and cannot be resolved;
- source data includes credentials/secrets;
- the row contains decision/action semantics;
- the row would require invented values;
- the source violates governance restrictions.

## 6. Source-Data Artifact Classes

### Tracked source data

Example:

- `data/portfolio/portfolio_metadata.csv`

Rules:

- may be committed only through explicit approved source-data PR;
- requires preview or automated approval;
- must preserve schema;
- must not include generated data;
- must not include credentials/secrets.

### Local ignored source data

Example:

- `data/raw/fundamentals.csv`

Rules:

- may be edited locally for validation;
- must remain ignored/untracked unless repository policy changes;
- should use local backups before updates;
- must not be committed;
- should be validated through builders/pipeline.

### Generated outputs

Examples:

- `data/processed/*.csv`;
- `data/logs/*.csv`;
- `reports/daily/*`.

Rules:

- generated outputs are validation artifacts;
- do not commit unless repository policy explicitly allows;
- inspect distributions and restore if needed.

## 7. Source-Data Categories

### Portfolio metadata

Fields:

- `ticker`;
- `sector`;
- `industry`;
- `asset_class`;
- `currency`;
- `metadata_source`;
- `metadata_last_updated`;
- optional `notes`, `country`, `region`, and `exchange` fields.

Approved current source method:

- Yahoo Finance for MVP batches, unless replaced by later governance.

Approval focus:

- sector/industry/asset_class/currency consistency;
- source and freshness present;
- no decision semantics.

### Fundamentals

Fields:

- `ticker`;
- `as_of_date`;
- `source_name`;
- `source_reference`;
- `source_freshness_date`;
- optional metrics such as revenue growth, EPS growth, margins, and debt/equity.

Approval focus:

- provenance present;
- metric definitions clear;
- values not invented;
- missing metrics classified explicitly;
- no buy/sell interpretation.

### Future provider-assisted data

Future provider-assisted data remains governed under `BL-0017`.

Provider/API access requires separate approval.

## 8. Batch Selection Flow

Recommended batch selection order:

1. Current portfolio holdings.
2. Existing source-supported fundamentals tickers.
3. A-grade scanner tickers.
4. B-grade scanner tickers.
5. Broader scanner universe only when justified.

Batch size rules:

- small preview batch: up to 10 tickers;
- standard controlled batch: up to 15 tickers;
- larger batches require explicit approval;
- full-universe expansion requires a separate plan.

Prioritization:

- A-grade before B-grade;
- existing metadata gaps before optional enrichment;
- source-supported rows before speculative expansion.

## 9. Preview → Approval → Update → Validation Lifecycle

### Step 1 — Preview

- inspect universe;
- identify missing rows;
- propose batch;
- produce preview table;
- do not modify files.

### Step 2 — Approval

- Automated Source Data Steward classifies rows;
- `APPROVED` rows may proceed;
- `REVIEW_REQUIRED` rows require human decision;
- `REJECTED` rows do not proceed.

### Step 3 — Update

- tracked source-data updates require PR;
- ignored local source-data updates remain local;
- no generated artifacts committed.

### Step 4 — Validation

- run focused tests;
- run relevant builder;
- run full pipeline when safe;
- inspect distributions;
- restore generated artifacts unless approved.

### Step 5 — Report

- report rows added/updated;
- report approval states;
- report distributions;
- report next blocker;
- report backlog impact.

## 10. Automated Approval Rules by Data Type

### Portfolio Metadata Automated Approval

A row may be automatically approved if:

- ticker is in approved batch;
- Yahoo Finance is the approved source method;
- sector is present;
- industry is present;
- `asset_class` is one of approved descriptive values, currently including:
  - `Equity`;
  - `REIT`;
  - `ETF`;
- currency is present;
- `metadata_source` is present;
- `metadata_last_updated` is valid;
- notes do not contain decision semantics.

A row must be `REVIEW_REQUIRED` if:

- source values are missing;
- sector or industry is unclear;
- asset_class is not recognized;
- ticker format is ambiguous;
- source lookup requires judgment.

### Fundamentals Automated Approval

A provenance-only row may be approved if:

- ticker is in approved batch;
- required provenance fields are present;
- dates are valid;
- optional metrics are blank rather than invented;
- notes clearly say metrics are not yet sourced.

A numerical fundamentals row may be approved only if:

- metric source is approved;
- metric definitions are clear;
- values are numeric and parseable;
- source date/freshness date is present;
- no metric is inferred from price action;
- missing values remain blank or review-required.

Numerical fundamentals should be `REVIEW_REQUIRED` if:

- definitions differ between sources;
- values are contradictory;
- values require calculations not yet governed;
- financial periods are unclear.

## 11. Human Review Triggers

Human review is required when:

- source method changes;
- new provider/API is proposed;
- credentials or secrets are involved;
- source values conflict;
- batch exceeds approved size;
- ticker mapping is ambiguous;
- a new asset class is needed;
- a new metric definition is needed;
- a row is marked `REVIEW_REQUIRED`;
- a generated artifact would be committed;
- Decision Engine semantics could be affected.

## 12. Codex Execution Permissions

Codex may:

- run preview-only inspections;
- classify rows using the protocol;
- update tracked source CSVs only after explicit approval;
- update ignored local raw files only when explicitly approved;
- run tests/builders/pipeline;
- open PRs for tracked source-data updates.

Codex may not:

- bypass preview/approval;
- silently expand batch scope;
- invent values;
- write to unrelated files;
- commit ignored raw fundamentals;
- commit generated outputs;
- modify Decision Engine/Reporting/Telegram/scanner logic as part of data provisioning.

## 13. Audit and Reporting Requirements

Every source-data provisioning task must report:

- batch scope;
- approved source method;
- files changed;
- rows added/updated;
- rows approved/review-required/rejected;
- tests run;
- builder/pipeline results;
- generated artifact handling;
- backlog impact;
- next blocker.

For local ignored updates, also report:

- backup file created;
- ignored/untracked status;
- final local row count.

## 14. Backlog Impact Assessment

Existing backlog items remain sufficient for this protocol:

- `BL-0015`;
- `BL-0016`;
- `BL-0017`;
- related `BL-0011`.

This protocol organizes existing source-data provisioning work and does not require a new backlog item.

Backlog impact assessment:
- No new backlog items identified.

## 15. Recommended Next Step

Recommended next step:

- review and merge this protocol;
- then create Codex prompts that explicitly invoke the Automated Source Data Steward role;
- first apply it to portfolio metadata expansion batches;
- then apply it to fundamentals metric sourcing batches;
- keep provider/API automation deferred under `BL-0017`.

## Validation Notes

This protocol is documentation-only.

Expected validation for this change:

- confirm changed files;
- confirm whether backlog was changed;
- confirm only documentation files were changed;
- confirm no runtime files, generated files, CSV files, tests, scripts, reports, workflows, portfolio files, raw fundamentals files, Decision Engine files, Reporting files, Telegram files, scanner files, Fundamental Layer files, or Portfolio Intelligence files were modified;
- do not run runtime tests unless needed;
- if tests are not run, state that validation was documentation-only.
