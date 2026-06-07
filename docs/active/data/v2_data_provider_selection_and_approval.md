# V2 Data Provider Selection and Approval

Status: ACTIVE
Reset stage: RESET-10L

## 1. Purpose

This document defines the v2 governance specification for selecting and approving fundamentals data providers.

No real fundamentals provider may be integrated until provider selection and approval criteria are documented, reviewed, and accepted.

This is a governance and specification document. It is not an implementation plan and does not authorize provider integration, API access, SEC or EDGAR ingestion, scraping, runtime logic, financial scoring, Decision Engine behavior, report generation, Telegram delivery, or production pipeline execution.

## 2. Scope

In scope:

- fundamentals data provider selection criteria;
- raw provider evidence requirements;
- normalized fundamentals eligibility;
- source-data readiness criteria;
- provenance and auditability requirements;
- implementation prerequisites.

Out of scope:

- provider integration;
- API calls;
- SEC or EDGAR integration;
- broker integration;
- scraping;
- runtime logic;
- financial scoring;
- recommendations;
- Decision Engine behavior;
- report generation;
- Telegram delivery;
- portfolio source-of-truth changes.

## 3. Provider Categories

Allowed future provider categories, subject to approval:

- primary regulatory filings and official company filings;
- paid financial data APIs;
- reputable market or fundamentals data providers;
- manually curated fundamentals files, if explicitly governed;
- synthetic fixtures for tests only.

Disallowed or not-yet-approved provider categories:

- anonymous scraped pages;
- unversioned web snippets;
- broker screen output as fundamentals source;
- AI-generated fundamentals;
- analyst opinions treated as raw fundamentals;
- social media or forum content;
- derived investment ratings treated as raw fundamentals.

RESET-10L does not approve a specific commercial provider.

## 4. Provider Approval Criteria

A fundamentals provider may be approved only after the following criteria are documented:

- licensing and terms-of-use compatibility;
- data provenance;
- update frequency;
- field coverage;
- historical availability;
- timestamp availability;
- currency and unit clarity;
- restatement handling;
- missing-value handling;
- rate limit and reliability profile;
- auditability;
- deterministic parsing feasibility;
- cost awareness;
- absence of hidden investment conclusions;
- compatibility with raw-to-normalized separation.

Provider approval must also confirm that raw source capture, normalized fundamentals input, source-data readiness, generated outputs, and reporting display inputs remain separate.

## 5. Raw Source Data Requirements

Raw provider evidence must preserve:

- provider name;
- provider record identifier, if available;
- ticker, symbol, or entity identifier;
- source timestamp;
- retrieval timestamp;
- reported period;
- fiscal year and fiscal quarter, if available;
- original field names;
- original values;
- currency;
- units;
- provenance metadata;
- provider status or error status;
- missing-field evidence.

Mandatory rules:

```text
Raw source data is immutable evidence.
Raw source data is not normalized input.
Raw source data is not investment quality.
Raw source data must not contain portfolio recommendations.
```

Raw source capture must not be overwritten by normalized inputs, generated outputs, reporting outputs, Telegram text, or Decision Engine records.

## 6. Normalized Fundamentals Eligibility

Normalized fundamentals may contain:

- program-ready field names;
- typed values;
- explicit missing values;
- currency and unit normalization metadata;
- source references back to raw evidence;
- period metadata;
- validation status.

Mandatory rules:

```text
Normalized fundamentals are program-ready input.
Normalized fundamentals are not investment conclusions.
Normalized fundamentals must not contain BUY, SELL, HOLD, allocation, conviction, urgency, or tradeability logic.
Missing values must never be converted to zero.
```

Normalized fundamentals must preserve traceability to raw provider evidence through source provider, source reference, and source record identity.

## 7. Source-Data Readiness

Source-data readiness is a neutral status based on:

- availability;
- completeness;
- freshness;
- validity;
- provenance;
- parseability;
- consistency.

Source-data readiness must not be interpreted as:

- company quality;
- valuation attractiveness;
- buy or sell signal;
- conviction;
- urgency;
- allocation advice;
- tradeability.

Existing governance decision:

```text
SOURCE_DATA_READINESS_IS_NOT_INVESTMENT_QUALITY
```

Readiness communicates source-data condition only. It does not approve a company, reject a company, rank a company, or authorize a portfolio action.

## 8. Provider Approval Workflow

Future provider approval must follow this workflow:

1. Candidate provider identified.
2. Terms and licensing checked.
3. Sample raw evidence reviewed.
4. Field coverage mapped.
5. Missing-value behavior documented.
6. Normalization mapping designed.
7. Source-data readiness rules specified.
8. Synthetic contract tests planned.
9. Real provider implementation approved in a separate sprint.

RESET-10L does not approve implementation.

## 9. Future Implementation Guardrails

Any future implementation sprint must:

- be separately approved;
- start from this document;
- add tests before or with implementation;
- use synthetic fixtures first;
- avoid live calls in tests;
- keep raw and normalized storage separate;
- avoid changing Decision Engine authority;
- avoid Telegram and reporting side effects;
- avoid production pipeline execution.

Future implementation must not introduce provider calls, SEC or EDGAR calls, broker calls, network calls, API keys, credentials, report generation, Telegram delivery, or investment logic without explicit sprint approval.

## 10. Explicit Non-Goals

RESET-10L does not:

- integrate a provider;
- add an API key;
- make a live call;
- make an SEC or EDGAR call;
- create or modify a data file;
- generate a report;
- send a Telegram message;
- add investment logic;
- convert missing values to zero.

## 11. Backlog Impact

RESET-10L adds backlog items to `docs/active/backlog.md` for:

- provider approval decision;
- provider integration design;
- synthetic provider contract tests;
- real provider implementation after approval, mapping, and synthetic tests exist.

These items do not authorize implementation.

## 12. Relationship to Existing V2 Contracts

This document must be read with:

- `docs/active/fundamentals_raw_to_normalized.md`
- `docs/active/data_contracts.md`
- `docs/active/data_lifecycle.md`
- `docs/active/source_data_strategy.md`
- `docs/active/reporting_input_aggregation.md`
- `docs/active/portfolio_source_of_truth.md`

The active v2 line remains:

```text
raw fundamentals source capture
-> normalized fundamentals program-ready input
-> source-data readiness
-> downstream Decision Engine / Reporting
```

Reporting input aggregation is not source-of-truth. Telegram renderer input is downstream and not source-of-truth.
