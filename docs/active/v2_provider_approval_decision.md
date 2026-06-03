# V2 Provider Approval Decision

Status: ACTIVE
Reset stage: RESET-10L-BL1

## 1. Purpose

This document records the first provider strategy approval decision for v2 fundamentals data.

It answers which provider strategy is approved as the first governed path toward real fundamentals data.

This is a governance decision. It is not an implementation sprint and does not authorize provider integration, API access, SEC or EDGAR ingestion, scraping, runtime logic, normalization code, financial scoring, Decision Engine behavior, report generation, Telegram delivery, or production pipeline execution.

## 2. Decision

The first approved v2 fundamentals provider strategy is a primary-first, provenance-first strategy.

This means:

- prefer official filings or official company/regulatory source evidence where feasible;
- allow future use of a commercial or third-party access layer only if it preserves traceability to original source evidence;
- require raw evidence capture before normalization;
- require normalized fundamentals to remain program-ready input only;
- require source-data readiness to remain neutral.

Commercial or third-party APIs may be considered later, but only if provider records preserve provenance, timestamps, reported periods, currency and unit clarity, missing-value behavior, auditability, and raw-to-normalized separation.

## 3. Rationale

The primary-first, provenance-first strategy is approved because it provides:

- highest auditability;
- strongest provenance;
- best alignment with raw-to-normalized separation;
- lower risk of hidden investment conclusions;
- clearer governance review before any Decision Engine use;
- safer foundation before later provider integration.

The strategy also protects the existing v2 doctrine:

```text
SOURCE_DATA_READINESS_IS_NOT_INVESTMENT_QUALITY
RAW_SOURCE_DATA_MUST_REMAIN_SEPARATE_FROM_NORMALIZED_PROGRAM_READY_INPUT
NORMALIZED_FUNDAMENTALS_ARE_PROGRAM_READY_INPUT_NOT_INVESTMENT_CONCLUSIONS
```

## 4. Approved Provider Categories for First Path

Approved for the first path:

- official company filings;
- regulatory filings;
- official company investor relations data, if timestamped and traceable;
- provider or API records that clearly map back to official filings and preserve provenance.

Conditionally allowed later:

- paid financial data APIs;
- reputable fundamentals providers;
- manually curated fundamentals files, but only with explicit governance and provenance metadata.

Not approved as fundamentals source:

- AI-generated fundamentals;
- analyst ratings or analyst opinions treated as raw fundamentals;
- anonymous scraped webpages;
- unversioned web snippets;
- broker screen output;
- social media or forum content;
- derived investment scores or rankings;
- recommendation feeds.

## 5. Mandatory Approval Constraints

Any later concrete provider must satisfy these constraints before implementation:

- licensing and terms-of-use review;
- provenance preservation;
- source timestamp and retrieval timestamp support;
- reported period support;
- fiscal year and fiscal quarter support where applicable;
- currency and unit clarity;
- explicit missing-value behavior;
- no missing-to-zero conversion;
- deterministic parsing feasibility;
- raw and normalized data separation;
- audit trail from normalized field back to raw evidence;
- no embedded investment recommendation logic;
- no Decision Engine authority expansion;
- no reporting or Telegram side effects.

## 6. Non-Approval of Implementation

```text
This decision does not approve implementation.
This decision does not approve a specific API key, provider account, live call, scraper, SEC/EDGAR integration, broker integration, data file, or runtime pipeline.
```

Any future implementation requires a separate approved sprint.

## 7. Impact on Backlog

This decision resolves `RESET-10L-BL1 — Provider Approval Decision`.

The active backlog remains responsible for follow-up work:

- `RESET-10L-BL2 — Provider Integration Design`
- `RESET-10L-BL3 — Synthetic Provider Contract Tests`
- `RESET-10L-BL4 — Real Provider Implementation`

`RESET-10L-BL2` must follow this primary-first, provenance-first decision.

## 8. Future Implementation Gates

Before real implementation, the project still requires:

1. Provider integration design.
2. Raw capture field mapping.
3. Normalized fundamentals mapping.
4. Missing-value behavior definition.
5. Source-data readiness rules.
6. Synthetic contract tests.
7. Separate real implementation sprint approval.

## 9. Explicit Non-Goals

RESET-10L-BL1 does not include:

- provider integration;
- API calls;
- SEC or EDGAR calls;
- broker calls;
- scrapers;
- live-data calls;
- credentials;
- data file creation or modification;
- source code changes;
- tests;
- reports;
- Telegram delivery;
- production pipeline execution;
- scoring;
- recommendations;
- BUY, SELL, HOLD, allocation, conviction, urgency, or tradeability logic;
- missing-value-to-zero conversion.

## 10. Relationship to Existing V2 Contracts

This decision must be read with:

- `docs/active/v2_data_provider_selection_and_approval.md`
- `docs/active/fundamentals_raw_to_normalized.md`
- `docs/active/data_contracts.md`
- `docs/active/data_lifecycle.md`
- `docs/active/source_data_strategy.md`

The active fundamentals/source-data line remains:

```text
raw fundamentals source capture
-> normalized fundamentals program-ready input
-> source-data readiness
-> downstream Decision Engine / Reporting
```
