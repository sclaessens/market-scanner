# ME-SC01 — SEC CompanyFacts Source Context Contract From Cached Raw Snapshots

Owner roles: Data Steward / Technical Architect / Financial Analyst / QA Lead / Governance Auditor

Sprint ID: ME-SC01

Job family: Source Context

Status: APPROVED CONTRACT FOR IMPLEMENTATION

## Purpose

ME-SC01 defines the Source Context contract for building SEC CompanyFacts source context from cached raw source snapshots produced by ME-SR01.

The purpose is to create a stable, source-only contract between raw SEC CompanyFacts snapshot evidence and later Fundamental Observation jobs.

ME-SC01 is a contract/documentation sprint. It does not implement runtime behavior.

## Background

The ME01–ME13 foundation phase established the Market Engine documentation structure, source intake flow, SEC CompanyFacts validation, field mapping, fundamental source context, non-decision observations, and job-based architecture.

ME-GOV01 established job-scoped sprint naming and requires all future work to use job-family sprint IDs.

ME-SR01 implemented raw SEC CompanyFacts snapshot persistence and cached raw loading under the Source Refresh job family.

ME-SC01 now defines the contract for the next layer: Source Context.

The Source Context job consumes cached raw SEC CompanyFacts source snapshots and emits canonical source-only context. It must not call SEC, refresh sources, create observations, analyze companies, issue recommendations, mutate portfolio/watchlist state, or deliver reports.

## Decision

Market Engine will build SEC CompanyFacts source context as an independent Source Context job.

The job must consume cached raw SEC CompanyFacts snapshots produced by the Source Refresh layer and emit source-only context snapshots under the approved Source Context persistence path.

The Source Context job may map, validate, normalize, and classify source availability. It may expose canonical fields, source readiness, missingness, provenance, and source diagnostics.

The Source Context job must not produce observations, derived calculations, analysis review, recommendation review, portfolio review, delivery output, Telegram output, or Decision Engine behavior.

## Job Boundary

ME-SC01 belongs to the `ME-SC` Source Context job family.

The Source Context boundary starts at cached raw SEC CompanyFacts snapshots and stops at persisted source-only context.

Input:

```text
data/market_engine/source_snapshots/sec_companyfacts/<source_refresh_run_id>/
```

Output:

```text
data/market_engine/source_contexts/fundamentals/<source_context_run_id>/
```

The job may read Source Refresh snapshots. It must not reach into provider/network behavior.

The job may use approved field mapping rules. It must not create financial interpretations beyond source readiness and field availability.

## Input Contract

The Source Context job consumes raw SEC CompanyFacts snapshot envelopes created by ME-SR01.

Expected raw snapshot envelope structure:

```json
{
  "metadata": {
    "ticker": "NVDA",
    "cik": "0001045810",
    "source_name": "sec_companyfacts",
    "fetched_at": "2026-06-15T00:00:00Z",
    "snapshot_id": "...",
    "payload_format_version": "sec-companyfacts-raw-v1"
  },
  "raw_payload": {
    "...": "exact SEC CompanyFacts payload"
  }
}
```

Required snapshot metadata:

- `ticker`;
- `cik`;
- `source_name`;
- `fetched_at`;
- `snapshot_id`;
- `payload_format_version`.

Required source name:

```text
sec_companyfacts
```

Required payload format version:

```text
sec-companyfacts-raw-v1
```

The job may also consume run-level Source Refresh metadata and manifests where available:

- `snapshot_metadata.json`;
- `ticker_manifest.csv`;
- `provider_errors.csv`.

Provider-error rows must remain explicit. They must not be converted into fake source context availability.

## Approved Field Mapping Scope

The initial SEC CompanyFacts Source Context contract is limited to the approved ME10/ME11 SEC field mapping set:

| Canonical field | Purpose | Source |
|---|---|---|
| `revenue` | Source-level revenue field availability and provenance | SEC CompanyFacts approved aliases |
| `net_income` | Source-level net income field availability and provenance | SEC CompanyFacts approved aliases |
| `operating_cash_flow` | Source-level operating cash flow field availability and provenance | SEC CompanyFacts approved aliases |
| `capital_expenditures` | Source-level capital expenditures field availability and provenance | SEC CompanyFacts approved aliases |

No additional canonical financial fields are approved by ME-SC01.

Adding fields such as assets, liabilities, equity, EPS, debt, cash, margins, growth, valuation metrics, or ratios requires a later explicit sprint.

## Output Contract

The Source Context job must emit source-only context output.

Recommended persisted output path:

```text
data/market_engine/source_contexts/fundamentals/<source_context_run_id>/
```

Recommended files:

```text
fundamental_source_context.json
field_provenance.csv
missing_fields.csv
context_metadata.json
```

The exact implementation may use fewer files if the same contract is preserved, but the public output must expose:

- context run ID;
- source refresh run ID or source snapshot reference;
- ticker;
- CIK;
- source name;
- source snapshot path or snapshot ID;
- context build timestamp;
- context format version;
- source availability state;
- canonical fields;
- field-level status;
- field-level raw value where selected;
- field-level unit;
- field-level period metadata where available;
- field-level filing metadata where available;
- selected SEC tag;
- alias selection reason;
- missing fields;
- provider-error linkage where relevant;
- source limitations.

## Context Format Version

The first Source Context output format should use:

```text
sec-companyfacts-source-context-v1
```

Any later incompatible output schema change requires a new documented version and compatibility tests.

## Source Availability States

The Source Context job must make source availability explicit.

Approved context-level states:

| State | Meaning |
|---|---|
| `AVAILABLE` | Raw snapshot is valid and all required canonical fields are available under approved mapping rules. |
| `PARTIAL` | Raw snapshot is valid, but one or more required canonical fields are missing or unavailable. |
| `MISSING` | No usable raw snapshot exists for the requested ticker/CIK. |
| `INVALID` | Raw snapshot exists but cannot be used because metadata, format, or payload structure is invalid. |
| `PROVIDER_ERROR` | Source Refresh recorded a provider error for the entity instead of a successful raw payload. |
| `UNSUPPORTED` | The entity or snapshot format is unsupported by the current Source Context contract. |

The implementation may add lower-level internal error categories, but the public context state must map to one of these approved states unless a later contract sprint extends the state model.

## Field-Level States

Each approved canonical field must expose a field-level state.

Approved field-level states:

| State | Meaning |
|---|---|
| `PRESENT` | Approved SEC tag was selected and source value/provenance are available. |
| `MISSING` | No approved SEC tag/value was found for the field. |
| `INVALID` | Candidate source value exists but cannot be accepted under the mapping contract. |
| `UNSUPPORTED` | The field is not supported by the current context contract. |

Field-level missingness must remain explicit.

Missing numeric values must not be converted to zero.

## Provenance Requirements

For each `PRESENT` canonical field, the Source Context output must preserve enough provenance to trace the value back to raw SEC CompanyFacts evidence.

Required provenance where available:

- canonical field name;
- selected SEC tag;
- taxonomy namespace if available;
- unit;
- raw value;
- fiscal year / fiscal period where available;
- period start/end where available;
- form type where available;
- filed date where available;
- accession number where available;
- frame where available;
- source snapshot ID;
- source snapshot path or stable reference;
- alias selection reason.

The Source Context job may normalize selected field structure, but it must not discard raw source traceability.

## Missingness and Error Rules

The Source Context job must preserve missingness and source failures explicitly.

Rules:

1. Missing raw snapshots become `MISSING`, not fake available context.
2. Invalid raw snapshot JSON becomes `INVALID`.
3. Missing required snapshot metadata becomes `INVALID`.
4. Unsupported payload format becomes `UNSUPPORTED` or `INVALID`, depending on implementation detail documented in the audit.
5. Provider error manifest rows become `PROVIDER_ERROR`.
6. Missing required canonical fields make the context `PARTIAL` unless all fields are unavailable, in which case implementation may use `PARTIAL` with all fields `MISSING` or `MISSING` if no usable source evidence exists.
7. Missing numeric values must remain missing and must never be converted to zero.
8. Context output must not invent values, substitute unapproved aliases, or infer missing SEC facts.

## Authority Boundary

The Source Context job may say:

- a raw source snapshot exists;
- a source is usable or unusable;
- a field is present, missing, invalid, or unsupported;
- a selected SEC tag maps to an approved canonical field;
- provenance exists for a selected field;
- source limitations are present.

The Source Context job must not say:

- whether a company is financially strong or weak;
- whether a stock is attractive;
- whether to buy, sell, hold, watch, avoid, rank, score, or allocate;
- whether free cash flow is positive or negative;
- whether growth, margin, valuation, or quality is good or bad;
- whether the portfolio should change;
- whether Telegram/reporting should be sent.

## Persistence Contract

Source Context output must be persisted separately from raw source snapshots.

Raw source input remains under:

```text
data/market_engine/source_snapshots/sec_companyfacts/<source_refresh_run_id>/
```

Source Context output goes under:

```text
data/market_engine/source_contexts/fundamentals/<source_context_run_id>/
```

The Source Context job must not overwrite raw source snapshots.

The Source Context job must not write to old legacy paths:

```text
data/processed/
data/generated/
data/logs/
data/normalized/
reports/
data/portfolio/
data/watchlist/
```

## Test Contract

Implementation tests for this contract must use mocked, synthetic, fixture-based, or temporary local payloads only.

Automated tests must not make live SEC/provider calls.

Required tests for the implementation sprint:

1. Loads a valid cached raw SEC CompanyFacts snapshot and emits source context.
2. Emits `AVAILABLE` when all approved canonical fields are present.
3. Emits `PARTIAL` when one or more approved canonical fields are missing.
4. Emits `MISSING` when no usable raw snapshot exists.
5. Emits `INVALID` for invalid JSON or missing required snapshot metadata.
6. Emits `PROVIDER_ERROR` from provider error manifest evidence.
7. Preserves selected SEC tag, raw value, unit, period metadata, filing metadata, and snapshot reference where available.
8. Does not convert missing numeric values to zero.
9. Does not select unapproved aliases.
10. Does not create observations, analysis, recommendations, portfolio review, delivery output, Telegram output, or Decision Engine behavior.
11. Does not write to old legacy data/report paths.
12. Does not call live providers.

## Documentation Requirements for Implementation

The implementation sprint must document:

- sprint ID;
- job family;
- input contract;
- output contract;
- persistence path;
- context format version;
- state model;
- field-level state model;
- provenance behavior;
- missingness behavior;
- tests run;
- files changed;
- boundary confirmation.

## Recommended Implementation Sprint

The recommended implementation sprint after ME-SC01 is:

```text
ME-SC02 — Implement SEC CompanyFacts Source Context from cached raw snapshots
```

ME-SC02 should implement this contract without expanding into observations, derived calculations, analysis, recommendations, portfolio review, delivery, Telegram, or Decision Engine behavior.

## Acceptance Criteria

ME-SC01 is complete when:

- the SEC CompanyFacts Source Context contract is documented;
- the input contract from ME-SR01 raw snapshots is defined;
- the output contract for source-only context is defined;
- approved context-level states are defined;
- approved field-level states are defined;
- provenance requirements are defined;
- missingness and provider-error rules are defined;
- persistence paths are defined;
- test requirements are defined;
- authority boundaries are explicit;
- the next implementation sprint is identified as `ME-SC02`;
- backlog and audit records are updated;
- no Python code, tests, data files, generated files, provider calls, runtime behavior, recommendation behavior, portfolio behavior, delivery behavior, Telegram behavior, or Decision Engine behavior are changed.

## Governance Status

Status: Approved Source Context contract.

Effective from: immediately after ME-SR01.

Implementation is not part of ME-SC01. Runtime implementation requires a later Source Context implementation sprint.
