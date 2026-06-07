# V2 Provider Integration Design

Status: ACTIVE
Reset stage: RESET-10L-BL2

## 1. Purpose

This document designs the future v2 provider integration path for the first approved fundamentals data strategy.

This is a design document, not an implementation sprint.

```text
This document does not approve implementation.
This document does not add provider calls.
This document does not add source code, tests, reports, data files, or runtime behavior.
```

The design target is:

```text
approved provider strategy
-> raw source capture mapping
-> normalized fundamentals mapping
-> missing-value behavior
-> source-data readiness rules
-> synthetic contract test plan
```

## 2. Approved Strategy Alignment

RESET-10L-BL1 approved the first v2 fundamentals provider strategy:

```text
Primary-first, provenance-first
```

Future implementation must prioritize official or primary-source evidence where feasible.

Possible future access layers may include third-party or commercial providers only if they preserve:

- provenance;
- source timestamps;
- retrieval timestamps;
- reported periods;
- fiscal periods;
- currency and unit clarity;
- explicit missing-value behavior;
- raw-to-normalized separation;
- auditability.

Third-party access may simplify retrieval, but it must not replace traceability to original source evidence.

## 3. Scope

In scope:

- future provider integration design;
- raw source capture design;
- normalized fundamentals mapping design;
- source-data readiness design;
- missing-value behavior design;
- synthetic contract test planning;
- implementation gates.

Out of scope:

- provider implementation;
- API calls;
- SEC or EDGAR calls;
- broker calls;
- web scraping;
- source code changes;
- test creation;
- runtime pipeline changes;
- report generation;
- Telegram delivery;
- Decision Engine behavior changes;
- scoring or recommendations;
- real data files.

## 4. Provider Access Model

The future provider access model must support these possible access paths:

1. Official filings or official company/regulatory source evidence.
2. Provider or API records traceable back to official filings.
3. Manually curated fundamentals files only if explicitly governed later.

RESET-10L-BL2 does not select a concrete commercial provider.

This design does not define API credentials, endpoints, provider accounts, URLs, scripts, commands, or runtime implementation details.

## 5. Raw Source Capture Design

Future implementation must capture raw source evidence before normalization.

Minimum raw evidence fields:

- `provider_name`
- `provider_category`
- `provider_record_id`, if available
- `original_source_reference`, if available
- `ticker`
- `symbol`
- `entity_identifier`, if available
- `source_timestamp`
- `retrieval_timestamp`
- `reported_period`
- `fiscal_year`
- `fiscal_quarter`, if applicable
- `original_field_name`
- `original_field_value`
- `original_currency`
- `original_unit`
- `provider_status`
- `provider_error_status`, if applicable
- `missing_field_evidence`
- `provenance_metadata`
- `raw_payload_hash`, if feasible
- `capture_version`

Mandatory rules:

```text
Raw source capture is immutable evidence.
Raw source capture must not overwrite or reinterpret source values.
Raw source capture must not contain investment conclusions.
Raw source capture must not contain BUY, SELL, HOLD, allocation, conviction, urgency, or tradeability logic.
```

Raw capture records must preserve source evidence, not improve it, score it, rank it, or turn it into recommendations.

## 6. Normalized Fundamentals Mapping Design

Normalized fundamentals are program-ready input derived from raw evidence.

The future normalized mapping may use the following fields.

| Normalized field | Expected value type | Source evidence requirement | Currency/unit handling | Missing-value behavior | Allowed derived behavior | Required |
|---|---|---|---|---|---|---|
| `revenue` | Decimal-like numeric or explicit missing | Reported revenue field | Preserve reported currency and unit, normalize only with metadata | Missing remains missing | None | Optional |
| `gross_profit` | Decimal-like numeric or explicit missing | Reported gross profit field | Preserve reported currency and unit | Missing remains missing | None | Optional |
| `operating_income` | Decimal-like numeric or explicit missing | Reported operating income field | Preserve reported currency and unit | Missing remains missing | None | Optional |
| `net_income` | Decimal-like numeric or explicit missing | Reported net income field | Preserve reported currency and unit | Missing remains missing | None | Optional |
| `eps_basic` | Decimal-like numeric or explicit missing | Reported basic EPS field | Preserve reported currency and unit where applicable | Missing remains missing | None | Optional |
| `eps_diluted` | Decimal-like numeric or explicit missing | Reported diluted EPS field | Preserve reported currency and unit where applicable | Missing remains missing | None | Optional |
| `total_assets` | Decimal-like numeric or explicit missing | Reported total assets field | Preserve reported currency and unit | Missing remains missing | None | Optional |
| `total_liabilities` | Decimal-like numeric or explicit missing | Reported total liabilities field | Preserve reported currency and unit | Missing remains missing | None | Optional |
| `shareholders_equity` | Decimal-like numeric or explicit missing | Reported shareholders equity field | Preserve reported currency and unit | Missing remains missing | None | Optional |
| `cash_and_equivalents` | Decimal-like numeric or explicit missing | Reported cash and equivalents field | Preserve reported currency and unit | Missing remains missing | None | Optional |
| `operating_cash_flow` | Decimal-like numeric or explicit missing | Reported operating cash flow field | Preserve reported currency and unit | Missing remains missing | None | Optional |
| `capital_expenditures` | Decimal-like numeric or explicit missing | Reported capital expenditures field | Preserve reported currency and unit | Missing remains missing | None | Optional |
| `free_cash_flow` | Decimal-like numeric or explicit missing | Reported free cash flow or traceable inputs | Preserve reported currency and unit | Missing remains missing | May be derived only from complete, traceable required inputs | Optional |
| `shares_outstanding` | Decimal-like numeric or explicit missing | Reported shares outstanding field | Preserve share unit metadata | Missing remains missing | None | Optional |
| `currency` | Text | Raw evidence currency or provider metadata | Required for monetary fields where available | Missing remains missing | None | Required when monetary values exist |
| `unit` | Text | Raw evidence unit or provider metadata | Required for scaled values where available | Missing remains missing | None | Required when scaled values exist |
| `fiscal_year` | Text or integer-like value | Period metadata | Not applicable | Missing remains missing | None | Required |
| `fiscal_quarter` | Text or integer-like value | Period metadata where applicable | Not applicable | Missing remains missing | None | Optional |
| `period_start` | Date-like text or explicit missing | Period metadata where available | Not applicable | Missing remains missing | None | Optional |
| `period_end` | Date-like text | Period metadata | Not applicable | Missing remains missing | None | Required |
| `source_reference` | Text | Raw evidence record reference | Not applicable | Missing invalidates traceability | None | Required |
| `normalization_status` | Text | Normalization process metadata | Not applicable | Missing invalidates normalized row | None | Required |
| `validation_status` | Text | Validation process metadata | Not applicable | Missing invalidates normalized row | None | Required |

Mandatory rules:

```text
Normalized fundamentals are program-ready input.
Normalized fundamentals are not investment conclusions.
Normalized fundamentals must preserve missing values explicitly.
Missing values must never be converted to zero.
```

Normalized records must preserve source provider, source reference, source record identity, period metadata, currency metadata, and unit metadata wherever applicable.

## 7. Missing-Value Behavior

Mandatory missing-value rules:

- missing source fields remain missing;
- unavailable values remain explicit missing values;
- parse failures are not converted to zero;
- provider errors are not converted to zero;
- stale values are not silently reused as current values;
- derived values may only be calculated when all required inputs are present;
- missing derived inputs must produce missing derived outputs;
- every missing field must remain traceable to missing evidence or provider status.

Explicitly forbidden:

```text
missing -> 0
unknown -> 0
not reported -> 0
parse error -> 0
provider error -> 0
stale value -> current value
```

## 8. Source-Data Readiness Design

Readiness statuses are neutral source-data states.

| Readiness status | Meaning | Allowed trigger | Forbidden interpretation |
|---|---|---|---|
| `READY` | Source data appears available, parseable, current, and traceable for the mapped fields | Required source evidence is present and valid | Company quality or investment approval |
| `PARTIAL_DATA` | Some expected source data is present and some is missing | One or more optional or expected fields are missing | Weak company or low conviction |
| `INSUFFICIENT_DATA` | Required data is too incomplete for the next analytical step | Required source evidence or traceability is missing | Sell signal or rejection |
| `STALE_DATA` | Source evidence is older than the approved freshness policy | Source timestamp or retrieval timestamp is stale | Negative investment view |
| `INVALID_DATA` | Data is malformed, inconsistent, or not parseable under the contract | Parse failure, invalid period metadata, invalid unit, or failed validation | Company quality judgment |
| `SOURCE_MISSING` | Expected source evidence is unavailable | Provider lacks record, filing missing, or source absent | Company quality judgment |
| `PROVIDER_ERROR` | Provider/access layer reports an error | Provider error status, unavailable response, or failed access layer | Company quality judgment |

Mandatory rules:

```text
Source-data readiness is not investment quality.
Source-data readiness is not company quality.
Source-data readiness is not valuation attractiveness.
Source-data readiness is not BUY, SELL, HOLD, allocation, conviction, urgency, or tradeability logic.
```

## 9. Provenance and Auditability

Future implementation must preserve this traceability chain:

```text
normalized field
-> raw evidence record
-> provider/source record
-> source timestamp / retrieval timestamp
-> period metadata
```

Every normalized value must be traceable back to raw evidence.

Derived normalized values must document:

- formula;
- required inputs;
- raw source evidence for each input;
- missing-value behavior;
- derivation status.

Readiness status must be traceable to source availability, completeness, freshness, validity, provenance, parseability, and consistency.

## 10. Storage Boundary Design

Future implementation must respect the existing v2 data lifecycle:

```text
data/raw/
data/normalized/
data/generated/
data/local/
```

Conceptual storage boundaries:

- `data/raw/` is for raw source evidence.
- `data/normalized/` is for program-ready normalized fundamentals.
- `data/generated/` is for generated downstream artifacts, if approved later.
- `data/local/` is for local-only runtime or private material, if governed.

RESET-10L-BL2 does not create, modify, move, or populate any file under `data/`.

## 11. Synthetic Contract Test Plan

The next required step is:

```text
RESET-10L-BL3 — Synthetic Provider Contract Tests
```

Future synthetic tests should prove:

1. Synthetic raw provider evidence can be represented.
2. Synthetic raw evidence can be mapped to normalized fundamentals.
3. Missing values remain missing.
4. Missing values are not converted to zero.
5. Provider errors produce neutral readiness statuses.
6. Stale source timestamps produce neutral stale readiness.
7. Normalized fields preserve source references.
8. Derived fields are only produced when all required inputs exist.
9. Source-data readiness remains neutral and does not become investment quality.
10. No Decision Engine, reporting, Telegram, pipeline, or file input/output side effects occur.

RESET-10L-BL2 does not add tests.

## 12. Implementation Gates

Real provider implementation cannot begin until all gates are satisfied:

1. Provider approval decision completed.
2. Provider integration design completed.
3. Synthetic provider contract tests completed.
4. Real provider implementation sprint approved separately.
5. Provider licensing and terms reviewed.
6. Provider access method approved.
7. Raw and normalized schema boundaries confirmed.
8. Missing-value behavior confirmed.
9. Readiness status behavior confirmed.
10. No Decision Engine authority expansion approved.

## 13. Explicit Non-Goals

RESET-10L-BL2 does not include:

- provider integration;
- API calls;
- SEC or EDGAR calls;
- broker calls;
- scrapers;
- live-data calls;
- credentials;
- source code changes;
- test changes;
- data file creation;
- data file modification;
- report generation;
- Telegram delivery;
- production pipeline execution;
- scoring;
- recommendations;
- BUY, SELL, HOLD, allocation, conviction, urgency, or tradeability logic;
- missing-value-to-zero conversion.

## 14. Backlog Impact

This design resolves `RESET-10L-BL2 — Provider Integration Design`.

The active backlog remains responsible for follow-up work:

- `RESET-10L-BL3 — Synthetic Provider Contract Tests`
- `RESET-10L-BL4 — Real Provider Implementation`

`RESET-10L-BL3` must use this provider integration design.
