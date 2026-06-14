# V2 Persistence Contract and Fixture Design

Status: ACTIVE
Reset stage: RESET-10L-BL12

## Purpose

This document translates the `RESET-10L-BL11` persistence design into concrete
contract and fixture requirements for future v2 persistence work.

This is a documentation/design-only artifact. It does not authorize code,
contract-test implementation, fixture file creation, provider calls, data writes,
production pipeline execution, report generation, Telegram delivery, Decision
Engine behavior, scoring, recommendations, BUY, SELL, HOLD, allocation,
conviction, urgency, target-price, or tradeability logic.

The purpose is to define the schemas, fixture families, and contract-test
requirements that a future implementation sprint must satisfy before any
production persistence function is written.

## Scope

This design covers future contracts and fixtures for three separate persistence
families:

```text
raw source evidence
normalized fundamentals
source-data readiness
```

The three families must remain separate. Raw evidence is immutable input
evidence. Normalized fundamentals are program-ready input. Source-data readiness
is neutral source/data status. None of them are investment conclusions.

## Contract Family 1 — Raw Source Evidence

### Purpose

Raw source evidence preserves the original provider/source context and provenance
needed to audit future normalized fundamentals.

### Required schema fields

A future raw evidence schema should include:

- `raw_evidence_id`;
- `provider_name`;
- `provider_category`;
- `provider_record_id`;
- `original_source_reference`;
- `ticker`;
- `symbol`;
- `entity_identifier`;
- `source_timestamp`;
- `retrieval_timestamp`;
- `reported_period`;
- `fiscal_year`;
- `fiscal_quarter`;
- `currency`;
- `unit`;
- `raw_fields`;
- `missing_field_evidence`;
- `provenance_metadata`;
- `raw_payload_hash`;
- `capture_version`;
- `validation_warnings`.

### Required behavior

Future raw evidence contracts must require that:

- provenance is present;
- source reference is preserved;
- source and retrieval timestamps are preserved;
- raw field names are not silently renamed in the raw layer;
- missing field evidence is explicit;
- raw payload hash or equivalent evidence identifier is present;
- capture version is present;
- raw evidence is not converted into investment language;
- missing values are not converted to zero;
- raw evidence writes are append-oriented or versioned if writes are later
  approved.

### Forbidden behavior

Raw evidence contracts must reject or flag:

- BUY, SELL, or HOLD labels;
- recommendations;
- target prices;
- allocation instructions;
- conviction or urgency labels;
- tradeability decisions;
- portfolio action text;
- reporting or Telegram text;
- missing-to-zero conversion.

## Contract Family 2 — Normalized Fundamentals

### Purpose

Normalized fundamentals provide stable, typed, program-ready records derived from
raw evidence.

### Required schema fields

A future normalized fundamentals schema should include:

- `normalized_record_id`;
- `raw_evidence_id`;
- `provider_name`;
- `original_source_reference`;
- `ticker`;
- `entity_identifier`;
- `metric_name`;
- `metric_value`;
- `metric_value_status`;
- `currency`;
- `currency_status`;
- `unit`;
- `unit_status`;
- `reported_period`;
- `fiscal_year`;
- `fiscal_quarter`;
- `source_timestamp`;
- `retrieval_timestamp`;
- `normalization_version`;
- `validation_warnings`.

### Required behavior

Future normalized fundamentals contracts must require that:

- every normalized record links back to raw evidence;
- metric names use the governed normalized vocabulary;
- metric values are typed only when valid;
- missing metric values remain explicit;
- missing currency remains explicit;
- missing unit remains explicit;
- fiscal period metadata is preserved;
- normalization version is recorded;
- validation warnings remain visible;
- no normalized record implies investment quality;
- no missing value becomes `0`, `0.0`, `"0"`, `False`, or an empty string.

### Forbidden behavior

Normalized fundamentals contracts must reject or flag:

- investment conclusions;
- recommendations;
- target prices;
- BUY, SELL, or HOLD labels;
- allocation, conviction, urgency, or tradeability fields;
- hidden missing values;
- default zeros for unavailable fundamentals;
- records without raw evidence linkage.

## Contract Family 3 — Source-Data Readiness

### Purpose

Source-data readiness summarizes availability, completeness, freshness, validity,
provenance, parseability, and consistency conditions.

Readiness is not investment quality.

### Required schema fields

A future readiness schema should include:

- `readiness_record_id`;
- `raw_evidence_id`;
- `normalized_record_set_id`;
- `ticker`;
- `provider_name`;
- `readiness_state`;
- `source_data_status`;
- `missing_fundamentals_count`;
- `partial_data_count`;
- `stale_data_count`;
- `invalid_data_count`;
- `provenance_status`;
- `parseability_status`;
- `consistency_status`;
- `freshness_status`;
- `readiness_warnings`;
- `readiness_version`.

### Required behavior

Future readiness contracts must require that:

- readiness is derived only from source/data conditions;
- missing, partial, stale, invalid, or provenance gaps remain visible;
- readiness links back to raw evidence and normalized record sets;
- readiness does not imply company quality;
- readiness does not imply valuation attractiveness;
- readiness does not authorize Decision Engine action;
- readiness failure states fail closed.

### Forbidden behavior

Readiness contracts must reject or flag:

- BUY, SELL, or HOLD labels;
- recommendation strength;
- valuation scores;
- target prices;
- allocation guidance;
- conviction or urgency labels;
- tradeability decisions;
- hidden missing data;
- conversion of missing values to zero.

## Fixture Families

Future fixture files should be synthetic, small, deterministic, and committed
only under approved test fixture paths. They must not contain credentials, raw
live provider payloads, private data, or production data.

### Fixture family A — complete synthetic source

Purpose: prove that a complete source-shaped response can produce raw evidence,
normalized fundamentals, and ready/complete source-data readiness.

Required properties:

- all required provenance fields present;
- all required normalized metrics present;
- currency and unit present;
- fiscal period present;
- no validation warnings except expected neutral metadata;
- no investment semantics.

### Fixture family B — partial synthetic source

Purpose: prove that partial source data remains explicit and does not become
zero-filled.

Required properties:

- some required fundamentals missing;
- missing field evidence present;
- normalized records include explicit missing statuses;
- readiness state remains partial or equivalent neutral state;
- missing counts are non-zero;
- no default zero substitution.

### Fixture family C — invalid synthetic source

Purpose: prove that invalid source data fails closed.

Required properties:

- invalid or unparseable field value present;
- validation warning present;
- readiness reflects invalid data condition;
- no data is silently normalized;
- no investment output is produced.

### Fixture family D — stale synthetic source

Purpose: prove that stale source data remains visible.

Required properties:

- source timestamp outside approved freshness window;
- freshness warning present;
- readiness reflects stale condition;
- raw evidence remains traceable;
- no downstream action is authorized.

### Fixture family E — provenance gap source

Purpose: prove that absent or incomplete provenance fails closed.

Required properties:

- missing source reference, timestamp, or provider identity;
- provenance warning present;
- readiness reflects provenance gap;
- no normalized record may appear without explicit warning;
- no Decision Engine use is authorized.

### Fixture family F — forbidden semantics source

Purpose: prove that forbidden investment semantics are rejected or flagged.

Required properties:

- synthetic forbidden fields such as BUY, SELL, HOLD, target price, allocation,
  conviction, urgency, recommendation, or tradeability;
- contract expectation that these are rejected, ignored, or explicitly flagged;
- no forbidden field appears in persisted raw, normalized, or readiness outputs
  unless captured only as a validation issue.

## Contract-Test Requirements

A future `RESET-10L-BL13` implementation should add tests that verify the
persistence boundary without production writes.

Required test categories:

1. raw evidence schema tests;
2. normalized fundamentals schema tests;
3. readiness schema tests;
4. raw-to-normalized provenance-link tests;
5. missing-value preservation tests;
6. partial-data readiness tests;
7. invalid-data failure tests;
8. stale-data readiness tests;
9. provenance-gap failure tests;
10. forbidden-semantics rejection tests;
11. no-side-effect tests for reports, Telegram, pipeline, and production data;
12. fixture determinism tests.

All tests must use synthetic or fixture data only.

Tests must not:

- make live provider calls;
- make SEC, EDGAR, broker, or network calls;
- use credentials;
- write production data;
- generate reports;
- create Telegram artifacts;
- run the production pipeline;
- touch portfolio or watchlist data;
- add Decision Engine behavior.

## Proposed Future Fixture Paths

The following paths are candidates for future fixture work only. This document
does not create or approve the files.

```text
tests/fixtures/fundamentals/persistence/raw_complete_source.json
tests/fixtures/fundamentals/persistence/raw_partial_source.json
tests/fixtures/fundamentals/persistence/raw_invalid_source.json
tests/fixtures/fundamentals/persistence/raw_stale_source.json
tests/fixtures/fundamentals/persistence/raw_provenance_gap_source.json
tests/fixtures/fundamentals/persistence/raw_forbidden_semantics_source.json
```

The future implementation may choose different names, but it must preserve the
fixture intent and guardrails.

## Proposed Future Test Paths

The following paths are candidates for future contract tests only. This document
does not create or approve the files.

```text
tests/contract/test_v2_persistence_raw_evidence_contracts.py
tests/contract/test_v2_persistence_normalized_fundamentals_contracts.py
tests/contract/test_v2_persistence_readiness_contracts.py
tests/contract/test_v2_persistence_fixture_contracts.py
```

## Acceptance Criteria for RESET-10L-BL13

A future BL13 may be considered complete only if it adds synthetic contract tests
and fixtures that prove:

- raw evidence schema expectations are enforced;
- normalized fundamentals schema expectations are enforced;
- readiness schema expectations are enforced;
- provenance linkage is required;
- missing values remain explicit;
- missing values are not converted to zero;
- partial, invalid, stale, and provenance-gap states remain neutral;
- forbidden investment semantics are rejected or flagged;
- no provider calls are made;
- no production data files are written;
- no reports or Telegram artifacts are generated;
- no production pipeline behavior is invoked;
- no Decision Engine investment behavior is added.

## Non-Goals

RESET-10L-BL12 does not:

- add fixtures;
- add tests;
- add code;
- write production data;
- execute provider calls;
- execute SEC, EDGAR, broker, or network calls;
- add credentials;
- generate reports;
- create Telegram artifacts;
- run the production pipeline;
- add Decision Engine behavior;
- approve investment analysis;
- add BUY, SELL, HOLD, allocation, conviction, urgency, tradeability, scoring,
  target-price, or recommendation behavior.

## Next Step

The next candidate step is:

```text
RESET-10L-BL13 — Synthetic Persistence Contract Tests
```

That future step should create the approved synthetic fixtures and contract tests
without live provider calls, production data writes, reports, Telegram, pipeline
execution, or Decision Engine investment logic.
