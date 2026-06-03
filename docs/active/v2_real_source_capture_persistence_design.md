# V2 Real-Source Capture Persistence Design

Status: ACTIVE
Reset stage: RESET-10L-BL11

## Purpose

This document defines the proposed persistence design for v2 real-source capture,
normalized fundamentals, and neutral source-data readiness.

This is a design-only governance document. It does not authorize implementation,
provider calls, data writes, production pipeline execution, report generation,
Telegram delivery, Decision Engine behavior, scoring, recommendations, BUY, SELL,
HOLD, allocation, conviction, urgency, target-price, or tradeability logic.

The design exists to preserve the core v2 doctrine:

```text
raw source evidence
-> normalized program-ready fundamentals
-> neutral source-data readiness
```

Raw source evidence, normalized fundamentals, and readiness records must remain
separate. None of these records are investment conclusions.

## Design Scope

This design covers the future persistence boundary for:

- raw provider/source evidence;
- normalized program-ready fundamentals;
- neutral source-data readiness;
- source provenance and audit metadata;
- missing-value and validation metadata;
- controlled write authorization requirements.

This design does not cover:

- live provider implementation;
- credential handling implementation;
- automatic source fetching;
- production pipeline scheduling;
- Decision Engine investment use;
- reporting or Telegram rendering;
- portfolio or watchlist updates;
- generated daily reports;
- BUY, SELL, HOLD, allocation, conviction, urgency, target-price, scoring, or
  tradeability behavior.

## Persistence Principles

### Raw source evidence is immutable input evidence

Raw source evidence should preserve the original provider/source response context
as audit evidence. It must not be edited to fit downstream expectations.

The raw layer may contain:

- provider/source name;
- provider category;
- provider record identifier;
- original source reference;
- ticker and source entity identifier;
- source timestamp;
- retrieval timestamp;
- reported period;
- fiscal year and fiscal quarter where available;
- currency and unit metadata;
- raw field evidence;
- missing field evidence;
- provenance metadata;
- raw payload hash;
- capture version.

The raw layer must not contain:

- BUY, SELL, or HOLD labels;
- investment recommendations;
- target prices;
- allocation instructions;
- conviction or urgency labels;
- tradeability decisions;
- portfolio action instructions;
- downstream reporting text.

### Normalized fundamentals are program-ready input

Normalized fundamentals should map governed source fields into stable,
program-ready records. They are not investment conclusions and must not imply
company quality, valuation attractiveness, or portfolio action.

The normalized layer may contain:

- normalized metric name;
- typed metric value when available and valid;
- currency;
- unit;
- reported period;
- fiscal year and fiscal quarter;
- source reference;
- provider/source identifiers;
- validation status;
- missing-value status;
- normalization version.

The normalized layer must keep missing values explicit. Missing values must not
be converted to:

```text
0
0.0
"0"
False
""
```

### Source-data readiness is neutral

Readiness records should describe whether data is available, complete, fresh,
valid, parseable, traceable, and consistent enough for downstream review.

Source-data readiness is not investment quality.

Readiness may include:

- readiness state;
- source data status;
- missing fundamentals count;
- partial data count;
- stale data count;
- invalid data count;
- provenance completeness;
- parseability status;
- consistency status;
- warnings and validation issues.

Readiness must not include:

- BUY, SELL, or HOLD labels;
- recommendation strength;
- valuation attractiveness;
- allocation guidance;
- conviction;
- urgency;
- tradeability.

## Proposed Storage Boundary

Future persistence should use three separately governed output families.

```text
raw source evidence store
normalized fundamentals store
source-data readiness store
```

The exact filenames, schemas, and write mechanics require separate approval and
contract tests before implementation.

### Proposed raw evidence store

Purpose: preserve source evidence and provenance.

Recommended future path family:

```text
data/raw/fundamentals/
```

This path is not approved for writes by this document. It is a proposed future
storage boundary only.

Future records should be append-oriented or versioned. Existing raw evidence
should not be silently overwritten.

Required characteristics:

- provenance-first;
- source reference preserved;
- retrieval timestamp preserved;
- source timestamp preserved;
- raw payload hash preserved;
- capture version preserved;
- no downstream investment logic;
- no normalized-only assumptions;
- no missing-to-zero conversion.

### Proposed normalized fundamentals store

Purpose: store stable program-ready fundamentals derived from raw evidence.

Recommended future path family:

```text
data/processed/fundamentals/
```

This path is not approved for writes by this document. It is a proposed future
storage boundary only.

Required characteristics:

- each normalized record references raw source evidence;
- each metric keeps explicit missing status;
- currency and unit are preserved where available;
- validation issues remain visible;
- normalization version is recorded;
- records remain neutral program input;
- no investment conclusions.

### Proposed readiness store

Purpose: summarize availability, completeness, freshness, validity, provenance,
parseability, and consistency.

Recommended future path family:

```text
data/processed/fundamental_readiness/
```

This path is not approved for writes by this document. It is a proposed future
storage boundary only.

Required characteristics:

- readiness is derived from source and normalization conditions;
- readiness references raw and normalized record identifiers;
- missing counts remain explicit;
- partial and stale data remain visible;
- readiness does not imply investment quality;
- readiness does not authorize Decision Engine action.

## Write Authorization Model

No production data write is approved by this document.

A future implementation sprint must separately approve:

- exact file paths;
- exact schemas;
- fixture examples;
- contract tests;
- write functions;
- overwrite or append semantics;
- atomic write behavior;
- failure behavior;
- audit metadata;
- no-side-effect tests;
- rollback or cleanup policy;
- guardrail checks for forbidden paths and artifacts.

Before any real write is allowed, the project should have:

1. schema documentation;
2. synthetic persistence contract tests;
3. fixture-based persistence tests;
4. explicit no-report/no-Telegram/no-pipeline tests;
5. explicit missing-value preservation tests;
6. explicit provenance-link tests;
7. explicit forbidden-semantics checks for investment labels.

## Proposed Record Lifecycle

The future lifecycle should be:

```text
manual or controlled source response
-> raw evidence capture
-> raw evidence validation
-> normalized fundamentals mapping
-> normalized record validation
-> readiness derivation
-> persistence only if separately approved
-> downstream review only after source-data readiness exists
```

Any failure should remain neutral and source/data-focused. Provider errors,
missing data, invalid data, parse errors, stale data, or provenance gaps should
not become investment conclusions.

## Failure Handling

Future persistence logic should fail closed.

Allowed failure outcomes:

- provider error;
- source missing;
- insufficient data;
- partial data;
- invalid data;
- stale data;
- provenance incomplete;
- parse failure;
- persistence failure.

Forbidden failure outcomes:

- defaulting missing values to zero;
- generating BUY, SELL, or HOLD labels;
- producing recommendations;
- triggering reports or Telegram;
- changing portfolio or watchlist data;
- running the production pipeline;
- hiding missing or invalid records.

## Audit and Traceability Requirements

Every persisted normalized or readiness record should be traceable back to raw
source evidence.

At minimum, future persisted records should retain:

- provider/source name;
- provider category;
- original source reference;
- ticker and entity identifier;
- reported period;
- fiscal year and fiscal quarter where available;
- source timestamp;
- retrieval timestamp;
- capture version;
- normalization version;
- raw payload hash or raw evidence identifier;
- validation warnings.

## Non-Goals

RESET-10L-BL11 does not:

- add or modify code;
- add or modify tests;
- write files under `data/`;
- create generated artifacts;
- execute a provider call;
- execute SEC, EDGAR, broker, or network calls;
- add credentials or secrets;
- generate reports;
- create Telegram artifacts;
- run the production pipeline;
- add Decision Engine behavior;
- approve investment analysis;
- add BUY, SELL, HOLD, allocation, conviction, urgency, tradeability, scoring,
  target-price, or recommendation behavior.

## Completion Criteria for Future Implementation

A future persistence implementation may only be considered after a separate
approved sprint defines and tests:

- raw evidence schema;
- normalized fundamentals schema;
- readiness schema;
- path and filename policy;
- append/version semantics;
- validation and failure semantics;
- provenance linkage;
- missing-value preservation;
- no side effects outside approved paths;
- no reporting or Telegram writes;
- no Decision Engine investment behavior.

## Next Step

The next candidate step is:

```text
RESET-10L-BL12 — Persistence Contract and Fixture Design
```

That future step should translate this persistence design into explicit schemas,
fixtures, and contract-test requirements before any production write function is
implemented.
