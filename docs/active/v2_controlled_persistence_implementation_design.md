<<'EOF'
# V2 Controlled Persistence Implementation Design

Status: ACTIVE
Reset stage: RESET-10L-BL14

## Purpose

This document defines the controlled implementation design for future v2
persistence of raw source evidence, normalized fundamentals, and neutral
source-data readiness.

This is a documentation/design-only artifact. It does not authorize production
persistence code, data writes, provider calls, SEC or EDGAR calls, broker calls,
network calls, production pipeline execution, report generation, Telegram
delivery, Decision Engine behavior, scoring, recommendations, BUY, SELL, HOLD,
allocation, conviction, urgency, target-price, or tradeability logic.

The goal is to define the minimal future implementation shape that Codex may
build in a later sprint, under explicit synthetic-only and no-side-effect
guardrails.

## Current Approved Inputs

The implementation design depends on these completed gates:

- `RESET-10L-BL11 — Real-Source Capture Persistence Design`;
- `RESET-10L-BL12 — Persistence Contract and Fixture Design`;
- `RESET-10L-BL13 — Synthetic Persistence Contract Tests`.

The BL13 contract tests and fixtures define the current executable acceptance
boundary for persistence expectations.

## Implementation Boundary

A future implementation may introduce a small v2-only persistence boundary that
accepts already prepared in-memory records and returns in-memory or synthetic
write results.

The boundary must remain downstream of the provider/source boundary and must not
fetch source data itself.

Allowed future responsibilities:

- validate raw source evidence records;
- validate normalized fundamentals records;
- validate readiness records;
- preserve provenance linkage;
- preserve explicit missing-value states;
- serialize approved synthetic records for controlled tests;
- return deterministic write-result metadata;
- fail closed on invalid or forbidden records.

Forbidden future responsibilities:

- live provider access;
- SEC, EDGAR, broker, or network access;
- credential handling;
- automatic source fetching;
- production pipeline integration;
- report generation;
- Telegram delivery;
- portfolio or watchlist updates;
- Decision Engine investment behavior;
- BUY, SELL, HOLD, allocation, conviction, urgency, target-price, scoring,
  recommendation, or tradeability behavior.

## Proposed Module Boundary

If implementation is approved in a later sprint, it should be isolated under the
v2 fundamentals package.

Candidate module path:

```text
src/market_scanner/fundamentals/fundamentals_persistence.py

This path is proposed only. It is not created by this sprint.

The module should expose pure, side-effect-controlled functions. It must have no
import-time side effects.

Candidate public functions:

validate_raw_evidence_record(record)
validate_normalized_fundamental_record(record)
validate_readiness_record(record)
prepare_persistence_batch(raw_records, normalized_records, readiness_records)
write_synthetic_persistence_batch(batch, output_root)

The write function, if later approved, must be restricted to synthetic/test
output roots during the first implementation phase. It must not write under
production data/ paths.

Proposed Data Structures

A future implementation may use dataclasses, typed dictionaries, or small plain
Python mappings. The chosen form must remain simple and testable.

Required conceptual structures:

Raw evidence record

Must preserve:

raw evidence identifier;
provider/source identity;
original source reference;
ticker and entity identifier;
source timestamp;
retrieval timestamp;
reported fiscal period;
raw fields;
missing field evidence;
provenance metadata;
raw payload hash or equivalent evidence identifier;
capture version;
validation warnings.
Normalized fundamentals record

Must preserve:

normalized record identifier;
raw evidence identifier;
normalized metric name;
metric value and explicit value status;
currency and explicit currency status;
unit and explicit unit status;
reported fiscal period;
source and retrieval timestamps;
normalization version;
validation warnings.
Readiness record

Must preserve:

readiness record identifier;
raw evidence identifier;
normalized record set identifier;
readiness state;
source data status;
missing fundamentals count;
partial, stale, and invalid counts;
provenance status;
parseability status;
consistency status;
freshness status;
readiness warnings;
readiness version.
Controlled Write Design

The first future implementation must not write to production data paths.

Allowed future write root for tests only:

tmp_path / "persistence"

or an equivalent pytest-managed temporary directory.

Forbidden write roots:

data/raw/
data/normalized/
data/generated/
data/processed/
data/portfolio/
data/watchlist/
data/logs/
reports/
reports/daily/
.github/workflows/

A future write function must reject forbidden output roots explicitly.

File Layout Design for Synthetic Writes

If synthetic writes are approved, the first implementation should write only to a
temporary root with three separated directories:

<temporary_root>/raw_source_evidence/
<temporary_root>/normalized_fundamentals/
<temporary_root>/source_data_readiness/

Each family should remain separate. A write to one family must not silently
change another family.

Candidate synthetic file naming pattern:

<ticker>_<record_id>.json

The exact naming may differ, but must be deterministic, testable, and free of
investment semantics.

Failure Behavior

Future implementation must fail closed.

It must reject or fail safely when:

provenance is missing;
raw evidence identifier is missing;
normalized record lacks raw evidence linkage;
readiness record lacks raw or normalized linkage;
missing values are represented as zero;
stale or invalid data is hidden;
forbidden investment semantics are present in approved output fields;
output root points to production data or report paths;
write target attempts to touch Telegram artifacts;
pipeline behavior is invoked.

Failure output must remain source/data-focused and neutral. It must not produce
investment conclusions.

Missing-Value Behavior

Missing values must remain explicit across all future implementation layers.

Forbidden missing-value substitutions:

0
0.0
"0"
False
""

Missing values should instead use explicit statuses such as:

missing
not_reported
unavailable
invalid
not_parseable

The exact vocabulary should follow existing provider and normalization contracts
where available.

Forbidden Semantics Guardrail

Future implementation must not emit or persist investment semantics.

Forbidden concepts include:

BUY;
SELL;
HOLD;
target price;
allocation;
recommendation;
conviction;
urgency;
tradeability;
valuation attractiveness;
portfolio action.

Synthetic tests may contain these words only inside controlled forbidden-input
fixtures or validation-warning expectations. They must not appear as approved
persistence outputs.

Required Test Coverage for Future Implementation

A future implementation sprint must add tests that prove:

validators accept complete synthetic records;
validators flag partial records neutrally;
validators fail closed on invalid records;
validators fail closed on provenance gaps;
missing values remain explicit;
missing values are not converted to zero;
forbidden semantics are rejected or flagged;
synthetic writes stay inside pytest temporary directories;
production data paths are rejected;
report and Telegram paths are rejected;
no provider calls are made;
no network calls are made;
no pipeline behavior is invoked;
no Decision Engine investment behavior is imported or called.
Proposed Next Implementation Scope

The next implementation sprint should be limited to:

a v2-only persistence boundary module;
pure validators;
synthetic-only temporary write support;
tests using BL13 fixtures;
no production writes;
no live source access;
no runtime integration.

It should not connect the persistence boundary to provider execution, pipeline
execution, reports, Telegram, portfolio review, or Decision Engine behavior.

Non-Goals

RESET-10L-BL14 does not:

add implementation code;
add tests;
add fixtures;
write production data;
execute provider calls;
execute SEC, EDGAR, broker, or network calls;
add credentials;
generate reports;
create Telegram artifacts;
run the production pipeline;
add Decision Engine behavior;
approve investment analysis;
add BUY, SELL, HOLD, allocation, conviction, urgency, tradeability, scoring,
target-price, or recommendation behavior.
Next Step

The next candidate step is:

RESET-10L-BL15 — Controlled Synthetic Persistence Implementation

That future step may implement the approved v2-only synthetic persistence
boundary under the guardrails described here. It must remain test-only,
synthetic-only, and disconnected from live provider access, production data
paths, reports, Telegram, pipeline execution, and Decision Engine investment
logic.
EOF
