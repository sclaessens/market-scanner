# ME-SC02 — Implement SEC CompanyFacts Source Context from cached raw snapshots

## Status

COMPLETED BY ME-SC02

## Job family

ME-SC — Source Context jobs

## Purpose

ME-SC02 implements the SEC CompanyFacts Source Context job from cached raw Source Refresh snapshots.

The sprint turns raw SEC CompanyFacts snapshot envelopes from ME-SR01 into explicit source-context output. It preserves source availability, selected canonical source values, field-level states, provenance, and missingness without creating observations, analysis, recommendations, portfolio review, delivery output, or Decision Engine authority.

## Background

ME-GOV01 introduced job-scoped sprint naming and required Market Engine work to remain inside explicit job-family boundaries.

ME-SR01 added raw SEC CompanyFacts snapshot persistence and cached raw snapshot loading under the Source Refresh job family.

ME-SC01 defined the SEC CompanyFacts Source Context contract from cached raw snapshots.

ME-SC02 implements that contract.

## Scope

In scope:

- create a job-scoped Source Context module;
- consume cached raw SEC CompanyFacts snapshots from ME-SR01;
- map approved SEC CompanyFacts fields into source-context fields;
- expose context-level source availability state;
- expose field-level source states;
- preserve raw source provenance and period metadata;
- preserve missing fields explicitly;
- persist source-context JSON output;
- add local tests using temporary cached snapshots only.

## Non-scope

Out of scope:

- live SEC/provider calls;
- new Source Refresh behavior;
- source-intake provider behavior changes;
- fundamental observations;
- derived observations;
- analysis review;
- recommendation review;
- portfolio review;
- delivery or Telegram behavior;
- Decision Engine behavior;
- broad pipeline orchestration;
- production data writes;
- generated artifact commits.

## Implemented module

ME-SC02 adds:

```text
src/market_engine/source_context/sec_companyfacts_context.py
src/market_engine/source_context/__init__.py

The module exposes:

SecCompanyFactsSourceContext
SecCompanyFactsContextField
SecCompanyFactsContextState
SecCompanyFactsContextFieldState
SecCompanyFactsContextBuildError
build_sec_companyfacts_source_context_from_snapshot
build_sec_companyfacts_source_context_from_snapshot_path
persist_sec_companyfacts_source_context
Input contract

The input is a cached raw SEC CompanyFacts snapshot from ME-SR01.

Canonical input path shape:

data/market_engine/source_snapshots/sec_companyfacts/<source_refresh_run_id>/raw/<snapshot_id>.json

The implementation consumes the ME-SR01 SecCompanyFactsRawSnapshot envelope through:

load_sec_companyfacts_raw_snapshot(...)

The Source Context job does not make provider or network calls.

Output contract

The canonical output path shape is:

data/market_engine/source_contexts/fundamentals/<source_context_run_id>/<ticker>/source_context.json

The output format version is:

sec-companyfacts-source-context-v1

The output preserves:

ticker;
CIK;
source name;
provider name;
source context format version;
context-level source state;
source refresh snapshot ID;
source refresh fetched timestamp;
source refresh payload format version;
source refresh snapshot path;
canonical field values;
field-level states;
field-level provenance;
missing canonical fields;
explicit source-context-only mode.
Context-level states

ME-SC02 implements:

AVAILABLE
PARTIAL
MISSING

The ME-SC01 contract also reserves:

INVALID
PROVIDER_ERROR
UNSUPPORTED

These reserved states are not produced by the first implementation path because ME-SC02 only consumes successfully loaded raw snapshots. Snapshot load failures are exposed as controlled SecCompanyFactsContextBuildError failures rather than silently converting failed snapshots into context output.

Field-level states

ME-SC02 implements:

PRESENT
MISSING

The ME-SC01 contract also reserves:

INVALID
UNSUPPORTED

These reserved states are not produced by the first implementation path because the approved SEC CompanyFacts field mapper currently returns mapped fields or missing fields.

Canonical fields

The first implemented canonical fields are:

revenue
net_income
operating_cash_flow
capital_expenditures

These follow the existing approved SEC CompanyFacts mapping contract.

Failure behavior

ME-SC02 fails safely and explicitly when cached snapshot loading fails.

Examples:

missing snapshot file;
invalid snapshot JSON;
missing snapshot metadata;
unsupported raw snapshot format;
ticker/CIK mismatch.

These failures are wrapped as SecCompanyFactsContextBuildError.

Missing source values are not converted to zero.

A numeric zero remains a present value.

Testing approach

ME-SC02 tests use temporary local cached raw snapshots created through the ME-SR01 snapshot persistence API.

The tests do not make live SEC/provider calls.

The tests verify:

available source context;
partial source context;
missing source context;
numeric zero handling;
SEC provenance and period metadata preservation;
source context JSON persistence;
controlled cached snapshot failure handling;
controlled entity mismatch handling;
no analysis, recommendation, portfolio, delivery, Telegram, or Decision Engine authority;
no legacy runtime imports.
Tests run
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/source_context -q

Result:

10 passed
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/source_context tests/market_engine/source_refresh tests/market_engine/source_intake tests/market_engine/fundamentals -q

Result:

101 passed
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine -q

Result:

101 passed
Boundary confirmation

ME-SC02 remains inside the Source Context job family.

It does not introduce:

source refresh behavior;
live provider calls;
fundamental observations;
derived observations;
analysis review;
recommendation review;
portfolio review;
delivery output;
Telegram behavior;
Decision Engine behavior.
Acceptance criteria
SEC CompanyFacts Source Context implementation exists.
Cached raw SEC CompanyFacts snapshots can be converted into Source Context.
Context-level source states are explicit.
Field-level source states are explicit.
Provenance and period metadata are preserved.
Missing source values remain missing.
Numeric zero is treated as present.
Source Context JSON can be persisted.
Snapshot load failures are controlled.
Tests use temporary local snapshots only.
No runtime authority outside Source Context is introduced.
