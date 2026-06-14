# No-Write-Before-Validation Policy

## Backlog Item

RESET-10L-BL58 — Define No-Write-Before-Validation Policy

## Purpose

This document defines the no-write-before-validation policy for the `market-scanner` project.

The policy prevents live provider responses, smoke outputs, partially validated source data, or uncertain fundamentals from being written into production data areas before explicit validation, approval, and persistence governance are complete.

The policy exists to protect the project from accidental contamination of production data, generated reports, Telegram output, portfolio/watchlist state, and Decision Engine behavior.

## Current Status

As of BL58:

- the canonical SEC CompanyFacts smoke boundary exists;
- the controlled live SEC CompanyFacts smoke boundary exists;
- BL52 confirmed that the missing `SEC_USER_AGENT` failure was correct fail-closed behavior;
- BL53 documented live provider smoke governance;
- BL54 documented the SEC CompanyFacts source approval gate;
- SEC CompanyFacts is not yet an approved production fundamentals source;
- no automatic persistence is approved;
- no provider response may be written to production data by default;
- no Decision Engine authority increase is approved;
- no Telegram or portfolio intelligence impact is approved.

## Core Principle

No live provider response may be written to production data before validation and explicit persistence approval.

The project must avoid this unsafe shortcut:

```text
live provider response
  → production data write
  → downstream analysis
  → Decision Engine impact
  → Telegram or portfolio action language
```

The controlled path is:

```text
provider response
  → preflight validation
  → bounded smoke or capture
  → redacted audit evidence
  → schema validation
  → completeness/freshness validation
  → source approval
  → persistence boundary approval
  → controlled write
  → downstream analysis review
```

## Policy Scope

This policy applies to all provider-derived data and all write-capable project outputs, including:

- SEC CompanyFacts responses;
- future live provider responses;
- raw payloads;
- normalized fundamentals;
- source-data CSVs;
- quality-state CSVs;
- generated reports;
- Telegram artifacts;
- portfolio files;
- watchlist files;
- final decision outputs;
- audit artifacts when they contain provider-derived evidence.

This policy applies to both manual and automated execution.

## Write Categories

### Forbidden Before Validation

The following writes are forbidden before explicit validation and approval:

- writes to `data/processed/`;
- writes to production source-data CSVs;
- writes to portfolio or watchlist files;
- writes to final decision outputs;
- writes to generated reports;
- writes to Telegram delivery artifacts;
- writes of raw provider payloads;
- writes of broad provider caches;
- writes of unredacted SEC payload evidence;
- writes that change runtime behavior or downstream interpretation.

### Allowed Without Provider Validation

The following are allowed when they do not contain raw provider payloads or production data:

- documentation-only changes under `docs/`;
- policy documents;
- backlog entries;
- redacted audit evidence;
- test fixtures using synthetic or redacted data;
- temporary local-only files that are not committed and are explicitly documented as local scratch output.

### Allowed Only After Explicit Approval

The following writes require a separately approved backlog item:

- controlled persistence of validated source data;
- normalized fundamentals output;
- source quality output;
- provider cache output;
- multi-ticker provider capture;
- production report generation;
- Telegram message generation;
- portfolio/watchlist updates;
- final decision output changes.

## Validation Requirements Before Any Production Write

Before any provider-derived data may be written to production paths, the project must have documented evidence for all applicable checks below.

### 1. Source Approval

The provider must have passed the relevant source approval gate.

For SEC CompanyFacts, this requires compliance with:

- `docs/active/data/sec_companyfacts_source_approval_gate.md`

A smoke-passed state is not enough.

### 2. Schema Validation

The source response shape must be understood and validated.

Required:

- expected provider schema documented;
- required fields identified;
- unsupported fields handled safely;
- invalid JSON or unexpected shape fails closed;
- missing company identity fails closed or produces review status;
- ticker/CIK mismatch fails closed.

### 3. Completeness Validation

The project must define and evaluate completeness.

Required:

- required canonical fields listed;
- optional fields listed;
- derived fields documented;
- missing required values produce explicit missingness;
- partial data cannot silently become approved data;
- incomplete data cannot strengthen recommendations.

### 4. Freshness Validation

The project must define and evaluate freshness.

Required:

- source timestamp, filing date, or equivalent freshness marker captured;
- stale threshold defined;
- stale data produces explicit quality state;
- missing freshness evidence produces review status;
- stale data cannot increase Decision Engine confidence.

### 5. Provenance Validation

Every persisted value must remain traceable.

Required:

- provider name;
- source family;
- ticker or identifier;
- fiscal period or reporting context;
- retrieval timestamp;
- source timestamp or filing date when available;
- redacted source reference;
- normalization or derivation status.

### 6. Redaction Validation

Sensitive or excessive provider evidence must not be committed.

Required:

- raw payloads excluded unless separately approved;
- `SEC_USER_AGENT` and local operator configuration excluded;
- raw SEC JSON excluded;
- broad provider cache excluded;
- full accessions redacted where required by governance;
- audit evidence remains minimal and controlled.

### 7. Persistence Boundary Approval

The write path must be explicitly approved.

Required:

- output path approved;
- schema approved;
- write mode approved;
- overwrite behavior approved;
- rollback or cleanup behavior defined;
- generated files policy respected;
- no accidental writes outside the approved path.

## Production Path Guardrails

The following paths are protected and must not be written by smoke, validation, or source-approval tasks unless a later backlog item explicitly approves the write.

Protected areas include:

```text
data/
reports/
.github/workflows/
src/ runtime entrypoints
portfolio/watchlist files
Telegram delivery artifacts
final decision outputs
```

Documentation-only tasks may write only to approved `docs/` locations.

Audit evidence for provider smokes belongs in:

```text
docs/audits/provider_smokes/
```

Active governance and source approval policies belong in:

```text
docs/active/governance/
docs/active/data/
```

## Smoke Test Write Rules

Live provider smoke tests must not write production data.

A smoke may document:

- whether the request executed;
- request count;
- HTTP status category;
- failure category;
- redacted provider/source summary;
- canonical fields found or missing;
- readiness state;
- whether any data was written.

A smoke may not write:

- raw provider payloads;
- caches;
- normalized production fundamentals;
- quality-state CSVs;
- reports;
- Telegram artifacts;
- portfolio or watchlist state;
- final decisions.

A successful smoke does not authorize persistence.

## Temporary Local Output Rules

Temporary local output is allowed only when all of the following are true:

- the backlog item explicitly allows temporary local output;
- the output path is outside production data areas or uses a controlled temporary path;
- the output is not committed;
- the output does not contain secrets or unredacted raw payloads;
- cleanup expectations are documented;
- the output cannot be mistaken for production data.

Temporary local output must not become part of committed project state unless a later backlog item explicitly approves it.

## Audit Evidence Rules

Audit evidence may be committed when it is redacted, minimal, and policy-compliant.

Audit evidence must state:

- backlog item;
- source or provider;
- whether live execution occurred;
- whether a write occurred;
- request count when applicable;
- validation result;
- failure category or success status;
- protected areas confirmed untouched;
- follow-up recommendation.

Audit evidence must not include:

- raw provider payloads;
- local secrets or operator configuration;
- raw `SEC_USER_AGENT` values;
- generated production data;
- unapproved cache contents.

## Decision Engine Protection

No provider-derived write may change Decision Engine behavior until Decision Engine authority is explicitly reviewed.

Forbidden before authority review:

- changing BUY/SELL/REVIEW behavior;
- reducing REVIEW safeguards;
- increasing confidence scores;
- making missing or partial data look complete;
- allowing real provider data to override quality gates;
- writing final decisions from unapproved provider data.

Decision Engine authority requires a separate backlog item after source approval, persistence approval, completeness/freshness validation, and auditability are complete.

## Telegram Protection

No provider-derived write may produce Telegram output before Telegram UX and provider evidence rules are explicitly approved.

Forbidden before approval:

- Telegram message generation from unapproved provider data;
- action-oriented language based on unapproved fundamentals;
- hiding missingness or blockers;
- presenting smoke results as investment guidance.

Telegram must remain isolated from smoke, capture, source approval, and persistence validation unless a later backlog item explicitly approves integration.

## Portfolio and Watchlist Protection

No provider-derived write may modify portfolio or watchlist state unless explicitly approved.

Forbidden before approval:

- updating portfolio CSVs;
- updating watchlist files;
- generating portfolio-specific recommendations;
- using unapproved provider data to alter holdings interpretation;
- writing broker-derived or provider-derived state into portfolio files without preflight.

Portfolio intelligence requires separate data integrity preflight governance.

## Failure Behavior

If validation is incomplete, ambiguous, stale, missing, or failed, the system must not write provider-derived production data.

Failure must be explicit.

Valid failure outcomes include:

- fail closed before network execution;
- produce redacted audit evidence;
- produce review-required status;
- create a backlog item;
- require operator remediation.

Invalid failure outcomes include:

- silent partial write;
- fallback provider substitution;
- unapproved cache creation;
- production data update with incomplete validation;
- downstream Decision Engine or Telegram impact.

## Current BL58 Decision

BL58 does not approve any new write path.

BL58 approves only the no-write-before-validation policy.

As of this document:

```text
Provider production writes: NOT_APPROVED
SEC CompanyFacts persistence: NOT_APPROVED
Raw payload retention: NOT_APPROVED
Provider cache writes: NOT_APPROVED
Decision Engine authority change: NOT_APPROVED
Telegram output from provider data: NOT_APPROVED
Portfolio/watchlist writes: NOT_APPROVED
```

## Required Evidence Before Future Write Approval

Before any provider-derived production write is approved, the project must have evidence for:

1. source approval;
2. schema validation;
3. completeness validation;
4. freshness validation;
5. provenance validation;
6. redaction validation;
7. persistence boundary approval;
8. protected path review;
9. rollback or cleanup expectations;
10. Decision Engine, Telegram, portfolio, and watchlist isolation.

## Follow-up Backlog Items

Recommended follow-up items:

- RESET-10L-BL59 — Define SEC CompanyFacts Persistence Boundary
- RESET-10L-BL60 — Create Provider Data Audit Trail Requirements
- RESET-10L-BL57 — Portfolio Data Integrity Preflight
- RESET-10L-BL56 — Telegram Delivery UX Review

## Definition of Done

BL58 is complete when the project has a clear policy preventing provider-derived data from being written into production data, generated outputs, Decision Engine paths, Telegram artifacts, portfolio files, or watchlist state before validation and explicit persistence approval are complete.
