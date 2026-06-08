# SEC CompanyFacts Source Approval Gate

## Backlog Item

RESET-10L-BL54 — SEC CompanyFacts Source Approval Gate

## Purpose

This document defines the source approval gate for SEC CompanyFacts within the `market-scanner` project.

The gate determines when SEC CompanyFacts may move from a controlled smoke-tested provider boundary toward approved use as a fundamentals data source.

A successful live smoke is not sufficient to approve SEC CompanyFacts for production persistence, normalized fundamentals, Decision Engine authority, Telegram output, portfolio intelligence, or final recommendations.

## Current Status

As of BL54:

- the canonical SEC CompanyFacts smoke boundary exists;
- the controlled live SEC CompanyFacts smoke boundary exists;
- BL51 failed closed before network execution because `SEC_USER_AGENT` was missing;
- BL52 confirmed this was correct fail-closed behavior;
- BL53 documented live provider smoke governance;
- SEC CompanyFacts is not yet an approved production fundamentals source;
- no automatic persistence is approved;
- no Decision Engine authority increase is approved;
- no Telegram or portfolio intelligence impact is approved.

## Core Principle

Provider source approval must be staged.

The project must not treat the following states as equivalent:

```text
network reachable ≠ smoke passed ≠ capture validated ≠ source approved ≠ Decision Engine authority
```

The required sequence is:

```text
UNTESTED
  → SMOKE_READY
  → SMOKE_PASSED
  → CAPTURE_VALIDATED
  → NORMALIZATION_VALIDATED
  → SOURCE_APPROVED
  → DECISION_AUTHORITY_REVIEWED
```

Each stage must be explicitly documented before the next stage is allowed.

## Approval Statuses

### UNTESTED

SEC CompanyFacts has not yet been validated through a controlled live request.

Allowed:

- documentation;
- policy design;
- injected fixture testing;
- redacted/source-shaped smoke-boundary tests.

Not allowed:

- live provider use;
- persistence;
- production analysis;
- Decision Engine impact;
- Telegram output;
- portfolio intelligence impact.

### SMOKE_READY

Preflight governance exists and the live smoke boundary is ready for explicit local execution.

Required evidence:

- controlled live smoke governance exists;
- approved target is defined;
- `SEC_USER_AGENT` requirement is documented;
- live execution is disabled by default;
- no-write guardrails are documented;
- audit evidence path is defined.

Allowed:

- one explicitly approved controlled live smoke.

Not allowed:

- provider approval;
- persistence;
- multi-ticker capture;
- bulk ingestion;
- Decision Engine impact.

### SMOKE_PASSED

A controlled live SEC CompanyFacts request has executed successfully under approved guardrails.

Required evidence:

- explicit operator approval;
- approved ticker and CIK;
- request count;
- HTTP status category;
- response shape validation;
- no unauthorized writes;
- no raw payload commit;
- no cache commit;
- audit record.

Allowed:

- proceed to capture validation design.

Not allowed:

- production persistence;
- source approval;
- Decision Engine authority increase;
- Telegram recommendation language;
- portfolio intelligence impact.

### CAPTURE_VALIDATED

The project has validated how a live SEC CompanyFacts response can be captured safely.

Required evidence:

- capture boundary defined;
- raw payload handling policy defined;
- redaction rules defined;
- cache rules defined;
- local temporary handling defined;
- no-production-write behavior confirmed;
- failure behavior confirmed.

Allowed:

- controlled capture implementation planning;
- redacted evidence extraction;
- schema validation planning.

Not allowed:

- automatic production writes;
- broad ticker capture;
- Decision Engine authority changes.

### NORMALIZATION_VALIDATED

The project has validated that SEC CompanyFacts data can be converted into canonical fundamentals records safely.

Required evidence:

- supported SEC concepts defined;
- fiscal period selection rules defined;
- unit/currency rules defined;
- accession/provenance redaction rules defined;
- missingness rules defined;
- ambiguity handling defined;
- FreeCashFlow derivation policy confirmed;
- prior-year growth evidence policy confirmed;
- completeness and freshness rules defined;
- deterministic tests exist.

Allowed:

- limited normalized output in a controlled boundary;
- source quality evaluation;
- completeness/freshness assessment.

Not allowed:

- final Decision Engine authority increase;
- recommendation strengthening;
- Telegram action language.

### SOURCE_APPROVED

SEC CompanyFacts is approved as a fundamentals source candidate for controlled use.

Required evidence:

- smoke passed;
- capture validated;
- normalization validated;
- completeness rules approved;
- freshness rules approved;
- source quality statuses defined;
- audit trail defined;
- no-write-before-validation policy satisfied;
- source failure behavior defined;
- multi-ticker expansion rules defined or explicitly deferred.

Allowed:

- controlled use in fundamentals quality evaluation;
- controlled persistence if separately approved;
- source quality states may influence analysis readiness.

Not allowed:

- automatic BUY/SELL strengthening without Decision Engine authority review;
- Telegram recommendation strengthening without UX review;
- portfolio-specific recommendation behavior without portfolio preflight.

### DECISION_AUTHORITY_REVIEWED

Decision Engine impact has been explicitly reviewed and approved.

Required evidence:

- source approval complete;
- persistence boundary approved;
- completeness/freshness gates implemented;
- missingness behavior validated;
- REVIEW safeguards preserved;
- recommendation language reviewed;
- auditability confirmed.

Allowed:

- carefully bounded Decision Engine use, if approved by a separate backlog item.

Not allowed:

- unrestricted automated investment decisions;
- unreviewed Telegram action language;
- unvalidated portfolio-specific recommendations.

## Source Approval Criteria

SEC CompanyFacts may not be approved until all criteria below are satisfied.

### 1. Provider Identity

The provider identity must be stable and explicit.

Required:

- source family: `SEC EDGAR / SEC CompanyFacts`;
- provider name: `SEC CompanyFacts`;
- approved endpoint pattern;
- approved ticker/CIK mapping source or explicit CIK list;
- no fallback provider unless separately approved.

### 2. Access Governance

Access must be controlled.

Required:

- `SEC_USER_AGENT` is mandatory;
- User-Agent value remains local and uncommitted;
- live execution is disabled by default;
- explicit operator approval is required;
- request count is bounded;
- network failure fails closed;
- no retry loops unless approved;
- no bulk downloads unless separately approved.

### 3. Scope Control

Initial approval must remain narrow.

Required:

- one-ticker smoke before multi-ticker expansion;
- approved target documented;
- no broad capture;
- no production workflow integration;
- no scanner-triggered live provider execution;
- no Telegram-triggered provider execution;
- no portfolio-triggered provider execution.

### 4. Schema Stability

The expected provider response shape must be understood.

Required:

- expected SEC CompanyFacts JSON structure documented;
- `facts.us-gaap` handling defined;
- supported concepts listed;
- unsupported concepts ignored safely;
- unexpected provider shapes fail closed;
- invalid JSON fails closed;
- missing company identity fails closed;
- ticker/CIK mismatch fails closed.

### 5. Fact Selection Rules

Fact selection must be deterministic.

Required:

- annual facts are selected intentionally;
- fiscal year and fiscal period rules are defined;
- latest filed fact selection is deterministic;
- amended/duplicate facts are handled;
- ambiguous facts fail closed or produce explicit review status;
- missing provenance produces explicit review status;
- period-end context is required.

### 6. Unit and Currency Rules

Units must be controlled.

Required:

- supported unit/currency rules documented;
- unsupported units ignored or reviewed;
- mixed-unit conflicts handled;
- monetary fields use expected currency;
- non-monetary concepts require separate approval.

### 7. Provenance and Auditability

Every normalized value must remain traceable.

Required:

- ticker preserved;
- CIK preserved;
- fiscal year preserved;
- fiscal period preserved;
- period end preserved;
- source family preserved;
- provider name preserved;
- redacted source reference preserved;
- retrieval timestamp preserved;
- source timestamp or filing date preserved when available;
- raw User-Agent not logged;
- raw payload not committed.

### 8. Completeness Rules

The project must define which fields are required, optional, or derived.

Required:

- canonical fundamentals fields mapped;
- missing required fields produce explicit missingness;
- partial data produces review status;
- FreeCashFlow direct vs derived status preserved;
- prior-year evidence status preserved;
- incomplete data cannot silently strengthen recommendations.

### 9. Freshness Rules

Freshness must be explicit.

Required:

- source filing date or source timestamp captured;
- stale data threshold defined;
- stale data produces explicit quality state;
- missing freshness evidence produces review status;
- stale fundamentals cannot silently increase Decision Engine confidence.

### 10. Persistence Boundary

Persistence must be separately approved.

Required before writing:

- target output path approved;
- schema approved;
- no-write-before-validation policy satisfied;
- raw payload retention policy defined;
- generated files excluded unless intentionally committed;
- rollback/cleanup behavior defined;
- production data writes separated from smoke tests.

### 11. Failure Behavior

All failure states must be explicit.

Required failure states include:

- missing User-Agent;
- malformed User-Agent;
- explicit invocation missing;
- wrong ticker;
- wrong CIK;
- ticker/CIK mismatch;
- HTTP error;
- network error;
- invalid JSON;
- unexpected provider shape;
- missing company identity;
- missing required context;
- ambiguous facts;
- missing provenance;
- canonical boundary rejected input.

Failure must not lead to silent partial approval.

### 12. Decision Engine Isolation

Source approval must not automatically change final decisions.

Required:

- Decision Engine behavior remains unchanged until reviewed;
- REVIEW remains the safe default when source quality is insufficient;
- missing, stale, partial, or ambiguous data blocks stronger output;
- real provider data does not override quality gates;
- final recommendations require separate authority review.

### 13. Telegram Isolation

Telegram output must remain conservative.

Required:

- no live-smoke result may create Telegram output;
- no unapproved provider data may be presented as investment guidance;
- missingness and blockers must remain visible;
- REVIEW states must not be disguised as action signals.

### 14. Portfolio and Watchlist Isolation

Provider approval must not depend on portfolio state unless separately approved.

Required:

- no portfolio files touched during source approval;
- no watchlist files touched during source approval;
- no portfolio-specific recommendations generated;
- portfolio intelligence requires separate preflight.

## Approval Decision Matrix

| Status | Live Request | Data Write | Persistence | Decision Engine Impact | Telegram Impact | Portfolio Impact |
|---|---:|---:|---:|---:|---:|---:|
| UNTESTED | No | No | No | No | No | No |
| SMOKE_READY | Explicit only | No | No | No | No | No |
| SMOKE_PASSED | Completed once | No | No | No | No | No |
| CAPTURE_VALIDATED | Controlled only | Temp/local only if approved | No production | No | No | No |
| NORMALIZATION_VALIDATED | Controlled only | Controlled test output only | No production unless approved | No | No | No |
| SOURCE_APPROVED | Controlled | Separately approved only | Separately approved only | No automatic impact | No automatic impact | No automatic impact |
| DECISION_AUTHORITY_REVIEWED | Controlled | Approved only | Approved only | Bounded approved impact | Reviewed only | Reviewed only |

## Current BL54 Decision

BL54 does not approve SEC CompanyFacts as a production fundamentals source.

BL54 approves only the source approval framework.

As of this document:

```text
SEC CompanyFacts status: SMOKE_READY / FAILURE_RESOLVED_BY_GOVERNANCE
Production source approval: NOT_APPROVED
Persistence approval: NOT_APPROVED
Decision Engine authority: NOT_APPROVED
Telegram impact: NOT_APPROVED
Portfolio/watchlist impact: NOT_APPROVED
```

## Required Evidence Before Source Approval

Before SEC CompanyFacts can become `SOURCE_APPROVED`, the project must have evidence for:

1. successful controlled live smoke;
2. source response shape validation;
3. capture boundary validation;
4. normalization validation;
5. completeness rules;
6. freshness rules;
7. provenance and audit trail;
8. no-write-before-validation compliance;
9. controlled persistence design;
10. Decision Engine isolation confirmation.

## Follow-up Backlog Items

Recommended follow-up items:

- RESET-10L-BL55 — Legacy Script Archive Completion
- RESET-10L-BL58 — Define No-Write-Before-Validation Policy
- RESET-10L-BL59 — Define SEC CompanyFacts Persistence Boundary
- RESET-10L-BL60 — Create Provider Data Audit Trail Requirements

## Definition of Done

BL54 is complete when the project has a clear, staged approval gate that prevents SEC CompanyFacts from moving directly from live smoke to production persistence, Decision Engine impact, Telegram output, portfolio intelligence, or recommendations.
