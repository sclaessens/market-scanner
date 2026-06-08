# Decision Engine Pending Source Approval Review Guard

## Backlog Item

RESET-10L-BL63 — Keep Decision Engine in REVIEW While Source Approval Is Pending

## Purpose

This document defines the Decision Engine guard that must remain in place while SEC CompanyFacts and future live provider sources are still pending source approval, persistence approval, completeness/freshness validation, and auditability review.

The purpose is to prevent unapproved provider evidence from strengthening BUY or SELL signals before the provider has passed the required governance gates.

BL63 defines the guardrail only. It does not approve code changes, Decision Engine behavior changes, SEC CompanyFacts source approval, production persistence, Telegram output changes, portfolio intelligence changes, or watchlist updates.

## Current Status

As of BL63:

- BL53 documented live provider smoke governance;
- BL54 documented the SEC CompanyFacts source approval gate;
- BL58 documented the no-write-before-validation policy;
- BL59 documented the SEC CompanyFacts persistence boundary;
- BL60 documented provider data audit trail requirements;
- SEC CompanyFacts is not yet an approved production fundamentals source;
- SEC CompanyFacts persistence is not approved;
- provider audit trail implementation is not approved;
- Decision Engine authority for SEC CompanyFacts is not approved;
- Telegram, portfolio, and watchlist impact are not approved.

## Core Principle

The Decision Engine must remain defensive while provider source approval is pending.

The project must not treat the following states as equivalent:

```text
provider response ≠ source approval ≠ persistence approval ≠ completeness/freshness validation ≠ Decision Engine authority
```

Until those gates are complete, unapproved provider-derived fundamentals must not strengthen recommendations.

The safe default remains:

```text
REVIEW
```

## Guard Decision

BL63 establishes that pending provider approval must keep affected final decisions in REVIEW or equivalent defensive state.

This applies when any of the following is true:

- source approval is incomplete;
- persistence approval is incomplete;
- completeness validation is incomplete;
- freshness validation is incomplete;
- provenance validation is incomplete;
- audit trail requirements are incomplete;
- source quality state is missing, stale, partial, ambiguous, or insufficient;
- provider data is smoke-tested only;
- provider data is temporary local validation output only;
- provider data is not allowed to be written;
- provider data exists but downstream authority review has not been completed.

## Explicitly Not Approved

BL63 does not approve:

- BUY signal strengthening;
- SELL signal strengthening;
- confidence score increases;
- recommendation language changes;
- final decision logic changes;
- Telegram message changes;
- report wording changes;
- portfolio-specific recommendation behavior;
- watchlist mutation;
- SEC CompanyFacts persistence;
- provider audit trail persistence;
- normalized fundamentals writes.

## REVIEW Guard Conditions

The Decision Engine must remain in REVIEW or equivalent defensive behavior when any of the following conditions are present.

### 1. Source Not Approved

If SEC CompanyFacts or another provider is not explicitly source-approved, its data must not strengthen final decisions.

Allowed:

- identify blocker;
- preserve REVIEW;
- document missing approval;
- show source quality limitation in a future reporting layer if separately approved.

Not allowed:

- treating smoke success as approval;
- replacing missing fundamentals with unapproved provider data;
- increasing confidence based on unapproved source data.

### 2. Persistence Not Approved

If provider persistence is not approved, provider-derived data must not be treated as production fundamentals.

Allowed:

- use in documentation-only audit evidence;
- use in temporary local validation if explicitly allowed;
- plan future persistence.

Not allowed:

- writing provider data to production paths;
- using temporary local data as production source of truth;
- changing final decisions from temporary or unapproved data.

### 3. Completeness Not Validated

If completeness is missing, partial, or unknown, stronger recommendations must remain blocked.

Required behavior:

- produce or preserve review-required state;
- expose missingness as a blocker;
- prevent silent promotion from incomplete data to actionable output.

### 4. Freshness Not Validated

If freshness is stale, unknown, or missing, stronger recommendations must remain blocked.

Required behavior:

- preserve REVIEW;
- record stale or unknown freshness status;
- prevent stale data from increasing confidence.

### 5. Provenance Not Validated

If provenance is missing or ambiguous, stronger recommendations must remain blocked.

Required behavior:

- preserve REVIEW;
- record provenance blocker;
- prevent unverifiable values from becoming authoritative.

### 6. Audit Trail Not Available

If the project cannot explain where a provider-derived value came from, when it was retrieved, and why it is valid, that value must not strengthen final recommendations.

Required behavior:

- preserve REVIEW;
- require audit trail completion;
- prevent opaque provider values from becoming trusted project data.

## Decision Engine Authority Stages

Decision Engine authority must be staged.

```text
NO_PROVIDER_AUTHORITY
  → SOURCE_APPROVED_ONLY
  → PERSISTENCE_APPROVED_ONLY
  → QUALITY_VALIDATED
  → DECISION_AUTHORITY_REVIEWED
```

### NO_PROVIDER_AUTHORITY

Current state for SEC CompanyFacts.

Provider data may not influence final decisions.

### SOURCE_APPROVED_ONLY

A provider may have passed source approval, but this still does not authorize final decision impact.

### PERSISTENCE_APPROVED_ONLY

Provider data may be written under controlled conditions, but this still does not authorize final decision impact.

### QUALITY_VALIDATED

Completeness, freshness, and provenance may be validated, but Decision Engine authority still requires review.

### DECISION_AUTHORITY_REVIEWED

Only after a separate authority review may provider data influence bounded final decision logic.

## Required Evidence Before Decision Engine Authority

Before SEC CompanyFacts or any provider-derived fundamentals can influence BUY/SELL/REVIEW logic, the project must have evidence for:

1. source approval complete;
2. persistence approval complete;
3. no-write-before-validation compliance;
4. normalized fundamentals schema approved;
5. completeness rules implemented and validated;
6. freshness rules implemented and validated;
7. provenance rules implemented and validated;
8. audit trail requirements satisfied;
9. missingness behavior validated;
10. REVIEW safeguards preserved;
11. reporting and Telegram wording reviewed if user-facing output is affected;
12. portfolio/watchlist isolation confirmed if holdings-aware behavior is affected.

## Required Future Implementation Guardrails

Any future implementation task that modifies Decision Engine authority must specify:

- exact files allowed to change;
- exact files forbidden to change;
- source approval evidence;
- persistence approval evidence;
- quality validation evidence;
- audit trail evidence;
- expected Decision Engine behavior before and after;
- targeted tests;
- regression tests for REVIEW safeguards;
- proof that Telegram, reports, portfolio, and watchlist behavior remain unchanged unless explicitly in scope.

## Recommended Test Expectations for Future Code Work

Future code work should include tests that prove:

- unapproved provider data keeps final decision in REVIEW;
- missing source approval keeps final decision in REVIEW;
- missing persistence approval keeps final decision in REVIEW;
- incomplete fundamentals keep final decision in REVIEW;
- stale fundamentals keep final decision in REVIEW;
- missing provenance keeps final decision in REVIEW;
- smoke-passed provider data does not strengthen BUY or SELL;
- temporary local validation output does not become authoritative;
- Decision Engine behavior is unchanged unless the backlog item explicitly authorizes it.

BL63 itself does not add or modify tests.

## Telegram and Reporting Isolation

Decision Engine REVIEW guardrails must not be weakened by Telegram or reporting layers.

Until provider source approval and Decision Engine authority review are complete:

- Telegram must not present stronger action language based on unapproved provider data;
- reporting must not hide blockers;
- REVIEW states must remain visible;
- missingness, staleness, ambiguity, and source approval blockers must remain explainable.

## Portfolio and Watchlist Isolation

Pending source approval must not affect portfolio or watchlist behavior.

Forbidden before separate approval:

- holdings-aware recommendation strengthening;
- portfolio-specific BUY/SELL changes;
- watchlist additions or removals;
- portfolio CSV updates;
- watchlist file updates;
- ticker prioritization based on unapproved provider evidence.

## Current BL63 Decision

BL63 defines the pending-source-approval REVIEW guard.

It does not approve implementation or Decision Engine behavior changes.

As of this document:

```text
Decision Engine provider authority: NOT_APPROVED
SEC CompanyFacts source approval: NOT_APPROVED
SEC CompanyFacts persistence: NOT_APPROVED
Completeness/freshness authority: NOT_APPROVED
Decision Engine REVIEW guard: REQUIRED
Telegram impact: NOT_APPROVED
Portfolio/watchlist impact: NOT_APPROVED
```

## Relationship to Existing Governance

This document reinforces:

- `docs/active/governance/live_provider_smoke_governance.md`;
- `docs/active/data/sec_companyfacts_source_approval_gate.md`;
- `docs/active/governance/no_write_before_validation_policy.md`;
- `docs/active/data/sec_companyfacts_persistence_boundary.md`;
- `docs/active/data/provider_data_audit_trail_requirements.md`;
- `docs/active/pipeline/decision_engine_contract.md`.

The REVIEW guard must not weaken any existing Decision Engine contract.

## Follow-up Backlog Items

Recommended follow-up items:

- RESET-10L-BL61 — Normalize SEC CompanyFacts Fundamentals for One Ticker
- RESET-10L-BL62 — Validate Fundamentals Completeness and Freshness Rules
- RESET-10L-BL64 — Add Tests for Pending Provider Approval REVIEW Guard
- RESET-10L-BL56 — Telegram Delivery UX Review
- RESET-10L-BL57 — Portfolio Data Integrity Preflight

## Definition of Done

BL63 is complete when the project explicitly documents that pending provider source approval, persistence approval, quality validation, and auditability must keep Decision Engine behavior defensive and prevent unapproved provider data from strengthening BUY, SELL, confidence, Telegram, reporting, portfolio, watchlist, or recommendation behavior.
