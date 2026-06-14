# SEC CompanyFacts Persistence Boundary

## Backlog Item

RESET-10L-BL59 — Define SEC CompanyFacts Persistence Boundary

## Purpose

This document defines the persistence boundary for future SEC CompanyFacts-derived fundamentals data in the `market-scanner` project.

The boundary specifies when, where, and how SEC CompanyFacts data may eventually be written after source approval, validation, and explicit persistence approval.

BL59 does not approve production persistence. It defines the rules that a future implementation must satisfy before any SEC CompanyFacts-derived data may be written into project data areas.

## Current Status

As of BL59:

- the canonical SEC CompanyFacts smoke boundary exists;
- the controlled live SEC CompanyFacts smoke boundary exists;
- BL52 confirmed correct fail-closed behavior for missing `SEC_USER_AGENT`;
- BL53 documented live provider smoke governance;
- BL54 documented the SEC CompanyFacts source approval gate;
- BL58 documented the no-write-before-validation policy;
- SEC CompanyFacts is not yet an approved production fundamentals source;
- no SEC CompanyFacts production persistence is approved;
- no raw payload retention is approved;
- no provider cache write is approved;
- no Decision Engine, Telegram, portfolio, or watchlist impact is approved.

## Core Principle

Persistence is a separate authority from source access and source approval.

The project must not treat the following states as equivalent:

```text
live response ≠ smoke passed ≠ source approved ≠ persistence approved ≠ Decision Engine authority
```

The required sequence is:

```text
live provider response
  → smoke validation
  → source approval gate
  → no-write-before-validation compliance
  → persistence boundary approval
  → controlled write implementation
  → audit trail validation
  → downstream analysis review
```

## Boundary Decision

BL59 approves only the persistence boundary design.

BL59 does not approve:

- writing SEC CompanyFacts data to production paths;
- writing raw SEC JSON payloads;
- writing provider caches;
- writing normalized fundamentals;
- writing source quality files;
- writing generated reports;
- writing Telegram artifacts;
- writing portfolio or watchlist files;
- changing Decision Engine behavior.

## Persistence Status

As of BL59:

```text
SEC CompanyFacts persistence: NOT_APPROVED
Raw payload retention: NOT_APPROVED
Provider cache writes: NOT_APPROVED
Normalized fundamentals writes: NOT_APPROVED
Source quality writes: NOT_APPROVED
Generated report writes: NOT_APPROVED
Telegram writes: NOT_APPROVED
Portfolio/watchlist writes: NOT_APPROVED
Decision Engine impact: NOT_APPROVED
```

## Approved Persistence Candidate Types

Future persistence may be considered only through explicitly approved backlog items.

Potential future persistence types include:

1. redacted audit evidence;
2. temporary local validation output;
3. controlled normalized fundamentals output;
4. controlled source quality output;
5. controlled provider audit trail output.

Each type requires separate approval.

## Persistence Categories

### 1. Redacted Audit Evidence

Purpose:

- document whether a smoke, capture, validation, or persistence attempt occurred;
- preserve governance evidence;
- avoid raw payload retention.

Allowed location:

```text
docs/audits/provider_smokes/
```

Allowed contents:

- backlog item;
- provider name;
- source family;
- approved ticker and CIK;
- request execution status;
- request count;
- HTTP status category;
- failure category or success status;
- redacted provenance summary;
- validation command or inspection method;
- confirmation that protected paths were not touched.

Forbidden contents:

- raw SEC JSON;
- full raw payloads;
- local `SEC_USER_AGENT` values;
- broad provider caches;
- generated production data;
- unapproved normalized fundamentals output.

### 2. Temporary Local Validation Output

Purpose:

- support local inspection and validation before production persistence is approved.

Allowed only when:

- a backlog item explicitly permits temporary local output;
- the output path is local, temporary, and non-production;
- the output is not committed;
- cleanup expectations are documented;
- the output does not contain secrets or unredacted raw payloads.

Forbidden:

- committing temporary validation output;
- placing temporary output in production data paths;
- using temporary output as production source of truth.

### 3. Controlled Normalized Fundamentals Output

Purpose:

- persist validated, normalized SEC CompanyFacts-derived fundamentals in a future approved data path.

Status:

```text
NOT_APPROVED_BY_BL59
```

Future approval requires:

- source approval complete;
- no-write-before-validation compliance;
- schema approved;
- freshness and completeness rules approved;
- provenance fields approved;
- write path approved;
- rollback or cleanup behavior defined;
- deterministic tests implemented.

### 4. Controlled Source Quality Output

Purpose:

- persist provider/source quality status, missingness, freshness, and review state.

Status:

```text
NOT_APPROVED_BY_BL59
```

Future approval requires:

- source quality schema defined;
- quality states approved;
- stale/partial/missing behavior approved;
- Decision Engine isolation confirmed;
- no automatic recommendation strengthening.

### 5. Provider Audit Trail Output

Purpose:

- preserve minimal evidence of provider-derived records and validation outcomes.

Status:

```text
NOT_APPROVED_BY_BL59
```

Future approval requires:

- audit trail schema defined;
- raw payload exclusion rules confirmed;
- redaction rules confirmed;
- retention policy defined;
- protected path review complete.

## Protected Production Paths

The following areas are protected from SEC CompanyFacts persistence until a future backlog item explicitly approves writes:

```text
data/
reports/
.github/workflows/
portfolio files
watchlist files
Telegram delivery artifacts
final decision outputs
```

Smoke tests, source approval work, and documentation-only tasks must not write to these areas.

## Candidate Future Write Paths

BL59 does not approve these paths, but identifies them as candidates for future review.

Potential future paths may include:

```text
data/processed/fundamentals_raw.csv
data/processed/fundamental_quality.csv
data/processed/source_data_quality.csv
docs/audits/provider_smokes/<redacted-audit-record>.md
```

A candidate path is not an approved path.

Before use, each path must be validated against:

- schema ownership;
- overwrite behavior;
- generated-file policy;
- source of truth rules;
- rollback expectations;
- downstream consumers;
- Decision Engine isolation;
- Telegram/reporting isolation.

## Minimum Persisted Record Requirements

Any future persisted SEC CompanyFacts-derived normalized record must include enough metadata to remain auditable.

Required fields or equivalents:

- ticker;
- CIK;
- company name when available;
- source family;
- provider name;
- SEC concept or canonical metric;
- fiscal year;
- fiscal period;
- period end date;
- value;
- unit or currency;
- normalization status;
- derivation status when applicable;
- source timestamp or filing date;
- retrieval timestamp;
- redacted source reference;
- quality state;
- missingness or review reason when applicable.

Raw payloads are not required for production persistence and remain forbidden unless separately approved.

## Raw Payload Policy

Raw SEC CompanyFacts JSON persistence is not approved.

Raw payloads must not be committed to the repository.

Future raw payload retention would require a separate backlog item that explicitly defines:

- purpose;
- storage location;
- retention duration;
- redaction policy;
- access rules;
- size limits;
- cleanup behavior;
- repository inclusion/exclusion rules.

Until such approval exists, raw payloads remain local-only and non-committed if used at all.

## Cache Policy

Broad SEC CompanyFacts cache writes are not approved.

A future cache design would require separate approval and must define:

- exact cache scope;
- ticker/CIK limits;
- local vs committed status;
- invalidation rules;
- freshness rules;
- size constraints;
- cleanup behavior;
- audit trail;
- no automatic Decision Engine impact.

No cache may be created by a smoke test.

## Write Mode Rules

Any future persistence implementation must define its write mode explicitly.

Allowed future write modes may include:

- create-only;
- replace controlled file;
- append controlled audit row;
- write to temporary path only.

Forbidden by default:

- silent overwrite;
- broad directory writes;
- implicit cache writes;
- writes triggered by import;
- writes triggered by scanner without approval;
- writes triggered by Telegram/reporting;
- writes triggered by portfolio/watchlist state.

## Validation Before Write

Before any SEC CompanyFacts-derived production write is allowed, the implementation must verify:

- source approval status;
- persistence approval status;
- approved ticker/CIK scope;
- schema validity;
- completeness state;
- freshness state;
- provenance state;
- redaction compliance;
- output path authorization;
- no forbidden downstream side effects.

Failure in any validation step must block the write.

## Failure Behavior

If persistence validation fails, the system must not write production data.

Valid outcomes include:

- fail closed;
- produce review-required state;
- produce redacted audit evidence;
- create a backlog item;
- require operator remediation.

Invalid outcomes include:

- partial production write;
- silent overwrite;
- fallback provider substitution;
- unapproved cache creation;
- Decision Engine output changes;
- Telegram message generation;
- portfolio/watchlist mutation.

## Decision Engine Isolation

Persistence approval does not automatically approve Decision Engine authority.

Even if SEC CompanyFacts data is eventually persisted, Decision Engine impact requires a separate authority review.

Persisted data must preserve quality states so that missing, stale, partial, ambiguous, or review-required records cannot silently strengthen recommendations.

## Telegram and Reporting Isolation

Persistence approval does not automatically approve reporting or Telegram use.

A future reporting or Telegram integration must separately confirm:

- source approval;
- persistence approval;
- freshness/completeness status;
- REVIEW safeguards;
- visible blockers;
- cautious language;
- no action-oriented output from unapproved or partial data.

## Portfolio and Watchlist Isolation

Persistence approval does not automatically approve portfolio or watchlist integration.

A future portfolio-aware integration must separately confirm:

- portfolio data integrity;
- ticker mapping validity;
- currency/cash treatment;
- watchlist state validity;
- no mutation of portfolio/watchlist files without explicit approval.

## Required Future Implementation Guardrails

Any future implementation task that writes SEC CompanyFacts-derived data must specify:

- exact files allowed to change;
- exact files forbidden to change;
- approved output path;
- schema;
- write mode;
- validation command;
- rollback/cleanup expectations;
- proof that raw payloads are not committed;
- proof that Decision Engine, Telegram, portfolio, and watchlist behavior remain isolated unless explicitly in scope.

## Current BL59 Decision

BL59 defines the SEC CompanyFacts persistence boundary only.

It does not approve persistence implementation.

As of this document:

```text
Persistence boundary: DEFINED
Production persistence: NOT_APPROVED
Implementation: NOT_APPROVED
Raw payload retention: NOT_APPROVED
Cache writes: NOT_APPROVED
Downstream authority: NOT_APPROVED
```

## Follow-up Backlog Items

Recommended follow-up items:

- RESET-10L-BL60 — Create Provider Data Audit Trail Requirements
- RESET-10L-BL61 — Normalize SEC CompanyFacts Fundamentals for One Ticker
- RESET-10L-BL62 — Validate Fundamentals Completeness and Freshness Rules
- RESET-10L-BL63 — Keep Decision Engine in REVIEW While Source Approval Is Pending

## Definition of Done

BL59 is complete when the project has a clear persistence boundary that defines future SEC CompanyFacts write conditions without approving any production write, raw payload retention, cache creation, Decision Engine impact, Telegram output, portfolio intelligence, or watchlist mutation.
