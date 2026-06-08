# Provider Data Audit Trail Requirements

## Backlog Item

RESET-10L-BL60 — Create Provider Data Audit Trail Requirements

## Purpose

This document defines the provider data audit trail requirements for the `market-scanner` project.

The audit trail exists to make provider-derived data traceable, reviewable, and safe before it can support source approval, persistence, normalized fundamentals, reporting, Telegram output, portfolio intelligence, or Decision Engine authority.

BL60 defines requirements only. It does not approve production persistence, provider cache writes, raw payload retention, Decision Engine impact, Telegram output, portfolio updates, or watchlist updates.

## Current Status

As of BL60:

- BL53 documented live provider smoke governance;
- BL54 documented the SEC CompanyFacts source approval gate;
- BL58 documented the no-write-before-validation policy;
- BL59 documented the SEC CompanyFacts persistence boundary;
- SEC CompanyFacts is not yet an approved production fundamentals source;
- provider production persistence is not approved;
- raw payload retention is not approved;
- provider cache writes are not approved;
- Decision Engine, Telegram, portfolio, and watchlist impact are not approved.

## Core Principle

Provider-derived facts must be auditable before they can become trusted project data.

The project must preserve enough evidence to answer:

```text
Where did this value come from?
When was it retrieved?
What provider produced it?
What source context supports it?
Was it complete, fresh, and valid?
Was it normalized or derived?
Was anything missing, ambiguous, stale, or blocked?
Was it allowed to be written?
Did it influence downstream behavior?
```

If the project cannot answer these questions, the data must remain review-only and must not strengthen recommendations.

## Audit Trail Scope

This policy applies to audit trails for:

- live provider smoke attempts;
- provider response validation;
- source approval evidence;
- controlled capture evidence;
- normalization evidence;
- completeness and freshness evaluation;
- persistence boundary validation;
- future provider-derived records;
- future source quality records.

It applies first to SEC CompanyFacts and later to any other provider integrated into the project.

## Audit Trail Types

### 1. Smoke Audit Evidence

Purpose:

- document whether a controlled live smoke ran;
- document whether preflight failed closed;
- document whether a network request happened;
- prove that forbidden areas remained untouched.

Required fields or statements:

- backlog item;
- provider name;
- source family;
- approved target;
- explicit operator approval status;
- live execution allowed yes/no;
- request executed yes/no;
- request count;
- failure category or success status;
- validation command or inspection method;
- data written yes/no;
- raw payload committed yes/no;
- cache created yes/no;
- Decision Engine changed yes/no;
- Telegram changed yes/no;
- portfolio/watchlist changed yes/no;
- follow-up recommendation.

Allowed location:

```text
docs/audits/provider_smokes/
```

### 2. Source Approval Audit Evidence

Purpose:

- prove that a provider has met the source approval criteria;
- document each approval stage;
- prevent smoke success from being mistaken for source approval.

Required fields or statements:

- approval status;
- provider identity;
- supported endpoint or source pattern;
- ticker/identifier mapping method;
- schema validation result;
- completeness validation result;
- freshness validation result;
- provenance validation result;
- failure behavior validation result;
- source approval decision;
- remaining blockers.

### 3. Capture Audit Evidence

Purpose:

- document how a live provider response was captured or inspected without approving production persistence.

Required fields or statements:

- capture scope;
- capture mode;
- output path if any;
- temporary/local-only status;
- raw payload retained yes/no;
- raw payload committed yes/no;
- redaction status;
- cleanup expectation;
- validation result;
- blocked writes.

### 4. Normalization Audit Evidence

Purpose:

- trace normalized canonical fundamentals back to provider facts and source context.

Required fields or equivalents:

- ticker;
- provider identifier such as CIK;
- provider name;
- source family;
- canonical field;
- provider concept;
- fiscal year;
- fiscal period;
- period end date;
- value;
- unit or currency;
- normalization status;
- derivation status;
- source timestamp or filing date when available;
- retrieval timestamp;
- redacted source reference;
- quality state;
- missingness or review reason when applicable.

### 5. Persistence Audit Evidence

Purpose:

- prove that a provider-derived write was allowed and controlled.

Required fields or statements:

- persistence approval status;
- approved output path;
- schema version or schema reference;
- write mode;
- overwrite or append behavior;
- rollback or cleanup expectation;
- protected path review;
- generated-file policy status;
- downstream consumer review;
- validation command;
- data written yes/no;
- files written or changed.

## Minimum Audit Record Fields

Any future structured provider audit record must include, where applicable:

| Field | Purpose |
|---|---|
| `audit_event_id` | Unique audit event identifier |
| `backlog_item` | Governance or implementation item that allowed the action |
| `provider_name` | Provider identity |
| `source_family` | Source family, such as SEC EDGAR / SEC CompanyFacts |
| `ticker` | Market ticker where applicable |
| `provider_identifier` | Provider identifier such as CIK |
| `source_endpoint_family` | Endpoint family or source class, not necessarily full URL |
| `operation_type` | Smoke, capture, normalize, validate, persist, or review |
| `operation_status` | Passed, failed, blocked, reviewed, approved, or rejected |
| `request_executed` | Whether a live request happened |
| `request_count` | Number of provider requests |
| `retrieval_timestamp` | When data was retrieved or attempted |
| `source_timestamp` | Provider filing date or source timestamp when available |
| `schema_status` | Schema validation status |
| `completeness_status` | Completeness validation status |
| `freshness_status` | Freshness validation status |
| `provenance_status` | Provenance validation status |
| `redaction_status` | Redaction validation status |
| `write_status` | Whether a write happened and whether it was approved |
| `quality_state` | Source quality state |
| `review_reason` | Reason for REVIEW, BLOCKED, or REJECTED state |
| `decision_engine_impact` | Whether Decision Engine behavior changed |
| `telegram_impact` | Whether Telegram output changed |
| `portfolio_watchlist_impact` | Whether portfolio/watchlist state changed |

A future implementation may split these fields across multiple files or schemas, but the same information must remain available and traceable.

## Required Status Values

Audit records must use explicit statuses.

Recommended operation statuses:

```text
NOT_RUN
BLOCKED_PREFLIGHT
FAILED_CLOSED
SMOKE_PASSED
VALIDATION_PASSED
VALIDATION_FAILED
REVIEW_REQUIRED
APPROVED
REJECTED
```

Recommended write statuses:

```text
NO_WRITE
WRITE_BLOCKED
TEMP_LOCAL_ONLY
WRITE_APPROVED
WRITE_COMPLETED
WRITE_REJECTED
```

Recommended quality states:

```text
UNTESTED
INSUFFICIENT_DATA
PARTIAL_DATA
STALE_DATA
VALIDATED
REVIEW_REQUIRED
SOURCE_APPROVED
SOURCE_REJECTED
```

Statuses must not imply approval unless approval has been explicitly granted by a backlog item or policy gate.

## Redaction Requirements

Audit trails must be useful without exposing excessive or sensitive data.

The following must not be committed:

- raw SEC CompanyFacts JSON payloads;
- broad provider caches;
- local `SEC_USER_AGENT` values;
- local operator secrets or environment values;
- unredacted raw payload excerpts;
- generated production data that is not explicitly approved;
- temporary local validation output.

Allowed audit evidence may include:

- provider name;
- source family;
- approved ticker and CIK;
- request count;
- HTTP status category;
- failure category;
- redacted source reference;
- canonical fields found or missing;
- quality state;
- review reason;
- confirmation that protected areas were untouched.

## Audit Trail Storage Rules

BL60 does not approve a new production audit database or provider cache.

Documentation evidence belongs in:

```text
docs/audits/provider_smokes/
docs/audits/
```

Active policies and requirements belong in:

```text
docs/active/data/
docs/active/governance/
```

Future structured provider audit output may be considered only after a separate backlog item approves:

- path;
- schema;
- write mode;
- retention rules;
- redaction rules;
- generated-file policy;
- downstream consumer behavior.

## Relationship to Existing Governance

This document depends on and reinforces:

- `docs/active/governance/live_provider_smoke_governance.md`;
- `docs/active/data/sec_companyfacts_source_approval_gate.md`;
- `docs/active/governance/no_write_before_validation_policy.md`;
- `docs/active/data/sec_companyfacts_persistence_boundary.md`.

The audit trail must not weaken any of those policies.

## Decision Engine Guardrail

Audit trail evidence may support future Decision Engine review, but it must not automatically change Decision Engine behavior.

Forbidden by BL60:

- using audit evidence to strengthen BUY/SELL signals;
- reducing REVIEW safeguards;
- treating smoke success as source approval;
- treating persistence success as recommendation authority;
- hiding missingness, staleness, ambiguity, or review reasons.

## Telegram and Reporting Guardrail

Audit trail evidence may support future reporting and Telegram explanations, but it must not automatically create user-facing output.

Forbidden by BL60:

- Telegram generation from unapproved provider audit evidence;
- action-oriented recommendation language;
- hiding blockers or data quality states;
- presenting source approval as investment advice.

## Portfolio and Watchlist Guardrail

Audit trail evidence must not mutate portfolio or watchlist state.

Forbidden by BL60:

- portfolio CSV updates;
- watchlist file updates;
- holdings-based recommendation changes;
- ticker additions or removals based on provider audit evidence alone.

## Current BL60 Decision

BL60 defines provider data audit trail requirements only.

It does not approve implementation or persistence.

As of this document:

```text
Provider audit trail requirements: DEFINED
Structured audit trail implementation: NOT_APPROVED
Provider audit trail persistence: NOT_APPROVED
Raw payload retention: NOT_APPROVED
Provider cache writes: NOT_APPROVED
Decision Engine impact: NOT_APPROVED
Telegram impact: NOT_APPROVED
Portfolio/watchlist impact: NOT_APPROVED
```

## Required Evidence Before Future Audit Trail Implementation

A future implementation task must define:

1. exact output path;
2. schema fields;
3. write mode;
4. retention rules;
5. redaction rules;
6. validation command;
7. protected path review;
8. generated-file policy;
9. downstream consumer review;
10. proof that Decision Engine, Telegram, portfolio, and watchlist behavior remain isolated unless explicitly in scope.

## Follow-up Backlog Items

Recommended follow-up items:

- RESET-10L-BL61 — Normalize SEC CompanyFacts Fundamentals for One Ticker
- RESET-10L-BL62 — Validate Fundamentals Completeness and Freshness Rules
- RESET-10L-BL63 — Keep Decision Engine in REVIEW While Source Approval Is Pending
- RESET-10L-BL57 — Portfolio Data Integrity Preflight
- RESET-10L-BL56 — Telegram Delivery UX Review

## Definition of Done

BL60 is complete when the project has a clear provider data audit trail requirement set that defines traceability, required fields, status values, redaction rules, storage rules, and downstream isolation without approving implementation, persistence, raw payload retention, cache creation, Decision Engine impact, Telegram output, portfolio intelligence, or watchlist mutation.
