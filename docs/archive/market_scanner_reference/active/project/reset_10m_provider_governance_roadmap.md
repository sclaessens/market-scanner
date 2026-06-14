# RESET-10M — Provider Governance & Operational Readiness Roadmap

## Purpose

RESET-10M defines the next controlled roadmap phase for the `market-scanner` project.

The project is currently transitioning from reset/cleanup work toward controlled real-source fundamentals integration. The canonical runtime is being stabilized, legacy script-era risks have been reduced, and the first SEC CompanyFacts smoke boundary exists.

However, live SEC CompanyFacts data must not yet influence persistence, the Decision Engine, Telegram output, portfolio intelligence, or final recommendations.

The next roadmap phase must therefore enforce a strict sequence:

1. smoke validation;
2. live-provider governance;
3. source approval;
4. persistence boundary;
5. Decision Engine authority review;
6. Telegram and portfolio UX improvements.

## Current State

The canonical live SEC CompanyFacts smoke module exists.

The first BL51 live-smoke attempt failed safely before network execution because `SEC_USER_AGENT` was missing.

This safe failure is acceptable and confirms that the smoke boundary does not proceed without required configuration.

No live SEC request was executed.

No data was written.

No portfolio, watchlist, Telegram, or Decision Engine behavior was changed.

## Strategic Principle

A successful live smoke is not the same as provider approval.

Live SEC CompanyFacts data may only move through the system in controlled stages. Each stage must be explicitly approved before the next one is allowed.

The project must avoid the following shortcut:

```text
live provider response → persisted fundamentals → stronger Decision Engine output → Telegram recommendation
```

Instead, the project must follow this controlled path:

```text
live provider response
  → smoke validation
  → governance policy
  → source approval gate
  → persistence boundary
  → completeness/freshness validation
  → controlled analysis use
  → guarded Decision Engine authority
  → cautious Telegram/portfolio UX
```

## Roadmap Order

### BL52 — Resolve Live SEC CompanyFacts Smoke Failure

BL52 is the immediate next step.

The goal is to resolve the safe failure caused by the missing `SEC_USER_AGENT` and confirm the live SEC CompanyFacts smoke boundary under controlled conditions.

BL52 must stay limited to smoke validation.

BL52 may confirm:

* missing configuration fails safely;
* valid configuration allows a controlled smoke;
* live network access is only used when explicitly allowed;
* no data is written;
* Decision Engine, Telegram, portfolio, and watchlist behavior remain untouched.

BL52 must not approve SEC CompanyFacts as a production source.

BL52 must not enable persistence.

BL52 must not change final decision behavior.

### BL53 — Document Live Provider Smoke Governance

After BL52, the project needs explicit governance for live provider calls.

BL53 defines when live calls are allowed, what configuration is required, what a smoke test may and may not do, and how results must be documented.

BL53 must establish that:

* live provider calls require explicit permission;
* `SEC_USER_AGENT` is mandatory for SEC requests;
* smoke tests may inspect provider response structure;
* smoke tests may not write production data;
* smoke tests may not affect portfolio, watchlist, Telegram, or Decision Engine behavior;
* every live smoke must leave auditable evidence.

### BL54 — SEC CompanyFacts Source Approval Gate

After smoke governance is documented, the project needs a source approval gate.

BL54 defines the conditions under which SEC CompanyFacts can move from “smoke passed” to “approved source candidate.”

A live smoke only proves that a request can be made and a response can be received.

Provider approval requires additional checks, including:

* schema stability;
* ticker coverage;
* field availability;
* completeness;
* freshness;
* error handling;
* auditability;
* normalization readiness;
* no unintended data writes.

Until BL54 is satisfied, SEC CompanyFacts must not increase Decision Engine authority.

### BL55 — Legacy Script Archive Completion

BL55 confirms that old script-era runtime artifacts no longer have an operational role.

This item should verify that legacy scripts, archived execution flows, and old runtime assumptions are historical only.

The canonical runtime remains the only valid execution path.

BL55 should not revive old scripts or broaden runtime behavior.

### BL56 — Telegram Delivery UX Review

Telegram UX should only be reviewed after live-provider smoke and source governance are under control.

Telegram is the user-facing layer and can easily make uncertain data feel actionable.

BL56 should define how Telegram output communicates:

* REVIEW states;
* missing fundamentals;
* source quality;
* blockers;
* data uncertainty;
* evidence limitations.

Telegram must not present strong action signals while fundamentals remain unapproved.

### BL57 — Portfolio Data Integrity Preflight

Portfolio intelligence depends on correct portfolio and watchlist data.

BL57 defines the minimum data-quality checks needed before portfolio-aware analysis becomes more prominent.

The preflight should cover:

* missing tickers;
* duplicate positions;
* stale positions;
* unclear currencies;
* cash treatment;
* ticker mapping;
* incomplete transaction history;
* mismatch between broker reality and local CSV state.

BL57 must not change portfolio files directly unless a later implementation task explicitly allows it.

## Roadmap Guardrails

During RESET-10M, the following guardrails apply:

* Live SEC CompanyFacts calls are allowed only when explicitly authorized.
* `SEC_USER_AGENT` is mandatory for SEC access.
* Smoke tests must not write production data.
* Provider smoke success does not equal source approval.
* Source approval must come before persistence.
* Persistence must come before Decision Engine authority expansion.
* Decision Engine authority must remain defensive until fundamentals are approved.
* Telegram must show blockers and uncertainty before action language.
* Portfolio intelligence must depend on validated portfolio input.
* Legacy script-era runtime remains historical unless explicitly migrated.
* Canonical runtime remains the only valid operational path.

## Recommended Execution Sequence

```text
BL52 — Resolve Live SEC CompanyFacts Smoke Failure
  ↓
BL53 — Document Live Provider Smoke Governance
  ↓
BL54 — SEC CompanyFacts Source Approval Gate
  ↓
BL55 — Legacy Script Archive Completion
  ↓
BL57 — Portfolio Data Integrity Preflight
  ↓
BL56 — Telegram Delivery UX Review
```

BL56 can be moved before BL57 if user-facing reporting becomes urgent, but it must not create stronger recommendation language before provider approval and data-quality gates are in place.

## What Is Explicitly Not Being Done Yet

RESET-10M does not yet approve SEC CompanyFacts as a full production fundamentals source.

RESET-10M does not yet enable automatic persistence of live provider data.

RESET-10M does not yet strengthen BUY or SELL recommendations.

RESET-10M does not yet allow Telegram to present stronger action-oriented investment messages.

RESET-10M does not yet rely on portfolio intelligence unless portfolio data integrity has been checked.

## Definition of Done

RESET-10M is successful when the project has a clear, staged path from live SEC CompanyFacts smoke validation toward eventual approved fundamentals usage, without skipping governance, persistence, source approval, Decision Engine authority review, Telegram guardrails, or portfolio data integrity checks.
