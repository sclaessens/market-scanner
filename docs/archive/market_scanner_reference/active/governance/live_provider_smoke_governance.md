# Live Provider Smoke Governance

## Backlog Item

RESET-10L-BL53 — Document Live Provider Smoke Governance

## Purpose

This document defines the governance rules for controlled live provider smoke tests in the `market-scanner` project.

Live provider smokes are allowed only as explicitly approved validation steps. They must prove that a provider boundary can be reached and interpreted under controlled conditions, without enabling production data writes, changing Decision Engine authority, modifying Telegram output, or touching portfolio/watchlist data.

## Scope

This policy applies to live provider smoke tests, including the controlled SEC CompanyFacts one-ticker smoke boundary.

It governs:

* explicit operator approval;
* required local configuration;
* permitted live request scope;
* preflight validation;
* network containment;
* raw payload handling;
* cache and persistence restrictions;
* audit evidence requirements;
* separation from production runtime behavior;
* Decision Engine, Telegram, portfolio, and watchlist isolation.

## Core Principle

A successful live provider smoke does not equal provider approval.

A live smoke may confirm that a provider endpoint can be reached and that the response can pass a controlled boundary.

It must not automatically approve the provider for production persistence, normalized fundamentals, Decision Engine authority, Telegram messaging, portfolio intelligence, or recommendations.

The project must avoid this unsafe shortcut:

```text
live provider response
  → persisted fundamentals
  → stronger Decision Engine output
  → Telegram recommendation
```

The required controlled path is:

```text
live provider response
  → smoke validation
  → audit evidence
  → source approval gate
  → persistence boundary approval
  → completeness/freshness validation
  → controlled analysis use
  → guarded Decision Engine authority
  → cautious Telegram/portfolio UX
```

## Live Smoke Permission Rules

A live provider smoke may only run when all of the following are true:

1. the backlog item explicitly allows a live provider smoke;
2. the operator explicitly approves the live request;
3. the provider, ticker, identifier, and endpoint are approved in advance;
4. the live request count is bounded;
5. required local configuration is present;
6. no production data write is authorized;
7. no cache write is authorized unless explicitly approved;
8. no raw payload may be committed;
9. no workflow, scanner, Telegram, portfolio, watchlist, or Decision Engine integration is allowed;
10. the result will be documented as audit evidence.

If any required preflight condition fails, the smoke must fail closed before network execution.

## SEC CompanyFacts-Specific Rules

For the controlled SEC CompanyFacts smoke boundary:

* the only currently approved target is NVDA / CIK `0001045810`;
* the approved endpoint is the SEC CompanyFacts endpoint for that CIK;
* `SEC_USER_AGENT` is mandatory and must be supplied locally by the operator;
* the User-Agent value must not be committed, printed in audit records, or included in generated output;
* live execution must be disabled by default;
* explicit local invocation is required;
* the smoke may execute at most one SEC request unless a future backlog item explicitly changes this;
* no broad SEC CompanyFacts cache is approved;
* no raw SEC payload may be committed;
* no production persistence is approved;
* no Decision Engine authority increase is approved.

## Preflight Requirements

Before network execution, the smoke boundary must verify:

* explicit live execution is enabled;
* the approved ticker is used;
* the approved identifier or CIK is used;
* required local configuration is present;
* the User-Agent or equivalent provider-required configuration is not empty or malformed;
* request scope is bounded;
* no output path, cache path, report path, Telegram path, portfolio path, or watchlist path is enabled.

Preflight failure must produce a clear failure category and must not execute a network request.

## Network Containment

Live provider smoke tests must remain network-contained.

A smoke may only call the approved endpoint or approved provider boundary.

A smoke must not:

* perform retry loops unless explicitly approved;
* fall back to another provider;
* perform multi-ticker capture;
* perform bulk downloads;
* create local caches;
* trigger scanner or workflow execution;
* call Telegram;
* inspect or mutate portfolio/watchlist state;
* write generated reports.

## Data Handling Rules

Live provider smoke results must be redacted and minimal.

The following must not be committed:

* raw provider payloads;
* full SEC accessions from live payloads;
* raw SEC JSON;
* broad provider cache files;
* local operator secrets or identifiers;
* `SEC_USER_AGENT` values;
* generated production data;
* generated reports;
* Telegram artifacts.

A smoke may document:

* whether a request was executed;
* request count;
* provider name;
* source family;
* approved ticker and identifier;
* HTTP status category;
* failure category;
* redacted provenance summary;
* canonical fields found or missing;
* readiness state;
* missingness reasons;
* whether production data was written.

## Persistence Rules

Live provider smoke tests must not write production data.

A successful smoke does not authorize:

* writing to `data/processed/`;
* writing to portfolio/watchlist files;
* writing raw fundamentals;
* writing normalized fundamentals;
* writing source-data CSVs;
* writing reports;
* writing Telegram output;
* updating final decisions.

Persistence requires a separate approved backlog item and a defined persistence boundary.

## Decision Engine Rules

Decision Engine authority must remain unchanged by live provider smoke tests.

A live smoke may not:

* change BUY/SELL/REVIEW logic;
* reduce REVIEW safeguards;
* increase confidence scoring;
* change final decisions;
* allow real provider data to override missingness gates;
* make recommendations more action-oriented.

A provider may influence Decision Engine behavior only after source approval, persistence governance, completeness/freshness validation, and Decision Engine authority review are complete.

## Telegram Rules

Telegram output must remain isolated from live provider smoke tests.

A smoke may not:

* send Telegram messages;
* create Telegram artifacts;
* change Telegram copy;
* make REVIEW states look actionable;
* hide missingness or blockers;
* present unapproved provider data as investment guidance.

Telegram UX improvements must happen only after provider governance and source approval gates are clear.

## Portfolio and Watchlist Rules

Live provider smokes must not inspect, mutate, or depend on portfolio/watchlist state unless a future backlog item explicitly approves that scope.

A smoke may not:

* read portfolio positions for decision behavior;
* update portfolio CSVs;
* update watchlist files;
* infer recommendations from holdings;
* create portfolio-specific provider output.

Portfolio intelligence requires separate data integrity preflight governance.

## Audit Evidence Requirements

Every live provider smoke attempt must produce or update audit evidence.

Audit evidence must state:

* backlog item;
* provider;
* approved target;
* whether live execution was allowed;
* whether the request executed;
* request count;
* failure category or success status;
* validation command(s);
* whether any data was written;
* whether any raw payload/cache/report/Telegram artifact was created;
* whether Decision Engine, Telegram, portfolio, or watchlist behavior changed;
* recommended follow-up.

Audit evidence belongs in:

```text
docs/audits/provider_smokes/
```

It does not belong directly in `docs/active/`.

## Failure Categories

Failure categories should be explicit and operator-readable.

Recommended categories include:

* `explicit_invocation_missing`;
* `user_agent_missing`;
* `user_agent_malformed`;
* `ticker_cik_mismatch`;
* `http_error`;
* `network_error`;
* `invalid_json`;
* `unexpected_provider_shape`;
* `missing_required_context`;
* `ambiguous_facts`;
* `missing_provenance`;
* `canonical_boundary_rejected_input`;
* `other`.

A failure before network execution is acceptable when it protects the project from unsafe provider access or incomplete configuration.

## Definition of Done for a Live Smoke

A live provider smoke is complete only when:

* preflight behavior is verified;
* live execution status is known;
* request count is known;
* no unauthorized data writes occurred;
* raw payloads and sensitive local configuration are not committed;
* forbidden areas remain untouched;
* targeted tests or validation commands pass;
* audit evidence is created;
* follow-up is clearly identified.

## Current SEC CompanyFacts Status

As of BL53:

* the canonical SEC CompanyFacts smoke boundary exists;
* the controlled live SEC CompanyFacts smoke boundary exists;
* BL51 failed closed before network execution because `SEC_USER_AGENT` was missing;
* BL52 confirmed this was correct fail-closed behavior;
* SEC CompanyFacts is not yet an approved production fundamentals source;
* no automatic persistence is approved;
* no Decision Engine authority increase is approved;
* no Telegram or portfolio intelligence impact is approved.

## Next Recommended Step

The next recommended backlog item is:

RESET-10L-BL54 — SEC CompanyFacts Source Approval Gate

BL54 should define the criteria required before SEC CompanyFacts can move from smoke-tested provider boundary to approved source candidate.
