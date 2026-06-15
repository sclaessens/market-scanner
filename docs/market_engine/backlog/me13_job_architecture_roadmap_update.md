# ME13 — Job Architecture Roadmap Update

Owner roles: Product Owner / Scrum Master / Technical Architect / Data Steward / Governance Auditor

Status: ACTIVE ROADMAP UPDATE

## Purpose

This document updates the Market Engine roadmap after the ME13 job architecture and persistence decision.

The immediate roadmap now prioritizes job separation and raw source persistence before additional derived analysis layers.

## Roadmap Change

Previous recommended next sprint after ME12:

```text
ME13 — Add first derived cash-generation observation layer
```

Revised sequence:

```text
ME13 — Define Market Engine job architecture and data persistence contract
ME14 — Persist raw SEC CompanyFacts source snapshots and support cached source loading
ME15 — Build cached source-context and observation job execution from persisted snapshots
ME16 — Add first derived cash-generation observation layer
```

## Why the Roadmap Changes

The operator clarified that Market Engine must eventually run different tasks independently on GitHub:

* data refresh jobs;
* preparatory/source-context jobs;
* observation jobs;
* analysis/review jobs;
* later recommendation/review jobs;
* later portfolio/review jobs;
* later delivery jobs.

These jobs must not all run together and must not all run on the same schedule.

A future upgrade to one job should not require unrelated jobs to be changed.

## ME13 Status

ME13 is documentation and architecture only.

Created:

* `docs/market_engine/architecture/job_architecture_and_persistence_contract.md`
* `docs/market_engine/governance/job_based_working_method_coding_testing.md`
* `docs/market_engine/audits/me13_job_architecture_and_persistence_contract_audit.md`
* `docs/market_engine/backlog/me13_job_architecture_roadmap_update.md`

## ME14 — Persist raw SEC CompanyFacts source snapshots and support cached source loading

Recommended next implementation sprint.

Owner roles:

* Data Steward;
* Technical Architect;
* Development Lead;
* QA Lead;
* Governance Auditor.

Goal:

Persist bounded raw SEC CompanyFacts provider responses so future source mapping, context building, and observations can run from cached source snapshots instead of repeatedly calling SEC.

Scope:

* bounded SEC CompanyFacts raw JSON snapshot writing;
* snapshot metadata;
* ticker manifest;
* provider error manifest;
* cached snapshot loading;
* mapping/context/observation compatibility from cached input;
* persistence tests;
* old path prohibition tests.

Approved path:

```text
data/market_engine/source_snapshots/sec_companyfacts/<run_id>/
```

Recommended structure:

```text
raw/
  NVDA_companyfacts.json
  AMD_companyfacts.json
snapshot_metadata.json
ticker_manifest.csv
provider_errors.csv
```

Not in scope:

* derived observations;
* free cash flow;
* growth;
* margins;
* valuation metrics;
* score;
* ranking;
* recommendation;
* BUY / SELL / HOLD;
* portfolio mutation;
* watchlist mutation;
* Telegram;
* reporting;
* Decision Engine behavior.

## ME15 — Build cached source-context and observation job execution from persisted snapshots

Candidate follow-up after ME14.

Goal:

Run existing field mapping, source context, and non-decision observation logic from cached source snapshots without provider calls.

Scope:

* source-context job input from persisted raw snapshots;
* observation job input from persisted source contexts;
* job execution boundaries;
* output persistence under `data/market_engine/source_contexts/...` and `data/market_engine/observations/...` if approved;
* contract tests proving no live provider calls.

## ME16 — Add first derived cash-generation observation layer

Candidate follow-up after persistence and cached job execution are stable.

Goal:

Add first derived but still non-decision cash-generation observations.

Potential derived calculation:

```text
free_cash_flow = operating_cash_flow - capital_expenditures
```

Still forbidden:

* FCF yield;
* valuation;
* quality score;
* ranking;
* recommendation;
* BUY / SELL / HOLD;
* portfolio action;
* Telegram/reporting;
* Decision Engine behavior.

## Planning Rule Going Forward

Every future sprint must state which job family it changes and whether the job's public contract changes.

If a sprint changes a public output contract, dependent jobs must either remain compatible or receive explicit follow-up work.

## Current Next Sprint

```text
ME14 — Persist raw SEC CompanyFacts source snapshots and support cached source loading
```
