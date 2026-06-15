# ME13 — Job Architecture Roadmap Update

Owner roles: Product Owner / Scrum Master / Technical Architect / Data Steward / Governance Auditor

Status: UPDATED BY ME-GOV01

## Purpose

This document records the Market Engine roadmap change after the ME13 job architecture and persistence decision.

ME13 established that the Market Engine roadmap must prioritize job separation and raw source persistence before additional derived analysis layers.

ME-GOV01 updates the sprint naming convention so the post-foundation roadmap no longer continues as generic `ME14`, `ME15`, etc. Future work must use job-scoped sprint IDs.

## Roadmap Change

Previous recommended next sprint after ME12:

```text
ME13 — Add first derived cash-generation observation layer
```

ME13 revised the roadmap toward job architecture and persistence first.

ME-GOV01 then revised the naming convention for all future work.

Updated sequence:

```text
ME13 — Define Market Engine job architecture and data persistence contract
ME-GOV01 — Define job-scoped sprint naming convention
ME-SR01 — Persist raw SEC CompanyFacts source snapshots and support cached source loading
ME-SC01 — Build cached SEC CompanyFacts source context from persisted snapshots
ME-FO01 — Produce fundamental observations from cached source context
ME-DO01 — Add first derived cash-generation observation layer
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

## ME-GOV01 Status

ME-GOV01 is governance and documentation only.

Created:

* `docs/market_engine/governance/me_gov01_job_scoped_sprint_naming_convention.md`
* `docs/market_engine/audits/me_gov01_job_scoped_sprint_naming_convention_audit.md`

Updated:

* `docs/market_engine/backlog/market_engine_backlog.md`
* `docs/market_engine/backlog/me13_job_architecture_roadmap_update.md`

ME-GOV01 confirms that:

* `ME01–ME13` remain historical foundation sprints;
* future sprints must not continue as `ME14`, `ME15`, etc.;
* future sprints must use job-family prefixes;
* each job family has its own numbering sequence starting at `01`.

## ME-SR01 — Persist raw SEC CompanyFacts source snapshots and support cached source loading

Recommended next implementation sprint.

Job family: Source Refresh

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
* source-refresh-local tests;
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
* source context implementation beyond cached loading compatibility;
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
* Decision Engine behavior;
* broad pipeline refactor;
* monolithic run-everything implementation.

## ME-SC01 — Build cached SEC CompanyFacts source context from persisted snapshots

Candidate follow-up after ME-SR01.

Job family: Source Context

Goal:

Run approved SEC field mapping and source context logic from cached source snapshots without provider calls.

Scope:

* source-context job input from persisted raw snapshots;
* source availability context;
* source freshness context;
* source metadata;
* field presence diagnostics;
* source quality context;
* output persistence under `data/market_engine/source_contexts/...` if separately approved;
* contract tests proving no live provider calls.

## ME-FO01 — Produce fundamental observations from cached source context

Candidate follow-up after source context job execution is stable.

Job family: Fundamental Observation

Goal:

Run non-decision fundamental observation logic from approved cached source context snapshots.

Scope:

* observation job input from persisted source contexts;
* non-decision observation output;
* missing-data preservation;
* source limitation flags;
* output persistence under `data/market_engine/observations/...` if separately approved;
* tests proving no live provider calls and no downstream side effects.

## ME-DO01 — Add first derived cash-generation observation layer

Candidate follow-up after persistence, cached source context, and fundamental observation execution are stable.

Job family: Derived Observation

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

If a sprint crosses job families, it must be explicitly labeled as governance, QA, data governance, or integration contract work.

Analysis, recommendation, portfolio review, and delivery authority must remain separated.

## Current Next Sprint

```text
ME-SR01 — Persist raw SEC CompanyFacts source snapshots and support cached source loading
```
