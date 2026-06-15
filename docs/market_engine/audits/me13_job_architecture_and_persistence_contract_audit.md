# ME13 — Job Architecture and Persistence Contract Audit

Owner roles: Product Owner / Technical Architect / Data Steward / Development Lead / QA Lead / Governance Auditor

Status: COMPLETED BY ME13

## Purpose

ME13 records a structural change in Market Engine architecture and workflow.

The project will no longer evolve toward one monolithic run where source refresh, mapping, context building, observation, analysis, recommendation, reporting, delivery, and portfolio review are bundled together.

Market Engine is now governed as a job-oriented system with independent jobs, independent input/output contracts, independent persistence paths, independent execution cadences, and independent upgrade paths.

## Files Created

* `docs/market_engine/architecture/job_architecture_and_persistence_contract.md`
* `docs/market_engine/audits/me13_job_architecture_and_persistence_contract_audit.md`

## Files Updated

This audit is part of the ME13 documentation update. The backlog and other architecture documents should be aligned with this contract before the next implementation sprint.

## Triggering Decision

After ME12, Market Engine could already produce source-grounded non-decision fundamental observations.

The planned next step was a derived cash-generation observation layer. However, the operator identified a higher-priority architectural requirement:

* data refresh must be separate from analysis;
* preparatory jobs must be executable independently;
* not every job should run every day;
* future GitHub jobs/actions should run on different cadences;
* upgrading one job should not require unrelated jobs to change;
* raw data should be persisted so future analysis does not repeatedly require live provider calls.

ME13 therefore pauses additional derived analysis and formalizes job architecture and persistence first.

## Strategic Decision

Market Engine adopts job-based architecture.

Approved job chain:

```text
source refresh jobs
→ persisted source snapshots
→ source context jobs
→ persisted context snapshots
→ observation jobs
→ persisted observation outputs
→ later analysis/recommendation/review jobs
→ later delivery jobs
```

Every job must have:

* documented input contract;
* documented output contract;
* persistence path;
* execution cadence;
* authority boundary;
* side-effect boundary;
* tests;
* upgrade policy.

## Job Independence Decision

Jobs must be independently buildable, independently executable, independently testable, and independently upgradeable.

A job may depend on another job's public output contract, but not on its private implementation.

If a job changes internally while preserving its output contract, downstream jobs should not need changes.

If a job output contract changes, the change must be documented and tested before dependent jobs are updated.

## Approved Job Families

ME13 defines these job families:

* source refresh jobs;
* source mapping jobs;
* source context jobs;
* observation jobs;
* derived observation jobs;
* analysis review jobs;
* future recommendation review jobs;
* future portfolio review jobs;
* future delivery jobs.

Only the early source, context, and observation families are currently approved for implementation.

Recommendation, portfolio review, delivery, reporting, Telegram, and Decision Engine behavior remain not approved unless a later sprint explicitly authorizes them.

## Persistence Decision

Market Engine must persist job outputs by authority layer.

Approved paths:

```text
data/market_engine/source_snapshots/...
data/market_engine/source_contexts/...
data/market_engine/observations/...
data/market_engine/analysis_reviews/...
data/market_engine/smokes/...
```

Raw provider responses must be stored as JSON, not CSV.

CSV may be used for manifests, summaries, and tabular derivative evidence, but not as the canonical raw provider format for SEC CompanyFacts.

## Data Type Separation

ME13 distinguishes:

* smoke artifacts — bounded execution evidence;
* raw source snapshots — exact provider responses;
* source contexts — canonical mapped source data with provenance;
* observation outputs — non-decision observation results;
* analysis reviews — later human review summaries;
* recommendation outputs — not approved yet;
* delivery artifacts — not approved yet.

## Old Path Prohibition

ME13 reaffirms that Market Engine jobs must not write to old paths:

```text
data/processed/
data/generated/
data/logs/
data/normalized/
reports/
data/portfolio/
data/watchlist/
```

## GitHub Actions Direction

Future GitHub Actions must be job-specific.

Market Engine must not use one scheduled GitHub Action that performs all work.

Future action candidates:

```text
sec-source-refresh.yml
fundamental-source-context.yml
fundamental-observations.yml
cash-generation-observations.yml
analysis-review.yml
```

Each action must have explicit triggers, bounded inputs, clear outputs, and no hidden downstream authority.

## Impact on Coding Standards

Future code should be organized around job boundaries.

ME13 reinforces these coding rules:

* no one-off quick scripts as canonical runtime;
* no unnecessary new Python file for every small step;
* new modules only for clear ownership boundaries;
* provider access separate from mapping/context/observation/analysis/recommendation/reporting/delivery;
* persistence helpers reusable across jobs;
* downstream jobs consume documented output objects or persisted snapshots, not upstream internals.

## Impact on Testing Strategy

Future tests must cover job contracts and side-effect boundaries.

Required future test families include:

* source refresh tests;
* snapshot persistence tests;
* cached loading tests;
* source mapping tests;
* source context tests;
* observation tests;
* job boundary tests;
* forbidden side-effect tests;
* old path prohibition tests.

Automated tests must not call live providers unless a future integration-test policy explicitly authorizes bounded provider tests.

## Roadmap Change

The previously recommended next sprint was:

```text
ME13 — Add first derived cash-generation observation layer
```

ME13 changes the roadmap.

The new recommended next implementation sprint is:

```text
ME14 — Persist raw SEC CompanyFacts source snapshots and support cached source loading
```

The derived cash-generation observation layer remains important, but must wait until raw SEC snapshot persistence and cached loading are in place.

## Boundary Confirmation

ME13 is documentation and architecture only.

No Python runtime behavior was intentionally changed.

No tests were intentionally changed.

No provider calls were required.

No generated data was written.

No reports were generated.

No Telegram delivery was introduced.

No portfolio/watchlist mutation was introduced.

No Decision Engine behavior was introduced.

No BUY / SELL / HOLD, recommendation, allocation, ranking, score, conviction, urgency, tradeability, position sizing, or execution behavior was introduced.

## Known Limitations

ME13 defines the job architecture and persistence contract but does not implement persistence yet.

It does not create GitHub Actions.

It does not create raw source snapshots.

It does not implement cached loading.

It does not add derived observations.

## Recommended Next Sprint

```text
ME14 — Persist raw SEC CompanyFacts source snapshots and support cached source loading
```

ME14 should implement bounded SEC raw JSON snapshot persistence under:

```text
data/market_engine/source_snapshots/sec_companyfacts/<run_id>/
```

and support loading cached snapshots so mapping, source context, and observations can run without repeated live SEC calls.
