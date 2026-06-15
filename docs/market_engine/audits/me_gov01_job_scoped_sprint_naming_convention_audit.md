# ME-GOV01 — Job-Scoped Sprint Naming Convention Audit

Owner roles: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Status: COMPLETED BY ME-GOV01

## Purpose

This audit records completion of `ME-GOV01 — Job-Scoped Sprint Naming Convention`.

ME-GOV01 is a governance/documentation sprint that defines the job-scoped sprint naming convention for all future Market Engine work after the ME01–ME13 foundation phase.

## Files Created

* `docs/market_engine/governance/me_gov01_job_scoped_sprint_naming_convention.md`
* `docs/market_engine/audits/me_gov01_job_scoped_sprint_naming_convention_audit.md`

## Files Updated

* `docs/market_engine/backlog/market_engine_backlog.md`
* `docs/market_engine/backlog/me13_job_architecture_roadmap_update.md`

## Governance Decision Confirmed

ME-GOV01 confirms that:

* `ME` remains the Market Engine project prefix;
* `ME01–ME13` remain historical foundation sprints;
* `ME01–ME13` must not be renumbered;
* future Market Engine sprints must not continue as `ME14`, `ME15`, etc.;
* all future Market Engine sprints must use job-family scoped sprint IDs;
* each job family has its own sequence starting at `01`;
* code changes should usually happen inside one job family at a time;
* cross-job work must be explicitly labeled as governance, QA, data governance, or integration contract work;
* analysis, recommendation, portfolio review, and delivery authority must remain separated;
* a sprint must be split when it crosses unrelated job boundaries or combines incompatible authorities.

## Approved Job-Family Prefixes

```text
ME-GOV   Governance / architecture / working method
ME-SR    Source Refresh jobs
ME-SC    Source Context jobs
ME-FO    Fundamental Observation jobs
ME-DO    Derived Observation jobs
ME-AR    Analysis Review jobs
ME-RR    Recommendation Review jobs
ME-PR    Portfolio Review jobs
ME-DL    Delivery jobs
ME-QA    Cross-job quality/testing/CI
ME-DATA  Data governance / persistence / retention
```

## Backlog and Roadmap Impact

ME-GOV01 supersedes the temporary post-ME13 generic next-sprint label `ME14`.

The next approved sprint is now:

```text
ME-SR01 — Persist raw SEC CompanyFacts source snapshots and support cached source loading
```

ME-SR01 is a Source Refresh job sprint.

ME-SR01 must not include:

* recommendation review;
* portfolio review;
* delivery;
* Telegram;
* broad pipeline refactor;
* Decision Engine behavior;
* monolithic run-everything implementation.

## Boundary Confirmation

ME-GOV01 is documentation and governance only.

No Python code changed.

No tests changed.

No data files changed.

No generated files changed.

No provider calls were introduced.

No live provider calls were run.

No runtime behavior changed.

No source refresh implementation was introduced.

No source context implementation was introduced.

No observation logic changed.

No analysis behavior changed.

No recommendation behavior changed.

No portfolio behavior changed.

No delivery or Telegram behavior changed.

No Decision Engine behavior changed.

## Follow-Up

The next approved sprint is:

```text
ME-SR01 — Persist raw SEC CompanyFacts source snapshots and support cached source loading
```

ME-SR01 should be executed as a Source Refresh job sprint with strict job-family scope.
