# ME13 — Job-Based Working Method, Coding, Testing, and Analysis Policy

Owner roles: Product Owner / Scrum Master / Technical Architect / Development Lead / QA Lead / Governance Auditor

Status: ACTIVE MARKET ENGINE WORKING METHOD

## Purpose

This policy translates the ME13 job architecture decision into day-to-day Market Engine work.

The project must now be planned, coded, tested, reviewed, and upgraded as a set of independent jobs rather than one monolithic runtime pipeline.

This policy applies to future analysis work, coding work, testing work, persistence work, GitHub Actions work, and documentation work.

## Core Working Method Change

Previous implicit direction:

```text
data fetch → source mapping → context → observations → analysis → recommendation → reporting → delivery
```

New approved working method:

```text
Build one job at a time.
Document the job contract.
Persist the job output if it will be reused.
Test the job boundary.
Only then build the next job.
```

No sprint should silently expand one job into a full end-to-end pipeline.

## Job-First Planning Rule

Every future sprint must identify which job family it belongs to before implementation begins.

Allowed job families are defined in:

```text
docs/market_engine/architecture/job_architecture_and_persistence_contract.md
```

A sprint must state whether it affects:

* source refresh;
* source mapping;
* source context;
* observation;
* derived observation;
* analysis review;
* recommendation review;
* portfolio review;
* delivery;
* orchestration;
* persistence;
* testing infrastructure.

If a sprint affects more than one job family, the sprint must justify why the combination is safe.

## Analysis Documentation Rule

Analysis documentation must distinguish between:

* source data;
* mapped source context;
* non-decision observations;
* derived observations;
* analysis review;
* recommendation review;
* portfolio-aware review;
* delivery/reporting.

A document may not use the generic word `analysis` to blur these layers.

For example:

```text
fundamental_observation_pass
```

is not the same as:

```text
recommendation_review
```

and neither is the same as:

```text
Decision Engine
```

## Coding Rule: Stable Job Interfaces

Every new job implementation must expose a stable interface.

The interface should describe:

* expected input object or persisted input path;
* output object or persisted output path;
* readiness/failure states;
* allowed side effects;
* forbidden side effects;
* persistence behavior;
* whether external provider calls are allowed.

Downstream jobs must consume the public interface, not private implementation details.

## Coding Rule: Internal Upgrades Must Stay Local

A job may be upgraded internally without changing other jobs if its public contract remains stable.

Examples of acceptable local upgrades:

* improving SEC request timeout handling inside a source refresh job;
* adding a new approved alias inside a field mapping job while keeping canonical output unchanged;
* improving observation messages while preserving observation schema;
* optimizing cached loading while preserving snapshot format.

Examples that require explicit contract review:

* changing output field names;
* changing readiness semantics;
* changing persistence paths;
* adding a derived metric;
* adding recommendation language;
* adding portfolio mutation;
* adding delivery behavior.

## Coding Rule: No Hidden Job Chaining

A job must not secretly execute downstream jobs.

Forbidden example:

```text
source refresh automatically runs analysis and recommendation review
```

Allowed example:

```text
source refresh writes raw source snapshots and exits
```

Any orchestration that chains jobs must be introduced as a separate explicit orchestration contract.

## Coding Rule: Persistence Before Repeated Analysis

If data will be reused by later jobs, it should be persisted before additional analysis layers are built.

For SEC CompanyFacts, the next implementation priority is raw source snapshot persistence and cached loading.

Repeated live SEC calls should not be the default basis for future analysis work.

## Testing Rule: Test Job Contracts, Not Only Functions

Tests must prove job-level behavior.

A job test suite should verify:

* accepted inputs;
* output schema;
* readiness/failure states;
* persistence path rules;
* old path prohibition;
* missing-data behavior;
* no forbidden authority fields;
* no unintended provider calls;
* no unintended downstream job execution;
* no portfolio/watchlist/reporting/Telegram side effects.

## Testing Rule: Contract Compatibility

When a job is upgraded, tests must prove whether downstream contracts remain compatible.

If a downstream job must change because an upstream job changed, the sprint must document the contract break.

## Testing Rule: Live Provider Calls

Automated tests must not call live providers unless a future policy explicitly approves bounded integration tests.

Provider behavior should be tested with fake, mocked, or cached fixture data.

Live provider calls belong in explicit bounded manual jobs or future approved scheduled jobs.

## Persistence Rule: Raw Before Derived

Raw provider data must be persisted before derived analysis becomes operationally important.

For SEC CompanyFacts:

```text
raw SEC JSON snapshot
→ cached loading
→ field mapping
→ source context
→ observations
→ derived observations
→ later review layers
```

Do not skip raw persistence and build a chain that repeatedly depends on live SEC responses.

## GitHub Actions Rule

Future GitHub Actions must be job-specific.

Do not create a single scheduled workflow that performs all Market Engine work.

Each workflow must state:

* trigger;
* job family;
* input path or provider source;
* output path;
* whether provider calls are allowed;
* whether artifacts are uploaded;
* authority boundary;
* side-effect boundary.

## Review Rule

Every future PR description should state:

* which job family changed;
* whether public input/output contracts changed;
* whether any downstream jobs must change;
* whether persistence paths changed;
* whether old paths were touched;
* whether provider calls were used;
* whether tests prove job-boundary safety;
* whether forbidden authority fields were introduced.

## Immediate Roadmap Rule

The next sprint should be:

```text
ME14 — Persist raw SEC CompanyFacts source snapshots and support cached source loading
```

The previously planned derived cash-generation observation layer is deferred until after raw snapshot persistence and cached loading are implemented.

## Non-Negotiable Boundaries

The job-based workflow does not weaken existing governance.

Still forbidden unless explicitly approved:

* BUY / SELL / HOLD;
* recommendation behavior;
* allocation;
* ranking;
* score;
* conviction;
* urgency;
* tradeability;
* position sizing;
* execution advice;
* portfolio mutation;
* watchlist mutation;
* Telegram delivery;
* production reporting;
* Decision Engine changes.

## Summary

Market Engine will now be built as a set of independent jobs with stable contracts.

This is a structural change in how we plan, code, test, persist data, schedule GitHub Actions, and upgrade individual parts of the engine.

The system must become modular not only in code, but also in operations and governance.
