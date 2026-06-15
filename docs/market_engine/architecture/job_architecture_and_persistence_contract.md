# ME13 — Market Engine Job Architecture and Data Persistence Contract

Owner roles: Product Owner / Technical Architect / Data Steward / Development Lead / QA Lead / Governance Auditor

Status: APPROVED CONTRACT FOR JOB-BASED MARKET ENGINE WORKFLOW

## Purpose

This document changes the Market Engine architecture from a single monolithic run into a set of independently buildable, independently executable, and independently upgradeable jobs.

The goal is to prevent future Market Engine work from becoming one large pipeline where data refresh, source mapping, analysis, recommendations, reporting, delivery, and portfolio review all have to change together.

Market Engine must be built as a job-oriented system where each job has its own input, output, execution cadence, authority boundary, tests, persistence contract, and upgrade path.

## Strategic Decision

Market Engine must not be implemented as one all-in-one runtime command.

The approved architecture is:

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

Each job must be independently executable and independently testable.

A future upgrade to one job must not require unrelated jobs to be rewritten, revalidated, or redeployed unless that job's public input/output contract changes.

## Why This Matters

Market Engine will eventually need different tasks with different execution frequencies.

Examples:

* SEC CompanyFacts source refresh may only need to run weekly or monthly.
* Price data may need to run daily or intraday in a later provider family.
* Fundamental mapping can run after a SEC source refresh.
* Fundamental observations can run from cached context data without calling SEC again.
* Recommendation review can remain manual or scheduled separately.
* Portfolio review can run monthly or only after operator approval.
* Telegram delivery must remain explicitly downstream and separately controlled.

This separation avoids:

* repeated unnecessary provider calls;
* rate-limit pressure;
* slow full-pipeline runs;
* hidden side effects;
* accidental recommendation output from scheduled data jobs;
* tight coupling between unrelated components;
* difficult debugging;
* non-reproducible analysis due to changing live provider responses.

## Job Independence Rule

Every Market Engine job must be designed as a replaceable unit.

A job may depend on another job's documented output contract, but it must not depend on that job's internal implementation.

Allowed dependency:

```text
fundamental_observation_pass reads a documented source context snapshot
```

Forbidden dependency:

```text
fundamental_observation_pass reaches into SEC provider internals or assumes private mapper implementation details
```

If a job is upgraded internally while its output contract remains stable, downstream jobs should not need code changes.

If a job output contract changes, that change must be explicit, versioned or documented, and covered by tests.

## Approved Job Families

### 1. Source Refresh Jobs

Purpose:

Fetch raw provider/source data and persist it as raw source snapshots.

Examples:

```text
sec_companyfacts_source_refresh
future_price_source_refresh
future_news_source_refresh
```

Authority:

Source refresh jobs may fetch and persist raw data. They may not analyze, score, rank, recommend, report, deliver, or mutate portfolio/watchlist state.

Output:

```text
data/market_engine/source_snapshots/<provider>/<run_id>/
```

### 2. Source Mapping Jobs

Purpose:

Map raw provider data into canonical Market Engine source fields using approved field contracts.

Examples:

```text
sec_companyfacts_field_mapping
```

Authority:

Source mapping jobs may normalize, select approved aliases, preserve provenance, and classify source readiness. They may not create analysis conclusions, scores, rankings, or recommendations.

Output:

```text
data/market_engine/source_contexts/<provider_or_context>/<run_id>/
```

### 3. Source Context Jobs

Purpose:

Build source-only context objects or persisted snapshots for later observation and analysis layers.

Examples:

```text
fundamental_source_context_build
```

Authority:

Source context jobs may expose readiness, canonical fields, missing fields, and provenance. They may not emit financial interpretations beyond source readiness.

Output:

```text
data/market_engine/source_contexts/fundamentals/<run_id>/
```

### 4. Observation Jobs

Purpose:

Convert approved source context into source-grounded, non-decision observations.

Examples:

```text
fundamental_observation_pass
cash_generation_observation_pass
```

Authority:

Observation jobs may say that a value is present, missing, positive, negative, zero, complete, or incomplete. They may not emit scores, rankings, recommendations, BUY / SELL / HOLD, portfolio instructions, or delivery output.

Output:

```text
data/market_engine/observations/fundamentals/<run_id>/
```

### 5. Derived Observation Jobs

Purpose:

Calculate explicitly approved derived observations from approved source fields.

Example:

```text
cash_generation_observation_pass
```

Possible future derived value:

```text
free_cash_flow = operating_cash_flow - capital_expenditures
```

Authority:

Derived observation jobs may only calculate metrics that are explicitly approved by a contract. They may not produce valuation, scores, rankings, recommendations, or allocation logic.

Output:

```text
data/market_engine/observations/derived/<run_id>/
```

### 6. Analysis Review Jobs

Purpose:

Combine observations into higher-level analysis summaries for human/operator review.

Authority:

Analysis review jobs may interpret observations, but must remain clearly separated from recommendation authority until a later approved sprint.

Output:

```text
data/market_engine/analysis_reviews/<run_id>/
```

### 7. Recommendation Review Jobs

Purpose:

Future downstream jobs that may eventually propose candidate actions.

Authority:

Not approved yet.

Recommendation jobs must not exist until an explicit recommendation governance contract is approved.

### 8. Portfolio Review Jobs

Purpose:

Future jobs that compare analysis/recommendation candidates with the operator's actual portfolio.

Authority:

Not approved yet.

Portfolio jobs must not mutate portfolio source files without explicit operator approval.

### 9. Delivery Jobs

Purpose:

Future jobs that format and deliver approved output through channels such as Telegram or reports.

Authority:

Not approved yet.

Delivery jobs may only deliver already-approved output from upstream jobs. They must not fetch data, run analysis, create recommendations, or mutate portfolio/watchlist state.

## Job Frequency Contract

Jobs must not assume that every other job runs on the same schedule.

Suggested future frequencies:

| Job | Suggested frequency | Notes |
|---|---:|---|
| `sec_companyfacts_source_refresh` | weekly or monthly | SEC fundamentals update slowly and do not need daily refresh by default. |
| `source_context_build` | after source refresh or manual | Uses cached raw snapshots. |
| `fundamental_observation_pass` | after context build or manual | Uses cached source context. |
| `cash_generation_observation_pass` | after observation pass or manual | Future derived observation layer. |
| `analysis_review` | weekly or manual | Human review, not final recommendation authority. |
| `recommendation_review` | manual initially | Not approved yet. |
| `portfolio_review` | monthly or manual | Not approved yet. |
| `telegram_delivery` | explicit/manual approval only | Not approved yet. |

Future GitHub Actions should reflect these separate schedules.

## Data Persistence Contract

Market Engine must persist data by job family and authority layer.

### Raw Source Snapshots

Raw provider responses must be stored as JSON, not CSV.

Reason:

SEC CompanyFacts data is hierarchical and contains facts, units, periods, filing metadata, accession numbers, frames, and taxonomy namespaces. CSV is insufficient as the canonical raw source format.

Approved path:

```text
data/market_engine/source_snapshots/sec_companyfacts/<run_id>/
```

Recommended structure:

```text
data/market_engine/source_snapshots/sec_companyfacts/<run_id>/
  raw/
    NVDA_companyfacts.json
    AMD_companyfacts.json
  snapshot_metadata.json
  ticker_manifest.csv
  provider_errors.csv
```

### Source Context Snapshots

Canonical mapped source context may be persisted separately from raw data.

Approved path:

```text
data/market_engine/source_contexts/fundamentals/<run_id>/
```

Possible files:

```text
fundamental_source_context.json
field_provenance.csv
missing_fields.csv
context_metadata.json
```

### Observation Outputs

Non-decision observations may be persisted separately from source context.

Approved path:

```text
data/market_engine/observations/fundamentals/<run_id>/
```

Possible files:

```text
fundamental_observations.json
fundamental_observations.csv
observation_metadata.json
```

### Smoke Artifacts

Smoke artifacts remain separate from source snapshots.

Approved path:

```text
data/market_engine/smokes/<job_family>/<provider_or_scope>/<run_id>/
```

Smoke artifacts are evidence, not source truth.

## Data Type Separation

Market Engine must preserve this distinction:

| Data type | Purpose | Canonical location |
|---|---|---|
| Smoke artifacts | Evidence that a bounded smoke run worked | `data/market_engine/smokes/...` |
| Raw source snapshots | Exact provider responses for reproducibility | `data/market_engine/source_snapshots/...` |
| Source contexts | Canonical mapped source data with provenance | `data/market_engine/source_contexts/...` |
| Observation outputs | Non-decision observation results | `data/market_engine/observations/...` |
| Analysis reviews | Later human-readable analysis summaries | `data/market_engine/analysis_reviews/...` |
| Recommendation outputs | Not approved yet | To be defined later |
| Delivery artifacts | Not approved yet | To be defined later |

## Old Path Prohibition

Market Engine jobs must not write to old paths:

```text
data/processed/
data/generated/
data/logs/
data/normalized/
reports/
data/portfolio/
data/watchlist/
```

These paths are legacy or separately governed unless explicitly approved in a future sprint.

## Job Interface Contract

Every job must document:

* job name;
* owner roles;
* input contract;
* output contract;
* persistence path;
* execution cadence;
* whether it may call external providers;
* whether it may write data;
* whether it may emit observations;
* whether it may emit analysis;
* whether it may emit recommendations;
* whether it may mutate portfolio/watchlist data;
* whether it may deliver reports/messages;
* test requirements;
* side-effect boundaries.

## Upgrade Rule

A job may be upgraded independently when:

* its public input/output contract remains stable;
* its tests prove contract compatibility;
* it does not introduce authority outside its job family;
* downstream jobs do not require changes;
* persisted data versioning or metadata captures the job version where relevant.

If the output schema changes, the sprint must:

* document the schema change;
* update downstream contract tests;
* preserve backward compatibility where practical;
* document migration or re-run requirements.

## GitHub Actions Direction

Future GitHub Actions should be job-specific.

Do not create one scheduled action that performs all Market Engine work.

Preferred future action names:

```text
sec-source-refresh.yml
fundamental-source-context.yml
fundamental-observations.yml
cash-generation-observations.yml
analysis-review.yml
```

Each action should have:

* explicit trigger;
* bounded inputs;
* clear output path;
* no hidden downstream authority;
* artifact upload only if approved;
* no secrets printed;
* no portfolio/watchlist mutation unless explicitly approved.

## Testing Implications

Each job family must have its own tests.

Required test families:

* source refresh tests;
* snapshot persistence tests;
* cached loading tests;
* source mapping tests;
* source context tests;
* observation tests;
* job boundary tests;
* forbidden side-effect tests;
* old path prohibition tests.

Automated tests must not call live providers unless a future test policy explicitly authorizes bounded integration tests.

## Coding Implications

Market Engine code should be organized around stable job boundaries.

Guidelines:

* Do not create new loose Python files for every small step.
* Add a new module only when there is a clear ownership boundary.
* Keep provider access separate from mapping, context, observation, analysis, recommendation, reporting, delivery, portfolio, and watchlist logic.
* Keep persistence helpers focused and reusable across jobs.
* Avoid job implementations that directly call downstream jobs without an explicit orchestration contract.
* Prefer functions/classes that consume documented input objects and return documented output objects.

## Immediate Consequence for the Roadmap

The next implementation sprint should not add derived cash-generation observations yet.

Before adding more analysis layers, Market Engine should persist raw SEC CompanyFacts source snapshots and support cached loading.

Recommended next sprint:

```text
ME14 — Persist raw SEC CompanyFacts source snapshots and support cached source loading
```

This sprint should:

* fetch bounded SEC CompanyFacts raw responses;
* save raw JSON snapshots under `data/market_engine/source_snapshots/sec_companyfacts/<run_id>/`;
* create snapshot metadata and ticker manifest files;
* load cached source snapshots without provider calls;
* allow mapping/context/observations to run from cached source data;
* keep all analysis/recommendation/Decision Engine/reporting/Telegram/portfolio/watchlist behavior out of scope.

## ME13 Decision

Market Engine adopts job-based architecture and data persistence contracts before expanding derived analysis.

This changes the immediate roadmap:

```text
Old next step: ME13 — Add first derived cash-generation observation layer
New next step: ME14 — Persist raw SEC CompanyFacts source snapshots and support cached source loading
```

Derived observations remain important, but persistence and job separation are now the higher-priority architecture gate.
