# ME-GOV01 — Job-Scoped Sprint Naming Convention

Owner roles: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Status: APPROVED GOVERNANCE RULE

## Purpose

This document defines the sprint naming convention for all future Market Engine work after the ME01–ME13 foundation phase.

The purpose is to make every future sprint visibly scoped to a specific Market Engine job family, so Market Engine can be developed, tested, persisted, scheduled, and upgraded as a set of independent jobs instead of as one monolithic pipeline.

The convention must:

* keep `ME` as the Market Engine project prefix;
* preserve `ME01–ME13` as historical foundation sprints;
* introduce job-family sprint prefixes from this point forward;
* make the job family immediately visible in every sprint ID;
* support independent job development;
* support stable input/output contracts;
* support job-specific persistence paths;
* support job-specific tests;
* support job-specific schedules;
* support job-specific upgrade paths;
* prevent broad cross-cutting do-everything sprints;
* make Codex prompts sharper by forcing job-family scope;
* improve roadmap clarity.

## Background

The Market Engine foundation phase used the sprint sequence `ME01`, `ME02`, through `ME13`.

That numbering style worked while the project was defining the overall Market Engine foundation: documentation structure, source intake, SEC CompanyFacts validation, field mapping, fundamental source context, non-decision observations, and the first job-based architecture decision.

ME13 introduced a major structural decision:

Market Engine must no longer be built as one monolithic pipeline.

Instead, Market Engine must be built as a collection of independent jobs. Each job must have clear boundaries, inputs, outputs, persistence paths, test contracts, schedules, failure behavior, and upgrade paths.

A job should be upgradeable without forcing unrelated jobs to change, as long as the public input/output contract remains stable.

Because of this, the old generic numbering style `ME01`, `ME02`, and so on is no longer sufficient for future work. It does not show which job family a sprint belongs to and does not protect the project from broad cross-cutting sprint scopes.

## Decision

From this point forward, all new Market Engine sprints must use job-scoped sprint prefixes.

The `ME` project prefix remains unchanged.

The foundation sprints `ME01–ME13` remain unchanged and must not be renumbered.

All future sprints must use this format:

```text
ME-<JOB-FAMILY><NN> — <Sprint title>
```

Examples:

```text
ME-GOV01 — Define job-scoped sprint naming convention
ME-SR01 — Persist raw SEC CompanyFacts source snapshots and support cached source loading
ME-SC01 — Build SEC CompanyFacts source context from cached source snapshots
ME-FO01 — Produce non-decision fundamental observations from approved source context
ME-RR01 — Define recommendation review contract
```

The sprint ID must make the job family immediately visible.

## Naming Format

The required naming format is:

```text
ME-<FAMILY><NN> — <Action-oriented sprint title>
```

Where:

* `ME` is the fixed Market Engine project prefix;
* `<FAMILY>` is the approved job-family code;
* `<NN>` is a two-digit sequence number inside that job family;
* the sprint title describes the concrete outcome.

Correct:

```text
ME-SR01 — Persist raw SEC CompanyFacts source snapshots and support cached source loading
```

Incorrect:

```text
ME14 — Persist raw SEC CompanyFacts source snapshots
```

Reason: `ME14` continues the old generic sprint sequence and hides the job family.

## Historical Foundation Sprint Rule

`ME01–ME13` are historical foundation sprints.

They must not be renamed, renumbered, or retroactively moved into the new job-family numbering system.

Going forward, they should be referenced as:

```text
ME01–ME13 foundation phase
```

or individually where needed:

```text
ME13 — Job-based Market Engine architecture decision
```

Future sprints may depend on foundation work, but they must not continue the generic sequence as `ME14`, `ME15`, etc.

Correct reference:

```text
This sprint builds on the ME01–ME13 foundation phase, especially ME13, which established the job-based Market Engine architecture.
```

Incorrect reference:

```text
This is ME14, continuing the Market Engine work.
```

## Job-Family Prefix Table

| Prefix | Job family | Definition |
|---|---|---|
| `ME-GOV` | Governance / architecture / working method | Governance decisions, architecture rules, job-boundary doctrine, sprint rules, working method, roadmap structure, authority separation |
| `ME-SR` | Source Refresh jobs | Jobs that fetch, refresh, cache, validate, and persist raw external source data |
| `ME-SC` | Source Context jobs | Jobs that convert raw source data into source-aware context, availability states, metadata, and field-level diagnostics |
| `ME-FO` | Fundamental Observation jobs | Jobs that produce non-decision fundamental observations from approved source context |
| `ME-DO` | Derived Observation jobs | Jobs that produce derived observations, trends, deltas, ratios, comparisons, and computed analytical signals from existing observations |
| `ME-AR` | Analysis Review jobs | Jobs that review observations into analytical interpretation without recommendation authority |
| `ME-RR` | Recommendation Review jobs | Jobs that produce recommendation review output from approved analysis inputs |
| `ME-PR` | Portfolio Review jobs | Jobs that apply portfolio-specific context such as positions, allocation, exposure, concentration, and portfolio fit |
| `ME-DL` | Delivery jobs | Jobs that deliver already-approved outputs through reports, Telegram, dashboards, exports, or other user-facing channels |
| `ME-QA` | Cross-job quality / testing / CI | Contract tests, regression tests, compatibility checks, CI gates, cross-job quality enforcement |
| `ME-DATA` | Data governance / persistence / retention | Shared data layout, persistence policy, retention policy, cache lifecycle, schema storage, and data governance |

## When to Use Each Prefix

### `ME-GOV`

Use `ME-GOV` for governance, architecture, roadmap, and working-method decisions.

Use this prefix for:

* sprint naming conventions;
* job-family definitions;
* architecture doctrine;
* authority boundaries;
* roadmap structure;
* backlog rules;
* documentation rules;
* Codex prompt rules;
* cross-job governance;
* integration contract policy.

Do not use `ME-GOV` for normal implementation work inside a functional job.

### `ME-SR`

Use `ME-SR` for Source Refresh jobs.

Use this prefix for:

* fetching source data;
* refreshing source payloads;
* validating raw source payloads;
* persisting raw source snapshots;
* loading cached source snapshots;
* source refresh schedules;
* source refresh retry behavior;
* source-specific smoke tests.

`ME-SR` jobs must not create analysis, recommendations, portfolio review, or delivery output.

### `ME-SC`

Use `ME-SC` for Source Context jobs.

Use this prefix for:

* source availability context;
* source freshness context;
* source metadata;
* field presence diagnostics;
* source quality context;
* source mapping context;
* missing, partial, stale, or invalid source status.

`ME-SC` jobs interpret source availability and quality, but they must not create recommendation decisions.

### `ME-FO`

Use `ME-FO` for Fundamental Observation jobs.

Use this prefix for:

* revenue observations;
* margin observations;
* profitability observations;
* cash flow observations;
* capex observations;
* debt observations;
* balance sheet observations;
* fundamental quality observations.

`ME-FO` output must remain non-decision observation output.

### `ME-DO`

Use `ME-DO` for Derived Observation jobs.

Use this prefix for:

* ratios;
* deltas;
* trend observations;
* historical comparisons;
* peer-relative observations;
* composite observation signals;
* derived quality indicators.

`ME-DO` jobs may derive new observations from approved upstream observations, but they must not bypass source context or produce final recommendation authority.

### `ME-AR`

Use `ME-AR` for Analysis Review jobs.

Use this prefix for:

* analysis-level interpretation;
* strength and weakness reviews;
* risk interpretation;
* observation summaries;
* source-backed analysis reviews;
* analyst-readable conclusions.

`ME-AR` may interpret observations but must not create recommendation output.

Analysis authority and recommendation authority must remain separated.

### `ME-RR`

Use `ME-RR` for Recommendation Review jobs.

Use this prefix for:

* recommendation review contracts;
* recommendation state logic;
* recommendation rationale;
* buy, watch, or avoid review output;
* conviction review;
* recommendation review summaries.

`ME-RR` may produce recommendation review output, but it must remain separate from delivery jobs and portfolio review jobs.

Recommendation authority and delivery authority must remain separated.

### `ME-PR`

Use `ME-PR` for Portfolio Review jobs.

Use this prefix for:

* position-aware review;
* allocation checks;
* exposure checks;
* concentration risk;
* portfolio fit;
* portfolio constraints;
* portfolio-specific interpretation.

`ME-PR` must not silently merge with `ME-RR`.

Portfolio review may inform recommendation review through an explicit contract, but portfolio authority and general recommendation authority must remain separated.

### `ME-DL`

Use `ME-DL` for Delivery jobs.

Use this prefix for:

* Telegram delivery;
* report generation;
* dashboards;
* exports;
* formatting of approved outputs;
* delivery schedules;
* delivery failure behavior.

`ME-DL` must not create analysis, recommendations, or portfolio review logic.

Delivery jobs deliver outputs. They do not decide.

### `ME-QA`

Use `ME-QA` for quality, tests, and CI work that crosses job boundaries or enforces contracts.

Use this prefix for:

* contract tests;
* regression tests;
* compatibility tests;
* CI gates;
* schema compatibility checks;
* cross-job test fixtures;
* test architecture;
* boundary enforcement tests.

`ME-QA` may touch multiple job families only when the purpose is explicitly quality, testing, CI, or contract enforcement.

### `ME-DATA`

Use `ME-DATA` for data governance, persistence, retention, and data lifecycle work.

Use this prefix for:

* persistence layout;
* data directory structure;
* retention policy;
* cache invalidation policy;
* schema storage;
* snapshot lifecycle;
* cleanup rules;
* data governance.

`ME-DATA` may affect multiple job families only when the sprint is explicitly about shared data governance or persistence policy.

## Examples

Correct examples:

```text
ME-GOV01 — Define job-scoped sprint naming convention
ME-SR01 — Persist raw SEC CompanyFacts source snapshots and support cached source loading
ME-SC01 — Build SEC CompanyFacts source context from cached snapshots
ME-FO01 — Produce non-decision revenue observations from approved source context
ME-DO01 — Derive historical growth observations from fundamental observations
ME-AR01 — Define analysis review contract for fundamental observation summaries
ME-RR01 — Define recommendation review contract without delivery authority
ME-PR01 — Define portfolio exposure review contract
ME-DL01 — Define Telegram delivery contract for reviewed outputs
ME-QA01 — Add job contract tests for Source Refresh and Source Context compatibility
ME-DATA01 — Define raw source snapshot retention policy
```

Incorrect examples:

```text
ME14 — Continue Market Engine implementation
```

Reason: continues the old generic numbering and hides job-family scope.

```text
ME-SR01 — Fetch SEC data and generate recommendations
```

Reason: combines Source Refresh with Recommendation Review.

```text
ME-FO01 — Build observations, portfolio checks, and Telegram delivery
```

Reason: combines Fundamental Observation, Portfolio Review, and Delivery.

```text
ME-RR01 — Analyze fundamentals and send Telegram report
```

Reason: combines Analysis Review, Recommendation Review, and Delivery.

## Numbering Rules

Each job family has its own independent sprint sequence.

Examples:

```text
ME-GOV01
ME-GOV02

ME-SR01
ME-SR02

ME-FO01
ME-FO02

ME-RR01
ME-RR02
```

Rules:

1. Numbering starts at `01` inside each job family.
2. Numbers are sequential within each job family.
3. Numbers are not shared across job families.
4. `ME-SR03` does not imply that `ME-FO03` exists.
5. New sprints must not continue as `ME14`, `ME15`, etc.
6. Completed sprint IDs must not be reused.
7. Cancelled sprint IDs should remain recorded as cancelled rather than reused.
8. If a sprint is renamed before execution but remains in the same job family, it may keep the same number.
9. If a sprint changes job family before execution, it must receive a new ID in the correct family.
10. If a sprint expands beyond its original job family, it must either be split or reclassified as `ME-GOV`, `ME-QA`, `ME-DATA`, or an explicit integration contract sprint.

## Cross-Job Sprint Rules

Cross-job work is allowed only when the sprint is explicitly scoped as one of the following:

* governance;
* architecture;
* quality/testing/CI;
* data governance;
* integration contract work.

Cross-job work must normally use one of these prefixes:

```text
ME-GOV
ME-QA
ME-DATA
```

A sprint that touches multiple functional job families must explain why it is not split.

Acceptable cross-job sprint names:

```text
ME-GOV02 — Define public input/output contract rules for Market Engine jobs
ME-QA01 — Add contract tests for Source Refresh and Source Context compatibility
ME-DATA01 — Define shared persistence layout for source snapshots and observation outputs
```

Unacceptable cross-job sprint names:

```text
ME-SR02 — Refresh sources, build observations, analyze stocks, and deliver Telegram output
ME-FO03 — Update source refresh, derived observations, and portfolio review
ME-DL02 — Change recommendations and delivery formatting
```

If one job depends on another, the work should usually be split into:

1. a governance or contract sprint;
2. one implementation sprint per job family;
3. one optional QA sprint to verify compatibility.

## Split Rules

A sprint must be split when any of the following are true:

1. It changes more than one job's runtime behavior.
2. It changes both upstream source handling and downstream analytical output.
3. It combines source refresh with observations.
4. It combines observations with analysis review.
5. It combines analysis review with recommendation review.
6. It combines recommendation review with delivery.
7. It combines recommendation review with portfolio review without an explicit contract sprint.
8. It introduces shared persistence changes and job business logic changes at the same time.
9. It requires broad test changes across unrelated job families.
10. It cannot be described with one clear job-family prefix.
11. Its title contains a broad `and` joining separate responsibilities.
12. It would make Codex inspect or modify unrelated runtime areas.
13. It would create or reinforce a monolithic Market Engine pipeline.

A sprint does not need to be split when:

* all work belongs to one job family;
* persistence changes are local to that job;
* tests are local to that job;
* documentation updates are directly related to that job;
* the public input/output contract remains stable and explicit.

## Backlog Rules

The Market Engine backlog must use job-scoped sprint IDs for all future work.

Each backlog item must include:

* sprint ID;
* sprint title;
* job family;
* purpose;
* scope;
* non-scope;
* allowed files or directories;
* forbidden files or directories;
* input contract impact;
* output contract impact;
* persistence impact;
* test impact;
* dependency on previous sprints;
* acceptance criteria.

Backlog items should be grouped by job family where practical.

Recommended structure:

```text
## Governance / Architecture
- ME-GOV01 — Define job-scoped sprint naming convention

## Source Refresh
- ME-SR01 — Persist raw SEC CompanyFacts source snapshots and support cached source loading

## Source Context
- ME-SC01 — Build SEC CompanyFacts source context from cached snapshots

## Fundamental Observation
- ME-FO01 — Produce non-decision fundamental observations from approved source context
```

Backlog titles must not hide broad scope behind narrow labels.

Bad:

```text
ME-SR01 — Build source refresh
```

Better:

```text
ME-SR01 — Persist raw SEC CompanyFacts source snapshots and support cached source loading
```

Bad:

```text
ME-RR01 — Improve recommendations
```

Better:

```text
ME-RR01 — Define recommendation review input/output contract without delivery authority
```

## Documentation Rules

Each sprint must create or update documentation in the correct Market Engine documentation area.

Sprint documentation must state:

* sprint ID;
* job family;
* purpose;
* scope;
* non-scope;
* affected input contracts;
* affected output contracts;
* affected persistence paths;
* affected tests;
* dependencies;
* acceptance criteria.

Governance sprints should be stored under a governance or architecture path.

This document is stored at:

```text
docs/market_engine/governance/me_gov01_job_scoped_sprint_naming_convention.md
```

Job-specific documentation should live close to the job family it describes.

Example documentation areas:

```text
docs/market_engine/jobs/source_refresh/
docs/market_engine/jobs/source_context/
docs/market_engine/jobs/fundamental_observation/
docs/market_engine/jobs/derived_observation/
docs/market_engine/jobs/analysis_review/
docs/market_engine/jobs/recommendation_review/
docs/market_engine/jobs/portfolio_review/
docs/market_engine/jobs/delivery/
```

Do not place job-specific logic only in broad generic documents when a job-family document exists.

## Audit Rules

Every completed sprint should have an audit or sprint completion note.

Audit notes must confirm:

* sprint ID;
* branch name;
* files changed;
* whether code changed;
* whether tests changed;
* whether data files changed;
* whether persistence paths changed;
* whether runtime behavior changed;
* whether provider calls changed;
* whether cross-job behavior changed;
* whether the sprint stayed inside its declared job family;
* whether any follow-up sprint is required.

For governance/documentation-only sprints, the audit must explicitly confirm:

```text
No Python code changed.
No tests changed.
No data files changed.
No provider calls were introduced.
No runtime behavior changed.
No recommendation behavior changed.
No portfolio behavior changed.
No delivery or Telegram behavior changed.
```

## Testing Rules

Testing must follow the job-family scope.

For implementation sprints, tests should primarily verify:

* the job's input contract;
* the job's output contract;
* local persistence behavior;
* local failure behavior;
* local schedule or cache behavior where relevant;
* backward compatibility of public contracts.

Cross-job tests belong under `ME-QA`, unless they are small local compatibility checks required by a single job sprint.

A job implementation sprint must not become a broad test rewrite.

If a test change affects multiple jobs, it should usually become a separate `ME-QA` sprint.

## Codex Prompt Rules

Every Codex implementation prompt must include the sprint ID and job family.

A Codex prompt must state:

* sprint ID;
* job family;
* exact goal;
* allowed files or directories;
* forbidden files or directories;
* whether code changes are allowed;
* whether tests are allowed;
* whether documentation changes are required;
* whether data writes are allowed;
* whether provider/network calls are allowed;
* expected audit update;
* acceptance criteria.

Codex prompts must not ask for broad work such as:

```text
Improve the Market Engine pipeline.
```

or:

```text
Make Market Engine fetch data, analyze stocks, create recommendations, and send reports.
```

Good Codex prompt style:

```text
Execute ME-SR01 — Persist raw SEC CompanyFacts source snapshots and support cached source loading.

This is a Source Refresh job sprint.

Allowed scope:
- source refresh documentation;
- source refresh runtime code;
- local source refresh tests;
- ME-SR01 audit file.

Forbidden scope:
- recommendation review;
- portfolio review;
- delivery;
- Telegram;
- broad pipeline refactor;
- Decision Engine behavior.
```

Codex prompts must be narrow enough that Codex can execute the sprint without drifting into unrelated job families.

## Examples of Correct and Incorrect Sprint Names

### Correct

```text
ME-GOV01 — Define job-scoped sprint naming convention
```

Reason: governance decision.

```text
ME-SR01 — Persist raw SEC CompanyFacts source snapshots and support cached source loading
```

Reason: source refresh only.

```text
ME-SC01 — Build SEC CompanyFacts source context from cached source snapshots
```

Reason: source context only.

```text
ME-FO01 — Produce non-decision capex observations from approved source context
```

Reason: fundamental observation only.

```text
ME-DO01 — Derive historical growth observations from fundamental observations
```

Reason: derived observation only.

```text
ME-AR01 — Define analysis review contract for fundamental observation summaries
```

Reason: analysis review only.

```text
ME-RR01 — Define recommendation review contract without delivery authority
```

Reason: recommendation review only.

```text
ME-PR01 — Define portfolio exposure review contract
```

Reason: portfolio review only.

```text
ME-DL01 — Define Telegram delivery contract for reviewed outputs
```

Reason: delivery only.

```text
ME-QA01 — Add job contract tests for Source Refresh and Source Context compatibility
```

Reason: cross-job quality work is explicitly QA.

```text
ME-DATA01 — Define raw source snapshot retention policy
```

Reason: data governance and retention.

### Incorrect

```text
ME14 — Continue Market Engine implementation
```

Reason: generic numbering is no longer allowed after ME13.

```text
ME-SR01 — Fetch SEC data and create stock recommendations
```

Reason: Source Refresh and Recommendation Review are mixed.

```text
ME-SC01 — Build source context and generate Telegram output
```

Reason: Source Context and Delivery are mixed.

```text
ME-FO01 — Build all fundamental logic and reports
```

Reason: Fundamental Observation and Delivery are mixed.

```text
ME-AR01 — Analyze fundamentals and decide buy/sell recommendations
```

Reason: Analysis Review and Recommendation Review are mixed.

```text
ME-RR01 — Review recommendations and update Telegram format
```

Reason: Recommendation Review and Delivery are mixed.

```text
ME-PR01 — Change portfolio review and recommendation authority
```

Reason: Portfolio Review and Recommendation Review must remain separated unless governed by an explicit contract sprint.

```text
ME-DL01 — Decide which stocks to buy and send report
```

Reason: Delivery jobs must not own recommendation authority.

## Next Approved Sprint

The next approved sprint after ME-GOV01 is:

```text
ME-SR01 — Persist raw SEC CompanyFacts source snapshots and support cached source loading
```

This is a Source Refresh job sprint.

Expected scope:

* raw SEC CompanyFacts source snapshot persistence;
* cached source loading;
* local source refresh validation;
* source refresh documentation;
* local source refresh tests if implementation is included;
* ME-SR01 audit note.

Explicit non-scope:

* no recommendation review logic;
* no portfolio review logic;
* no delivery or Telegram changes;
* no analysis authority changes;
* no broad pipeline refactor;
* no monolithic run-everything implementation.

## Governance Status

Status: Approved.

Effective from: immediately after the ME01–ME13 foundation phase.

Applies to:

* all future Market Engine backlog items;
* all future Market Engine documentation sprints;
* all future Market Engine implementation sprints;
* all future Market Engine test sprints;
* all future Market Engine audit notes;
* all future Codex prompts.

`ME01–ME13` remain historical foundation sprints and must not be renumbered.
