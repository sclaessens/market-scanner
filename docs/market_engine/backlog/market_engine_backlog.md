# Market Engine Backlog

Owner role: Scrum Master / PM / Product Owner

Status: ACTIVE MARKET ENGINE BACKLOG

## Purpose

This backlog captures the Market Engine sprint line.

`ME01–ME13` are the historical foundation phase. From `ME-GOV01` onward, all future Market Engine work must use the job-scoped sprint naming convention defined in:

```text
docs/market_engine/governance/me_gov01_job_scoped_sprint_naming_convention.md
```

Backlog items do not authorize implementation unless the sprint scope explicitly does so and repository governance allows it.

## Backlog Rules

* Preserve old repository assets as reference material.
* Do not blindly copy old script-era code.
* Do not use old quick scripts as canonical runtime.
* Do not continue legacy cleanup as the active implementation path.
* Do not delete, archive, rename, or ignore old files as part of Market Engine backlog work unless a sprint explicitly authorizes it.
* Keep classification upstream and allocation downstream.
* Preserve Decision Engine authority as the only allocation authority.
* Keep source readiness separate from investment quality.
* Keep missing data explicit.
* Do not convert missing numeric values to zero.
* Do not introduce BUY / SELL / HOLD, recommendation, allocation, urgency, conviction, tradeability, or hidden ranking semantics outside an approved Decision Engine or recommendation-review boundary.
* Do not introduce Telegram, reporting, portfolio, watchlist, provider, or runtime side effects unless a sprint explicitly authorizes them.
* Future Market Engine sprints must use job-scoped sprint IDs.
* Code changes should usually happen inside one job family at a time.
* Cross-job work must be explicitly labeled as governance, QA, data governance, or integration contract work.
* Analysis, recommendation, portfolio review, and delivery authority must remain separated.

## Historical Foundation Phase

`ME01–ME13` remain historical foundation sprints and must not be renumbered.

They may be referenced as:

```text
ME01–ME13 foundation phase
```

or individually where needed.

Future work must not continue as `ME14`, `ME15`, etc.

## Foundation Sprint Roadmap

### ME01 - Reset Market Engine documentation structure and knowledge extraction policy

Owner roles: PM / Product Owner, Scrum Master, Governance Auditor, Technical Architect

Status: COMPLETED BY ME01

Goal: Create the Market Engine documentation root, knowledge extraction policy, source inventory, baseline coding and testing standards, placeholders, audit record, and backlog.

Scope: Documentation and backlog only.

Outcome: `docs/market_engine/` was established as the active Market Engine documentation root with required baseline documentation and backlog structure.

### ME02 - Extract and write Market Engine functional flow

Owner roles: Functional Analyst, Product Owner, Scrum Master, Governance Auditor

Status: COMPLETED BY ME02

Goal: Extract the Market Engine functional flow from existing documentation, code, tests, audits, and backlog items.

Scope: Functional flow specification, role responsibilities, user/operator workflows, classification flow, state boundaries, and implementation/testing implications.

Outcome: Source intake, analysis, operator review, and downstream layers were separated and documented.

### ME03 - Extract and write Market Engine financial, scanner, and fundamental logic

Owner roles: Financial Analyst, Data Steward, Functional Analyst, Governance Auditor

Status: COMPLETED BY ME03

Goal: Extract financial, scanner, fundamental, and source-readiness logic for Market Engine specifications.

Scope: Financial logic, scanner classification lessons, fundamental data lessons, provider/source readiness, data implications, missing-data rules, quality-state rules, ticker failure handling, source-intake boundaries, analysis boundaries, and failure modes.

Outcome: Financial/scanner/fundamental rules were documented while preserving the boundary between source intake, analysis, recommendation, and allocation authority.

### ME04-PREP - Archive old active documentation and make Market Engine the only active docs root

Owner roles: Scrum Master, Governance Auditor, Technical Architect

Status: COMPLETED BY ME04-PREP

Goal: Preserve former active v2, BL, and reset documentation as historical reference material while making `docs/market_engine/` the only active Market Engine documentation root.

Scope: Documentation structure only.

Outcome: Former active documentation was preserved under `docs/archive/market_scanner_reference/active/`.

### ME04-PREP-B - Inventory remaining legacy documentation outside Market Engine

Owner roles: Scrum Master, Governance Auditor

Status: COMPLETED BY ME04-PREP-B

Goal: Inventory remaining documentation and reference material outside `docs/market_engine/` and outside the Market Scanner reference archive.

Scope: Documentation inventory only.

Outcome: Remaining legacy documentation candidates were inventoried before consolidation.

### ME04-PREP-C - Consolidate remaining legacy documentation under Market Scanner reference archive

Owner roles: Scrum Master, Governance Auditor

Status: COMPLETED BY ME04-PREP-C

Goal: Move clear legacy documentation candidates under `docs/archive/market_scanner_reference/` while keeping `docs/market_engine/` as the only active Market Engine documentation root.

Scope: Documentation structure only.

Outcome: Legacy documentation/reference areas were preserved under `docs/archive/market_scanner_reference/`.

### ME04-PREP-D - Inventory legacy runtime, tests, and data before Market Engine cutover

Owner roles: Technical Architect, Development Lead, Data Steward, QA / Test Lead, Governance Auditor

Status: COMPLETED BY ME04-PREP-D

Goal: Inventory legacy runtime, scripts, tests, data, reports, and root-level files before Market Engine cutover.

Scope: Documentation-only inventory.

Outcome: Old runtime, scripts, tests, data, reports, and root-level files were classified without moving, deleting, or modifying runtime assets.

### ME04 - Extract and write Market Engine technical, coding, and testing architecture

Owner roles: Technical Architect, Development Lead, QA / Test Lead, Governance Auditor

Status: COMPLETED BY ME04

Goal: Extract technical architecture, coding rules, and testing architecture for Market Engine.

Scope: Module ownership, provider/data/analysis/decision separation, runtime boundaries, side-effect controls, test-family conventions, manual smoke harness standards, forbidden field policy, and file strategy.

Outcome: Market Engine technical ownership, provider boundaries, test boundaries, and file/module strategy were documented.

### ME05 - Build all-ticker source intake smoke

Owner roles: Development Lead, Data Steward, QA / Test Lead, Governance Auditor

Status: COMPLETED BY ME05

Goal: Build an explicit all-ticker source intake smoke harness after ME02 through ME04 specifications authorize the boundary.

Scope: Bounded manual source intake smoke harness, source availability capture, per-ticker failure capture, raw evidence feasibility, normalized data feasibility, missingness preservation, and source-readiness states.

Outcome: A clean `src/market_engine/source_intake/` package, fake provider scenarios, readiness statuses, per-ticker intake results, batch summaries, missing-field frequency tracking, targeted tests, fake-provider manual smoke entrypoint, and audit/documentation updates were added.

### ME06 - Add bounded real provider source intake smoke and coverage review

Owner roles: Data Steward, QA / Test Lead, Governance Auditor, Operator / User

Status: COMPLETED BY ME06

Goal: Add a bounded real-provider source intake smoke and review coverage evidence without entering analysis or recommendation behavior.

Scope: First real provider selection, explicit manual invocation, ticker limit, source coverage evidence, failure triage, source-readiness implications, missing-data observations, provider/source limitations, data-owner review, generated-output/archive decision inputs, and backlog follow-up.

Outcome: A SEC CompanyFacts provider adapter, mocked provider tests, explicit real-provider manual smoke flags, ticker limit enforcement, and local source coverage review were added.

### ME07 - Review real-provider coverage and define source-data owner decisions

Owner roles: Data Steward, Technical Architect, Development Lead, QA / Test Lead, Governance Auditor, Operator / User

Status: COMPLETED BY ME07

Goal: Review ME06 real-provider coverage behavior and define source-data owner decisions before building first fundamental source context.

Scope: Provider availability review, SEC access/user-agent/network follow-up, ticker-to-CIK ownership decision, smoke evidence retention policy, required-field alias review, source artifact handling, and readiness criteria for first fundamental source context.

Outcome: The bounded SEC smoke failure was triaged as a controlled network/DNS access failure in the environment; SEC CompanyFacts remained approved for bounded smoke only until access and ownership decisions were resolved.

### ME08 - Repair SEC CompanyFacts network access and rerun bounded coverage review

Owner roles: Data Steward, Technical Architect, Development Lead, QA / Test Lead, Governance Auditor, Operator / User

Status: COMPLETED BY ME08

Goal: Resolve the SEC CompanyFacts network/request access issue and rerun bounded source coverage before approving source context work.

Scope: SEC access diagnostics, User-Agent/contact policy review, environment/network review, bounded manual SEC smoke rerun, ticker-to-CIK ownership decision, source evidence retention decision, and coverage review documentation.

Outcome: Local runtime DNS and HTTPS access succeeded. The bounded SEC CompanyFacts smoke reached `AVAILABLE=4` for `NVDA`, `AMD`, `META`, and `COST` with no missing fields or provider errors. SEC CompanyFacts was approved for bounded coverage review only.

### ME09 - Run bounded multi-ticker SEC CompanyFacts coverage artifact review

Owner roles: Data Steward, Technical Architect, Development Lead, QA / Test Lead, Governance Auditor, Operator / User

Status: COMPLETED BY ME09

Goal: Run a bounded multi-ticker SEC CompanyFacts coverage review and evaluate isolated smoke artifacts before any source context or analysis sprint.

Scope: Explicit bounded ticker set, SEC CompanyFacts coverage review, isolated non-production smoke artifacts if explicitly requested, missing-field evidence, provider-error evidence, ticker-to-CIK source ownership review, artifact retention review, and readiness criteria for first source context.

Outcome: A bounded 10-ticker SEC CompanyFacts coverage review reached `AVAILABLE=10` for `NVDA`, `AMD`, `META`, `COST`, `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `TSLA`, and `AVGO`. Non-production smoke artifacts were written locally and intentionally not committed.

### ME10 - Define approved SEC CompanyFacts field mapping and source coverage contract

Owner roles: Data Steward, Technical Architect, Development Lead, QA / Test Lead, Governance Auditor

Status: COMPLETED BY ME10

Goal: Convert bounded SEC CompanyFacts smoke evidence into an approved source-field mapping and source coverage contract before source context or analysis work.

Scope: SEC field alias review, required-field contract, ticker-to-CIK ownership decision, source coverage contract, artifact retention policy, missing-field semantics, provider-error semantics, and readiness criteria for first fundamental source context.

Outcome: The first SEC CompanyFacts field mapping and source coverage contract was approved for `revenue`, `net_income`, `operating_cash_flow`, and `capital_expenditures`. SEC CompanyFacts was approved for field mapping implementation, but not for analysis.

### ME11 - Implement SEC field mapping and first fundamental source context

Owner roles: Financial Analyst, Data Steward, Technical Architect, Development Lead, QA / Test Lead, Governance Auditor

Status: COMPLETED BY ME11

Goal: Implement the approved SEC CompanyFacts field mapping contract and create the first source-only Market Engine fundamental context.

Scope: Deterministic SEC alias priority, canonical source field mapping, raw source value preservation, SEC fact provenance, missing-data preservation, source readiness, and source-only context objects.

Outcome: SEC CompanyFacts contract mapping, a source-only fundamental context, tests for approved mappings and forbidden substitutions, provenance checks, missing-data checks, and documentation/audit updates were added.

### ME12 - Build first non-decision fundamental analysis pass

Owner roles: Financial Analyst, Data Steward, Technical Architect, Development Lead, QA / Test Lead, Governance Auditor

Status: COMPLETED BY ME12

Goal: Build the first non-decision fundamental analysis pass from approved Market Engine source context.

Scope: Source-backed financial observations, explicit missing-data handling, source limitation flags, and deterministic non-decision context suitable for later operator review.

Outcome: The first non-decision fundamental analysis pass was added. It consumes ME11 source context and emits source-grounded observations without free cash flow, growth, margins, ratios, valuation metrics, scores, rankings, recommendations, or Decision Engine behavior.

### ME13 - Define Market Engine job architecture and data persistence contract

Owner roles: Product Owner / Technical Architect / Data Steward / Development Lead / QA Lead / Governance Auditor

Status: COMPLETED BY ME13

Goal: Define the Market Engine job architecture and data persistence contract before additional derived analysis layers.

Scope: Job-oriented architecture, independent input/output contracts, independent persistence paths, independent execution cadences, authority boundaries, side-effect boundaries, tests, upgrade policy, and GitHub Actions direction.

Outcome: Market Engine is governed as a job-oriented system with independent jobs, independent input/output contracts, independent persistence paths, independent execution cadences, and independent upgrade paths. The previously generic post-ME13 `ME14` next-sprint label is superseded by ME-GOV01.

## Job-Scoped Sprint Governance

### ME-GOV01 — Define job-scoped sprint naming convention

Owner roles: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Job family: Governance / architecture / working method

Status: COMPLETED BY ME-GOV01

Goal: Define the job-scoped sprint naming convention for all future Market Engine work after the ME01–ME13 foundation phase.

Scope: Sprint naming convention, job-family prefixes, numbering rules, cross-job sprint rules, split rules, backlog rules, documentation rules, audit rules, testing rules, Codex prompt rules, foundation sprint reference rules, and next sprint approval.

Not in scope: Python code changes, tests, provider calls, runtime execution, generated data, reports, Telegram, portfolio/watchlist mutation, recommendation behavior, or Decision Engine behavior.

Acceptance criteria:

* `ME` remains the project prefix.
* `ME01–ME13` are preserved as historical foundation sprints.
* Future sprints do not continue as `ME14`, `ME15`, etc.
* Future sprints use job-family prefixes.
* Each job family has its own numbering sequence starting at `01`.
* Cross-job work is explicitly labeled as governance, QA, data governance, or integration contract work.
* Split rules protect job independence.
* Analysis, recommendation, portfolio review, and delivery authority remain separated.
* The next approved sprint is `ME-SR01`.

Outcome: Job-scoped sprint naming is approved and documented in `docs/market_engine/governance/me_gov01_job_scoped_sprint_naming_convention.md`.

## Approved Job Families

| Prefix | Job family | Scope |
|---|---|---|
| `ME-GOV` | Governance / architecture / working method | Governance decisions, architecture rules, job-boundary doctrine, sprint rules, working method, roadmap structure, authority separation |
| `ME-SR` | Source Refresh jobs | Fetch, refresh, cache, validate, and persist raw external source data |
| `ME-SC` | Source Context jobs | Convert raw source data into source-aware context, availability states, metadata, and diagnostics |
| `ME-FO` | Fundamental Observation jobs | Produce non-decision fundamental observations from approved source context |
| `ME-DO` | Derived Observation jobs | Produce derived observations, trends, deltas, ratios, comparisons, and computed analytical signals |
| `ME-AR` | Analysis Review jobs | Review observations into analytical interpretation without recommendation authority |
| `ME-RR` | Recommendation Review jobs | Produce recommendation review output from approved analysis inputs |
| `ME-PR` | Portfolio Review jobs | Apply portfolio-specific context such as positions, allocation, exposure, concentration, and portfolio fit |
| `ME-DL` | Delivery jobs | Deliver already-approved outputs through reports, Telegram, dashboards, exports, or other user-facing channels |
| `ME-QA` | Cross-job quality / testing / CI | Contract tests, regression tests, compatibility checks, CI gates, cross-job quality enforcement |
| `ME-DATA` | Data governance / persistence / retention | Shared data layout, persistence policy, retention policy, cache lifecycle, schema storage, and data governance |

## Next Approved Sprint

### ME-SR01 — Persist raw SEC CompanyFacts source snapshots and support cached source loading

Owner roles: Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: Source Refresh

Status: NEXT APPROVED SPRINT

Goal: Persist bounded raw SEC CompanyFacts provider responses so future source mapping, context building, and observations can run from cached source snapshots instead of repeatedly calling SEC.

Scope:

* bounded SEC CompanyFacts raw JSON snapshot writing;
* snapshot metadata;
* ticker manifest;
* provider error manifest;
* cached snapshot loading;
* source refresh documentation;
* local source refresh tests;
* old path prohibition tests;
* ME-SR01 audit note.

Approved persistence path:

```text
data/market_engine/source_snapshots/sec_companyfacts/<run_id>/
```

Recommended snapshot structure:

```text
raw/
  NVDA_companyfacts.json
  AMD_companyfacts.json
snapshot_metadata.json
ticker_manifest.csv
provider_errors.csv
```

Explicit non-scope:

* no recommendation review;
* no portfolio review;
* no delivery;
* no Telegram;
* no broad pipeline refactor;
* no Decision Engine behavior;
* no monolithic run-everything implementation;
* no derived observations;
* no free cash flow;
* no growth;
* no margins;
* no valuation metrics;
* no score;
* no ranking;
* no BUY / SELL / HOLD;
* no portfolio mutation;
* no watchlist mutation;
* no reporting.

Acceptance criteria:

* Raw SEC CompanyFacts snapshots are persisted as raw JSON, not CSV.
* Cached source loading is supported for downstream mapping/context/observation jobs.
* Provider errors are persisted separately from successful raw payloads.
* Snapshot metadata and ticker manifest are written in the approved source snapshot path.
* No old data/report paths are written.
* Tests remain local to Source Refresh unless an explicit `ME-QA` sprint is created.
* No analysis, recommendation, portfolio, delivery, Telegram, or Decision Engine behavior is introduced.

## Deferred / Future Candidate Sprints

These candidate sprints must receive job-scoped IDs before execution.

### ME-SC01 — Build cached SEC CompanyFacts source context from persisted snapshots

Candidate follow-up after ME-SR01.

Job family: Source Context

Goal: Build source context from cached SEC CompanyFacts source snapshots without live provider calls.

### ME-FO01 — Produce fundamental observations from cached source context

Candidate follow-up after source context job execution is stable.

Job family: Fundamental Observation

Goal: Produce non-decision fundamental observations from approved source context snapshots.

### ME-DO01 — Add first derived cash-generation observation layer

Candidate follow-up after persistence, cached context, and fundamental observation job execution are stable.

Job family: Derived Observation

Goal: Add first derived but still non-decision cash-generation observations.

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

### Future — Produce local operator review output

Candidate future sprint requiring job-scoped classification before execution.

Likely job family: Analysis Review or Delivery, depending on authority and output scope.

Goal: Produce local operator review output for human review without creating allocation authority outside the Decision Engine.

Scope must remain communication/review only unless a later sprint explicitly authorizes delivery, recommendation, or portfolio authority.
