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
* Future sprints must be preserved in the backlog and roadmap as soon as they are identified as logical next steps.
* Planned sprint sequence may only be interrupted when a real problem, blocker, architectural gap, governance risk, test gap, data-quality issue, or newly discovered dependency requires insertion.
* When a sprint is inserted ahead of the planned sequence, the insertion reason must be documented in the backlog and roadmap.
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
| `ME-SD` | Setup Detection jobs | Detect non-actionable setups and patterns from approved observation inputs |
| `ME-PR` | Portfolio Review jobs | Apply portfolio-specific context such as positions, allocation, exposure, concentration, and portfolio fit |
| `ME-DL` | Delivery jobs | Deliver already-approved outputs through reports, Telegram, dashboards, exports, or other user-facing channels |
| `ME-QA` | Cross-job quality / testing / CI | Contract tests, regression tests, compatibility checks, CI gates, cross-job quality enforcement |
| `ME-DATA` | Data governance / persistence / retention | Shared data layout, persistence policy, retention policy, cache lifecycle, schema storage, and data governance |

## Completed Job-Scoped Sprints

### ME-SR01 — Persist raw SEC CompanyFacts source snapshots and support cached source loading

Owner roles: Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: Source Refresh

Status: COMPLETED BY ME-SR01

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

Outcome: ME-SR01 added SEC CompanyFacts raw snapshot persistence, cached raw snapshot loading, provider error manifest writing, metadata validation, latest cached snapshot selection, and an explicit cached SEC provider path that avoids provider/network calls when supplied a cached snapshot file. Tests use temporary local payloads only.

### ME-SC01 — Define SEC CompanyFacts Source Context contract from cached raw snapshots

Owner roles: Data Steward / Technical Architect / Financial Analyst / QA Lead / Governance Auditor

Job family: Source Context

Status: COMPLETED BY ME-SC01

Goal: Define the Source Context contract for building SEC CompanyFacts source context from cached raw source snapshots produced by ME-SR01.

Scope:

* Source Context input contract from cached raw SEC CompanyFacts snapshot envelopes;
* Source Context output contract;
* approved context-level source availability states;
* approved field-level states;
* provenance requirements;
* missingness and provider-error rules;
* persistence paths;
* test requirements for later implementation;
* authority boundaries;
* next implementation sprint identification.

Approved input path:

```text
data/market_engine/source_snapshots/sec_companyfacts/<source_refresh_run_id>/
```

Approved output path:

```text
data/market_engine/source_contexts/fundamentals/<source_context_run_id>/
```

Approved context format version:

```text
sec-companyfacts-source-context-v1
```

Approved context-level states:

* `AVAILABLE`;
* `PARTIAL`;
* `MISSING`;
* `INVALID`;
* `PROVIDER_ERROR`;
* `UNSUPPORTED`.

Approved field-level states:

* `PRESENT`;
* `MISSING`;
* `INVALID`;
* `UNSUPPORTED`.

Approved initial canonical fields:

* `revenue`;
* `net_income`;
* `operating_cash_flow`;
* `capital_expenditures`.

Explicit non-scope:

* no Python implementation;
* no tests;
* no provider calls;
* no runtime behavior;
* no source refresh behavior;
* no fundamental observations;
* no derived observations;
* no free cash flow;
* no growth;
* no margins;
* no valuation metrics;
* no analysis review;
* no recommendation review;
* no portfolio review;
* no delivery;
* no Telegram;
* no Decision Engine behavior.

Acceptance criteria:

* Source Context contract is documented.
* Input contract from ME-SR01 raw snapshots is defined.
* Output contract for source-only context is defined.
* Context-level and field-level states are defined.
* Provenance requirements are defined.
* Missingness and provider-error rules are defined.
* Persistence paths are defined.
* Test requirements are defined for implementation.
* Authority boundaries are explicit.
* Next implementation sprint is identified as `ME-SC02`.
* No runtime, code, test, provider, data, generated artifact, recommendation, portfolio, delivery, Telegram, or Decision Engine behavior is changed.

Outcome: ME-SC01 approved the SEC CompanyFacts Source Context contract in `docs/market_engine/source_context/me_sc01_sec_companyfacts_source_context_contract.md` and recorded the audit in `docs/market_engine/audits/me_sc01_sec_companyfacts_source_context_contract_audit.md`.

## Next Approved Sprint

### ME-SC02 — Implement SEC CompanyFacts Source Context from cached raw snapshots

Owner roles: Data Steward / Technical Architect / Financial Analyst / Development Lead / QA Lead / Governance Auditor

Job family: Source Context

Status: COMPLETED BY ME-SC02

Goal: Implement the SEC CompanyFacts Source Context contract defined by ME-SC01.

Scope:

* load cached raw SEC CompanyFacts snapshot envelopes from the ME-SR01 persistence path;
* build source-only context output;
* emit implemented context-level states: `AVAILABLE`, `PARTIAL`, and `MISSING`;
* reserve contract-level states: `INVALID`, `PROVIDER_ERROR`, and `UNSUPPORTED`;
* emit implemented field-level states: `PRESENT` and `MISSING`;
* reserve contract-level field states: `INVALID` and `UNSUPPORTED`;
* preserve source provenance;
* preserve source refresh snapshot metadata;
* preserve missingness explicitly;
* fail safely with controlled `SecCompanyFactsContextBuildError` when cached snapshot loading fails;
* persist Source Context output under `data/market_engine/source_contexts/fundamentals/<source_context_run_id>/<ticker>/source_context.json`;
* add local Source Context tests using synthetic/temporary cached payloads only;
* document implementation and audit results.

Explicit non-scope:

* no live provider calls in automated tests;
* no source refresh job runner;
* no source refresh behavior change;
* no source intake provider behavior change;
* no fundamental observations;
* no derived observations;
* no free cash flow;
* no growth;
* no margins;
* no valuation metrics;
* no score;
* no ranking;
* no BUY / SELL / HOLD;
* no analysis review;
* no recommendation review;
* no portfolio review;
* no delivery;
* no Telegram;
* no Decision Engine behavior.

Acceptance criteria:

* ME-SC02 implements the ME-SC01 contract for cached raw snapshot consumption.
* Source Context can be built from cached raw SEC CompanyFacts snapshots without live provider calls.
* Source Context output remains source-only.
* Missingness remains explicit.
* Numeric zero is treated as present, not missing.
* Raw source provenance and period metadata are preserved.
* Source refresh snapshot metadata is preserved.
* Cached snapshot failures are controlled and explicit.
* Tests prove boundary compliance and old path prohibition.
* Documentation, backlog, and audit are updated.

Outcome: ME-SC02 added a job-scoped Source Context implementation in `src/market_engine/source_context/`, with tests in `tests/market_engine/source_context/`. The implementation consumes ME-SR01 cached raw SEC CompanyFacts snapshots, emits source-only context output, preserves canonical field values, field states, source provenance, source refresh metadata, and missingness, and can persist context JSON under the approved Source Context path. Automated tests use temporary local cached snapshots only and do not make live provider calls.

## Candidate Follow-Up Sprints

### ME-SR02 — Build bounded SEC CompanyFacts source refresh job runner

Owner roles: Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: Source Refresh

Status: CANDIDATE FOLLOW-UP

Goal: Build a bounded Source Refresh job runner that fetches a controlled ticker set, persists raw SEC CompanyFacts snapshots, and records provider errors under the approved source snapshot path.

Scope:

* bounded ticker input;
* explicit SEC CompanyFacts provider use;
* raw snapshot writing;
* provider error manifest writing;
* run metadata;
* no downstream source context, observations, analysis, recommendation, portfolio, delivery, Telegram, or Decision Engine behavior.

Acceptance criteria:

* Job runner is explicit and bounded.
* Raw successful payloads are persisted under `data/market_engine/source_snapshots/sec_companyfacts/<run_id>/`.
* Provider errors are persisted separately.
* No old data/report paths are written.
* Automated tests do not call live providers.
* No analysis, recommendation, portfolio, delivery, Telegram, or Decision Engine behavior is introduced.

### ME-FO01 — Define Fundamental Observation contract from SEC CompanyFacts Source Context

Owner roles: Financial Analyst / Data Steward / Technical Architect / QA Lead / Governance Auditor

Job family: Fundamental Observation

Status: COMPLETED BY ME-FO01

Goal: Define the non-decision Fundamental Observation contract from approved SEC CompanyFacts Source Context.

Scope:

* define approved Fundamental Observation input contract;
* define approved Fundamental Observation output contract;
* define observation categories;
* define observation states;
* define Source Context state handling;
* define provenance requirements;
* define forbidden authority semantics;
* define persistence path recommendation;
* define ME-FO02 implementation boundaries.

Explicit non-scope:

* no Python code changes;
* no tests;
* no data files;
* no provider calls;
* no runtime behavior;
* no Source Refresh changes;
* no Source Context changes;
* no derived calculations;
* no analysis review;
* no recommendation review;
* no portfolio review;
* no delivery;
* no Telegram;
* no Decision Engine behavior.

Acceptance criteria:

* Fundamental Observation job boundary is defined.
* Source Context input contract is defined.
* Fundamental Observation output contract is defined.
* Approved observation categories and states are defined.
* Source Context state handling is defined.
* Provenance requirements are defined.
* Forbidden authority semantics are defined.
* Persistence path recommendation is defined.
* ME-FO02 implementation scope is clear.
* Sprint remains documentation/contract only.

Outcome: ME-FO01 defined the Fundamental Observation contract from SEC CompanyFacts Source Context. Implementation is deferred to `ME-FO02 — Implement Fundamental Observations from SEC CompanyFacts Source Context`.

### ME-FO02 — Implement Fundamental Observations from SEC CompanyFacts Source Context

Owner roles: Financial Analyst / Data Steward / Technical Architect / QA Lead / Governance Auditor

Job family: Fundamental Observation

Status: COMPLETED BY ME-FO02

Goal: Implement non-decision Fundamental Observations from approved SEC CompanyFacts Source Context.

Scope:

* consume `SecCompanyFactsSourceContext` objects from ME-SC02;
* emit `sec-companyfacts-fundamental-observations-v1` output;
* implement approved ME-FO01 observation categories;
* implement approved ME-FO01 observation states;
* preserve source context state;
* preserve source refresh metadata;
* preserve source values;
* preserve source provenance;
* preserve missingness explicitly;
* treat numeric zero as present;
* persist Fundamental Observation output under `data/market_engine/fundamental_observations/<fundamental_observation_run_id>/<ticker>/fundamental_observations.json`;
* refuse overwrite of existing Fundamental Observation output;
* add local tests using synthetic/temporary cached Source Context input only;
* document implementation and audit results.

Explicit non-scope:

* no raw SEC CompanyFacts fetching;
* no cached raw snapshot loading as a primary input;
* no Source Refresh behavior changes;
* no Source Context behavior changes;
* no derived calculations;
* no free cash flow;
* no growth;
* no margins;
* no ratios;
* no valuation metrics;
* no peer comparison;
* no trend analysis;
* no scoring;
* no ranking;
* no BUY / SELL / HOLD;
* no recommendation review;
* no portfolio review;
* no delivery;
* no Telegram;
* no reporting;
* no Decision Engine behavior;
* no position sizing;
* no execution advice.

Acceptance criteria:

* Available Source Context produces approved observations.
* Partial Source Context preserves missingness.
* Missing Source Context produces `NOT_ASSESSED` and `MISSING_DATA` observations.
* Positive, negative, zero, and missing source values are handled correctly.
* Numeric zero remains present and produces `ZERO_SOURCE_VALUE` where applicable.
* Source values and provenance are preserved.
* Source refresh metadata is preserved.
* Derived calculations are not emitted.
* Recommendation, score, ranking, portfolio, delivery, Telegram, and Decision Engine authority are not emitted.
* Persistence writes JSON under the approved Fundamental Observation path.
* Persistence refuses overwrite.
* Tests do not use live SEC/provider calls.
* Tests do not import legacy runtime modules.
* Documentation, backlog, and audit are updated.

Outcome: ME-FO02 implemented non-decision Fundamental Observations from SEC CompanyFacts Source Context in `src/market_engine/fundamental_observations/`, with tests in `tests/market_engine/fundamental_observations/`. The implementation consumes ME-SC02 Source Context objects, emits source-grounded observation output, preserves source values, missingness, Source Context state, source refresh metadata, and provenance, and stays inside the ME-FO job family without introducing derived calculations, analysis review, recommendation review, portfolio review, delivery, Telegram, or Decision Engine behavior.

### ME-DO01 — Add first derived cash-generation observation layer

Owner roles: Financial Analyst / Data Steward / Technical Architect / QA Lead / Governance Auditor

Job family: Derived Observation

Status: COMPLETED BY ME-DO01

Goal: Add the first derived but still non-decision cash-generation observation layer from approved ME-FO02 Fundamental Observations.

Scope:

* consume `SecCompanyFactsFundamentalObservationSet` objects from ME-FO02;
* emit `sec-companyfacts-derived-cash-generation-observations-v1` output;
* calculate only `free_cash_flow = operating_cash_flow - capital_expenditures`;
* preserve upstream Fundamental Observation references;
* preserve upstream source values;
* preserve upstream source references;
* preserve source context state;
* preserve source refresh metadata;
* preserve missingness explicitly;
* treat numeric zero as present;
* emit positive, negative, and zero derived source-value states;
* emit limitation observations when required source fields are missing;
* persist Derived Cash Generation output under `data/market_engine/derived_observations/cash_generation/<derived_observation_run_id>/<ticker>/derived_cash_generation_observations.json`;
* refuse overwrite of existing Derived Cash Generation output;
* add local tests using synthetic/temporary upstream observations only;
* document implementation and audit results.

Explicit non-scope:

* no raw SEC CompanyFacts fetching;
* no cached raw snapshot loading as a primary input;
* no Source Refresh behavior changes;
* no Source Context behavior changes;
* no Fundamental Observation behavior changes;
* no FCF yield;
* no margins;
* no growth;
* no ratios;
* no valuation metrics;
* no peer comparison;
* no trend analysis;
* no scoring;
* no ranking;
* no BUY / SELL / HOLD;
* no recommendation review;
* no portfolio review;
* no delivery;
* no Telegram;
* no reporting;
* no Decision Engine behavior;
* no position sizing;
* no execution advice.

Acceptance criteria:

* Positive free cash flow is derived correctly.
* Negative free cash flow is derived correctly.
* Zero free cash flow is derived correctly.
* Zero operating cash flow remains present and can be used in derivation.
* Missing operating cash flow limits derivation explicitly.
* Missing capital expenditures limits derivation explicitly.
* Upstream Fundamental Observation references are preserved.
* Upstream source values and source references are preserved.
* Source context metadata is preserved.
* Source refresh metadata is preserved.
* Persistence writes JSON under the approved Derived Observation path.
* Persistence refuses overwrite.
* Analysis, recommendation, score, ranking, portfolio, delivery, Telegram, and Decision Engine authority are not emitted.
* Tests do not use live SEC/provider calls.
* Tests do not import legacy runtime modules.
* Documentation, backlog, and audit are updated.

Outcome: ME-DO01 implemented the first non-decision Derived Observation layer in `src/market_engine/derived_observations/`, with tests in `tests/market_engine/derived_observations/`. The implementation consumes ME-FO02 Fundamental Observations, derives only free cash flow from operating cash flow and capital expenditures, preserves upstream source values, source references, Source Context state, source refresh metadata, and missingness, and stays inside the ME-DO job family without introducing analysis review, recommendation review, portfolio review, delivery, Telegram, reporting, or Decision Engine behavior.

### ME-AR01 — Define Analysis Review contract from Fundamental and Derived Observations

Owner roles: Financial Analyst / Functional Analyst / Data Steward / Technical Architect / QA Lead / Governance Auditor

Job family: Analysis Review

Status: COMPLETED BY ME-AR01

Goal: Define the non-recommendation Analysis Review contract from approved ME-FO02 Fundamental Observations and ME-DO01 Derived Observations.

Scope:

* define approved Analysis Review input families;
* define approved upstream input formats;
* define recommended Analysis Review output format;
* define recommended Analysis Review persistence path;
* define approved Analysis Review categories;
* define approved Analysis Review states;
* define state semantics;
* define recommended review item structure;
* define approved and forbidden message style;
* define provenance requirements;
* define persistence requirements;
* define ME-AR02 implementation requirements;
* preserve recommendation, portfolio, delivery, Telegram, reporting, and Decision Engine boundaries.

Approved input families:

* ME-FO — Fundamental Observations;
* ME-DO — Derived Observations.

Approved initial input formats:

* `sec-companyfacts-fundamental-observations-v1`;
* `sec-companyfacts-derived-cash-generation-observations-v1`.

Recommended output format:

* `sec-companyfacts-analysis-review-v1`.

Recommended output path:

* `data/market_engine/analysis_reviews/<analysis_review_run_id>/<ticker>/analysis_review.json`.

Approved Analysis Review categories:

* `SOURCE_AVAILABILITY_REVIEW`;
* `FUNDAMENTAL_OBSERVATION_COMPLETENESS_REVIEW`;
* `CASH_GENERATION_REVIEW`;
* `FREE_CASH_FLOW_REVIEW`;
* `DATA_LIMITATION_REVIEW`;
* `HUMAN_REVIEW_REQUIREMENT`.

Approved Analysis Review states:

* `SOURCE_HEALTHY`;
* `SOURCE_LIMITED`;
* `OBSERVATIONS_COMPLETE`;
* `OBSERVATIONS_LIMITED`;
* `CASH_GENERATION_POSITIVE`;
* `CASH_GENERATION_NEGATIVE`;
* `CASH_GENERATION_NEUTRAL`;
* `DATA_LIMITED`;
* `REQUIRES_HUMAN_REVIEW`;
* `NOT_ASSESSED`.

Explicit non-scope:

* no Python implementation;
* no tests;
* no runtime behavior;
* no provider calls;
* no data writes;
* no generated artifacts;
* no raw SEC CompanyFacts fetching;
* no Source Refresh changes;
* no Source Context changes;
* no Fundamental Observation changes;
* no Derived Observation changes;
* no Recommendation Review behavior;
* no Portfolio Review behavior;
* no Delivery behavior;
* no Telegram;
* no reporting;
* no Decision Engine behavior;
* no BUY / SELL / HOLD;
* no target price;
* no score;
* no ranking;
* no rating;
* no conviction;
* no urgency;
* no tradeability;
* no allocation;
* no position sizing;
* no execution advice;
* no watchlist mutation;
* no portfolio mutation.

Acceptance criteria:

* Analysis Review job boundary is defined.
* Approved upstream observation families are defined.
* Approved upstream input formats are defined.
* Recommended Analysis Review output format is defined.
* Recommended Analysis Review persistence path is defined.
* Approved Analysis Review categories are defined.
* Approved Analysis Review states are defined.
* State semantics are documented.
* Approved message style is documented.
* Forbidden message style is documented.
* Provenance requirements are documented.
* Persistence requirements are documented.
* ME-AR02 implementation requirements are documented.
* Recommendation, portfolio, delivery, Telegram, reporting, and Decision Engine boundaries remain explicit.
* Sprint remains documentation/contract only.

Outcome: ME-AR01 defined the non-recommendation Analysis Review contract from ME-FO02 Fundamental Observations and ME-DO01 Derived Observations. The contract approves initial Analysis Review categories and states, defines provenance and persistence requirements, and prepares ME-AR02 implementation without introducing Python code, tests, runtime behavior, provider calls, data writes, Recommendation Review, Portfolio Review, Delivery, Telegram, reporting, or Decision Engine authority.

### ME-AR02 — Implement Analysis Review from Fundamental and Derived Observations

Owner roles: Financial Analyst / Functional Analyst / Data Steward / Technical Architect / QA Lead / Governance Auditor

Job family: Analysis Review

Status: COMPLETED BY ME-AR02

Goal: Implement non-recommendation Analysis Review from approved ME-FO02 Fundamental Observations and ME-DO01 Derived Observations.

Scope:

* consume `SecCompanyFactsFundamentalObservationSet` objects from ME-FO02;
* consume `SecCompanyFactsDerivedCashGenerationObservationSet` objects from ME-DO01;
* emit `sec-companyfacts-analysis-review-v1` output;
* implement approved ME-AR01 Analysis Review categories;
* implement approved ME-AR01 Analysis Review states;
* validate upstream observation-set alignment;
* preserve upstream Fundamental Observation references;
* preserve upstream Derived Observation references;
* preserve upstream source values;
* preserve upstream derived values;
* preserve source context state;
* preserve source refresh metadata;
* preserve missingness and limitation states;
* emit data limitation review when upstream observations are limited;
* emit human review requirement when upstream observations are incomplete or limited;
* persist Analysis Review output under `data/market_engine/analysis_reviews/<analysis_review_run_id>/<ticker>/analysis_review.json`;
* refuse overwrite of existing Analysis Review output;
* add local tests using synthetic/temporary upstream observations only;
* document implementation and audit results.

Approved Analysis Review categories implemented:

* `SOURCE_AVAILABILITY_REVIEW`;
* `FUNDAMENTAL_OBSERVATION_COMPLETENESS_REVIEW`;
* `CASH_GENERATION_REVIEW`;
* `FREE_CASH_FLOW_REVIEW`;
* `DATA_LIMITATION_REVIEW`;
* `HUMAN_REVIEW_REQUIREMENT`.

Approved Analysis Review states implemented:

* `SOURCE_HEALTHY`;
* `SOURCE_LIMITED`;
* `OBSERVATIONS_COMPLETE`;
* `OBSERVATIONS_LIMITED`;
* `CASH_GENERATION_POSITIVE`;
* `CASH_GENERATION_NEGATIVE`;
* `CASH_GENERATION_NEUTRAL`;
* `DATA_LIMITED`;
* `REQUIRES_HUMAN_REVIEW`;
* `NOT_ASSESSED`.

Explicit non-scope:

* no raw SEC CompanyFacts fetching;
* no provider calls;
* no Source Refresh behavior changes;
* no Source Context behavior changes;
* no Fundamental Observation behavior changes;
* no Derived Observation behavior changes;
* no Recommendation Review behavior;
* no Portfolio Review behavior;
* no Delivery behavior;
* no Telegram;
* no reporting;
* no Decision Engine behavior;
* no BUY / SELL / HOLD;
* no target price;
* no score;
* no ranking;
* no rating;
* no conviction;
* no urgency;
* no tradeability;
* no allocation;
* no position sizing;
* no execution advice;
* no watchlist mutation;
* no portfolio mutation.

Acceptance criteria:

* Analysis Review output is emitted using `sec-companyfacts-analysis-review-v1`.
* Complete positive upstream observations produce non-recommendation Analysis Review.
* Negative cash generation is reviewed without recommendation authority.
* Neutral cash generation is reviewed without recommendation authority.
* Limited upstream observations emit data limitation review.
* Limited upstream observations emit human review requirement.
* Upstream Fundamental Observation references are preserved.
* Upstream Derived Observation references are preserved.
* Upstream source values and derived values are preserved.
* Source context metadata is preserved.
* Source refresh metadata is preserved.
* Upstream observation-set mismatch fails safely.
* Persistence writes JSON under the approved Analysis Review path.
* Persistence refuses overwrite.
* Recommendation, score, ranking, portfolio, delivery, Telegram, reporting, and Decision Engine authority are not emitted.
* Tests do not use live SEC/provider calls.
* Tests do not import legacy runtime modules.
* Documentation, backlog, and audit are updated.

Outcome: ME-AR02 implemented non-recommendation Analysis Review in `src/market_engine/analysis_review/`, with tests in `tests/market_engine/analysis_review/`. The implementation consumes ME-FO02 Fundamental Observations and ME-DO01 Derived Cash Generation Observations, emits approved ME-AR01 review categories and states, preserves upstream observation references, source values, derived values, Source Context state, source refresh metadata, missingness, and limitation states, and stays inside the ME-AR job family without introducing Recommendation Review, Portfolio Review, Delivery, Telegram, reporting, or Decision Engine behavior.

### ME-RR01 — Define Recommendation Review contract from Analysis Review

Owner roles: Financial Analyst / Functional Analyst / Data Steward / Technical Architect / QA Lead / Governance Auditor

Job family: Recommendation Review

Status: COMPLETED BY ME-RR01

Goal: Define the Recommendation Review contract from approved Analysis Review output.

Scope:

* define the Recommendation Review contract boundary;
* define allowed input contract sec-companyfacts-analysis-review-v1;
* define recommended output contract sec-companyfacts-recommendation-review-v1;
* define recommended future output path data/market_engine/recommendation_reviews/<recommendation_review_run_id>/<ticker>/recommendation_review.json;
* define review states;
* define review categories;
* define allowed message semantics;
* define forbidden message semantics;
* define missing-data and numeric-zero requirements;
* define provenance requirements;
* define boundaries with Analysis Review, Portfolio Review, Decision Engine, Delivery, Reporting, Telegram, providers, and legacy runtime;
* define ME-RR02 implementation requirements.

Approved input contract:

* sec-companyfacts-analysis-review-v1.

Recommended output contract:

* sec-companyfacts-recommendation-review-v1.

Recommended future output path:

* data/market_engine/recommendation_reviews/<recommendation_review_run_id>/<ticker>/recommendation_review.json.

Approved review states:

* human_review_required;
* insufficient_evidence;
* blocked_by_missing_data;
* not_applicable.

Approved review categories:

* analysis_supportive_but_not_actionable;
* analysis_mixed_or_conflicted;
* analysis_blocked_by_missing_data;
* analysis_not_supported;
* input_contract_invalid.

Explicit non-scope:

* no Python implementation;
* no tests;
* no runtime behavior;
* no provider calls;
* no data writes;
* no generated artifacts;
* no portfolio review;
* no portfolio action;
* no allocation;
* no position sizing;
* no execution advice;
* no Telegram;
* no reporting;
* no delivery;
* no Decision Engine behavior;
* no BUY / SELL / HOLD as direct trading instructions;
* no score;
* no ranking;
* no conviction;
* no urgency;
* no tradeability;
* no watchlist mutation;
* no portfolio mutation.

Acceptance criteria:

* Recommendation Review job boundary is defined.
* Approved Analysis Review input contract is defined.
* Recommended Recommendation Review output contract is defined.
* Recommended future persistence path is defined.
* Review states are defined.
* Review categories are defined.
* Allowed message semantics are defined.
* Forbidden message semantics are defined.
* Missing-data rules are defined.
* Numeric-zero rules are defined.
* Provenance requirements are defined.
* Boundaries with Portfolio Review, Delivery, Reporting, Telegram, and Decision Engine remain explicit.
* ME-RR02 implementation requirements are documented.
* Sprint remains documentation/contract only.

Outcome: ME-RR01 defined Recommendation Review as a non-actionable, source-grounded, human-review routing layer from sec-companyfacts-analysis-review-v1. The contract approves initial review states and review categories, defines provenance and boundary requirements, and prepares ME-RR02 implementation without introducing Python code, tests, runtime behavior, provider calls, data writes, Portfolio Review, Delivery, Telegram, reporting, or Decision Engine authority.

### ME-RR02 — Implement Recommendation Review from Analysis Review

Owner roles: Financial Analyst / Functional Analyst / Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: Recommendation Review

Status: COMPLETED BY ME-RR02

Goal: Implement the minimum viable non-actionable Recommendation Review builder from approved Analysis Review output.

Scope remained inside the ME-RR job family and did not introduce portfolio review, delivery, Telegram, reporting, Decision Engine behavior, BUY / SELL / HOLD action semantics, allocation, position sizing, execution advice, ranking, scoring, conviction, urgency, tradeability, watchlist mutation, or portfolio mutation.

Implemented input contract:

* sec-companyfacts-analysis-review-v1.

Implemented output contract:

* sec-companyfacts-recommendation-review-v1.

Implemented runtime module:

* `src/market_engine/recommendation_review/sec_companyfacts_recommendation_review.py`.

Implemented package export:

* `src/market_engine/recommendation_review/__init__.py`.

Implemented tests:

* `tests/market_engine/recommendation_review/test_sec_companyfacts_recommendation_review.py`.

Implemented audit:

* `docs/market_engine/audits/me_rr02_recommendation_review_implementation_audit.md`.

Implemented review states:

* human_review_required;
* insufficient_evidence;
* blocked_by_missing_data;
* not_applicable.

Implemented review categories:

* analysis_supportive_but_not_actionable;
* analysis_mixed_or_conflicted;
* analysis_blocked_by_missing_data;
* analysis_not_supported;
* input_contract_invalid.

Implemented behavior:

* supportive Analysis Review input creates a non-actionable human-review candidate;
* limited Analysis Review input blocks Recommendation Review with explicit missing data;
* unsupported Analysis Review contracts fail closed;
* Recommendation Review JSON can be persisted under `data/market_engine/recommendation_reviews`;
* persistence refuses overwrite;
* normal review text does not emit action-authority terms;
* legacy `scripts` and `market_scanner` imports are not introduced.

Validation:

* targeted Recommendation Review tests passed: 7 passed;
* full Market Engine test suite passed: 136 passed.

Outcome: ME-RR02 implemented the first non-actionable SEC CompanyFacts Recommendation Review layer. The layer consumes `sec-companyfacts-analysis-review-v1`, emits `sec-companyfacts-recommendation-review-v1`, preserves upstream provenance and missing-data state, persists JSON safely, refuses overwrite, and keeps portfolio, delivery, Telegram, reporting, and Decision Engine authority out of scope.

### ME-RM01 — Align Market Engine roadmap and sprint sequence

Owner roles: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Job family: Roadmap / Governance

Status: COMPLETED BY ME-RM01

Goal: Align the Market Engine roadmap and backlog after Recommendation Review implementation and preserve the required future sprint sequence.

Scope: Roadmap documentation, backlog sequence update, Setup Detection insertion before Portfolio Review, and governance rule for preserving future sprint sequence.

Outcome: ME-RM01 created `docs/market_engine/roadmap/market_engine_roadmap.md`, inserted Setup Detection before Portfolio Review, moved Portfolio Review after Setup Detection-aware Analysis Review and Recommendation Review work, and added the governance rule that future logical next sprints must be preserved in the backlog and roadmap when identified.

Insertion reason: Setup Detection was identified as a missing architectural layer between Derived Observations and downstream Analysis Review / Recommendation Review / Portfolio Review. Without this insertion, the roadmap would jump too quickly from Recommendation Review to Portfolio Review and skip a required pattern/setup layer.

## Completed Sprint

### ME-SD01 — Define Setup Detection contract

Owner roles: Financial Analyst / Functional Analyst / Data Steward / Technical Architect / QA Lead / Governance Auditor

Job family: Setup Detection

Status: COMPLETED BY ME-SD01

Goal: Define the contract for detecting patterns and setups from Fundamental Observations and Derived Observations.

Scope: Documentation-only contract sprint.

Implemented input contracts:

* `sec-companyfacts-fundamental-observations-v1`;
* `sec-companyfacts-derived-cash-generation-observations-v1`.

Implemented output contract:

* `sec-companyfacts-setup-detection-v1`.

Implemented documentation:

* `docs/market_engine/setup_detection/me_sd01_setup_detection_contract.md`.

Implemented audit:

* `docs/market_engine/audits/me_sd01_setup_detection_contract_audit.md`.

ME-SD01 defined:

* Setup Detection job-family boundary;
* setup definition;
* initial setup families;
* setup categories;
* setup states;
* required evidence model;
* missing-data handling;
* non-actionable boundary;
* provenance requirements;
* future ME-SD02 implementation requirements;
* future ME-SD02 persistence requirements;
* future ME-SD02 test requirements;
* relationship to Analysis Review, Recommendation Review, Portfolio Review, Decision Engine, and Delivery / Reporting.

ME-SD01 did not introduce Python code, tests, provider calls, data writes, BUY / SELL / HOLD action semantics, portfolio mutation, Decision Engine behavior, Telegram, reporting, recommendation authority, allocation, execution advice, or delivery behavior.

Outcome: ME-SD01 defined Setup Detection as the missing non-actionable pattern/setup layer between Derived Observations and Analysis Review. The contract allows future ME-SD02 implementation to detect structured setups from approved Fundamental Observations and Derived Cash Generation Observations while preserving provenance, missing-data state, source grounding, numeric-zero semantics, and authority boundaries.

### ME-SD02 — Implement first Setup Detection layer

Owner roles: Financial Analyst / Functional Analyst / Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: Setup Detection

Status: COMPLETED BY ME-SD02

Goal: Implement the first non-actionable Setup Detection builder from approved observation inputs.

Scope: Local synthetic tests only; no live providers; no portfolio mutation; no Decision Engine behavior; no BUY / SELL / HOLD action semantics.

Implemented runtime module:

* `src/market_engine/setup_detection/sec_companyfacts_setup_detection.py`

Implemented package export:

* `src/market_engine/setup_detection/__init__.py`

Implemented tests:

* `tests/market_engine/setup_detection/test_sec_companyfacts_setup_detection.py`

Implemented audit:

* `docs/market_engine/audits/me_sd02_setup_detection_implementation_audit.md`

Implemented input contracts:

* `sec-companyfacts-fundamental-observations-v1`;
* `sec-companyfacts-derived-cash-generation-observations-v1`.

Implemented output contract:

* `sec-companyfacts-setup-detection-v1`.

ME-SD02 implemented:

* Setup Detection runtime module under the active `market_engine` package;
* builder equivalent to `build_sec_companyfacts_setup_detection(...)`;
* output contract `sec-companyfacts-setup-detection-v1`;
* setup items using the categories and states defined by ME-SD01;
* source and derived observation references;
* explicit missing-data preservation;
* numeric-zero preservation;
* fail-closed behavior for unsupported input contracts;
* JSON persistence equivalent to `persist_sec_companyfacts_setup_detection(...)`;
* overwrite refusal for persisted setup detection output;
* local synthetic tests.

ME-SD02 tested:

* complete positive setup evidence produces setup detection output;
* partial evidence produces `setup_partially_detected`;
* missing required observations produce `setup_blocked_by_missing_data`;
* conflicted evidence produces `setup_conflicted`;
* unsupported input contract fails closed;
* numeric zero is preserved and not treated as missing;
* source and derived references are preserved;
* forbidden action-authority terms are not emitted in normal setup text;
* persistence writes JSON under temporary root;
* persistence refuses overwrite;
* no legacy `scripts` or old `market_scanner` imports are introduced.

ME-SD02 did not introduce:

* live provider calls;
* SEC or EDGAR calls;
* yfinance calls;
* production data writes;
* Analysis Review behavior changes;
* Recommendation Review behavior changes;
* Portfolio Review behavior;
* Decision Engine behavior;
* Telegram delivery;
* reporting output;
* BUY / SELL / HOLD action semantics;
* allocation;
* position sizing;
* execution advice;
* ranking;
* scoring;
* conviction scoring;
* urgency scoring;
* tradeability scoring.

Outcome: ME-SD02 implemented the first non-actionable Setup Detection layer in `src/market_engine/setup_detection/`, with tests in `tests/market_engine/setup_detection/`. The implementation consumes approved SEC CompanyFacts Fundamental Observations and Derived Cash Generation Observations, emits `sec-companyfacts-setup-detection-v1`, preserves source and derived observation references, preserves missing-data and numeric-zero semantics, implements JSON persistence under `data/market_engine/setup_detections/<run_id>/<ticker>/setup_detection.json`, refuses overwrite, and does not introduce Analysis Review, Recommendation Review, Portfolio Review, Delivery, Telegram, reporting, or Decision Engine behavior.

### ME-AR03 — Extend Analysis Review contract for Setup Detection input

Owner roles: Financial Analyst / Functional Analyst / Data Steward / Technical Architect / QA Lead / Governance Auditor

Job family: Analysis Review

Status: COMPLETED BY ME-AR03

Goal: Define how Analysis Review can consume Setup Detection output without recommendation authority.

Scope: Documentation-only contract update.

Implemented contract:

* `docs/market_engine/analysis_review/me_ar03_setup_detection_input_contract.md`

Implemented audit:

* `docs/market_engine/audits/me_ar03_setup_detection_input_contract_audit.md`

ME-AR03 defined:

* how `sec-companyfacts-setup-detection-v1` becomes an approved Analysis Review input;
* how Setup Detection evidence is referenced from Analysis Review items;
* how setup limitations and missing-data states are preserved;
* how Setup Detection categories map to Analysis Review categories;
* how Setup Detection states map to Analysis Review states;
* how conflicted, partial, blocked, and not-assessed setup evidence should be handled;
* how numeric zero remains present and must not be treated as missing;
* how Analysis Review remains non-recommendation and non-actionable;
* how ME-AR04 must implement Setup Detection-aware Analysis Review behavior.

ME-AR03 did not introduce Python code, tests, provider calls, data writes, Recommendation Review behavior, Portfolio Review behavior, Telegram, reporting, Decision Engine behavior, BUY / SELL / HOLD action semantics, allocation, position sizing, execution advice, ranking, scoring, conviction, urgency, or tradeability authority.

Outcome: ME-AR03 extended the Analysis Review contract so a future ME-AR04 implementation can consume `sec-companyfacts-setup-detection-v1` while preserving Analysis Review as descriptive, provenance-preserving, missing-data-aware, numeric-zero-safe, non-recommendation, and non-actionable.

### ME-AR04 — Implement Analysis Review consumption of Setup Detection

Owner roles: Financial Analyst / Functional Analyst / Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: Analysis Review

Status: COMPLETED BY ME-AR04

Goal: Implement Analysis Review support for Setup Detection input.

Scope: Local synthetic tests only; no provider calls; no Recommendation Review, Portfolio Review, Delivery, Telegram, reporting, or Decision Engine behavior.

ME-AR04 must implement Analysis Review consumption of Setup Detection according to the ME-AR03 contract.

ME-AR04 must:

* consume `sec-companyfacts-setup-detection-v1`;
* preserve existing Fundamental Observation and Derived Observation behavior;
* validate input alignment across Fundamental Observations, Derived Observations, and Setup Detection;
* emit Setup Detection-aware Analysis Review items;
* preserve Setup Detection categories and states;
* preserve Setup Detection evidence;
* preserve Setup Detection limitations;
* preserve missing observations;
* preserve source and derived references;
* preserve numeric-zero semantics;
* preserve non-recommendation and non-actionable boundary markers;
* fail closed or emit controlled limitation output for unsupported Setup Detection input contracts;
* add local synthetic tests only.

ME-AR04 must test:

* complete Setup Detection input creates Setup Detection-aware Analysis Review;
* partial setup input creates partial Setup Detection-aware review;
* missing setup evidence creates blocked or data-limited review;
* conflicted setup input creates conflicted review and human-review routing;
* not-assessed setup input remains not assessed;
* unsupported Setup Detection input contract fails closed;
* numeric zero remains present and is not treated as missing;
* Setup Detection references are preserved;
* Fundamental Observation and Derived Observation references remain preserved;
* existing Analysis Review behavior is not broken;
* forbidden action-authority terms are not emitted;
* no legacy `scripts` or old `market_scanner` imports are introduced.

ME-AR04 must not introduce recommendation authority, portfolio mutation, delivery behavior, Telegram behavior, reporting behavior, Decision Engine behavior, BUY / SELL / HOLD action semantics, allocation, position sizing, execution advice, ranking, scoring, conviction, urgency, or tradeability authority.

Implemented runtime:

* `src/market_engine/analysis_review/sec_companyfacts_analysis_review.py`

Implemented tests:

* `tests/market_engine/analysis_review/test_sec_companyfacts_analysis_review.py`

Implemented documentation:

* `docs/market_engine/analysis_review/me_ar04_analysis_review_setup_detection_implementation.md`
* `docs/market_engine/audits/me_ar04_analysis_review_setup_detection_implementation_audit.md`

Outcome: ME-AR04 extended the existing `sec-companyfacts-analysis-review-v1` implementation with optional Setup Detection input. The implementation preserves existing ME-AR02 behavior without Setup Detection input, validates Setup Detection alignment and contract version, emits Setup Detection-aware Analysis Review items, preserves setup evidence, setup limitations, missing observations, source and derived references, numeric-zero semantics, and remains non-recommendation and non-actionable.

## Completed Sprint

### ME-RR03 — Extend Recommendation Review contract for Setup Detection-aware Analysis Review

Owner roles: Financial Analyst / Functional Analyst / Data Steward / Technical Architect / QA Lead / Governance Auditor

Job family: Recommendation Review

Status: COMPLETED BY ME-RR03

Goal: Define how Recommendation Review consumes Setup Detection-aware Analysis Review.

Scope: Documentation-only contract update.

Recommendation Review remains non-actionable.

Implemented contract:

* `docs/market_engine/recommendation_review/me_rr03_setup_detection_aware_analysis_review_contract.md`

Implemented audit:

* `docs/market_engine/audits/me_rr03_setup_detection_aware_analysis_review_contract_audit.md`

ME-RR03 defined:

* how Setup Detection-aware Analysis Review becomes approved Recommendation Review input;
* how `sec-companyfacts-analysis-review-v1` remains the approved input contract;
* how setup-aware evidence is preserved in Recommendation Review provenance;
* how detected setup states route only to non-actionable human review;
* how partial setup states preserve uncertainty;
* how conflicted setup states preserve conflict;
* how blocked setup states preserve explicit missing-data blocking;
* how not-assessed setup states remain not assessed or insufficient evidence;
* how not-detected setup states must not become negative recommendations;
* how missing setup data remains explicit;
* how numeric zero remains present and must not be treated as missing;
* how Recommendation Review remains downstream of Analysis Review;
* ME-RR04 implementation requirements.

ME-RR03 did not introduce Python code, tests, provider calls, data writes, Recommendation Review runtime changes, Analysis Review runtime changes, Setup Detection runtime changes, Portfolio Review behavior, Telegram, reporting, delivery, Decision Engine behavior, BUY / SELL / HOLD action semantics, allocation, position sizing, execution advice, ranking, scoring, conviction, urgency, or tradeability authority.

Outcome: ME-RR03 extended the Recommendation Review contract on paper so ME-RR04 can later implement consumption of Setup Detection-aware `sec-companyfacts-analysis-review-v1` output. The contract preserves setup-aware evidence and provenance, keeps missing setup data explicit, preserves numeric-zero semantics, routes setup states only to non-actionable human-review or blocked/insufficient-evidence outcomes, and prevents action, portfolio, delivery, ranking, scoring, or Decision Engine authority.

## Recommended Next Sprint

### ME-RR04 — Implement Setup Detection-aware Recommendation Review behavior

Owner roles: Financial Analyst / Functional Analyst / Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: Recommendation Review

Status: COMPLETED BY ME-RR04

Goal: Implement Setup Detection-aware Recommendation Review behavior.

Scope: Non-actionable Recommendation Review only; no action authority, portfolio mutation, delivery behavior, Telegram, reporting, or Decision Engine behavior.

ME-RR04 must implement Setup Detection-aware Recommendation Review only after ME-RR03 defines the contract.

ME-RR04 must:

* consume only validated `sec-companyfacts-analysis-review-v1`;
* preserve existing ME-RR02 behavior when Setup Detection-aware Analysis Review items are absent;
* detect Setup Detection-aware Analysis Review items where present;
* preserve setup-aware provenance;
* preserve setup categories and states;
* preserve setup evidence and limitations;
* preserve missing setup observations;
* preserve source and derived references;
* preserve numeric-zero semantics;
* preserve non-actionable boundary markers;
* route detected setup evidence to human review only;
* route partial setup evidence to human review with explicit uncertainty;
* route conflicted setup evidence to human review with explicit conflict;
* route blocked setup evidence to blocked-by-missing-data;
* route not-assessed setup evidence to insufficient-evidence or blocked routing;
* fail closed for unsupported Analysis Review input contracts;
* add local synthetic tests only;
* avoid live provider calls;
* avoid production data writes;
* avoid legacy `scripts` or old `market_scanner` imports.

ME-RR04 must not introduce portfolio mutation, watchlist mutation, delivery behavior, Telegram behavior, reporting behavior, Decision Engine behavior, BUY / SELL / HOLD action semantics, allocation, position sizing, execution advice, ranking, scoring, conviction, urgency, or tradeability authority.

Implemented runtime:

* `src/market_engine/recommendation_review/sec_companyfacts_recommendation_review.py`

Implemented tests:

* `tests/market_engine/recommendation_review/test_sec_companyfacts_recommendation_review.py`

Implemented documentation:

* `docs/market_engine/recommendation_review/me_rr04_setup_detection_aware_recommendation_review_implementation.md`
* `docs/market_engine/audits/me_rr04_setup_detection_aware_recommendation_review_implementation_audit.md`

Outcome: ME-RR04 implemented Recommendation Review consumption of Setup Detection-aware `sec-companyfacts-analysis-review-v1` output. The implementation preserves existing ME-RR02 behavior when setup-aware fields are absent, preserves setup categories, setup states, setup evidence, setup limitations, missing setup observations, source and derived references, numeric-zero semantics, and routes setup-aware evidence only to non-actionable human-review, blocked-by-missing-data, or insufficient-evidence Recommendation Review outcomes.

## Completed Sprint

### ME-PR01 — Define Portfolio Review contract from Recommendation Review

Owner roles: Financial Analyst / Functional Analyst / Data Steward / Technical Architect / QA Lead / Governance Auditor

Job family: Portfolio Review

Status: COMPLETED BY ME-PR01

Goal: Define the Portfolio Review contract after Setup Detection-aware Recommendation Review exists.

Scope: Documentation-only contract sprint.

Implemented contract:

* `docs/market_engine/portfolio_review/me_pr01_portfolio_review_contract.md`

Implemented audit:

* `docs/market_engine/audits/me_pr01_portfolio_review_contract_audit.md`

Implemented backlog update:

* `docs/market_engine/backlog/me_pr01_backlog_update.md`

ME-PR01 defined:

* Portfolio Review job-family boundary;
* approved Recommendation Review input requirements;
* required explicit portfolio-context input family;
* position, exposure, concentration, and fit review semantics;
* allowed portfolio-review states;
* allowed portfolio-review categories;
* missing-data and stale-data rules;
* numeric-zero preservation rules;
* provenance requirements;
* authority boundary between Portfolio Review and Decision Engine;
* ME-PR02 implementation requirements.

Approved Recommendation Review input contract:

* `sec-companyfacts-recommendation-review-v1`

Approved portfolio-context input family:

* `market-engine-portfolio-context-v1`

Approved Portfolio Review output contract:

* `sec-companyfacts-portfolio-review-v1`

Outcome: ME-PR01 defined Portfolio Review as a non-actionable, explicit-portfolio-context-dependent review layer downstream of Recommendation Review and upstream of Decision Engine handoff. The contract preserves Recommendation Review provenance, Setup Detection-aware provenance when present, portfolio-context evidence, missing portfolio-context data, stale portfolio-context data, and numeric-zero semantics.

ME-PR01 did not introduce Python code, tests, runtime behavior, provider calls, broker calls, data writes, generated artifacts, portfolio mutation, watchlist mutation, Telegram, reporting, delivery, Decision Engine behavior, BUY / SELL / HOLD action semantics, allocation execution, target weights, order generation, position sizing instructions, ranking, scoring, conviction, urgency, or tradeability authority.

## Recommended Next Sprint

### ME-PR02 — Implement Portfolio Review

Owner roles: Financial Analyst / Functional Analyst / Data Steward / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: Portfolio Review

Status: COMPLETED BY ME-PR02

Goal: Implement Portfolio Review after the ME-PR01 contract definition.

Scope: Non-actionable Portfolio Review only. No portfolio mutation, no broker calls, no Decision Engine behavior, no delivery behavior, no Telegram, no reporting, no BUY / SELL / HOLD action semantics, no allocation execution, no target weights, no order generation, no ranking, no scoring, no conviction, no urgency, and no tradeability authority.

ME-PR02 must implement Portfolio Review only after ME-PR01 defines the contract.

ME-PR02 must:

* consume approved `sec-companyfacts-recommendation-review-v1` input;
* consume explicitly supplied `market-engine-portfolio-context-v1` input;
* emit `sec-companyfacts-portfolio-review-v1`;
* preserve Recommendation Review provenance;
* preserve Setup Detection-aware provenance when present;
* preserve portfolio-context provenance;
* preserve missing portfolio-context data explicitly;
* preserve stale portfolio-context data explicitly;
* preserve numeric-zero semantics;
* produce non-actionable position, exposure, concentration, and portfolio-fit review output;
* fail closed for unsupported input contracts;
* add local synthetic tests only;
* avoid live provider calls;
* avoid broker calls;
* avoid production data writes;
* avoid legacy `scripts` or old `market_scanner` imports.

ME-PR02 must preserve Decision Engine authority and must not execute allocations, orders, rebalances, alerts, reports, delivery actions, portfolio mutations, or watchlist mutations.

Implemented runtime:

* `src/market_engine/portfolio_review/sec_companyfacts_portfolio_review.py`
* `src/market_engine/portfolio_review/__init__.py`

Implemented tests:

* `tests/market_engine/portfolio_review/test_sec_companyfacts_portfolio_review.py`

Implemented documentation:

* `docs/market_engine/portfolio_review/me_pr02_portfolio_review_implementation.md`
* `docs/market_engine/audits/me_pr02_portfolio_review_implementation_audit.md`

Outcome: ME-PR02 implemented non-actionable Portfolio Review from validated `sec-companyfacts-recommendation-review-v1` input and explicitly supplied `market-engine-portfolio-context-v1` input. The implementation emits `sec-companyfacts-portfolio-review-v1`, preserves Recommendation Review provenance and Setup Detection-aware provenance when present, preserves portfolio-context provenance, missing and stale portfolio-context markers, numeric-zero semantics, and produces review-only position, exposure, concentration, portfolio-fit, data-limitation, and downstream-handoff-readiness items.

Possible future Portfolio Review follow-up candidate: `ME-PR03 — Define approved portfolio context source and persistence contract`. This candidate is not inserted ahead of ME-DE01 because ME-PR02 did not uncover a blocker; it should be added formally only if a later Decision Engine handoff or portfolio-context sprint requires persisted portfolio-context sourcing beyond caller-supplied context.

## Completed Sprint

### ME-DE01 — Define Decision Engine handoff contract

Owner roles: Product Owner / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: Decision Engine handoff

Status: COMPLETED BY ME-DE01

Goal: Define the boundary between Market Engine review output and actual decision/action authority.

Scope: Documentation-only contract sprint unless explicitly re-scoped.

ME-DE01 defined:

* approved upstream input from Portfolio Review;
* handoff payload requirements;
* what the Market Engine may request from the Decision Engine;
* what only the Decision Engine may decide;
* action/allocation authority boundaries;
* fail-closed rules;
* audit and traceability requirements;
* ME-DE02 implementation requirements.

Implemented contract:

* `docs/market_engine/decision_engine/me_de01_decision_engine_handoff_contract.md`

Implemented audit:

* `docs/market_engine/audits/me_de01_decision_engine_handoff_contract_audit.md`

Outcome: ME-DE01 defined `market-engine-decision-engine-handoff-v1` as the future handoff payload downstream of `sec-companyfacts-portfolio-review-v1`. The contract defines Portfolio Review eligibility, blocked handoff states, fail-closed rules, numeric-zero preservation, provenance requirements, prohibited payload fields, and ME-DE02 implementation requirements while preserving Decision Engine as the only future action and allocation authority.

ME-DE01 did not introduce Python code, tests, provider calls, data writes, Telegram, reporting, delivery behavior, portfolio mutation, BUY / SELL / HOLD execution semantics, allocation execution, order generation, or live Decision Engine behavior.

## Completed Sprint

### ME-DE02 — Implement controlled Decision Engine handoff

Owner roles: Product Owner / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: Decision Engine handoff

Status: COMPLETED BY ME-DE02

Goal: Implement controlled handoff according to the ME-DE01 contract.

Scope: Must preserve Decision Engine as the only action/allocation authority.

ME-DE02 must implement handoff behavior only after ME-DE01 defines the contract.

ME-DE02 must not bypass Portfolio Review, Recommendation Review, Analysis Review, Setup Detection, or authority boundaries.

ME-DE02 must:

* consume approved `sec-companyfacts-portfolio-review-v1` input only;
* emit `market-engine-decision-engine-handoff-v1`;
* validate Portfolio Review contract and version;
* validate ticker identity;
* validate portfolio-context version and state;
* validate Portfolio Review handoff-readiness evidence;
* preserve Recommendation Review and Setup Detection-aware provenance when present;
* preserve missing-data markers;
* preserve stale-data markers;
* preserve numeric-zero semantics;
* produce only a handoff-readiness payload;
* avoid Decision Engine decisions, actions, allocation, ranking, scoring, execution, delivery, Telegram, and reporting behavior.

Implemented runtime:

* `src/market_engine/decision_engine_handoff/sec_companyfacts_handoff.py`
* `src/market_engine/decision_engine_handoff/__init__.py`

Implemented tests:

* `tests/market_engine/decision_engine_handoff/test_sec_companyfacts_handoff.py`

Implemented documentation:

* `docs/market_engine/decision_engine/me_de02_decision_engine_handoff_implementation.md`
* `docs/market_engine/audits/me_de02_decision_engine_handoff_implementation_audit.md`

Outcome: ME-DE02 implemented deterministic `market-engine-decision-engine-handoff-v1` construction from approved `sec-companyfacts-portfolio-review-v1` input. Eligible Portfolio Review output produces `ready_for_decision_engine_review`; ineligible input produces explicit blocked handoff states with deterministic blocked reasons. The implementation preserves Portfolio Review, portfolio-context, Recommendation Review, Analysis Review, Setup Detection-aware, missing-data, stale-data, and numeric-zero evidence without introducing Decision Engine action or allocation authority.

ME-DE02 did not introduce provider calls, broker calls, live data access, portfolio writes, watchlist writes, Telegram, reporting, delivery behavior, Decision Engine runtime decisions, trade instructions, allocation advice, target weights, order generation, position sizing, urgency, conviction, tradeability, ranking, scoring, or execution advice.

## Completed Sprint

### ME-DL01 — Define Delivery / Reporting contract

Owner roles: Product Owner / Operator / User / Technical Architect / QA Lead / Governance Auditor

Job family: Delivery / Reporting

Status: COMPLETED BY ME-DL01

Goal: Define how approved outputs may be delivered or reported.

Scope: Documentation-only contract sprint unless explicitly re-scoped.

ME-DL01 defined:

* approved upstream input requirements;
* delivery eligibility;
* reporting eligibility;
* Telegram/reporting boundaries;
* user-facing output contract;
* audit and traceability requirements;
* fail-closed delivery rules;
* ME-DL02 implementation requirements.

Implemented contract:

* `docs/market_engine/delivery_reporting/me_dl01_delivery_reporting_contract.md`

Implemented audit:

* `docs/market_engine/audits/me_dl01_delivery_reporting_contract_audit.md`

Outcome: ME-DL01 defined `market-engine-delivery-report-v1` as the future Delivery / Reporting payload downstream of `market-engine-decision-engine-handoff-v1`. The contract defines approved input, delivery states, allowed reporting categories, forbidden reporting behavior, presentation rules, blocked/upstream handling, missing-data handling, stale-data handling, numeric-zero safety, provenance preservation, and ME-DL02 implementation requirements.

ME-DL01 did not introduce Python code, tests, provider calls, data writes, portfolio mutation, Decision Engine behavior, BUY / SELL / HOLD action semantics, allocation, position sizing, execution advice, ranking, scoring, conviction, urgency, tradeability authority, Telegram delivery, email delivery, broker integration, scheduler behavior, report generation, or user-facing alerts.

## Completed Sprint

### ME-DL02 — Implement controlled Delivery / Reporting output

Owner roles: Product Owner / Operator / User / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: Delivery / Reporting

Status: COMPLETED BY ME-DL02

Goal: Implement the Delivery / Reporting contract defined by ME-DL01.

Scope: Must not bypass Recommendation Review, Portfolio Review, or Decision Engine handoff authority boundaries.

ME-DL02 must implement delivery/reporting only after ME-DL01 defines the contract.

ME-DL02 must:

* consume only approved `market-engine-decision-engine-handoff-v1` payloads;
* emit `market-engine-delivery-report-v1`;
* preserve blocked upstream states as blocked;
* preserve missing-data markers;
* preserve stale-data markers;
* preserve numeric-zero semantics;
* preserve upstream provenance;
* emit only allowed non-actionable reporting categories;
* use local synthetic tests only;
* avoid provider calls;
* avoid Telegram, email, broker, portfolio, watchlist, scheduler, and production report writes;
* avoid ranking, conviction, urgency, target-price, BUY / SELL / HOLD, allocation, or execution semantics.

Implemented runtime:

* `src/market_engine/delivery_reporting/sec_companyfacts_delivery_report.py`
* `src/market_engine/delivery_reporting/__init__.py`

Implemented tests:

* `tests/market_engine/delivery_reporting/test_sec_companyfacts_delivery_report.py`

Implemented documentation:

* `docs/market_engine/delivery_reporting/me_dl02_delivery_reporting_implementation.md`
* `docs/market_engine/audits/me_dl02_delivery_reporting_implementation_audit.md`

Outcome: ME-DL02 implemented deterministic `market-engine-delivery-report-v1` construction from approved `market-engine-decision-engine-handoff-v1` input. The implementation preserves blocked upstream states, missing-data markers, stale-data markers, numeric-zero evidence, and upstream provenance while emitting only non-actionable reporting payload sections.

ME-DL02 did not introduce provider calls, live market data calls, Telegram delivery, email delivery, broker integration, portfolio writes, watchlist writes, scheduler behavior, UI behavior, Decision Engine behavior, Recommendation Review behavior, Portfolio Review behavior, new financial analysis logic, trade instructions, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability, or execution advice.

## Completed Sprint

### ME-RUN05 — Implement local dry-run artifact persistence

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN05

Goal: Implement local dry-run artifact persistence for generated Market Engine runtime artifacts.

Scope: Optional local non-production JSON persistence for already-built `market-engine-end-to-end-dry-run-v1` payloads only.

ME-RUN05 implemented:

* local dry-run artifact format: `market-engine-local-dry-run-artifact-v1`;
* local dry-run artifact manifest format: `market-engine-local-dry-run-artifact-manifest-v1`;
* approved path category: `artifacts/market_engine/dry_runs/`;
* explicit `--write-local-artifact` command behavior;
* deterministic artifact metadata through caller-supplied artifact timestamp;
* safe path validation;
* overwrite refusal by default;
* stable, human-readable JSON serialization;
* local synthetic tests only.

Implemented runtime:

* `src/market_engine/run/local_dry_run_artifacts.py`
* `src/market_engine/run/end_to_end_dry_run_command.py`
* `src/market_engine/run/__init__.py`

Implemented tests:

* `tests/market_engine/run/test_local_dry_run_artifacts.py`

Implemented documentation:

* `docs/market_engine/run/me_run05_local_dry_run_artifact_persistence_implementation.md`
* `docs/market_engine/audits/me_run05_local_dry_run_artifact_persistence_audit.md`

Outcome: ME-RUN05 lets local dry-run executions persist deterministic, inspectable, non-production JSON artifacts while preserving the stdout-only default command behavior. The writer preserves dry-run contract identity, missing-data markers, stale-data markers, blocked states, blocked reasons, numeric-zero values, provenance, delivery report references, forbidden-side-effect confirmation, and authority-boundary confirmation.

ME-RUN05 did not introduce provider calls, live market data calls, Telegram delivery, email delivery, broker integration, production report generation, portfolio writes, watchlist writes, scheduler behavior, UI behavior, Decision Engine behavior, Recommendation Review behavior, Portfolio Review behavior, new financial analysis logic, trade instructions, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability, or execution advice.

## Completed Sprint

### ME-RUN10 — Implement cached-source end-to-end local execution path

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN10

Goal: Implement the cached-source end-to-end local execution path defined by ME-RUN09.

Scope: Local cached-source input loading, command input mode, fail-closed validation, downstream contract construction through approved builders, tests, documentation, and audit only.

ME-RUN10 implemented:

* `cached_source_snapshot` dry-run input mode;
* `market-engine-cached-source-local-execution-input-v1` wrapper support;
* cached SEC CompanyFacts source snapshot path containment validation;
* cached source snapshot to Source Context construction;
* downstream contract construction through the implemented Market Engine chain;
* explicit local portfolio-context input support;
* optional local dry-run artifact writing through the existing `--write-local-artifact` flag;
* local synthetic tests only.

Implemented runtime:

* `src/market_engine/run/cached_source_execution.py`
* `src/market_engine/run/end_to_end_dry_run.py`
* `src/market_engine/run/end_to_end_dry_run_command.py`
* `src/market_engine/run/local_dry_run_artifacts.py`
* `src/market_engine/run/__init__.py`

Implemented tests:

* `tests/market_engine/run/test_me_run10_cached_source_local_execution.py`

Implemented documentation:

* `docs/market_engine/run/me_run10_cached_source_local_execution_implementation.md`
* `docs/market_engine/audits/me_run10_cached_source_local_execution_implementation_audit.md`
* `docs/market_engine/backlog/me_run10_cached_source_local_execution_backlog_entry.md`
* `docs/market_engine/roadmap/me_run10_cached_source_local_execution_roadmap_entry.md`

Outcome: ME-RUN10 proves Market Engine can run the local dry-run chain from an already-existing cached SEC CompanyFacts source snapshot and explicitly supplied local portfolio context without live provider calls or production side effects. The final output remains `market-engine-end-to-end-dry-run-v1`, and artifact persistence remains opt-in through the existing local artifact path.

ME-RUN10 did not introduce provider calls, SEC/EDGAR live calls, yfinance calls, live market data calls, broker calls, Telegram/email delivery, production report generation, portfolio writes, watchlist writes, scheduler behavior, UI behavior, all-ticker production runs, automatic cache refresh, automatic cache cleanup, Decision Engine decisions, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, execution advice, ranking, scoring, urgency, conviction, tradeability, or execution advice.

## Completed Sprint

### ME-RUN11 — Run cached-source local execution against a broader deterministic ticker bundle

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN11

Goal: Validate cached-source local execution against a small deterministic ticker bundle.

Scope: Local synthetic cached-source fixtures, ticker-by-ticker command coverage, opt-in artifact validation, fail-closed malformed snapshot validation, tests, documentation, and audit only.

ME-RUN11 implemented:

* deterministic bundle coverage for `NVDA`, `MSFT`, and `AMD`;
* per-ticker validation of `market-engine-end-to-end-dry-run-v1`;
* per-ticker validation of `cached_source_snapshot`;
* cached-source provenance checks;
* source refresh snapshot ID provenance checks;
* numeric-zero source and portfolio-context evidence checks;
* artifact writing default-off validation across bundle runs;
* opt-in artifact writing validation for one selected ticker;
* malformed cached-source fail-closed validation.

Implemented tests:

* `tests/market_engine/run/test_me_run11_cached_source_ticker_bundle_execution.py`

Implemented documentation:

* `docs/market_engine/run/me_run11_cached_source_ticker_bundle_execution.md`
* `docs/market_engine/audits/me_run11_cached_source_ticker_bundle_execution_audit.md`
* `docs/market_engine/backlog/me_run11_cached_source_ticker_bundle_execution_backlog_entry.md`
* `docs/market_engine/roadmap/me_run11_cached_source_ticker_bundle_execution_roadmap_entry.md`

Outcome: ME-RUN11 proves the ME-RUN10 cached-source local execution path can run against a small deterministic ticker bundle by invoking the approved command path ticker-by-ticker. The sprint does not add a broad batch runner or production execution contract. The final per-ticker output remains `market-engine-end-to-end-dry-run-v1`, and artifact persistence remains opt-in through `--write-local-artifact`.

ME-RUN11 did not introduce provider refresh, SEC/EDGAR live calls, yfinance calls, live market data calls, broker calls, Telegram/email delivery, production report generation, portfolio writes, watchlist writes, scheduler behavior, UI behavior, all-ticker production execution, automatic cache refresh, automatic cache cleanup, Decision Engine decisions, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, execution advice, ranking, scoring, urgency, conviction, tradeability, or execution advice.

## Future Sprint Candidates

Recommended next sprint after ME-RUN11:

```text
ME-RUN12 - Define safe all-ticker cached-source batch dry-run contract
```

Rationale: ME-RUN11 validates a small deterministic per-ticker bundle. Any broader cached-source batch behavior should be contract-defined before implementation so that production boundaries, cached-source discovery, artifact semantics, failure isolation, and operator visibility remain explicit.

## Completed Sprint

### ME-RUN13 - Implement safe all-ticker cached-source batch dry-run behavior

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN13

Goal: Implement the safe local cached-source batch dry-run path defined by ME-RUN12.

Scope: Local cached-source batch dry-run implementation, deterministic local tests, documentation, and audit only.

ME-RUN13 implemented:

* batch contract: `market-engine-cached-source-batch-dry-run-v1`;
* per-ticker output preservation as `market-engine-end-to-end-dry-run-v1`;
* deterministic cached-source discovery under an explicit local root;
* explicit requested ticker support;
* deterministic cached ticker discovery mode;
* missing cached-source ticker blocking;
* invalid cached-source ticker blocking;
* unsupported cached-source ticker blocking;
* ambiguous cached-source ticker blocking;
* downstream contract failure isolation;
* unexpected local error isolation;
* batch counts and per-ticker result summaries;
* numeric-zero evidence preservation;
* opt-in local batch artifact writing;
* overwrite protection for batch artifacts.

Implemented runtime:

* `src/market_engine/run/cached_source_batch_execution.py`
* `src/market_engine/run/__init__.py`

Implemented tests:

* `tests/market_engine/run/test_me_run13_cached_source_batch_dry_run.py`

Implemented documentation:

* `docs/market_engine/run/me_run13_safe_all_ticker_cached_source_batch_dry_run_implementation.md`
* `docs/market_engine/audits/me_run13_safe_all_ticker_cached_source_batch_dry_run_implementation_audit.md`

Outcome: ME-RUN13 implements the ME-RUN12 contract as a local cached-source batch wrapper over approved per-ticker dry-runs. It preserves per-ticker failure isolation, deterministic counts, local-only provenance, numeric-zero evidence, opt-in artifact behavior, and non-actionable boundaries.

ME-RUN13 did not introduce live provider calls, SEC/EDGAR fetches, yfinance calls, live market data calls, broker calls, Telegram/email delivery, production report generation, portfolio writes, watchlist writes, scheduler behavior, UI behavior, automatic cache refresh, automatic cache cleanup, Decision Engine decisions, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, execution advice, ranking, scoring, urgency, conviction, tradeability, or execution advice.

## Next Implementation Candidate

### ME-RUN14 - Add cached-source batch dry-run command interface

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: CANDIDATE AFTER ME-RUN13

Goal: Add a narrow operator-facing command interface for the ME-RUN13 cached-source batch dry-run runtime behavior.

Rationale: ME-RUN13 implements the safe batch behavior as a runtime function and artifact writer. A separate sprint should add any operator-facing command interface so command arguments, terminal output, artifact flags, and failure messages remain explicit and reviewable.

Scope: Command interface, local argument parsing, terminal JSON output, opt-in artifact wiring, deterministic local tests, documentation, and audit only unless explicitly re-scoped.

Non-goals: no provider refresh, live market data, external API calls, broker calls, Telegram/email delivery, production reports, portfolio writes, watchlist writes, scheduler behavior, UI behavior, automatic cache refresh, automatic cache cleanup, new financial logic, Decision Engine decisions, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability, or execution advice.

## Completed Sprint

### ME-UNI02 - Implement canonical ticker universe loading and validation

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI02

Goal: implement the canonical ticker universe loader and validation layer defined by ME-UNI01.

Scope: loader, validator, typed result models, deterministic selection, synthetic tests, documentation, backlog, roadmap and audit only.

ME-UNI02 implemented:

* canonical contract version: `market-engine-canonical-ticker-universe-v1`;
* canonical default path: `data/market_engine/ticker_universe/ticker_universe.csv`;
* explicit path override support;
* required-column validation;
* required field validation;
* allowed-value validation;
* ticker trim and uppercase normalization only;
* duplicate normalized ticker and market rejection;
* active cached-source default selection;
* explicit inactive row inclusion when requested;
* optional metadata preservation;
* deterministic ordering;
* operator-readable validation errors.

Implemented runtime:

```text
src/market_engine/ticker_universe/
```

Implemented tests:

```text
tests/market_engine/ticker_universe/test_canonical_ticker_universe.py
```

Implemented documentation:

```text
docs/market_engine/ticker_universe/me_uni02_canonical_ticker_universe_loader_implementation.md
docs/market_engine/audits/me_uni02_canonical_ticker_universe_loader_audit.md
```

Outcome: the canonical ticker universe can be loaded and validated deterministically before downstream RUN consumption.

ME-UNI02 did not introduce provider calls, live network calls, source refresh jobs, batch execution, Telegram behavior, email delivery, broker integration, portfolio writes, watchlist writes, scheduler behavior, UI behavior, Decision Engine behavior, BUY / SELL / HOLD semantics, allocation advice, target prices, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Next Canonical-Universe RUN Candidate

### ME-RUN16 - Execute first real cached-source batch dry-run using canonical ticker universe

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: CANDIDATE AFTER ME-UNI02

Goal: consume the ME-UNI02 canonical ticker universe loader in the cached-source batch dry-run path and execute the first real cached-source batch dry-run using the canonical universe.

Scope: cached-source/local-only RUN integration, canonical universe visibility, fail-closed invalid-universe behavior, local tests, documentation and audit only unless explicitly re-scoped.

ME-RUN16 must remain blocked from provider refresh, live data calls, Telegram delivery, portfolio writes, watchlist writes, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability and execution advice.

## Completed Sprint

### ME-RUN16 - Execute first real cached-source batch dry-run using canonical ticker universe

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED WITH BLOCKED TICKER OUTCOME BY ME-RUN16

Goal: execute the first cached-source batch dry-run selected from the canonical ticker universe.

Outcome:

* canonical universe loaded from `data/market_engine/ticker_universe/ticker_universe.csv`;
* 14 canonical rows loaded;
* 13 active `cached_source_only` tickers selected;
* SMCI excluded because `source_policy=manual_review_only`;
* all 13 selected tickers returned `blocked_missing_cached_source`;
* no provider or live data fallback occurred;
* generated local batch manifest under `artifacts/market_engine/...`, not committed.

Implemented runtime change:

```text
src/market_engine/run/cached_source_batch_dry_run_command.py
```

Implemented tests:

```text
tests/market_engine/run/test_cached_source_batch_dry_run_command.py
```

Implemented documentation:

```text
docs/market_engine/run/me_run16_first_canonical_universe_cached_source_batch_dry_run_execution.md
docs/market_engine/audits/me_run16_first_canonical_universe_cached_source_batch_dry_run_audit.md
docs/market_engine/backlog/me_run16_first_canonical_universe_cached_source_batch_dry_run_backlog_entry.md
docs/market_engine/roadmap/me_run16_first_canonical_universe_cached_source_batch_dry_run_roadmap_entry.md
```

ME-RUN16 did not introduce provider calls, live data calls, Telegram delivery, portfolio writes, watchlist writes, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Next Source Refresh Candidate

### ME-SR02 - Produce bounded canonical-universe SEC CompanyFacts cached source snapshots

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh

Status: CANDIDATE AFTER ME-RUN16

Goal: produce or validate bounded local SEC CompanyFacts cached source snapshots for the canonical universe so a later RUN sprint can execute downstream dry-runs from real cached source evidence.

Rationale: ME-RUN16 proved canonical-universe RUN selection and fail-closed behavior. It also showed that this checkout has no cached source snapshots under `data/market_engine/source_snapshots`, so every selected ticker blocks before downstream dry-run execution.

Scope: Source Refresh only. No portfolio writes, watchlist writes, Telegram delivery, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Completed Sprint

### ME-SR03 - Resolve canonical-universe cached-source coverage blockers

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh / Source Coverage

Status: COMPLETED BY ME-SR03

Goal: resolve or precisely document the canonical-universe cached-source coverage blockers for HO, ASML, and TSM.

Outcome:

* ASML resolved by preserving annual `20-F` `us-gaap` facts in `EUR`;
* TSM resolved by preserving annual `20-F` `ifrs-full` facts in `USD`;
* HO remains blocked because no approved cached SEC CompanyFacts snapshot exists locally;
* canonical rerun improved to 12 completed tickers and 1 blocked ticker.

Implemented runtime change:

```text
src/market_engine/source_intake/sec_companyfacts_fields.py
```

Implemented tests:

```text
tests/market_engine/source_intake/test_sec_companyfacts_field_mapping.py
```

Implemented documentation:

```text
docs/market_engine/source_data/me_sr03_canonical_universe_cached_source_coverage_blockers.md
docs/market_engine/audits/me_sr03_canonical_universe_cached_source_coverage_blockers_audit.md
docs/market_engine/backlog/me_sr03_canonical_universe_cached_source_coverage_blockers_backlog_entry.md
docs/market_engine/roadmap/me_sr03_canonical_universe_cached_source_coverage_blockers_roadmap_entry.md
```

ME-SR03 did not introduce provider calls, live data calls, Telegram delivery, portfolio writes, watchlist writes, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Completed Sprint

### ME-SR04 - Resolve HO canonical-universe source identity or exclusion decision

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh / Source Coverage

Status: COMPLETED BY ME-SR04

Goal: decide whether HO should receive an approved source identity/backfill path or be moved out of default cached-source execution until a valid source exists.

Rationale: ME-SR03 resolved ASML and TSM using existing cached source data. HO remains blocked because ME-SR02 recorded it as unsupported and no local cached SEC CompanyFacts snapshot exists.

Outcome:

* HO remains in the canonical universe as Thales on Euronext;
* HO source policy changed from `cached_source_only` to `manual_review_only`;
* HO is excluded from default canonical SEC CompanyFacts cached-source execution;
* HO is not eligible for future Telegram preview or delivery until a separate approved source identity decision changes that status;
* canonical cached-source rerun selected 12 supported tickers and completed 12 with zero blocked tickers.

Implemented configuration/test changes:

```text
data/market_engine/ticker_universe/ticker_universe.csv
tests/market_engine/ticker_universe/test_canonical_ticker_universe.py
```

Implemented documentation:

```text
docs/market_engine/source_data/me_sr04_ho_canonical_universe_source_identity_decision.md
docs/market_engine/audits/me_sr04_ho_canonical_universe_source_identity_decision_audit.md
docs/market_engine/backlog/me_sr04_ho_canonical_universe_source_identity_decision_backlog_entry.md
docs/market_engine/roadmap/me_sr04_ho_canonical_universe_source_identity_decision_roadmap_entry.md
```

Scope: Source Refresh / source identity only. No portfolio writes, watchlist writes, Telegram delivery, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Completed Universe Governance Sprint

### ME-UNI08 - Add first-class Professional Swing Universe CLI flag

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI08

Goal: expose the approved editable Professional Swing Universe as a first-class local cached-source batch runtime CLI choice.

Outcome:

* added `--professional-swing-universe` to the cached-source batch dry-run command;
* routed the flag through the existing ME-UNI07 runtime-input builder;
* preserved the ME-UNI06 Professional Swing Universe loader and validation path;
* preserved custom `--canonical-ticker-universe <path>` behavior;
* made the Professional Swing flag and custom universe path mutually exclusive;
* added focused command tests and sprint documentation.

Implemented runtime/test changes:

```text
src/market_engine/run/cached_source_batch_dry_run_command.py
tests/market_engine/run/test_cached_source_batch_dry_run_command.py
```

Implemented documentation:

```text
docs/market_engine/ticker_universe/me_uni08_professional_swing_universe_cli_flag.md
docs/market_engine/audits/me_uni08_professional_swing_universe_cli_flag_audit.md
docs/market_engine/backlog/me_uni08_professional_swing_universe_cli_flag_backlog_entry.md
docs/market_engine/roadmap/me_uni08_professional_swing_universe_cli_flag_roadmap_entry.md
```

Scope: ME-UNI08 did not introduce provider calls, source refresh, output/reporting behavior, delivery behavior, scheduler behavior, portfolio/watchlist writes, Decision Engine action semantics, or trading authority.

## Next Source Support Candidate

### ME-SR05 - Classify source support for Professional Swing Universe

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh / Source Coverage

Status: COMPLETED BY ME-SR05

Goal: classify actual cached-source support for Professional Swing Universe rows before broad supported-universe cached-source scanning.

Rationale: ME-UNI08 makes the editable Professional Swing Universe easy to select at runtime, but operator source-policy hints are not authoritative source-support truth.

Outcome:

* implemented deterministic Professional Swing Universe source-support classification;
* consumed the validated editable Professional Swing Universe loader;
* classified local SEC CompanyFacts source support from approved cached snapshot artifacts and provider error records;
* emitted explicit statuses for `supported_cached`, `missing_snapshot`, `unsupported_sec_companyfacts`, `missing_required_source_field`, `malformed_or_unreadable_source_artifact`, `ambiguous_identity`, `manual_review_only`, and `excluded`;
* preserved source artifact references, provider error references, required source field status, missing-field evidence, universe row references, and numeric-zero evidence;
* preserved the source-support-only boundary with no provider calls, source refresh, reporting, portfolio/watchlist mutation, Decision Engine behavior, action semantics, allocation, ranking, scoring, urgency, conviction, tradeability, position sizing, order, or execution behavior.

Planned sequence after ME-SR05:

```text
ME-RUN20 - Execute clean supported-universe cached-source scan
ME-OUT01 - Define readable operator report from dry-run artifacts
ME-CANDIDATE01 - Define non-actionable candidate classification contract
```

### ME-RUN20 - Execute clean supported-universe cached-source scan

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN20

Goal: execute a local cached-source scan against the currently supported active subset of the editable Professional Swing Universe and produce inspectable local artifacts.

Scope: ME-RUN20 should consume ME-SR05 source-support classification results. Unsupported, missing, malformed, ambiguous, manual-review-only, and excluded rows must remain visible but must not be silently treated as clean supported cached-source rows.

Outcome:

* executed the 12 ME-SR05-supported cached-source tickers through the existing local cached-source batch dry-run path;
* requested 12, discovered 12 cached snapshots, executed 12, completed 12;
* observed 0 blocked, 0 failed, 0 skipped, 0 missing, 0 ambiguous, 0 unsupported, and 0 stale source results inside the supported subset;
* wrote local non-production artifacts under `artifacts/market_engine/me-run20-supported-universe-20260623T120000Z/`;
* did not commit generated artifacts by default;
* preserved all cached-source/local-only and non-actionable boundaries.

### ME-OUT01 - Define readable operator report from dry-run artifacts

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-OUT - Output / Operator Reporting

Status: RECOMMENDED NEXT AFTER ME-RUN20

Goal: define a readable, non-actionable operator report contract from generated dry-run artifacts without introducing delivery, trading authority, ranking, scoring, allocation, or execution behavior.

## Completed Sprint

### ME-RUN17 - Canonical-universe cached-source batch dry-run with ME-SR02 snapshots

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED WITH DOWNSTREAM BLOCKED OUTCOME BY ME-RUN17

Goal: execute and fix canonical-universe cached-source batch dry-run behavior using ME-SR02 snapshots.

Outcome:

* fixed RUN discovery for `sec_companyfacts/<snapshot_id>/raw/*.json`;
* preserved older `*/raw/*.json` discovery;
* selected 13 canonical active `cached_source_only` tickers;
* excluded SMCI as `manual_review_only`;
* discovered 12 ME-SR02 raw snapshots;
* executed 12 local end-to-end dry-run payloads;
* kept HO blocked as missing cached source;
* generated 12 local per-ticker artifacts plus a batch manifest;
* preserved provider-free, local-only, non-actionable boundaries.

Implemented runtime change:

```text
src/market_engine/run/cached_source_batch_execution.py
```

Implemented test change:

```text
tests/market_engine/run/test_cached_source_batch_dry_run_command.py
```

Implemented documentation:

```text
docs/market_engine/run/me_run17_canonical_universe_cached_source_batch_dry_run_with_me_sr02_snapshots.md
docs/market_engine/audits/me_run17_canonical_universe_cached_source_batch_dry_run_audit.md
docs/market_engine/backlog/me_run17_canonical_universe_cached_source_batch_dry_run_backlog_entry.md
docs/market_engine/roadmap/me_run17_canonical_universe_cached_source_batch_dry_run_roadmap_entry.md
```

ME-RUN17 did not introduce provider calls, live data calls, Telegram delivery, portfolio writes, watchlist writes, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Next RUN Candidate

### ME-RUN18 - Provide portfolio context for canonical-universe cached-source dry-runs

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: CANDIDATE AFTER ME-RUN17

Goal: provide approved local portfolio context to canonical-universe cached-source dry-runs so downstream review stages can progress without production portfolio writes.

Rationale: ME-RUN17 now discovers ME-SR02 snapshots and executes 12 dry-run payloads, but the chain remains blocked downstream because required local portfolio context is not provided.

Scope: local cached-source RUN behavior only. No provider refresh, live market data, portfolio writes, watchlist writes, Telegram/email delivery, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Completed Sprint

### ME-RUN19 - Portfolio-context-aware canonical cached-source dry-run

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN19

Goal: execute the canonical-universe cached-source batch dry-run with ME-SR02 snapshots and approved local non-production portfolio context.

Outcome:

* created the local non-production portfolio context file at `data/market_engine/portfolio_contexts/local_portfolio_context.json`;
* ran the existing ME-RUN18 `--portfolio-context` command path without runtime code changes;
* selected 13 active `cached_source_only` tickers from the canonical universe;
* excluded SMCI because `source_policy=manual_review_only`;
* discovered 12 ME-SR02 cached source snapshots;
* executed 12 per-ticker dry-runs;
* completed 10 tickers through Portfolio Review, Decision Engine handoff, Delivery / Reporting, and dry-run summary;
* preserved 3 blocked tickers: ASML, HO, and TSM;
* kept generated artifacts under `artifacts/market_engine/me-run19-20260622T103000Z/` uncommitted.

Implemented documentation:

```text
docs/market_engine/run/me_run19_portfolio_context_aware_canonical_cached_source_dry_run_execution.md
docs/market_engine/audits/me_run19_portfolio_context_aware_canonical_cached_source_dry_run_audit.md
docs/market_engine/backlog/me_run19_portfolio_context_aware_canonical_cached_source_dry_run_backlog_entry.md
docs/market_engine/roadmap/me_run19_portfolio_context_aware_canonical_cached_source_dry_run_roadmap_entry.md
```

ME-RUN19 did not introduce provider calls, live data calls, Telegram delivery, portfolio writes, watchlist writes, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Next Source Refresh Candidate

### ME-SR03 - Resolve canonical-universe cached-source coverage blockers

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-SR - Source Refresh

Status: CANDIDATE AFTER ME-RUN19

Goal: resolve the remaining cached-source coverage blockers exposed by ME-RUN19 before broader canonical-universe validation or Telegram preview work.

Rationale: ME-RUN19 proved that local portfolio context unlocks Portfolio Review, Decision Engine handoff, and Delivery / Reporting for complete cached-source tickers. Remaining blockers are source-coverage issues: HO lacks a cached source snapshot, while ASML and TSM preserve upstream missing-field evidence and block at Recommendation Review.

Scope: Source Refresh only. No portfolio writes, watchlist writes, Telegram delivery, production reports, scheduler behavior, UI behavior, Decision Engine action semantics, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability or execution advice.
