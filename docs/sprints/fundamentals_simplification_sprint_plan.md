# Fundamentals Simplification Sprint Plan

Status: ACTIVE SPRINT PLAN

## Purpose

This plan splits the fundamentals simplification program into separate future sprints.

The program exists to reduce complexity around fundamentals data, active documentation, analysis ownership, and future implementation sequencing.

This current sprint is documentation-only. It does not authorize code changes, data changes, raw fundamentals edits, provider/API calls, scraping, generated artifact updates, pipeline runs, tests, file moves, file deletion, archive moves, Decision Engine changes, Reporting changes, Telegram changes, or implementation.

## Current sprint confirmation

This task is documentation-only.

No code changes are authorized.

No data changes are authorized.

This sprint is meant to reduce complexity by consolidating active fundamentals doctrine into a smaller set of authoritative documents.

## Sprint A - Active Doctrine Consolidation

Status: current sprint.

Purpose:

- create the active fundamentals platform contract;
- create the dedicated technical calculation specification;
- separate financial, functional, and technical analysis contracts;
- create the active role matrix;
- stop using scattered sprint-level documents as active fundamentals doctrine;
- preserve Decision Engine authority and upstream descriptive-only semantics.

Required outputs:

- `docs/active/contracts/fundamentals_platform_contract.md`
- `docs/active/contracts/fundamental_calculations_technical_spec.md`
- `docs/active/analysis/financial_analysis_contract.md`
- `docs/active/analysis/functional_analysis_contract.md`
- `docs/active/analysis/technical_analysis_contract.md`
- `docs/active/roles_and_responsibilities.md`
- `docs/sprints/fundamentals_simplification_sprint_plan.md`

Sprint A does not implement code. Sprint A does not move or archive old files.

## Sprint B - Documentation Archive and Cleanup

Status: future sprint.

Purpose:

- move historical sprint previews, validations, pilots, closeouts, and superseded documents to archive;
- keep active doctrine small;
- preserve auditability;
- prevent historical validation notes from acting as active doctrine;
- reduce `docs/sprints/` clutter.

Do not execute Sprint B in this sprint.

Expected work:

- identify historical sprint-level fundamentals documents;
- identify portfolio metadata and source lookup documents that are no longer active;
- move or mark documents historical according to repository archive policy;
- preserve all audit evidence;
- avoid deleting files.

## Sprint C - Technical Code Inventory

Status: future sprint.

Purpose:

- inspect `.py` files related to fundamentals;
- classify scripts as keep, refactor, replace, or archive candidate;
- identify compatibility wrappers;
- identify obsolete scripts;
- map current Fundamental Layer behavior to the new target architecture.

Do not execute code changes in this sprint.

Expected work:

- inspect current builder scripts;
- inspect current tests;
- inspect current raw and generated artifact assumptions;
- identify migration risk;
- create a developer-ready inventory without modifying runtime behavior.

## Sprint D - Fundamentals History Implementation Spec

Status: future sprint.

Purpose:

- create a developer-ready implementation specification for `fundamentals_history.csv`;
- define intake validation;
- define the metrics layer;
- define the quality layer;
- define the analysis layer;
- define migration behavior from current MVP artifacts;
- define tests and validation commands.

Expected outputs may include:

- developer specification;
- test plan;
- migration plan;
- fixture strategy;
- explicit generated artifact policy;
- source-data policy.

Sprint D remains documentation/specification work unless implementation is explicitly authorized.

## Sprint E - Implementation by Codex

Status: future sprint.

Purpose:

- implement approved specs;
- add tests;
- migrate or wrap current MVP behavior;
- validate focused builders;
- update runtime only as explicitly approved.

Expected scope may include:

- `scripts/core/build_fundamentals_history_intake.py`
- `scripts/core/build_fundamental_metrics.py`
- `scripts/core/build_fundamental_quality.py`
- `scripts/core/build_fundamental_analysis.py`
- focused tests;
- compatibility handling for the existing Fundamental Layer.

Sprint E is the first sprint where code changes may be allowed, but only after explicit implementation authorization.

## Sprint F - Raw Data Collection Program

Status: future sprint.

Purpose:

- collect raw historical data for many tickers;
- avoid analysis until raw coverage is sufficient;
- start with the active scanner/metadata universe;
- target 3 to 5 fiscal years per ticker;
- preserve source references and date metadata.

Rules:

- no return to per-3-ticker data creation as the main workflow;
- no analysis before adequate raw history coverage;
- no provider/API calls unless explicitly approved;
- no scraping unless explicitly approved;
- no source-data inference without evidence.

## Keep active

The following documents should remain active:

- `docs/active/architecture_current_state.md`
- `docs/active/governance_v2.md`
- `docs/active/contracts/pipeline_contracts.md`
- `docs/active/contracts/fundamentals_platform_contract.md`
- `docs/active/contracts/fundamental_calculations_technical_spec.md`
- `docs/active/analysis/financial_analysis_contract.md`
- `docs/active/analysis/functional_analysis_contract.md`
- `docs/active/analysis/technical_analysis_contract.md`
- `docs/active/roles_and_responsibilities.md`
- `docs/sprints/project_backlog.md`
- `docs/sprints/sprint_status_tracker.md`

## Move or archive later

The following types of documents should move to archive or be marked historical in Sprint B after their findings are consolidated:

- recent sprint-level fundamentals preview documents;
- recent fundamentals source lookup documents;
- fundamentals provenance-only update documents;
- numerical fundamentals pilot documents;
- numerical fundamentals standard batch documents;
- numerical fundamentals preflight validation documents;
- numerical fundamentals upstream refresh validation documents;
- numerical fundamentals contract scaling alignment documents;
- operational source-data steward protocol documents after active runbook consolidation;
- portfolio metadata source lookup and preview documents after their findings are consolidated;
- portfolio metadata validation documents after active contracts absorb relevant findings;
- completed operational sprint closeouts under `docs/sprints/`;
- superseded role documentation, including `docs/project_roles_and_responsibilities.md`, after the active role matrix is accepted.

Specific recent candidates include:

- `docs/sprints/numerical_fundamentals_contract_scaling_alignment.md`
- `docs/sprints/numerical_fundamentals_pilot_1.md`
- `docs/sprints/numerical_fundamentals_standard_batch_1.md`
- `docs/sprints/numerical_fundamentals_standard_batch_1_preflight_validation.md`
- `docs/sprints/numerical_fundamentals_standard_batch_1_upstream_refresh_validation.md`
- `docs/sprints/fundamentals_provenance_only_update_1.md`
- `docs/sprints/operational_sprint_automated_source_data_steward_protocol.md`
- recent fundamentals source-data expansion previews;
- recent portfolio metadata expansion previews;
- recent portfolio metadata source lookup documents;
- recent operational sprint follow-up documents that have already fed active doctrine.

Do not move anything in Sprint A.

## Backlog impact assessment

No new backlog items identified.

Existing backlog coverage is sufficient for this sprint:

| Backlog item | Relevance |
|---|---|
| `BL-0015` | Covers approved fundamental data source and quality classification contract need. |
| `BL-0017` | Covers governed automated or provider-assisted ingestion strategy. |
| `BL-0019` | Covers net margin schema pressure, now redirected toward calculated metrics. |
| `BL-0020` | Covers overloaded date alignment problem, now addressed by split date semantics. |
| `BL-0021` | Covers simplified raw historical fundamentals architecture. |

No backlog changes are required in Sprint A.

## Governance guardrails

All future sprints must preserve:

- classification upstream;
- allocation downstream;
- Decision Engine as only allocation authority;
- Reporting as communication only;
- no upstream tradeability;
- no hidden filtering;
- no ranking or scoring authority outside Decision Engine;
- deterministic outputs;
- auditability;
- English-only repository content.

## Validation approach for Sprint A

Sprint A validation is documentation-only.

Required confirmation:

- only documentation files changed;
- no code files changed;
- no tests changed;
- no CSV files changed;
- no raw fundamentals changed;
- no generated files changed;
- no provider APIs called;
- no scraping performed;
- no pipeline run;
- no tests run unless needed.

## Recommended next step

After Sprint A PR review and approval, start Sprint B as a separate documentation cleanup sprint.

Sprint B should archive or mark historical the superseded sprint-level fundamentals and source-data documents, while preserving auditability and avoiding deletion.