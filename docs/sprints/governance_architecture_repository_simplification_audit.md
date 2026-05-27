# Governance, Architecture, Repository Structure, and Role Optimization Audit

## 1. Status and Scope

This is a documentation-only audit.

This audit does not:

- modify code;
- modify tests;
- modify CSV files;
- modify source data;
- modify generated outputs;
- delete files;
- move files;
- archive files;
- change runtime behavior;
- run the pipeline;
- authorize implementation.

No sprint is closed or certified by this audit.

## 2. Executive Summary

The project has reached a useful but heavy governance point. The recent portfolio metadata, provenance-only fundamentals, numerical pilot, standard batch, preflight validation, and upstream refresh validation work exposed real structural issues. The value of governance is clear: it found schema mismatch, source-data ambiguity, local ignored artifact drift, and date-alignment failure before these issues were scaled.

The process now needs consolidation before more fundamentals data is created. Governance has become too heavy for the current operational need, the fundamentals workflow is carrying too many responsibilities in one artifact, documentation has accumulated sprint-level ballast, and role boundaries need to be sharper.

The fundamentals pipeline should be simplified from the current ticker/date raw-metric model toward a clearer chain:

```text
raw fundamentals history -> calculated metrics -> quality classification -> fundamental analysis classification -> portfolio and Decision Engine layers
```

Further fundamentals data expansion should pause until this architecture and the active documentation surface are simplified.

## 3. Repository Structure Review

Current repository structure is broadly coherent, but the operational surface is noisy.

Active areas:

- `docs/active/`: current operational doctrine, governance, architecture, repository structure, contracts, runbooks, roadmap, cleanup guidance.
- `docs/sprints/project_backlog.md`: deferred-work capture source of truth.
- `docs/sprints/sprint_status_tracker.md`: sprint lifecycle status source of truth.
- `scripts/`: runtime implementation, including scanner, core layers, data-source helpers, diagnostics, operations, portfolio, reporting, Telegram, watchlist, and utility modules.
- `tests/`: focused regression coverage for core layers, data sources, diagnostics, portfolio, reporting, and operator visibility.

Historical or archival areas:

- `docs/archive/`: archived sprint, audit, migration, technical, functional, execution, and superseded documents.
- `docs/audits/`: currently a notice area, with audit history mainly under `docs/archive/audits/`.
- `legacy/`: legacy Telegram and watchlist surfaces.

Generated-output areas:

- `data/processed/`: generated scanner, classification, Decision Engine, reporting, per-ticker, and historical artifacts.
- `data/logs/`: generated logs and operational trace files.
- `reports/`: generated reports and daily communications.
- `.pytest_cache/`, `__pycache__/`: local generated caches.

Source-data areas:

- `data/raw/`: local fundamentals source artifacts and local backups.
- `data/portfolio/`: portfolio transactions, positions, metadata, summaries, and generated review artifacts.
- `data/intake/`: source-data intake templates and pilots.
- `data/watchlist/`: watchlist source and status artifacts.

Potential ballast:

- Many recent `docs/sprints/` files are validation records, previews, source lookup notes, and sprint planning artifacts that are useful evidence but should not all remain in active working context.
- `docs/project_roles_and_responsibilities.md` contains legacy mixed-language role doctrine and overlaps with newer active governance.
- `docs/functional/`, `docs/technical/`, `docs/financial/`, `docs/execution/`, and `docs/vision/` contain high-value reference material, but their active authority is less clear than `docs/active/`.
- `data/raw/` contains multiple fundamentals backup files that are useful local evidence but create confusion if treated as source-of-truth alternatives.

Potential confusion points:

- `data/raw/fundamentals.csv` is both source-like and metric-like.
- `as_of_date` currently does too much work and does not separate fiscal period, reporting date, source freshness, and extraction date.
- `docs/active/` already states that governance should reduce ceremony, but the recent sprint surface still creates heavy ceremony.
- Some historical validation documents may be mistaken for active doctrine.

## 4. Documentation Inventory

### Active doctrine

Representative files:

- `AGENTS.md`
- `docs/active/architecture_current_state.md`
- `docs/active/governance_v2.md`
- `docs/active/repository_structure.md`
- `docs/active/contracts/pipeline_contracts.md`

Rationale:

These files define the current operating doctrine: classification upstream, allocation downstream, Decision Engine as the only allocation authority, reporting as communication only, active documentation hierarchy, governance levels, and runtime contract boundaries.

### Active operating procedure

Representative files:

- `docs/active/runbooks/local_development.md`
- `docs/active/simplified_sprint_lifecycle.md`
- `docs/sprints/project_backlog.md`
- `docs/sprints/sprint_status_tracker.md`
- `docs/sprints/operational_sprint_automated_source_data_steward_protocol.md`

Rationale:

These documents define repeatable workflow, backlog capture, sprint status handling, local development, and source-data steward procedure. They should remain operational until consolidated.

### Historical evidence

Representative files:

- `docs/archive/audits/*`
- `docs/archive/sprints/*`
- `docs/archive/migration/*`
- `docs/sprints/fundamentals_provenance_only_update_1.md`
- `docs/sprints/numerical_fundamentals_pilot_1.md`
- `docs/sprints/numerical_fundamentals_standard_batch_1.md`
- `docs/sprints/numerical_fundamentals_standard_batch_1_preflight_validation.md`
- `docs/sprints/numerical_fundamentals_standard_batch_1_upstream_refresh_validation.md`
- `docs/sprints/portfolio_metadata_update_1_post_merge_validation.md`

Rationale:

These files record decisions, source-data validation, implementation history, and audit findings. They should remain preserved, but most should not act as day-to-day instructions.

### Candidate for archive

Representative files:

- completed operational sprint closeouts under `docs/sprints/`
- preview documents such as `fundamentals_source_data_expansion_preview_1.md`, `portfolio_metadata_expansion_preview_1.md`, and source lookup previews
- numerical fundamentals validation records after their findings are consolidated
- portfolio metadata source lookup and preview documents after their findings are consolidated

Rationale:

These files have audit value but add active-context clutter. They should be moved or marked historical in a later cleanup PR, not deleted.

### Candidate for consolidation

Representative files:

- `docs/project_roles_and_responsibilities.md`
- `docs/sprints/operational_sprint_5_data_steward_role.md`
- `docs/sprints/operational_sprint_automated_source_data_steward_protocol.md`
- `docs/sprints/numerical_fundamentals_contract_scaling_alignment.md`
- `docs/sprints/fundamentals_provenance_only_update_1.md`
- `docs/sprints/numerical_fundamentals_pilot_1.md`
- `docs/sprints/numerical_fundamentals_standard_batch_1_preflight_validation.md`
- `docs/sprints/numerical_fundamentals_standard_batch_1_upstream_refresh_validation.md`

Rationale:

The content overlaps around roles, source-data stewardship, fundamentals metric governance, validation procedure, and date semantics. It should feed a smaller active set instead of remaining scattered across sprint artifacts.

## 5. Fundamentals Architecture Review

The current fundamentals approach is too complex for further scaling.

Recent work found:

- provenance-only rows helped bootstrap traceability but did not provide useful numerical coverage;
- the numerical pilot proved that narrow manual approved-source extraction can work;
- the standard batch showed that source-data work can scale partly, but review-required metric cells remain common;
- preflight validation showed that generated upstream artifacts can be stale relative to local ignored raw fundamentals;
- upstream refresh validation proved that focused context and Fundamental Layer rebuilds can include batch tickers without provider calls;
- the same validation exposed an as-of-date alignment issue: opportunity rows used `date = 2026-05-22`, while local raw fundamentals used `as_of_date = 2026-05-24`, causing the Fundamental Layer to treat present rows as unavailable;
- `net_margin` is analytically useful but unsupported by the current raw fundamentals schema;
- local ignored raw fundamentals can diverge from tracked documentation and generated artifacts;
- Fundamental Layer output states such as `INSUFFICIENT_DATA`, `PARTIAL_DATA`, and `SUFFICIENT_DATA` are useful, but they are being asked to absorb source presence, metric sufficiency, provenance, date matching, and schema limitations at once.

The current approach mixes source data, provenance, metric values, quality classification, date matching, and analysis too early. This creates repeated governance friction and makes each data update behave like a mini-architecture review.

## 6. Proposed Simplified Fundamentals Architecture

### Raw Fundamentals History Layer

Potential artifact:

- `data/raw/fundamentals_history.csv`

Purpose:

Store raw reported financial statement data by ticker and fiscal period.

Example fields:

- `ticker`
- `fiscal_year`
- `fiscal_period`
- `period_end_date`
- `report_date`
- `currency`
- `revenue`
- `gross_profit`
- `operating_income`
- `net_income`
- `diluted_eps`
- `total_debt`
- `total_equity`
- `free_cash_flow`
- `source_name`
- `source_reference`
- `source_freshness_date`
- `extraction_date`
- `notes`

Rules:

- raw data only;
- no analysis;
- no score;
- no buy/sell semantics;
- no classification;
- local ignored unless repository policy changes.

### Fundamental Metrics Layer

Potential artifact:

- `data/processed/fundamental_metrics.csv`

Purpose:

Calculate metrics from raw history.

Example metrics:

- `revenue_growth_yoy`
- `eps_growth_yoy`
- `gross_margin`
- `operating_margin`
- `net_margin`
- `debt_to_equity`
- `return_on_equity`
- `free_cash_flow_margin`

Rules:

- calculations only;
- deterministic;
- no Decision Engine semantics;
- no ranking or allocation authority.

### Fundamental Quality Layer

Potential artifact:

- `data/processed/fundamental_quality.csv`

Purpose:

Classify completeness and reliability.

Example states:

- `SOURCE_MISSING`
- `RAW_HISTORY_PRESENT`
- `METRICS_PARTIAL`
- `METRICS_COMPLETE`
- `REVIEW_REQUIRED`

Rules:

- descriptive only;
- no buy/sell;
- no scoring authority;
- no ranking authority.

### Fundamental Analysis Layer

Potential artifact:

- `data/processed/fundamental_analysis.csv`

Purpose:

Classify business and fundamental characteristics descriptively.

Example states:

- `growth_state`
- `margin_state`
- `debt_state`
- `profitability_state`
- `cashflow_state`
- `trend_state`

Rules:

- descriptive only;
- no allocation authority;
- no tradeability semantics.

### Downstream Layers

Portfolio Intelligence and the Decision Engine consume descriptive outputs only.

The Decision Engine remains the only allocation authority.

## 7. Recommended Data Model Direction

The current `data/raw/fundamentals.csv` model should remain a temporary MVP only.

The future source of truth should move toward multi-year raw fundamentals history. Manual source extraction should target raw statement values, not precomputed ratios, whenever possible.

Recommended date semantics:

- replace or split `as_of_date`;
- use `period_end_date` for the fiscal period end;
- use `report_date` for when the company reported or filed the period;
- use `source_freshness_date` for source verification freshness;
- use `extraction_date` for local data extraction.

Metrics should be calculated downstream from raw history where possible. Storing precomputed metrics directly in raw source artifacts should be limited to temporary MVP support or cases where the metric is directly reported and the source term is clearly defined.

This direction reduces schema churn. For example, `net_margin` becomes a deterministic calculated metric when `net_income` and `revenue` exist, rather than a raw schema expansion issue.

## 8. Role Optimization Review

### Product Owner / User

Responsibilities:

- own priorities, risk appetite, and final approval for governance exceptions;
- decide whether to pause or resume data expansion;
- approve major architecture direction.

May approve:

- priorities;
- acceptance of simplification roadmap;
- governance exceptions after risk explanation.

May not approve:

- hidden allocation semantics outside the Decision Engine;
- source-data inference without evidence.

Handoff points:

- receives PM and analyst recommendations;
- authorizes planning or implementation direction.

Outputs:

- priority decisions;
- approval or rejection of proposed simplification phases.

### PM / Scrum Master

Responsibilities:

- own sprint structure, scope control, sequencing, backlog hygiene, and process simplification.

May approve:

- sprint framing;
- backlog capture;
- documentation-only sequencing.

May not approve:

- runtime implementation without developer specification;
- architecture boundary changes without governance review.

Handoff points:

- hands accepted scope to Functional Analyst, Technical Analyst, or Developer.

Outputs:

- sprint plan;
- backlog hygiene;
- concise status tracker updates.

### Functional Analyst

Responsibilities:

- own business requirements, data meaning, user workflows, and acceptance criteria.

May approve:

- business meaning of descriptive states;
- acceptance criteria.

May not approve:

- code implementation details;
- allocation authority outside the Decision Engine.

Handoff points:

- translates user intent into requirements for Technical Analyst and Developer.

Outputs:

- functional requirements;
- acceptance criteria.

### Data Steward

Responsibilities:

- own source-data quality, provenance, schema compliance, source approval, and local ignored data handling.

May approve:

- source acceptability;
- extraction evidence;
- data completeness status.

May not approve:

- financial metric relevance alone;
- runtime schema changes;
- allocation conclusions.

Handoff points:

- hands approved raw evidence to Financial Analyst and Technical Analyst.

Outputs:

- source approval notes;
- data quality findings;
- provenance records.

### Financial Analyst

Responsibilities:

- own financial metric definitions, statement interpretation, source reliability, and metric relevance.

May approve:

- metric formulas;
- financial statement interpretation;
- review triggers for financial ambiguity.

May not approve:

- runtime implementation;
- Decision Engine allocation outcomes.

Handoff points:

- hands metric definitions to Technical Analyst for contract design.

Outputs:

- metric dictionary;
- formula definitions;
- financial interpretation notes.

### Technical Analyst / Architect

Responsibilities:

- own architecture boundaries, layer responsibilities, data contracts, and simplification design.

May approve:

- data contract proposals;
- layer boundaries;
- technical architecture direction.

May not approve:

- Product Owner priorities;
- hidden allocation semantics.

Handoff points:

- hands developer-ready specifications to Developer / Codex.

Outputs:

- contract design;
- architecture notes;
- developer specification.

### Developer / Codex

Responsibilities:

- own implementation, tests, refactoring, scripts, and repository changes only when explicitly authorized.

May approve:

- implementation feasibility notes;
- test approach details.

May not approve:

- implementation scope without authorization;
- source-data values;
- governance exceptions;
- allocation semantics outside the Decision Engine.

Handoff points:

- receives approved specifications;
- returns code changes, tests, validation evidence, and implementation notes.

Outputs:

- code changes;
- tests;
- validation results;
- implementation commits.

### Governance Auditor

Responsibilities:

- own doctrine compliance, separation of concerns, and documentation consistency checks.

May approve:

- compliance findings;
- audit recommendations.

May not approve:

- implementation by itself;
- new allocation logic.

Handoff points:

- reports risks back to PM, Technical Analyst, and Product Owner.

Outputs:

- audit findings;
- risk assessment;
- backlog impact assessment.

### Decision Engine

The Decision Engine is not a human role. It is the only allocation authority.

It owns final decision semantics. No human role, upstream layer, report, validation artifact, or source-data process may create allocation, tradeability, urgency, conviction, eligibility, ranking, scoring, or hidden filtering semantics outside it.

## 9. Current Role Problems

Current role problems:

- too many governance docs can act as pseudo-authority;
- Data Steward and Financial Analyst responsibilities overlap around metrics and source interpretation;
- PM/Scrum Master documents have become heavy relative to operational need;
- Codex is sometimes used for both planning and execution without a clear authorization boundary;
- validation artifacts can become active doctrine accidentally;
- the Decision Engine authority boundary must stay isolated as fundamentals become richer;
- legacy role documentation contains mixed-language content and older framing that should be consolidated.

Recommended simplification:

- one active role matrix;
- one source-data operating procedure;
- one fundamentals data contract;
- explicit handoff from Data Steward to Financial Analyst to Technical Analyst to Developer;
- documentation-only audits should produce backlog and roadmap inputs, not implementation authority.

## 10. Documentation Simplification Proposal

Recommended smaller active documentation set:

- one current architecture doctrine document: consolidate into `docs/active/architecture_current_state.md`;
- one project backlog: keep `docs/sprints/project_backlog.md`;
- one sprint status tracker: keep `docs/sprints/sprint_status_tracker.md`;
- one fundamentals data contract: create or consolidate into `docs/active/contracts/fundamentals_data_contract.md`;
- one source-data operating procedure: consolidate source-steward protocols into `docs/active/runbooks/source_data_operations.md`;
- one reporting and Decision Engine doctrine document: keep core doctrine in `docs/active/architecture_current_state.md` and detailed runtime boundaries in `docs/active/contracts/pipeline_contracts.md`;
- one role and responsibility matrix: replace or supersede `docs/project_roles_and_responsibilities.md` with an English-only active matrix.

Existing docs that should feed consolidation:

- `docs/sprints/numerical_fundamentals_contract_scaling_alignment.md`
- `docs/sprints/numerical_fundamentals_pilot_1.md`
- `docs/sprints/numerical_fundamentals_standard_batch_1_upstream_refresh_validation.md`
- `docs/sprints/operational_sprint_automated_source_data_steward_protocol.md`
- `docs/sprints/operational_sprint_5_data_steward_role.md`
- `docs/project_roles_and_responsibilities.md`
- `docs/active/contracts/pipeline_contracts.md`

Do not edit or consolidate these documents in this audit.

## 11. Archive Strategy

Recommended archive strategy:

- keep `docs/archive/` as the historical evidence area;
- move completed sprint docs from `docs/sprints/` to `docs/archive/sprints/` in a later cleanup PR;
- group operational sprint-era docs by phase if the file count becomes hard to navigate;
- move completed validation audits and source-data previews after their findings are consolidated;
- preserve auditability by retaining every decision, validation, and closeout record;
- mark archive files as historical if moving them is too much path churn.

What should remain active:

- current architecture doctrine;
- governance v2;
- repository structure;
- pipeline contracts;
- backlog;
- sprint tracker;
- local development and source-data runbooks;
- approved fundamentals data contract once created.

What should move or be marked historical later:

- completed sprint closeouts;
- preview-only documents;
- superseded validation reports;
- older role doctrine after replacement;
- source lookup notes after their findings are consolidated.

Do not move anything yet.

## 12. Backlog Review

`docs/sprints/project_backlog.md` remains useful as the deferred-work source of truth, but some items now overlap with the need for a simpler fundamentals architecture.

Item assessment:

- `BL-0015` remains active and relevant, but should be reframed later around the new raw-history architecture rather than the current raw metric MVP alone.
- `BL-0016` remains relevant for portfolio metadata and sector exposure, separate from fundamentals simplification.
- `BL-0017` remains relevant but should depend on the simplified raw-history and source-data contract before provider-assisted ingestion work resumes.
- `BL-0019` remains relevant as a symptom of the current schema model; under the proposed architecture, `net_margin` should likely become a calculated metric rather than a raw schema field.
- `BL-0020` remains highly relevant; the proposed model directly addresses it by splitting `as_of_date` into period, report, freshness, and extraction dates.

Duplicate or overlapping areas:

- `BL-0015`, `BL-0017`, `BL-0019`, and `BL-0020` overlap around fundamentals source-data contract design.
- They should not be deleted. They should be consolidated conceptually under a new architecture simplification backlog item.

New backlog item needed:

- `BL-0021`: Simplify fundamentals data architecture around raw historical financial statement data.

The item was added to `docs/sprints/project_backlog.md` with status `CAPTURED`. It does not authorize implementation.

## 13. Proposed Simplification Roadmap

### Phase 1 - Stop and consolidate

- pause further fundamentals data expansion;
- merge pending documentation validation PRs;
- create simplified active doctrine map;
- decide the fundamentals raw history direction.

### Phase 2 - Contract redesign

- define `fundamentals_history.csv`;
- define metrics calculation contract;
- define quality classification contract;
- define date semantics.

### Phase 3 - Repository cleanup

- archive historical sprint docs;
- consolidate overlapping governance docs;
- reduce active documentation surface.

### Phase 4 - Implementation by Codex

- proceed only after docs are approved;
- create new raw history model;
- create calculation layer;
- adjust Fundamental Layer;
- write tests.

### Phase 5 - Controlled data creation

- start with 3 tickers by 3 years;
- validate calculations;
- scale to 10 tickers;
- then resume broader batches.

## 14. Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| Over-governance slows progress | Data work becomes slower than learning value justifies | Consolidate active docs and use event-driven review triggers |
| Under-governance risks bad data | Source values, dates, or formulas can drift | Keep Data Steward and Financial Analyst approvals explicit |
| Local ignored raw data can drift from GitHub documentation | Validation can pass locally but remain unreproducible for reviewers | Document local ignored policy and use clear extraction evidence |
| Generated artifacts can confuse validation | Stale generated files may be mistaken for current source truth | Treat generated outputs as validation evidence only, not doctrine |
| External provider calls can stall validation | Full pipeline runs can time out or refresh unrelated outputs | Use focused builder validation before full pipeline runs |
| Role confusion can cause Codex to overreach | Planning can become unauthorized implementation | Require explicit Developer / Codex authorization for code, tests, data, and runtime changes |
| Too many historical docs obscure current doctrine | Old decisions may be treated as active instructions | Archive or mark historical after consolidation |
| Metric schema expansion repeats with every new ratio | Each metric becomes a governance/code issue | Store raw statement values and calculate metrics downstream |
| Date ambiguity blocks valid source rows | Present raw data can be classified as missing | Split period, report, freshness, and extraction date semantics |

## 15. Final Recommendation

Pause further fundamentals data expansion.

Perform a controlled simplification sprint before adding more fundamentals data. Move toward raw historical fundamentals as the source of truth, calculate metrics downstream, classify quality separately, and keep analysis descriptive. Consolidate docs and roles, then resume data creation with smaller, clearer algorithms.

## 16. Backlog Impact Assessment

Backlog impact assessment:

- New backlog item added: `BL-0021` - Simplify fundamentals data architecture around raw historical financial statement data.
- Status: `CAPTURED`.
- Implementation is not authorized.

## 17. Branch and PR

Documentation-only branch:

- `docs/governance-architecture-repository-simplification-audit`

Intended commit scope:

- `docs/sprints/governance_architecture_repository_simplification_audit.md`
- `docs/sprints/project_backlog.md`

No code, tests, source CSVs, raw fundamentals, generated outputs, reports, or workflow files are in scope.

Draft PR title:

- `docs: audit governance architecture and repository simplification`

## 18. Validation Requirements

Validation is documentation-only.

Confirmed:

- branch name: `docs/governance-architecture-repository-simplification-audit`;
- files changed: audit document and project backlog only;
- backlog changed: yes, `BL-0021` added;
- only documentation files changed;
- no runtime files changed;
- no generated files changed;
- no CSV files changed;
- no raw fundamentals files changed;
- no provider APIs were called;
- no scraping was performed;
- no pipeline was run;
- no tests were run.

Validation limitation:

- This audit reviewed repository structure and documentation evidence only. It did not execute runtime validation because the task is documentation-only and explicitly avoids implementation.
