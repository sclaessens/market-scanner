# Project Backlog

Status: ACTIVE
Last reviewed: 2026-05-27

## 1. Purpose

This document is the operational source of truth for deferred project work.

The backlog captures:

- future improvements;
- optional corrections;
- technical debt;
- governance refinements;
- future features;
- research questions;
- data-quality improvements;
- reporting improvements;
- test expansion ideas;
- non-blocking audit findings;
- deferred out-of-scope ideas;
- governance gaps;
- architectural follow-up;
- operational risks;
- future sprint candidates;
- implementation limitations;
- non-blocking follow-up work.

The backlog is a capture and triage mechanism only. It does not authorize implementation.

## 2. Governance Role

All backlog items inherit the certified project doctrine:

- classification upstream;
- allocation downstream;
- Decision Engine = ONLY allocation authority;
- upstream layers classify only;
- reporting communicates only;
- no upstream tradeability;
- no hidden filtering;
- no hidden allocation semantics outside Decision Engine;
- no decision semantics outside Decision Engine;
- no ranking authority outside Decision Engine;
- no scoring authority outside Decision Engine.

A backlog item does not authorize:

- implementation;
- architecture change;
- sprint scope change;
- strategy optimization;
- threshold tuning;
- new decision logic;
- new allocation logic;
- new filtering logic;
- new tradeability semantics;
- new ranking or scoring semantics;
- execution changes.

## 3. Relationship to Active Operating Model

Backlog triage, sprint capacity, effort points, governance risk ratings, sprint selection, sprint closeout review, document replacement policy, calculation governance, and logic review cadence are governed by:

- `docs/active/backlog_and_sprint_operating_model.md`

This backlog applies that operating model to concrete backlog items.

## 4. Relationship to `sprint_status_tracker.md`

`docs/sprints/sprint_status_tracker.md` is the operational sprint lifecycle status source of truth.

This backlog is the operational deferred-work source of truth.

The tracker answers: what phase is each sprint in?

The backlog answers: what deferred work, optional improvements, technical debt, and research questions have been captured for future governance review?

Backlog items may support an active sprint, but they do not change sprint status. Sprint phase transitions must still be updated in `docs/sprints/sprint_status_tracker.md`.

## 5. Mandatory Backlog Reconciliation

Mandatory Backlog Reconciliation is a required sprint lifecycle control.

Every sprint audit, implementation audit, and sprint closeout must explicitly evaluate whether new deferred work, governance gaps, technical debt, architectural follow-up, operational risks, future sprint candidates, implementation limitations, or non-blocking follow-up work were identified during:

- preparation;
- governance audit;
- developer specification;
- implementation;
- implementation audit;
- closeout.

If new backlog items are identified, they must be added to this document before the sprint may be considered fully closed.

Every future sprint audit and closeout document must contain a dedicated section named:

```text
Backlog Impact Assessment
```

That section must conclude with exactly one of the following forms:

```text
Backlog impact assessment:
- No new backlog items identified.
```

or:

```text
Backlog impact assessment:
- New backlog items identified and added to project_backlog.md
```

## 6. Backlog Item Lifecycle

| Lifecycle State | Meaning |
|---|---|
| `CAPTURED` | Item has been recorded but not triaged into active scope. |
| `TRIAGED` | Item has been reviewed for relevance, category, priority, effort, risk, dependencies, and possible sprint fit. |
| `ANALYSIS REQUIRED` | Item needs analysis before sprint proposal or rejection. |
| `CANDIDATE SPRINT` | Item may be proposed for a future sprint. |
| `APPROVED FOR PLANNING` | Item is approved for sprint planning but not implementation. |
| `ACTIVE SPRINT` | Item is included in an approved sprint scope. |
| `IMPLEMENTED` | Item has been implemented or fully satisfied through approved sprint governance. |
| `REJECTED` | Item was reviewed and intentionally rejected or superseded by a better direction. |
| `DEFERRED` | Item remains valid but is intentionally postponed. |

## 7. Backlog Categories

| Category | Use |
|---|---|
| Governance | Governance rules, audit findings, status protocols, certification controls. |
| Technical Debt | Cleanup or maintainability work not currently in scope. |
| Data Contract | Schema, artifact, row identity, compatibility, or data quality work. |
| Testing | Test hardening, fixtures, regression coverage, CI checks. |
| Reporting | Presentation, summaries, observability, report hygiene. |
| Research | Questions requiring analysis before scope definition. |
| Documentation | Documentation corrections, alignment, archival notes. |
| Architecture Candidate | Potential future architecture changes requiring governance review. |
| Future Feature | New functionality not approved for current implementation. |
| Operational Reliability | Runtime safety, logging, observability, deterministic operation. |
| Developer Experience | Tooling, scripts, setup, local workflows. |
| Contracts | Runtime contract documentation, artifact semantics, row identity, boundary preservation. |
| Runbooks | Operational procedures, local workflows, recovery steps, daily operating practices. |
| Operational Intelligence | Observability, historical storage, diagnostics, and platform intelligence without allocation authority. |
| Observational Research | Prediction tracking, feedback loops, and performance analysis that remain research-only unless routed through Decision Engine governance. |
| Strategy / Logic | Trading logic, calculation placement, ticker-category logic, and strategy review before implementation. |

## 8. Priority Definitions

| Priority | Meaning |
|---|---|
| `P0` | Critical governance or production risk. |
| `P1` | Important next-cycle improvement. |
| `P2` | Useful enhancement. |
| `P3` | Optional or long-term idea. |

Priority does not authorize implementation. It only guides future triage.

## 9. Effort Points

| Points | Meaning |
|---:|---|
| 1 | Small documentation update, reference cleanup, or narrow analysis note. |
| 2 | Small isolated implementation or cleanup with limited tests and no contract change. |
| 3 | Normal feature/refactor/spec with focused tests or moderate documentation impact. |
| 5 | Larger layer, contract, schema, or orchestration change requiring careful validation. |
| 8 | Complex architecture or multi-layer change. Must usually be split before implementation. |
| `N/A` | Not applicable because item is already implemented, rejected, or superseded. |

## 10. Governance Risk Rating

| Risk | Meaning |
|---|---|
| `LOW` | Documentation, reference, or isolated cleanup. No runtime authority boundary risk. |
| `MEDIUM` | Runtime code, tests, or operational behavior change with clear boundaries. |
| `HIGH` | Data contracts, pipeline sequencing, generated artifact contracts, or cross-layer dependencies. |
| `CRITICAL` | Decision Engine authority, allocation semantics, hidden filtering risk, or reporting decision semantics. |
| `N/A` | Not applicable because item is already implemented, rejected, or superseded. |

## 11. Required Fields Per Backlog Item

Every backlog item must include:

- ID;
- title;
- category;
- source document;
- source sprint;
- description;
- rationale;
- governance risk;
- governance risk rating;
- effort points;
- sprint candidate;
- dependency or blocking notes;
- proposed next step;
- status;
- priority;
- owner role;
- created date;
- last reviewed date;
- related documents;
- notes.

## 12. Sprint Candidate Rules

Sprint candidate values:

| Value | Meaning |
|---|---|
| `YES` | Can be considered for the next sprint after capacity and dependency review. |
| `NO` | Not a current sprint candidate. |
| `LATER` | Valid, but intentionally postponed. |
| `BLOCKED` | Valid, but a dependency must be resolved first. |
| `SPLIT FIRST` | Too broad for one sprint; must be split before planning. |

A sprint may include at most 5 effort points and one primary theme unless the Product Owner explicitly approves an exception after risk review.

## 13. Current Backlog Triage Table

| ID | Title | Category | Status | Priority | Effort | Risk Rating | Sprint Candidate | Dependencies / Blocking Notes | Proposed Next Step | Owner Role | Created | Last Reviewed | Related Documents | Notes |
|---|---|---|---|---|---:|---|---|---|---|---|---|---|---|---|
| BL-0001 | Define exact upstream input universe for Fundamental Layer | Data Contract | IMPLEMENTED | P1 | N/A | N/A | NO | None. | None. Keep as historical implemented item. | Technical Lead / Developer Specification | 2026-05-09 | 2026-05-27 | `docs/archive/sprints/sprint_3_developer_spec.md`; `docs/archive/audits/sprint_3_implementation_audit.md` | Implemented as `data/processed/context_strength.csv` input contract. |
| BL-0002 | Define ticker/date row-key and duplicate handling for Fundamental Layer | Data Contract | IMPLEMENTED | P1 | N/A | N/A | NO | None. | None. Keep as historical implemented item. | Technical Lead / Developer Specification | 2026-05-09 | 2026-05-27 | `docs/archive/sprints/sprint_3_developer_spec.md`; `docs/archive/audits/sprint_3_implementation_audit.md` | Implemented with fail-fast duplicate and missing-key validation. |
| BL-0003 | Expand forbidden-field checks for Fundamental Layer | Testing | IMPLEMENTED | P1 | N/A | N/A | NO | None. | None. Keep as historical implemented item. | Technical Lead / Developer Specification | 2026-05-09 | 2026-05-27 | `docs/archive/sprints/sprint_3_developer_spec.md`; `docs/archive/audits/sprint_3_implementation_audit.md` | Implemented in focused tests and grep review. |
| BL-0004 | Clarify deterministic ordering is not ranking or priority | Governance | IMPLEMENTED | P1 | N/A | N/A | NO | None. | None. Keep as historical implemented item. | Technical Lead / Developer Specification | 2026-05-09 | 2026-05-27 | `docs/archive/sprints/sprint_3_developer_spec.md`; `docs/archive/audits/sprint_3_implementation_audit.md` | Implemented as upstream-order preservation and tested as non-quality-based deterministic ordering. |
| BL-0005 | Normalize legacy mixed-language sprint and roadmap documentation | Documentation | DEFERRED | P3 | 2 | LOW | LATER | Historical docs have been archived; remaining value is optional cleanup of legacy mixed-language archive material. | Defer until active documentation has higher-priority gaps resolved. | Documentation Steward / Governance | 2026-05-10 | 2026-05-27 | `AGENTS.md`; `docs/archive/sprints/sprint_7_stability_persistence.md`; `docs/archive/superseded/execution_roadmap_v2.md` | Not a current sprint candidate because active docs now control current work. |
| BL-0006 | Remediate legacy Reporting and Telegram semantic drift | Reporting | TRIAGED | P2 | 2 | MEDIUM | LATER | Sprint 8 remediated core reporting doctrine; `build_telegram_summary.py` remains a protected compatibility wrapper per C.4. | Handle only through a narrow cleanup scope if active references are removed or wrappers are explicitly retained. | Reporting Architect / Technical Analyst | 2026-05-10 | 2026-05-27 | `docs/active/inventory/python_runtime_reference_dependency_verification.md`; `scripts/reporting/build_telegram_summary.py`; `scripts/reporting/build_reporting_layer.py` | Core risk is reduced; remaining work is cleanup/wrapper hygiene, not broad reporting redesign. |
| BL-0007 | Expand active pipeline contract documentation | Contracts | DEFERRED | P2 | 1 | LOW | LATER | Current active contracts are sufficient until implementation exposes new ambiguity. | Update contracts only when a concrete sprint changes or stresses runtime artifact semantics. | Documentation Steward / Technical Analyst | 2026-05-18 | 2026-05-27 | `docs/active/contracts/pipeline_contracts.md`; `docs/active/architecture_current_state.md` | Keep as low-effort future documentation hardening. |
| BL-0008 | Add operational runbooks beyond local development | Runbooks | CANDIDATE SPRINT | P2 | 2 | LOW | YES | Should wait for next actively exercised operational workflow or cleanup implementation. | Create focused runbook update when an implementation sprint needs it. | Operations / Developer Experience | 2026-05-18 | 2026-05-27 | `docs/active/runbooks/local_development.md` | Good supporting item for a future 3-point technical sprint if capacity allows. |
| BL-0009 | Define operational intelligence storage and observability scope | Operational Intelligence | ANALYSIS REQUIRED | P1 | 3 | HIGH | BLOCKED | Requires strategy/logic review and authority-boundary design before storage or diagnostics implementation. | Produce design note only after strategy/logic rationalization clarifies what should be observed. | Technical Analyst / Data Platform / Governance | 2026-05-18 | 2026-05-27 | `docs/active/roadmap_current.md`; `docs/active/governance_v2.md`; `docs/active/contracts/pipeline_contracts.md` | Observational only unless later routed through Decision Engine governance. |
| BL-0010 | Frame prediction tracking and feedback loops as observational research | Observational Research | ANALYSIS REQUIRED | P1 | 3 | HIGH | BLOCKED | Requires research-label design, historical outcome definition, and explicit non-allocation boundary. | Keep research-only until strategy/logic and operational-intelligence scope are defined. | Research / Technical Analyst / Governance | 2026-05-18 | 2026-05-27 | `docs/active/roadmap_current.md`; `docs/active/architecture_current_state.md` | Must not become hidden scoring, ranking, or allocation. |
| BL-0011 | Define and repair authoritative active portfolio source | Data Contract | TRIAGED | P1 | 3 | HIGH | LATER | Requires data-steward confirmation of current portfolio source and no inference from historical transactions. | Revalidate after fundamentals and strategy/logic priorities are selected; implement only with explicit data repair scope. | Data Steward / Technical Analyst | 2026-05-19 | 2026-05-27 | `data/portfolio/portfolio_positions.csv`; archived OS3 investigation docs | Keep valid but not next sprint unless portfolio state blocks execution. |
| BL-0012 | Govern full pipeline freshness before Decision Engine execution | Operational Reliability | CANDIDATE SPRINT | P1 | 3 | HIGH | YES | C.4 protects `run_scan.py` and `run_full_pipeline.py`; freshness work needs strict do-not-touch boundaries and tests. | Candidate for a focused implementation/spec sprint after backlog review, or after strategy/logic rationalization if sequencing is preferred. | Technical Analyst / Developer Experience | 2026-05-19 | 2026-05-27 | `scripts/run_scan.py`; `scripts/run_full_pipeline.py`; `scripts/core/decision_engine.py` | Strong candidate because stale inputs can mislead final decisions without obvious failure. |
| BL-0013 | Decide portfolio-only holdings contract | Architecture Candidate | ANALYSIS REQUIRED | P1 | 5 | CRITICAL | BLOCKED | Touches row universe, Reporting communication, portfolio holdings, and possible Decision Engine authority. | Produce Level 2/3 architecture decision only after portfolio source contract and logic model are clearer. | Technical Analyst / Decision Engine Owner / Reporting Architect | 2026-05-19 | 2026-05-27 | `docs/active/contracts/pipeline_contracts.md`; archived OS3 investigation docs | Do not combine with other major work. |
| BL-0014 | Define governed boundary and trigger pass-through for reporting | Contracts | ANALYSIS REQUIRED | P1 | 3 | CRITICAL | BLOCKED | Entry/stop/target display risks execution guidance and hidden tradeability semantics. | Requires dedicated reporting/contract design before any schema or Telegram change. | Contracts / Reporting Architect / Governance | 2026-05-19 | 2026-05-27 | `data/processed/scanner_ranked.csv`; `docs/active/contracts/pipeline_contracts.md` | Useful operator feature, but high semantic risk. |
| BL-0015 | Define and implement approved Fundamental data source and quality classification contract | Data Contract / Operational Intelligence / Contracts | APPROVED FOR PLANNING | P1 | 5 | HIGH | YES | Sprint A-D created doctrine, inventory, and implementation specification; Sprint E can implement only with Option A compatibility wrapper first. | Convert into Sprint E1/E2 implementation only after final sprint capacity decision and do-not-touch boundaries are accepted. | Data Steward / Technical Analyst / Governance | 2026-05-20 | 2026-05-27 | `docs/active/specs/fundamentals_history_implementation_spec.md`; `docs/active/contracts/fundamentals_platform_contract.md` | Main implementation candidate, but should not be combined with unrelated cleanup. |
| BL-0016 | Define approved Portfolio Metadata and Sector Exposure contract | Data Contract / Operational Intelligence / Contracts | DEFERRED | P1 | 5 | HIGH | LATER | Portfolio metadata docs were archived; future contract still valid but should wait until fundamentals/source-data architecture stabilizes. | Revisit after fundamentals raw-history and ticker-category logic are clarified. | Data Steward / Technical Analyst / Governance | 2026-05-20 | 2026-05-27 | `data/processed/portfolio_intelligence.csv`; archived portfolio metadata docs | Still relevant, but not next sprint. |
| BL-0017 | Define governed automated data ingestion strategy for fundamentals and portfolio metadata | Data Contract / Operational Intelligence / Operational Reliability | ANALYSIS REQUIRED | P1 | 5 | HIGH | BLOCKED | Provider/API integration must wait for raw-history schema, source policy, and data-steward workflow. | Defer until manual/local raw-history architecture and validation model are proven. | Functional Analyst / Technical Analyst / Data Steward / Governance | 2026-05-20 | 2026-05-27 | `docs/active/specs/fundamentals_history_implementation_spec.md`; `data/raw/fundamentals.csv`; `data/portfolio/portfolio_metadata.csv` | No provider/API work before governed source-data scope. |
| BL-0018 | Define governed analyst expectations and historical validation research strategy | Observational Research / Operational Intelligence / Data Contract | ANALYSIS REQUIRED | P1 | 5 | CRITICAL | BLOCKED | Analyst expectations can become hidden scoring/conviction if not isolated as point-in-time research. | Defer until observational research and operational-intelligence scope are designed. | Research / Data Steward / Technical Analyst / Governance | 2026-05-21 | 2026-05-27 | `docs/research/analyst_expectations_and_backtesting_research_plan.md` | Important long-term idea, but too risky before research governance. |
| BL-0019 | Add optional net margin support to raw fundamentals schema and Fundamental Layer contract | Data Contract / Fundamental Layer / Schema | REJECTED | P2 | N/A | N/A | NO | Superseded by the new fundamentals architecture: net margin should be calculated in the metrics layer, not added as raw metric-like source data. | Do not implement original raw-schema approach. Preserve net margin as a planned calculated metric. | Financial Analyst / Technical Analyst | 2026-05-24 | 2026-05-27 | `docs/active/contracts/fundamental_calculations_technical_spec.md`; archived numerical fundamentals docs | Original direction rejected/superseded, not the concept of net margin. |
| BL-0020 | Define raw fundamentals as-of-date alignment policy for opportunity-date validation | Data Contract / Fundamental Layer / Validation | IMPLEMENTED | P1 | N/A | N/A | NO | Addressed by new split date semantics: `period_end_date`, `report_date`, `source_freshness_date`, and `extraction_date`. | None. Preserve through future implementation tests. | Data Steward / Technical Analyst | 2026-05-25 | 2026-05-27 | `docs/active/contracts/fundamentals_platform_contract.md`; `docs/active/specs/fundamentals_history_implementation_spec.md` | Policy need satisfied at documentation/spec level; future code must implement it. |
| BL-0021 | Simplify fundamentals data architecture around raw historical financial statement data | Architecture Candidate / Data Contract / Documentation | IMPLEMENTED | P1 | N/A | N/A | NO | Sprint A-D produced active doctrine, calculation spec, inventory, and implementation spec. | None as backlog item; future implementation is tracked through BL-0015. | Technical Analyst / Data Steward / Financial Analyst / Governance | 2026-05-27 | 2026-05-27 | `docs/active/contracts/fundamentals_platform_contract.md`; `docs/active/inventory/fundamentals_code_inventory.md`; `docs/active/specs/fundamentals_history_implementation_spec.md` | Architecture simplification is now active doctrine. |
| BL-0022 | Rationalize trading strategy, calculation placement, and ticker-category logic | Strategy / Logic / Architecture Candidate | CANDIDATE SPRINT | P1 | 5 | HIGH | YES | New operating model and calculation registry identify this as needed before adding complex new logic. | Create a documentation-only strategy/logic rationalization sprint before broad implementation or deletion of useful logic. | Strategy and Logic Steward / Financial Analyst / Technical Analyst / Governance | 2026-05-27 | 2026-05-27 | `docs/active/backlog_and_sprint_operating_model.md`; `docs/active/logic/calculation_registry.md`; `docs/active/specs/python_runtime_cleanup_developer_spec.md` | Captures Product Owner request to review logic, calculations, ticker categories, and future-proof simplification before forcing new logic into old structure. |
| BL-0023 | Define narrow Python runtime cleanup implementation scope | Technical Debt / Developer Experience / Governance | APPROVED FOR PLANNING | P2 | 2 | MEDIUM | YES | Sprint C.4 found no Python file approved for deletion now and recommended a very narrow cleanup scope. | Define or approve a tiny cleanup batch before moving/deleting code; keep protected files untouched. | Technical Analyst / Developer / Governance Auditor | 2026-05-27 | 2026-05-27 | `docs/active/specs/python_runtime_cleanup_developer_spec.md`; `docs/active/inventory/python_runtime_reference_dependency_verification.md` | Candidate supporting sprint, but should not be combined with BL-0015 or BL-0022 unless only documentation scope is used. |

## 14. Triage Summary

### Implemented or satisfied

- BL-0001 through BL-0004 are implemented historical Sprint 3 items.
- BL-0020 is satisfied at documentation/spec level through split date semantics.
- BL-0021 is satisfied at architecture/specification level through the fundamentals simplification sequence.

### Rejected or superseded

- BL-0019 original raw-schema direction is rejected/superseded. Net margin remains valid as a calculated metric in the future Fundamental Metrics layer.

### Current strongest sprint candidates

| Candidate | Effort | Risk | Reason |
|---|---:|---|---|
| BL-0022 | 5 | HIGH | Best aligns with the Product Owner's request to review strategy, calculations, ticker categories, and complexity before more implementation. |
| BL-0015 | 5 | HIGH | Main controlled fundamentals implementation candidate after Sprint D, but should follow or explicitly incorporate logic-rationalization findings. |
| BL-0012 | 3 | HIGH | Pipeline freshness affects correctness of Decision Engine inputs and may be a focused reliability sprint. |
| BL-0023 | 2 | MEDIUM | Very narrow cleanup scope can reduce technical debt safely, but no deletion is approved yet. |
| BL-0008 | 2 | LOW | Runbook support can accompany a technical sprint if capacity allows. |

### Recommended next sprint

Recommended next sprint:

```text
Sprint P3 / C.5 — Strategy, Logic, Calculation, and Ticker-Category Rationalization
```

Recommended backlog driver:

```text
BL-0022
```

Rationale:

- It directly addresses the need to review whether current logic is still correct before implementation expands.
- It supports calculation registry completion.
- It helps decide which logic belongs in which layer or module.
- It can guide future cleanup and Sprint E implementation.
- It keeps the project from forcing new logic into old files or outdated abstractions.

Do not combine BL-0022 with BL-0015 implementation in the same sprint. BL-0022 is a 5-point high-risk analysis sprint and consumes the default sprint capacity.

## 15. Backlog Update Protocol

During every future:

- preparation document;
- governance audit;
- re-audit;
- execution plan;
- execution review;
- developer specification;
- implementation audit;
- sprint closeout;

capture the following in this backlog unless the item already exists:

- non-blocking corrections;
- optional improvements;
- future enhancements;
- deferred ideas;
- out-of-scope suggestions;
- technical debt;
- research questions;
- documentation improvements;
- test expansion opportunities;
- reporting improvements;
- operational reliability improvements.

Every future governance audit, implementation audit, and closeout must include the dedicated `Backlog Impact Assessment` section required by section 5. A sprint must not be marked fully closed unless the closeout includes one of the two mandatory backlog impact conclusions and all identified backlog items have been added to this document.

When adding an item:

1. assign the next `BL-####` ID;
2. fill every required field;
3. link the source document and sprint;
4. keep the status as `CAPTURED` unless formal triage has occurred;
5. do not treat the item as active implementation scope.

## 16. Rules for Converting Backlog Items Into Future Sprints

A backlog item may become active work only after:

1. explicit analysis;
2. prioritization;
3. effort and governance-risk scoring;
4. sprint-capacity check;
5. sprint proposal;
6. governance review;
7. execution planning;
8. Product Owner approval;
9. developer specification where implementation is involved;
10. implementation authorization where code, tests, data, generated artifacts, or runtime behavior are involved.

Backlog conversion must preserve:

- certified architecture;
- classification-first doctrine;
- Decision Engine authority;
- separation of concerns;
- distribution preservation;
- deterministic outputs;
- forbidden-field governance;
- calculation registry discipline;
- code-placement review.

## 17. Anti-Scope-Creep Controls

Backlog items must not:

- change runtime code;
- change tests;
- change generated CSV/data files;
- start implementation;
- redesign architecture;
- reprioritize active sprints;
- alter certified sprint doctrine;
- convert themselves into active scope;
- authorize implementation;
- create new sprint proposals unless explicitly requested;
- invent completed work unsupported by documentation;
- remove roadmap content.

If a backlog item appears urgent, it still requires formal governance before implementation.

## 18. Backlog Impact Assessment

Backlog impact assessment:

- New backlog items identified and added to project_backlog.md

New items added:

- BL-0022: Rationalize trading strategy, calculation placement, and ticker-category logic.
- BL-0023: Define narrow Python runtime cleanup implementation scope.

## 19. Final Scrum Master Recommendation

PROJECT BACKLOG TRIAGED AND SCORED

The backlog is now ready for capacity-based sprint selection under `docs/active/backlog_and_sprint_operating_model.md`.

Recommended next sprint driver: BL-0022.

Do not start Sprint E implementation until the Product Owner accepts either:

1. BL-0022 strategy/logic rationalization first; or
2. Sprint E1 with strict do-not-touch boundaries and explicit acknowledgement that broader strategy/logic rationalization remains pending.