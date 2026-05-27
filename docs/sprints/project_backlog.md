# Project Backlog

## 1. Purpose

This document is the operational source of truth for deferred project work.

The backlog captures:

- future improvements
- optional corrections
- technical debt
- governance refinements
- future features
- research questions
- data-quality improvements
- reporting improvements
- test expansion ideas
- non-blocking audit findings
- deferred out-of-scope ideas
- governance gaps
- architectural follow-up
- operational risks
- future sprint candidates
- implementation limitations
- non-blocking follow-up work

The backlog is a capture mechanism only. It does not authorize implementation.

## 2. Governance Role

All backlog items inherit the certified project doctrine:

- classification upstream
- allocation downstream
- Decision Engine = ONLY allocation authority
- upstream layers classify only
- reporting communicates only
- no upstream tradeability
- no hidden filtering
- no hidden allocation semantics outside Decision Engine
- no decision semantics outside Decision Engine
- no ranking authority outside Decision Engine
- no scoring authority outside Decision Engine

A backlog item does not authorize:

- implementation
- architecture change
- sprint scope change
- strategy optimization
- threshold tuning
- new decision logic
- new allocation logic
- new filtering logic
- new tradeability semantics
- new ranking or scoring semantics
- execution changes

## 3. Relationship to `sprint_status_tracker.md`

`docs/sprints/sprint_status_tracker.md` is the operational sprint lifecycle status source of truth.

This backlog is the operational deferred-work source of truth.

The tracker answers: what phase is each sprint in?

The backlog answers: what deferred work, optional improvements, technical debt, and research questions have been captured for future governance review?

Backlog items may support an active sprint, but they do not change sprint status. Sprint phase transitions must still be updated in `docs/sprints/sprint_status_tracker.md`.

## 3.1 Mandatory Backlog Reconciliation

Mandatory Backlog Reconciliation is a required sprint lifecycle control.

Every sprint audit, implementation audit, and sprint closeout must explicitly evaluate whether new deferred work, governance gaps, technical debt, architectural follow-up, operational risks, future sprint candidates, implementation limitations, or non-blocking follow-up work were identified during:

- preparation
- governance audit
- developer specification
- implementation
- implementation audit
- closeout

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

This reconciliation is mandatory, explicit, auditable, deterministic, repeatable, and lifecycle-integrated. It ensures no deferred work disappears, no governance risk remains implicit, no architectural debt remains undocumented, no future sprint candidate remains hidden, and no implementation limitation remains untracked.

## 4. Relationship to `execution_roadmap_v2.md`

`docs/archive/superseded/execution_roadmap_v2.md` remains the roadmap and doctrine document.

The roadmap defines sequencing and architectural intent. It is not the backlog.

The backlog captures deferred or candidate work that may later inform a roadmap update, sprint proposal, or developer specification only after formal governance review.

## 5. Backlog Item Lifecycle

| Lifecycle State | Meaning |
|---|---|
| CAPTURED | Item has been recorded but not triaged into active scope |
| TRIAGED | Item has been reviewed for category, priority, and governance risk |
| ANALYSIS REQUIRED | Item needs analysis before sprint proposal or rejection |
| CANDIDATE SPRINT | Item may be proposed for a future sprint |
| APPROVED FOR PLANNING | Item is approved for sprint planning but not implementation |
| ACTIVE SPRINT | Item is included in an approved sprint scope |
| IMPLEMENTED | Item has been implemented through approved sprint governance |
| REJECTED | Item was reviewed and intentionally rejected |
| DEFERRED | Item remains valid but is intentionally postponed |

## 6. Backlog Categories

| Category | Use |
|---|---|
| Governance | Governance rules, audit findings, status protocols, certification controls |
| Technical Debt | Cleanup or maintainability work not currently in scope |
| Data Contract | Schema, artifact, row identity, compatibility, or data quality work |
| Testing | Test hardening, fixtures, regression coverage, CI checks |
| Reporting | Presentation, summaries, observability, report hygiene |
| Research | Questions requiring analysis before scope definition |
| Documentation | Documentation corrections, alignment, archival notes |
| Architecture Candidate | Potential future architecture changes requiring governance review |
| Future Feature | New functionality not approved for current implementation |
| Operational Reliability | Runtime safety, logging, observability, deterministic operation |
| Developer Experience | Tooling, scripts, setup, local workflows |
| Contracts | Runtime contract documentation, artifact semantics, row identity, boundary preservation |
| Runbooks | Operational procedures, local workflows, recovery steps, daily operating practices |
| Operational Intelligence | Observability, historical storage, diagnostics, and platform intelligence without allocation authority |
| Observational Research | Prediction tracking, feedback loops, and performance analysis that remain research-only unless routed through Decision Engine governance |

## 7. Backlog Status Definitions

| Status | Meaning |
|---|---|
| CAPTURED | Recorded as a future or supporting item |
| TRIAGED | Reviewed and categorized |
| ANALYSIS REQUIRED | Needs explicit analysis before planning |
| CANDIDATE SPRINT | Candidate for future sprint proposal |
| APPROVED FOR PLANNING | Approved for planning only |
| ACTIVE SPRINT | Included in current approved sprint scope |
| IMPLEMENTED | Completed through approved sprint execution |
| REJECTED | Not accepted for future work |
| DEFERRED | Valid but postponed |

## 8. Priority Definitions

| Priority | Meaning |
|---|---|
| P0 | Critical governance or production risk |
| P1 | Important next-cycle improvement |
| P2 | Useful enhancement |
| P3 | Optional or long-term idea |

Priority does not authorize implementation. It only guides future triage.

## 9. Required Fields Per Backlog Item

Every backlog item must include:

- ID
- Title
- Category
- Source document
- Source sprint
- Description
- Rationale
- Governance risk
- Proposed next step
- Status
- Priority
- Owner role
- Created date
- Related documents
- Notes

## 10. Current Backlog Table

| ID | Title | Category | Source Document | Source Sprint | Description | Rationale | Governance Risk | Proposed Next Step | Status | Priority | Owner Role | Created Date | Related Documents | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BL-0001 | Define exact upstream input universe for Fundamental Layer | Data Contract | `docs/archive/audits/sprint_3_execution_review.md` | Sprint 3 | Define the authoritative input source for the future Fundamental Quality Layer and any fallback behavior. | The Sprint 3 execution review found input-universe ambiguity before developer specification. | Ambiguous input source could allow row loss, hidden filtering, or inconsistent contract enforcement. | Completed through Sprint 3 implementation and verified by implementation audit. | IMPLEMENTED | P1 | Technical Lead / Developer Specification | 2026-05-09 | `docs/archive/sprints/sprint_3_execution_plan.md`; `docs/archive/audits/sprint_3_execution_review.md`; `docs/archive/sprints/sprint_3_developer_spec.md`; `docs/archive/audits/sprint_3_implementation_audit.md` | Implemented as `data/processed/context_strength.csv` input contract. Verified during Sprint 3 implementation audit. |
| BL-0002 | Define ticker/date row-key and duplicate handling for Fundamental Layer | Data Contract | `docs/archive/audits/sprint_3_execution_review.md` | Sprint 3 | Define one-row-per-ticker/date enforcement and duplicate-key behavior for future Fundamental output. | Distribution preservation requires precise row identity and deterministic duplicate handling. | Duplicate ambiguity could create suppressed, duplicated, or unstable opportunity rows. | Completed through Sprint 3 implementation and verified by implementation audit. | IMPLEMENTED | P1 | Technical Lead / Developer Specification | 2026-05-09 | `docs/archive/sprints/sprint_3_execution_plan.md`; `docs/archive/audits/sprint_3_execution_review.md`; `docs/archive/sprints/sprint_3_developer_spec.md`; `docs/archive/audits/sprint_3_implementation_audit.md` | Implemented with fail-fast duplicate and missing-key validation. Verified during Sprint 3 implementation audit. |
| BL-0003 | Expand forbidden-field checks for Fundamental Layer | Testing | `docs/archive/audits/sprint_3_execution_review.md` | Sprint 3 | Expand future Sprint 3 forbidden-field checks to cover all forbidden schema and semantic examples. | Execution review found the example grep list directionally correct but incomplete. | Missing checks could allow upstream tradeability, allocation, conviction, urgency, ranking, scoring, or actionability fields to drift in. | Completed through Sprint 3 implementation and verified by implementation audit. | IMPLEMENTED | P1 | Technical Lead / Developer Specification | 2026-05-09 | `docs/archive/sprints/sprint_3_execution_plan.md`; `docs/archive/audits/sprint_3_execution_review.md`; `docs/archive/sprints/sprint_3_developer_spec.md`; `docs/archive/audits/sprint_3_implementation_audit.md` | Implemented in focused tests and grep review. Forbidden terms appear only as negative assertions in tests. |
| BL-0004 | Clarify deterministic ordering is not ranking or priority | Governance | `docs/archive/audits/sprint_3_execution_review.md` | Sprint 3 | Clarify that any deterministic output ordering for Fundamental artifacts is operational only and must not imply ranking, priority, or score authority. | The execution plan forbids opportunity reordering, but implementation may require deterministic file ordering. | Output order could be misinterpreted as upstream ranking or prioritization. | Completed through Sprint 3 implementation and verified by implementation audit. | IMPLEMENTED | P1 | Technical Lead / Developer Specification | 2026-05-09 | `docs/archive/sprints/sprint_3_execution_plan.md`; `docs/archive/audits/sprint_3_execution_review.md`; `docs/archive/sprints/sprint_3_developer_spec.md`; `docs/archive/audits/sprint_3_implementation_audit.md` | Implemented as upstream-order preservation and tested as non-quality-based deterministic ordering. |
| BL-0005 | Normalize legacy mixed-language sprint and roadmap documentation | Documentation | `docs/archive/audits/sprint_7_implementation_audit.md` | Sprint 7 | Review legacy sprint, roadmap, audit, technical, and functional documentation for mixed-language content and normalize it to the permanent English-only repository standard where appropriate. | Permanent language governance now requires repository documentation and governance artifacts to remain English-only, while existing legacy Sprint 7 and roadmap documentation contains mixed Dutch and English text. | Mixed-language governance artifacts can create inconsistent institutional interpretation and future documentation drift. | Propose a documentation-only normalization task under formal governance; preserve meaning and avoid changing runtime scope or certified architecture. | CAPTURED | P1 | Technical Lead / Documentation Governance | 2026-05-10 | `AGENTS.md`; `README.md`; `docs/sprints/sprint_status_tracker.md`; `docs/archive/sprints/sprint_7_stability_persistence.md`; `docs/archive/superseded/execution_roadmap_v2.md` | Captured during Sprint 7 implementation audit. This item does not authorize runtime changes or sprint closure. |
| BL-0006 | Remediate legacy Reporting and Telegram semantic drift | Reporting | `docs/archive/sprints/sprint_8_reporting_preparation.md` | Sprint 8 | Review and remediate legacy reporting and Telegram behaviours that summarize, omit, rank, score, label actionable setups, expose execution-like entry/stop/target language, or contain mixed-language operator-facing text. | Sprint 8 preparation found current reporting artifacts and Telegram code paths that predate the certified reporting doctrine and may communicate hidden prioritization, hidden execution semantics, hidden suppression, or non-English repository text. | Reporting could be misread as allocation authority, execution signalling, ranking authority, or hidden filtering if legacy semantics remain active. | Create a governed Sprint 8 execution plan and developer specification that replaces legacy semantics with traceable, deterministic, row-preserving communication rules. | CAPTURED | P1 | Technical Lead / Reporting Architect | 2026-05-10 | `scripts/reporting/build_telegram_summary.py`; `scripts/reporting/send_telegram.py`; `scripts/telegram/process_telegram_commands.py`; `reports/daily/telegram_message.txt`; `reports/daily/market_scan_2026-04-30.md`; `docs/archive/sprints/sprint_8_reporting_preparation.md` | Captured during Sprint 8 preparation. This item does not authorize implementation outside approved Sprint 8 execution planning. |
| BL-0007 | Expand active pipeline contract documentation | Contracts | `docs/active/contracts/pipeline_contracts.md` | Operational Documentation Review | Expand the compact pipeline contract overview into detailed layer-specific contracts if implementation work exposes ambiguity. | Active contracts reduce dependency on archived sprint documents for runtime semantics. | Ambiguous contracts could cause accidental schema drift, row loss, hidden filtering, or authority leakage. | Triage after the first operational sprint that changes runtime orchestration or artifact handling. | CAPTURED | P1 | Technical Lead / Documentation Governance | 2026-05-18 | `docs/active/architecture_current_state.md`; `docs/active/contracts/pipeline_contracts.md` | Documentation-only backlog item. Does not authorize contract or runtime changes. |
| BL-0008 | Add operational runbooks beyond local development | Runbooks | `docs/active/runbooks/local_development.md` | Operational Documentation Review | Add practical runbooks for full pipeline runs, GitHub Actions recovery, runtime failure triage, and daily operation. | Operational runbooks improve repeatability without changing runtime behavior. | Missing runbooks can slow recovery and increase ad hoc operational changes. | Create runbooks incrementally when the related workflow is actively exercised or repaired. | CAPTURED | P2 | Operations / Developer Experience | 2026-05-18 | `docs/active/runbooks/local_development.md`; `docs/active/repository_cleanup_recommendations.md` | Runbooks must remain procedural and must not alter runtime contracts. |
| BL-0009 | Define operational intelligence storage and observability scope | Operational Intelligence | `docs/active/roadmap_current.md` | Operational Documentation Review | Define future historical decision storage, diagnostics, and platform intelligence scope as observational unless governed into Decision Engine consumption. | Operational intelligence is a priority area, but authority boundaries must stay explicit. | Historical analysis could be mistaken for allocation authority if not scoped clearly. | Produce a lightweight design note before implementing persistent intelligence artifacts. | CAPTURED | P1 | Technical Lead / Data Platform | 2026-05-18 | `docs/active/roadmap_current.md`; `docs/active/governance_v2.md`; `docs/active/contracts/pipeline_contracts.md` | Does not authorize data model, runtime, or Decision Engine changes. |
| BL-0010 | Frame prediction tracking and feedback loops as observational research | Observational Research | `docs/active/roadmap_current.md` | Operational Documentation Review | Keep prediction tracking, feedback loops, and historical performance analysis research-only unless a future governed change explicitly routes approved signals through the Decision Engine. | The next platform phase needs learning loops without eroding classification/allocation separation. | Research output could create hidden scoring, ranking, tradeability, or allocation semantics upstream. | Define research outputs, labels, and governance gates before any implementation. | CAPTURED | P1 | Research / Technical Lead | 2026-05-18 | `docs/active/roadmap_current.md`; `docs/active/architecture_current_state.md` | Observational only. No upstream tradeability, hidden filtering, or allocation authority. |
| BL-0011 | Define and repair authoritative active portfolio source | Data Contract | `docs/sprints/operational_sprint_3_investigation_followup.md` | Operational Sprint 3 | Define the authoritative active portfolio source and repair stale or incomplete active portfolio CSV state. | OS3 flow testing found that active `portfolio_positions.csv` contained only ASML while backup files indicated expected broader holdings, preventing Telegram and Reporting from showing a source-supported portfolio view. | Incorrect active portfolio data can produce misleading portfolio metadata, missing portfolio rows, or stale Decision Engine context. Repair must not infer current allocation decisions from historical transaction fields. | Create a governed data repair and source-of-truth validation task before OS3 completion. | CAPTURED | P1 | PM / Data Steward / Technical Lead | 2026-05-19 | `data/portfolio/portfolio_positions.csv`; `data/portfolio/portfolio_positions_backup.csv`; `docs/sprints/operational_sprint_3_investigation_followup.md` | Backlog only. Does not authorize CSV edits or runtime changes. |
| BL-0012 | Govern full pipeline freshness before Decision Engine execution | Operational Reliability | `docs/sprints/operational_sprint_3_investigation_followup.md` | Operational Sprint 3 | Ensure governed orchestration rebuilds required upstream artifacts before the Decision Engine consumes Portfolio Intelligence. | OS3 investigation found that the scan runner can call the Decision Engine without rebuilding `fundamental_quality.csv`, `timing_state_layer.csv`, or `portfolio_intelligence.csv`, allowing stale Decision Engine input. | Stale Portfolio Intelligence can create incorrect final decisions, incomplete portfolio metadata, and misleading reporting while preserving old rows. | Propose a Level 2 orchestration design and validation plan for full pipeline freshness. | CAPTURED | P1 | Technical Lead / Developer Experience | 2026-05-19 | `scripts/run_scan.py`; `scripts/core/build_portfolio_intelligence.py`; `scripts/core/decision_engine.py`; `docs/sprints/operational_sprint_3_investigation_followup.md` | Backlog only. Does not authorize code changes. |
| BL-0013 | Decide portfolio-only holdings contract | Architecture Candidate | `docs/sprints/operational_sprint_3_investigation_followup.md` | Operational Sprint 3 | Decide whether portfolio-only holdings should become Decision Engine input/output rows or whether Reporting should receive a separate communication-only portfolio source summary. | Current Portfolio Intelligence annotates upstream opportunity rows only, so holdings not present in the timing/opportunity universe cannot appear as Decision Engine rows or Telegram portfolio actions. | Adding portfolio-only rows changes row universe and contracts; treating portfolio history as current action authority risks Decision Engine bypass and reporting-based decision semantics. | Produce a Level 2 design decision before changing row universe or Reporting input sources; escalate to Level 3 if decision authority changes are proposed. | CAPTURED | P1 | Technical Lead / Reporting Architect / Decision Engine Owner | 2026-05-19 | `docs/active/contracts/pipeline_contracts.md`; `docs/sprints/operational_sprint_3_investigation_followup.md` | Backlog only. Does not authorize Decision Engine or Reporting contract changes. |
| BL-0014 | Define governed boundary and trigger pass-through for reporting | Contracts | `docs/sprints/operational_sprint_3_investigation_followup.md` | Operational Sprint 3 | Define whether entry, stop, target, breakout, pullback, or support fields should be passed through to Reporting and Telegram as source-supported communication fields. | OS3 flow testing found that operator-needed boundaries exist in upstream or legacy artifacts but are not currently Decision Engine or Reporting supported fields. | Boundary display can be useful communication, but it risks becoming execution guidance, hidden tradeability, or allocation semantics if not governed and source-supported. | Create a Level 2 contract proposal for pass-through-only boundary fields, with Level 3 escalation if any new decision semantics are introduced. | CAPTURED | P1 | Technical Lead / Contracts / Reporting Architect | 2026-05-19 | `data/processed/scanner_ranked.csv`; `docs/active/contracts/pipeline_contracts.md`; `docs/sprints/operational_sprint_3_investigation_followup.md` | Backlog only. Does not authorize schema, runtime, or Telegram changes. |
| BL-0015 | Define and implement approved Fundamental data source and quality classification contract | Data Contract / Operational Intelligence / Contracts | `docs/sprints/operational_sprint_3c_fundamental_data_followup.md` | Operational Sprint 3C investigation | Define an approved raw fundamentals artifact or provider integration and the descriptive Fundamental Layer quality classification contract required before real quality metadata can influence Decision Engine inputs. | OS3C investigation found that `fundamental_quality.csv` contains 291 fallback rows with `quality_state = INSUFFICIENT_DATA`, `quality_metadata_status = source_missing`, and `source_data_status = source_missing`; all final decisions remain `REVIEW` because approved fundamental quality data is absent. | Missing fundamentals prevent meaningful Decision Engine decisions; any fix must remain descriptive/classification-only and must not introduce ranking, tradeability, allocation, urgency, conviction, hidden filtering, or Decision Engine bypass. | Create a governed Level 2 design and implementation plan for fundamental data source integration. | CAPTURED | P1 | PM / Data Steward / Technical Lead / Governance | 2026-05-20 | `docs/sprints/operational_sprint_3c_fundamental_data_followup.md`; `scripts/core/build_fundamental_layer.py`; `data/processed/fundamental_quality.csv` | Backlog only. Does not authorize implementation, provider integration, runtime changes, tests, generated files, CSV edits, or Decision Engine loosening. |
| BL-0016 | Define approved Portfolio Metadata and Sector Exposure contract | Data Contract / Operational Intelligence / Contracts | `docs/sprints/operational_sprint_4_portfolio_metadata_followup.md` | Operational Sprint 4 Portfolio Metadata follow-up | Define the approved portfolio metadata and sector exposure source contract required for Portfolio Intelligence to provide complete descriptive sector metadata to the Decision Engine. | Post-Fundamental-MVP verification showed `SUFFICIENT_DATA` reaches Decision Engine inputs, but all 291 final decisions remain `REVIEW` because `portfolio_metadata_status = PARTIAL` and portfolio source files contain no explicit sector, industry, asset class, metadata source, or metadata freshness fields. | Partial portfolio sector metadata prevents Decision Engine decisions; any fix must remain descriptive/classification-only upstream and must not introduce allocation, ranking, urgency, conviction, hidden filtering, Reporting-based decision logic, or Decision Engine bypass. | Create a governed Level 2 design for portfolio metadata and sector exposure source integration. | CAPTURED | P1 | PM / Data Steward / Technical Lead / Governance | 2026-05-20 | `docs/sprints/operational_sprint_4_portfolio_metadata_followup.md`; `data/processed/portfolio_intelligence.csv`; `data/processed/final_decisions.csv`; `data/portfolio/portfolio_positions.csv`; `data/portfolio/portfolio_review.csv` | Backlog only. Does not authorize runtime changes, CSV edits, generated artifacts, Decision Engine loosening, Reporting changes, or Telegram changes. |
| BL-0017 | Define governed automated data ingestion strategy for fundamentals and portfolio metadata | Data Contract / Operational Intelligence / Operational Reliability | `docs/sprints/operational_sprint_4_data_source_strategy_followup.md` | Operational Sprint 4 Data Source Strategy follow-up | Define a governed automated or provider-assisted ingestion strategy for fundamentals and portfolio metadata after the manual CSV MVP contracts proved the source-artifact path works. | Manual CSV artifacts validated the Fundamental and Portfolio Metadata contracts, but long-term operational quality requires governed source selection, provider-assisted ingestion, automated refresh, source provenance, and freshness metadata. | Ad hoc provider/API integration or manual data drift could create nondeterministic inputs, stale source metadata, hidden inference, unverified data quality, or Decision Engine bypass pressure. | Create a governed Level 2 design for automated or provider-assisted data ingestion covering fundamentals and portfolio metadata. | CAPTURED | P1 | PM / Functional Analyst / Technical Analyst / Scrum Master / Governance | 2026-05-20 | `docs/sprints/operational_sprint_4_data_source_strategy_followup.md`; `data/raw/fundamentals.csv`; `data/portfolio/portfolio_metadata.csv`; `data/processed/fundamental_quality.csv`; `data/processed/portfolio_intelligence.csv`; `data/processed/final_decisions.csv` | Backlog only. Does not authorize implementation, provider/API integration, credentials, runtime orchestration changes, generated files, CSV edits, Decision Engine loosening, Reporting inference, or Telegram inference. |
| BL-0018 | Define governed analyst expectations and historical validation research strategy | Observational Research / Operational Intelligence / Data Contract | `docs/research/analyst_expectations_and_backtesting_research_plan.md` | Operational Sprint 5 research planning | Define a governed research strategy for analyst consensus, analyst ratings, price targets, estimate data, point-in-time storage, historical validation, and future backtesting pipeline integration. | The operator identified analyst expectations and historical outcome validation as potentially valuable for improving analysis reliability, but this data must be source-supported, point-in-time controlled, and validated before any future Decision Engine consideration. | Analyst expectations could be misused as hidden buy/sell signals, ranking authority, conviction, tradeability, allocation pressure, or Decision Engine bypass if integrated without research governance and historical validation controls. | Create a documentation-only source-policy and validation-design sprint for analyst expectations before any data collection, provider/API integration, runtime ingestion, backtesting code, or Decision Engine integration. | CAPTURED | P1 | PM / Data Steward / Research / Technical Lead / Governance | 2026-05-21 | `docs/research/analyst_expectations_and_backtesting_research_plan.md`; `docs/sprints/operational_sprint_5_analyst_expectations_backlog_capture.md`; `docs/sprints/operational_sprint_5_data_steward_role.md`; `docs/sprints/operational_sprint_5_source_data_expansion_plan.md` | Backlog only. Does not authorize implementation, provider/API calls, daily ingestion, runtime changes, generated files, CSV edits, analyst consensus scoring, backtesting code, Decision Engine changes, Reporting changes, or Telegram changes. |
| BL-0019 | Add optional net margin support to raw fundamentals schema and Fundamental Layer contract | Data Contract / Fundamental Layer / Schema | `docs/sprints/numerical_fundamentals_contract_scaling_alignment.md` | Numerical Fundamentals Contract and Scaling Alignment | Define and implement optional `net_margin` support in the raw fundamentals schema and Fundamental Layer contract if the project chooses to promote net margin from candidate metric to writable metric. | Numerical Fundamentals Pilot 1 found that `net_margin` can be source-supported, but the current local raw fundamentals schema does not include a writable `net_margin` column. | Adding schema support without governance could create schema drift, inconsistent sufficiency rules, or runtime parsing ambiguity. The change must remain descriptive/classification-only and must not alter Decision Engine authority. | Create a governed developer specification before any code, test, schema, or source-data changes. | CAPTURED | P2 | Technical Lead / Data Steward / Developer Specification | 2026-05-24 | `docs/sprints/numerical_fundamentals_contract_scaling_alignment.md`; `docs/sprints/numerical_fundamentals_pilot_1.md`; `scripts/core/build_fundamental_layer.py`; `data/raw/fundamentals.csv` | Backlog only. Does not authorize implementation, raw CSV edits, generated files, provider/API integration, runtime changes, Decision Engine changes, Reporting changes, or Telegram changes. |
| BL-0020 | Define raw fundamentals as-of-date alignment policy for opportunity-date validation | Data Contract / Fundamental Layer / Validation | `docs/sprints/numerical_fundamentals_standard_batch_1_upstream_refresh_validation.md` | Numerical Fundamentals Standard Batch 1 upstream refresh validation | Define how local ignored raw fundamentals `as_of_date` values should align with scanner opportunity dates when validating source-supported fundamentals in refreshed full-context artifacts. | Upstream refresh validation found that Standard Batch 1 rows were present locally but had `as_of_date = 2026-05-24`, while refreshed opportunity rows used `date = 2026-05-22`; the Fundamental Layer only selects raw rows with `as_of_date <= opportunity_date`, so the batch remained `row_missing`. | Ambiguous date alignment can make source-supported local raw values appear missing, block sufficiency validation, or encourage unsafe manual date edits if not governed. The policy must remain descriptive/classification-only and must not loosen Decision Engine authority. | Create a governed data-contract or data-steward specification for raw fundamentals date alignment before changing raw data, code, tests, or validation workflows. | CAPTURED | P1 | Data Steward / Technical Lead / Fundamental Layer Contract | 2026-05-25 | `docs/sprints/numerical_fundamentals_standard_batch_1.md`; `docs/sprints/numerical_fundamentals_standard_batch_1_preflight_validation.md`; `docs/sprints/numerical_fundamentals_standard_batch_1_upstream_refresh_validation.md`; `scripts/core/build_fundamental_layer.py`; `data/raw/fundamentals.csv` | Backlog only. Does not authorize raw CSV edits, generated file commits, code changes, provider/API integration, runtime changes, Decision Engine changes, Reporting changes, or Telegram changes. |
| BL-0021 | Simplify fundamentals data architecture around raw historical financial statement data | Architecture Candidate / Data Contract / Documentation | `docs/sprints/governance_architecture_repository_simplification_audit.md` | Governance, Architecture, Repository Structure, and Role Optimization Audit | Define a simplified fundamentals architecture that separates raw historical statement data, calculated metrics, quality classification, and descriptive fundamental analysis before further source-data expansion. | The simplification audit found that the current `data/raw/fundamentals.csv` MVP mixes source data, provenance, metric values, sufficiency, date matching, and analysis too early, creating repeated governance friction and blocking scalable data creation. | Architecture simplification touches data contracts and Fundamental Layer boundaries. It must remain descriptive/classification-only and must not introduce allocation, tradeability, urgency, conviction, ranking, scoring, eligibility, hidden filtering, Decision Engine bypass, or reporting decision logic. | Create a governed fundamentals data-contract redesign before implementation, source-data expansion, raw CSV changes, tests, generated output changes, provider/API integration, or pipeline behavior changes. | CAPTURED | P1 | Technical Analyst / Data Steward / Financial Analyst / Governance | 2026-05-27 | `docs/sprints/governance_architecture_repository_simplification_audit.md`; `docs/sprints/numerical_fundamentals_contract_scaling_alignment.md`; `docs/sprints/numerical_fundamentals_standard_batch_1_upstream_refresh_validation.md`; `docs/active/contracts/pipeline_contracts.md` | Backlog only. Does not authorize implementation, data collection, raw CSV edits, generated files, provider/API integration, runtime changes, Decision Engine changes, Reporting changes, or Telegram changes. |

## 11. Seeded Backlog Items From Sprint 3 Non-Blocking Findings

Seeded Sprint 3 items:

- BL-0001: Define exact upstream input universe for Fundamental Layer
- BL-0002: Define ticker/date row-key and duplicate handling for Fundamental Layer
- BL-0003: Expand forbidden-field checks for Fundamental Layer
- BL-0004: Clarify deterministic ordering is not ranking or priority

These items were incorporated into `docs/archive/sprints/sprint_3_developer_spec.md`, implemented during Sprint 3, and verified by `docs/archive/audits/sprint_3_implementation_audit.md`. Further work remains controlled by the sprint status tracker and future sprint governance, not by this backlog.

## 12. Backlog Update Protocol

During every future:

- preparation document
- governance audit
- re-audit
- execution plan
- execution review
- developer specification
- implementation audit
- sprint closeout

the following must be captured in this backlog unless the item already exists:

- non-blocking corrections
- optional improvements
- future enhancements
- deferred ideas
- out-of-scope suggestions
- technical debt
- research questions
- documentation improvements
- test expansion opportunities
- reporting improvements
- operational reliability improvements

Every future governance audit, implementation audit, and closeout must include the dedicated `Backlog Impact Assessment` section required by section 3.1. A sprint must not be marked fully closed unless the closeout includes one of the two mandatory backlog impact conclusions and all identified backlog items have been added to this document.

When adding an item:

1. Assign the next `BL-####` ID.
2. Fill every required field.
3. Link the source document and sprint.
4. Keep the status as `CAPTURED` unless formal triage has occurred.
5. Do not treat the item as active implementation scope.

## 13. Rules for Converting Backlog Items Into Future Sprints

A backlog item may become active work only after:

1. explicit analysis
2. prioritization
3. sprint proposal
4. governance review
5. execution planning
6. Technical Lead approval
7. developer specification
8. implementation authorization

Backlog conversion must preserve:

- certified architecture
- classification-first doctrine
- Decision Engine authority
- separation of concerns
- distribution preservation
- deterministic outputs
- forbidden-field governance

## 14. Anti-Scope-Creep Controls

Backlog items must not:

- change runtime code
- change tests
- change generated CSV/data files
- start implementation
- redesign architecture
- reprioritize active sprints
- alter certified sprint doctrine
- convert themselves into active scope
- authorize implementation
- create new sprint proposals unless explicitly requested
- invent completed work unsupported by documentation
- remove roadmap content

If a backlog item appears urgent, it still requires formal governance before implementation.

## 15. Final Scrum Master Recommendation

PROJECT BACKLOG READY
