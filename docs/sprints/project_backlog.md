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
