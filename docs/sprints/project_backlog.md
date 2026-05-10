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

`docs/sprints/execution_roadmap_v2.md` remains the roadmap and doctrine document.

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
| BL-0001 | Define exact upstream input universe for Fundamental Layer | Data Contract | `docs/audits/sprint_3_execution_review.md` | Sprint 3 | Define the authoritative input source for the future Fundamental Quality Layer and any fallback behavior. | The Sprint 3 execution review found input-universe ambiguity before developer specification. | Ambiguous input source could allow row loss, hidden filtering, or inconsistent contract enforcement. | Completed through Sprint 3 implementation and verified by implementation audit. | IMPLEMENTED | P1 | Technical Lead / Developer Specification | 2026-05-09 | `docs/sprints/sprint_3_execution_plan.md`; `docs/audits/sprint_3_execution_review.md`; `docs/sprints/sprint_3_developer_spec.md`; `docs/audits/sprint_3_implementation_audit.md` | Implemented as `data/processed/context_strength.csv` input contract. Verified during Sprint 3 implementation audit. |
| BL-0002 | Define ticker/date row-key and duplicate handling for Fundamental Layer | Data Contract | `docs/audits/sprint_3_execution_review.md` | Sprint 3 | Define one-row-per-ticker/date enforcement and duplicate-key behavior for future Fundamental output. | Distribution preservation requires precise row identity and deterministic duplicate handling. | Duplicate ambiguity could create suppressed, duplicated, or unstable opportunity rows. | Completed through Sprint 3 implementation and verified by implementation audit. | IMPLEMENTED | P1 | Technical Lead / Developer Specification | 2026-05-09 | `docs/sprints/sprint_3_execution_plan.md`; `docs/audits/sprint_3_execution_review.md`; `docs/sprints/sprint_3_developer_spec.md`; `docs/audits/sprint_3_implementation_audit.md` | Implemented with fail-fast duplicate and missing-key validation. Verified during Sprint 3 implementation audit. |
| BL-0003 | Expand forbidden-field checks for Fundamental Layer | Testing | `docs/audits/sprint_3_execution_review.md` | Sprint 3 | Expand future Sprint 3 forbidden-field checks to cover all forbidden schema and semantic examples. | Execution review found the example grep list directionally correct but incomplete. | Missing checks could allow upstream tradeability, allocation, conviction, urgency, ranking, scoring, or actionability fields to drift in. | Completed through Sprint 3 implementation and verified by implementation audit. | IMPLEMENTED | P1 | Technical Lead / Developer Specification | 2026-05-09 | `docs/sprints/sprint_3_execution_plan.md`; `docs/audits/sprint_3_execution_review.md`; `docs/sprints/sprint_3_developer_spec.md`; `docs/audits/sprint_3_implementation_audit.md` | Implemented in focused tests and grep review. Forbidden terms appear only as negative assertions in tests. |
| BL-0004 | Clarify deterministic ordering is not ranking or priority | Governance | `docs/audits/sprint_3_execution_review.md` | Sprint 3 | Clarify that any deterministic output ordering for Fundamental artifacts is operational only and must not imply ranking, priority, or score authority. | The execution plan forbids opportunity reordering, but implementation may require deterministic file ordering. | Output order could be misinterpreted as upstream ranking or prioritization. | Completed through Sprint 3 implementation and verified by implementation audit. | IMPLEMENTED | P1 | Technical Lead / Developer Specification | 2026-05-09 | `docs/sprints/sprint_3_execution_plan.md`; `docs/audits/sprint_3_execution_review.md`; `docs/sprints/sprint_3_developer_spec.md`; `docs/audits/sprint_3_implementation_audit.md` | Implemented as upstream-order preservation and tested as non-quality-based deterministic ordering. |

## 11. Seeded Backlog Items From Sprint 3 Non-Blocking Findings

Seeded Sprint 3 items:

- BL-0001: Define exact upstream input universe for Fundamental Layer
- BL-0002: Define ticker/date row-key and duplicate handling for Fundamental Layer
- BL-0003: Expand forbidden-field checks for Fundamental Layer
- BL-0004: Clarify deterministic ordering is not ranking or priority

These items were incorporated into `docs/sprints/sprint_3_developer_spec.md`, implemented during Sprint 3, and verified by `docs/audits/sprint_3_implementation_audit.md`. Further work remains controlled by the sprint status tracker and future sprint governance, not by this backlog.

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
