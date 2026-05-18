# Governance v2

Status: ACTIVE

This document defines the operational governance model for the post-purification phase of the market-scanner repository.

## Purpose

Governance v2 preserves institutional architecture safety while reducing unnecessary sprint ceremony and documentation overhead.

The system has completed certified Sprints 0 through 8. The architecture purification phase is complete. Governance now shifts from migration governance to operational governance.

## Scope and Source of Truth

This document defines how changes are classified and reviewed in the operational phase. It does not replace the architecture source of truth, roadmap, backlog, or runtime contracts.

Use these sources for current operational decisions:

- Architecture boundaries: `docs/active/architecture_current_state.md`
- Governance level and review triggers: `docs/active/governance_v2.md`
- Repository/documentation navigation: `docs/active/repository_structure.md`
- Runtime contract overview: `docs/active/contracts/pipeline_contracts.md`
- Forward planning: `docs/active/roadmap_current.md`
- Deferred work: `docs/sprints/project_backlog.md`

## Governance Objective

Preserve:

- certified architecture
- Decision Engine authority
- classification-first doctrine
- deterministic runtime behavior
- reporting neutrality
- auditability
- separation of concerns

Reduce:

- redundant audit cycles
- excessive sprint lifecycle documents
- repeated doctrine statements
- governance friction
- historical artifact clutter
- delivery slowdown

## Governance Levels

### Level 1: Standard Operational Change

Examples:

- logging improvements
- orchestration improvements
- runbook updates
- GitHub Actions recovery
- operational visibility
- documentation cleanup
- reporting presentation improvements that do not change source semantics
- test coverage improvements that do not change runtime contracts

Required governance:

- concise implementation summary
- tests or validation notes
- confirmation that architecture boundaries are unchanged

Formal governance audit is not required.

### Level 2: Architectural Surface Change

Examples:

- layer contract changes
- data schema changes
- pipeline sequencing changes
- new persistent stores
- changes to orchestration semantics
- reporting contract changes
- changes affecting row preservation or deterministic behavior

Required governance:

- lightweight design note
- contract impact review
- targeted architecture review
- validation summary

Formal audit is required only if authority boundaries may be affected.

### Level 3: Governance-Critical Change

Examples:

- Decision Engine authority changes
- allocation semantics changes
- execution semantics changes
- hidden filtering risk
- upstream tradeability semantics
- reporting decision semantics
- classification/allocation boundary changes

Required governance:

- formal governance audit
- architecture certification review
- explicit approval before implementation
- implementation audit after completion

Level 3 should be rare.

## Architecture Review Triggers

Architecture review is event-driven, not mandatory for every sprint.

It is triggered when a change may affect:

- Decision Engine authority
- allocation or execution semantics
- classification versus allocation boundaries
- reporting neutrality
- row preservation
- deterministic runtime contracts
- required output schemas
- pipeline sequencing
- data lineage or auditability

## Mandatory Safety Check

Every operational change must answer:

1. Does this change alter allocation authority?
2. Does this change introduce upstream tradeability or hidden filtering?
3. Does this change alter reporting from communication into decision-making?
4. Does this change modify runtime output contracts?
5. Does this change weaken determinism, row preservation, or auditability?

If all answers are no, the change is usually Level 1.

If any answer is yes or uncertain, escalate to Level 2 or Level 3.

## Backlog Governance

The backlog remains a planning tool, not an implementation authorization mechanism.

Backlog entries should be concise and operational. They should identify future work, risk, or improvement opportunities without recreating sprint-level audit ceremony.

## Documentation Governance

Active doctrine belongs in `docs/active/`.

Historical sprint and audit files remain preserved for traceability but are not active operational instructions unless referenced by active documentation.

## Governance Anti-Pattern

Avoid governance theater.

Do not create audits, reviews, certifications, or closeout documents when the change is low-risk and does not touch architecture authority boundaries.

Institutional quality is preserved through clear contracts, tests, deterministic behavior, and event-driven review triggers, not through unnecessary document volume.
