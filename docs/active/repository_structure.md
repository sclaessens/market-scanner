# Repository Structure v2

Status: ACTIVE

This document defines the active documentation structure for the operational phase of the market-scanner repository.

## Objective

The repository is transitioning from architecture purification to operational intelligence platform evolution. The documentation structure must preserve institutional traceability while reducing governance noise and making current-state development easier.

## Documentation Tiers

### Active documentation

Location: `docs/active/`

Active documentation is authoritative for current development and operations. These files define the current architecture, governance model, runtime contracts, operational workflow, calculation governance, backlog/sprint operating model, and roadmap.

Active documents must be:

- current-state first
- concise
- operationally actionable
- free of historical sprint ceremony
- free of duplicated doctrine
- aligned with certified architecture principles

### Reference documentation

Location: `docs/reference/`

Reference documentation preserves institutional rationale. These files explain why the system uses its certified architecture, governance model, separation of concerns, and deterministic pipeline design.

Reference documents are explanatory, not operationally authoritative.

### Archive documentation

Location: `docs/archive/`

Archive documentation preserves historical sprint artifacts, audits, migration documents, deprecated plans, and superseded governance material.

Archive documents are historical records. They must not be treated as active implementation instructions unless explicitly referenced by an active document.

## Active Documentation Map

The active documentation set is intentionally small:

- `docs/active/architecture_current_state.md`
- `docs/active/governance_v2.md`
- `docs/active/contracts/pipeline_contracts.md`
- `docs/active/backlog_and_sprint_operating_model.md`
- `docs/active/logic/calculation_registry.md`
- `docs/active/roles_and_responsibilities.md`
- `docs/active/operational_development_model.md`
- `docs/active/simplified_sprint_lifecycle.md`
- `docs/active/archive_strategy.md`
- `docs/active/active_reference_archive_classification.md`
- `docs/active/operational_phase_preparation.md`
- `docs/active/repository_cleanup_recommendations.md`
- `docs/active/repository_structure.md`
- `docs/active/roadmap_current.md`
- `docs/active/runbooks/local_development.md`

Active specialized documents may also exist under:

- `docs/active/analysis/`
- `docs/active/contracts/`
- `docs/active/inventory/`
- `docs/active/logic/`
- `docs/active/specs/`
- `docs/active/runbooks/`

These folders should remain focused. Do not create new active documents if an existing active document can be updated cleanly.

## Where To Look Before Changing X

| Change area | Start here |
|---|---|
| Architecture or layer responsibility | `docs/active/architecture_current_state.md` |
| Governance boundary or review level | `docs/active/governance_v2.md` |
| Runtime contracts or artifact semantics | `docs/active/contracts/pipeline_contracts.md` |
| Backlog triage, sprint capacity, sprint closeout discipline | `docs/active/backlog_and_sprint_operating_model.md` |
| Calculation placement, formula ownership, logic review | `docs/active/logic/calculation_registry.md` |
| Role ownership and handoff boundaries | `docs/active/roles_and_responsibilities.md` |
| Local workflow or development hygiene | `docs/active/runbooks/local_development.md` |
| Roadmap planning | `docs/active/roadmap_current.md` |
| Backlog or deferred work | `docs/sprints/project_backlog.md` |
| Historical sprint or audit evidence | `docs/archive/` |
| Repository organization | `docs/active/repository_structure.md` |

## Historical Documentation Handling

Historical evidence for certified Sprints 0 through 8 and completed operational sprint planning is preserved under `docs/archive/`.

The remaining `docs/sprints/` files are operational backlog, sprint status, current simplification sprint planning, and recent simplification audit records. The remaining `docs/audits/` file is a notice. Archived sprint, audit, migration, and superseded documents are superseded by `docs/active/` unless an active document explicitly delegates authority to an archived file.

## Source of Truth Rule

When files conflict:

1. `AGENTS.md` remains the repository-level AI governance authority.
2. `docs/active/architecture_current_state.md` is the architecture source of truth.
3. `docs/active/governance_v2.md` is the operational governance source of truth.
4. `docs/active/backlog_and_sprint_operating_model.md` is the backlog triage, sprint capacity, and sprint closeout operating source of truth.
5. `docs/active/logic/calculation_registry.md` is the calculation placement and calculation-governance starting point.
6. `docs/active/*` supersedes legacy sprint, audit, and migration documents for operational development.
7. Archived and historical documents preserve context but do not authorize implementation.

## Runtime Scope

This restructuring changes documentation and governance organization only. It does not change runtime code, pipeline outputs, allocation semantics, reporting semantics, data contracts, or Decision Engine authority.