# Active, Reference, and Archive Classification

Status: ACTIVE

This document classifies repository documentation by operational authority.

## Active

Active documents are the current operational sources of truth.

Examples:

- `AGENTS.md`
- `README.md`
- `docs/active/architecture_current_state.md`
- `docs/active/governance_v2.md`
- `docs/active/repository_structure.md`
- `docs/active/simplified_sprint_lifecycle.md`
- `docs/active/operational_development_model.md`
- `docs/active/roadmap_current.md`
- `docs/active/repository_cleanup_recommendations.md`
- `docs/active/operational_phase_preparation.md`

## Reference

Reference documents explain institutional reasoning and architecture rationale. They help future maintainers understand why the system is designed this way, but they are not the default source for current operational instructions.

## Archive

Archive documents preserve historical execution and review evidence. Completed sprint plans, completed sprint reviews, completed sprint closeouts, and historical migration materials belong in this tier.

## Legacy Folder Classification

The existing `docs/sprints/` and `docs/audits/` folders mainly preserve the certified history of Sprints 0 through 8.

`docs/sprints/project_backlog.md` remains operationally relevant until backlog content is migrated or mirrored into the active planning structure.

`docs/sprints/sprint_status_tracker.md` remains the historical certification record for Sprints 0 through 8.

`docs/sprints/execution_roadmap_v2.md` is replaced for forward planning by `docs/active/roadmap_current.md`, while remaining useful historical context.

## Classification Rules

A document is Active when it answers current operational questions.

A document is Reference when it explains rationale.

A document is Archive when it records historical execution, review, certification, or migration context.

## Conflict Rule

When active, reference, and archive documents conflict, active documentation governs current work.