# Market Scanner

Governance-certified institutional market scanner and decision-support project.

## Current Phase

The project is in the reset and controlled clean rebuild phase.

RESET-0 decided to proceed with a controlled rebuild. RESET-1 created the canonical v2 documentation baseline. RESET-2 defined the staged repository structure and archive plan. Batch A detaches superseded active documentation from the active source-of-truth path.

New feature work on the old active architecture is paused. Old code, old tests, old generated artifacts, and legacy documentation are preserved as reference material only unless explicitly carried forward by canonical v2 documentation.

## Start Here

For onboarding or before making changes, read the current source-of-truth documents in this order:

1. `AGENTS.md`
2. `docs/active/project_charter.md`
3. `docs/active/product_vision.md`
4. `docs/active/roles_and_responsibilities.md`
5. `docs/active/pm_operating_model.md`
6. `docs/active/technical_architecture.md`
7. `docs/active/data_architecture.md`
8. `docs/active/source_data_strategy.md`
9. `docs/active/pipeline_contract.md`
10. `docs/active/decision_engine_contract.md`
11. `docs/active/reporting_contract.md`
12. `docs/active/testing_strategy.md`
13. `docs/active/repository_structure.md`
14. `docs/active/backlog.md`
15. `docs/active/roadmap.md`

Reset records are preserved in `docs/resets/`.

## Core Architecture Doctrine

- classification upstream
- allocation downstream
- Decision Engine = only allocation authority
- reporting communicates only
- no hidden filtering
- no upstream tradeability
- deterministic behavior
- row preservation where contractually required
- auditability
- explicit contracts before implementation
- separation of concerns

## Active V2 Planning Baseline

The active v2 planning baseline is the RESET-1 canonical documentation set under `docs/active/`.

The intended v2 architecture is contract-first and should be implemented only after the relevant reset stage is approved. Old Python files under `scripts/` are reference-only for v2 and must not be used as the v2 implementation base.

## Legacy Documentation

Superseded active documentation is preserved under:

- `docs/legacy/active_superseded/`

Historical sprint, audit, migration, and archive documents remain preserved for traceability. They do not override the canonical v2 documentation unless a current active document explicitly carries them forward.

## Repository Language Governance

Repository content must remain English-only. This applies to documentation, sprint and audit artifacts, technical and functional specifications, source code comments, tests, logs, generated reports, CSV schemas, configuration descriptions, CI output, and governance documents.

Dutch is permitted only in direct chat communication with the user and must not be introduced into repository content or generated artifacts.

## Source of Truth Order

When documentation conflicts:

1. `AGENTS.md` remains the repository-level AI governance authority until explicitly updated.
2. RESET-1 canonical documents in `docs/active/` define current v2 planning authority.
3. `docs/resets/` preserves reset decisions and closeout records.
4. `docs/legacy/active_superseded/` and older historical documents preserve traceability only.
5. Old code, tests, generated data, reports, workflows, sprint docs, and audit docs do not authorize v2 implementation by themselves.
