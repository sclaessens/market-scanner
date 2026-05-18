# Market Scanner

Governance-certified institutional market scanner and decision pipeline.

## Current Phase

The repository has completed certified architecture purification across Sprints 0 through 8.

The project is now in the operational intelligence platform evolution phase.

Future work should focus on reliability, orchestration, operational visibility, historical decision storage, feedback loops, reporting usability, and production readiness while preserving the certified architecture.

## Core Architecture Doctrine

- classification upstream
- allocation downstream
- Decision Engine = only allocation authority
- reporting communicates only
- no hidden filtering
- no upstream tradeability
- deterministic architecture
- row preservation
- auditability
- separation of concerns

Upstream layers classify and preserve opportunity distribution. They do not determine tradeability, conviction, allocation eligibility, or final actions.

## Active Architecture

The certified active pipeline is:

```text
scanner
  -> validation_layer
  -> context_layer
  -> fundamental_layer
  -> timing_state_layer
  -> portfolio_intelligence_layer
  -> decision_engine
  -> reporting
```

Detailed current-state architecture is maintained in:

- `docs/active/architecture_current_state.md`

## Active Documentation

The operational documentation entry point is:

- `docs/active/`

Key active documents:

- `docs/active/repository_structure.md`
- `docs/active/architecture_current_state.md`
- `docs/active/governance_v2.md`
- `docs/active/simplified_sprint_lifecycle.md`
- `docs/active/operational_development_model.md`
- `docs/active/archive_strategy.md`
- `docs/active/active_reference_archive_classification.md`
- `docs/active/roadmap_current.md`
- `docs/active/repository_cleanup_recommendations.md`
- `docs/active/operational_phase_preparation.md`

## Governance v2

Governance has moved from migration governance to operational governance.

Standard operational changes should use lightweight governance. Architecture review is event-driven and required only when changes may affect authority boundaries, runtime contracts, reporting neutrality, deterministic behavior, or row preservation.

Governance v2 is maintained in:

- `docs/active/governance_v2.md`

## Historical Documentation

Historical sprint and audit documents are preserved for institutional traceability.

The previous `docs/sprints/` and `docs/audits/` folders mainly contain certification history for Sprints 0 through 8. They are no longer the default operational entry point unless explicitly referenced by active documentation.

## Repository Language Governance

The repository language standard is English-only. This applies to documentation, sprint and audit artifacts, technical and functional specifications, source code comments, tests, logs, generated reports, CSV schemas, configuration descriptions, CI output, and governance documents.

Dutch is permitted only in direct chat communication with the user and must not be introduced into repository content or generated artifacts.

## Source of Truth Order

When documentation conflicts:

1. `AGENTS.md` remains the repository-level AI governance authority.
2. `docs/active/architecture_current_state.md` defines current architecture.
3. `docs/active/governance_v2.md` defines current governance.
4. Other `docs/active/` documents define operational workflow and planning.
5. Historical sprint, audit, migration, and archive documents preserve traceability but do not override active documentation.