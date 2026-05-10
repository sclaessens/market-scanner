# Market Scanner

Governance-certified institutional market scanner and decision pipeline.

## Current Architecture Doctrine

- classification upstream
- allocation downstream
- Decision Engine = only allocation authority

Upstream layers classify and preserve opportunity distribution. They do not determine tradeability, conviction, allocation eligibility, or final actions.

## Authoritative Governance Docs

- `AGENTS.md`
- `docs/sprints/sprint_0_governance_status.md`
- `docs/audits/sprint_0_final_governance_audit.md`
- `docs/sprints/sprint_status_tracker.md`
- `docs/sprints/project_backlog.md`

These documents override older sprint plans, archive notes, research drafts, and pre-Sprint-0 analysis where terminology conflicts.

Sprint lifecycle status is maintained in `docs/sprints/sprint_status_tracker.md`. Deferred improvements, optional corrections, technical debt, research questions, and future enhancement ideas are maintained in `docs/sprints/project_backlog.md`. Backlog entries do not authorize implementation.

## Mandatory Backlog Reconciliation

Every future sprint audit, implementation audit, and closeout must include a dedicated `Backlog Impact Assessment` section.

The section must conclude exactly one of:

```text
Backlog impact assessment:
- No new backlog items identified.
```

or:

```text
Backlog impact assessment:
- New backlog items identified and added to project_backlog.md
```

Any newly identified deferred work, governance gaps, technical debt, architectural follow-up, operational risks, future sprint candidates, implementation limitations, or non-blocking follow-up work must be added to `docs/sprints/project_backlog.md` before sprint closure.
