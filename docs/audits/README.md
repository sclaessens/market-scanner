# Audit Documents

Audit documents are governance evidence.

Current authoritative certification:

- `docs/audits/sprint_0_final_governance_audit.md`

Certified doctrine:

- classification upstream
- allocation downstream
- Decision Engine = ONLY allocation authority

Audit documents may quote legacy terms while documenting findings. Quoted legacy terminology is not active implementation guidance unless explicitly marked as current.

## Mandatory Backlog Reconciliation

Every future sprint audit and implementation audit must include a dedicated section named:

```text
Backlog Impact Assessment
```

The section must explicitly conclude exactly one of:

```text
Backlog impact assessment:
- No new backlog items identified.
```

or:

```text
Backlog impact assessment:
- New backlog items identified and added to project_backlog.md
```

If new deferred work, governance gaps, technical debt, architectural follow-up, operational risks, future sprint candidates, implementation limitations, or non-blocking follow-up work are identified during audit, they must be added to `docs/sprints/project_backlog.md` before sprint closeout.
