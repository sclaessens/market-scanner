# Sprint Documentation Notice

Sprint documents are living delivery plans and historical execution records.

Post-Sprint-0, all sprint work must inherit the certified governance doctrine:

- classification upstream
- allocation downstream
- Decision Engine = ONLY allocation authority
- upstream layers classify only
- reporting communicates only
- no hidden filtering
- no upstream tradeability

Authoritative Sprint 0 references:

- `docs/sprints/sprint_0_governance_status.md`
- `docs/audits/sprint_0_final_governance_audit.md`
- `AGENTS.md`

Operational sprint lifecycle status is maintained in:

- `docs/sprints/sprint_status_tracker.md`

The roadmap defines doctrine and sequencing. It must not be used as the operational status tracker. Any sprint phase transition, certification, audit, execution review, developer specification approval, implementation completion, implementation audit, closeout, or closure must update `docs/sprints/sprint_status_tracker.md`.

Deferred project work is maintained in:

- `docs/sprints/project_backlog.md`

The backlog is the source of truth for non-blocking improvements, optional corrections, future enhancements, technical debt, research questions, and out-of-scope ideas. Backlog entries require formal sprint governance before implementation and do not authorize runtime, test, data, strategy, architecture, or allocation changes.

Mandatory Backlog Reconciliation is required for all future sprint audits, implementation audits, and closeouts.

Every future audit and closeout document must contain a dedicated section named:

```text
Backlog Impact Assessment
```

That section must explicitly conclude exactly one of:

```text
Backlog impact assessment:
- No new backlog items identified.
```

or:

```text
Backlog impact assessment:
- New backlog items identified and added to project_backlog.md
```

If new deferred work, governance gaps, technical debt, architectural follow-up, operational risks, future sprint candidates, implementation limitations, or non-blocking follow-up work are identified, they must be added to `docs/sprints/project_backlog.md` before sprint closure.

Closed sprint certifications:

- Sprint 1: `docs/sprints/sprint_1_closeout.md`
- Sprint 2: `docs/sprints/sprint_2_closeout.md`
- Sprint 3: `docs/sprints/sprint_3_closeout.md`

If any older sprint document mentions `tradeable_setup`, `context_tradeable`, upstream conviction, execution gating, or filtering-first behavior, interpret it as a historical problem statement or explicit anti-pattern unless the current Sprint 0 status document says otherwise.
