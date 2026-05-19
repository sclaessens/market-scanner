# Operational Sprint 3 Investigation Follow-up — Telegram UX Data Contract Blocker

## 1. Status

Status: INVESTIGATION FOLLOW-UP

Operational Sprint 3 is not closed, not certified complete, and not ready for successful implementation completion.

A local implementation attempt produced a safer operator-first Telegram layout, but flow testing showed that the desired operator Telegram UX cannot currently be populated from source-supported Decision Engine and Reporting data.

No runtime implementation is authorized by this document.

## 2. Investigation Context

Operational Sprint 3 — Telegram UX & Reporting Usability attempted to improve the Telegram output from an audit-oriented reporting summary into an operator-first daily summary.

The attempted layout included:

1. Portfolio
2. Buy Candidates Now
3. Breakout Watch
4. Pullback Watch
5. General Watchlist / Review
6. Run & Traceability

The layout direction is governance-safe, but the current source-supported data is insufficient to populate the intended operator sections without inventing portfolio decisions, buy signals, trigger prices, stop levels, breakout levels, pullback levels, or support levels.

## 3. Flow Test Finding

The generated Telegram message reported:

- no source-supported portfolio-position rows
- no source-supported rows with existing Decision Engine buy action fields
- no source-supported breakout, trigger, or entry boundary fields
- no source-supported pullback or support boundary fields
- all current rows are `REVIEW`

This is not a Telegram formatting failure. It is a data lineage and contract limitation exposed by the Telegram UX attempt.

Telegram is correctly refusing to invent missing portfolio actions or boundary values.

## 4. Root Cause Summary

Portfolio holdings do not reach Reporting as Decision Engine rows.

The current path is:

```text
timing_state_layer.csv -> portfolio_intelligence.csv -> final_decisions.csv -> reporting_dashboard_data.csv -> Telegram
```

Current Portfolio Intelligence preserves and annotates upstream timing/opportunity rows only. It does not add portfolio-only holdings as new rows.

ASML exists in `data/portfolio/portfolio_positions.csv`, but does not appear in current `timing_state_layer.csv`, `portfolio_intelligence.csv`, `final_decisions.csv`, or reporting output.

The active portfolio source contains only ASML, while backup files indicate expected broader holdings: COST, MRVL, ON, and TECK.

The scan runner currently does not rebuild all intermediate artifacts before the Decision Engine. In particular, stale `portfolio_intelligence.csv` can remain the Decision Engine input if `fundamental_quality.csv`, `timing_state_layer.csv`, and `portfolio_intelligence.csv` are not rebuilt before final decisions are produced.

Boundary fields such as entry, stop, target, breakout trigger, support, or pullback may exist in upstream or legacy artifacts, but they are not currently Decision Engine or Reporting supported fields.

## 5. Governance Interpretation

The Telegram layer must remain communication-only.

Telegram must not:

- create portfolio actions
- infer current hold, sell, trim, buy, or review decisions from historical portfolio fields
- treat `last_action` as current Decision Engine authority
- invent trigger prices, stop prices, support levels, pullback zones, or breakout levels
- introduce ranking, scoring, conviction, urgency, tradeability, or hidden filtering

Reporting must continue to communicate source-supported Decision Engine outputs only unless a future governed contract explicitly adds a communication-only portfolio source summary or passes through approved source-supported boundary fields.

## 6. Required Contract Decisions Before OS3 Completion

Operational Sprint 3 should not proceed to implementation completion until the following decisions are made:

1. Authoritative active portfolio source: determine whether active portfolio CSV files must be repaired from backup files or rebuilt from another source of truth.
2. Pipeline freshness: govern the full pipeline sequence so the Decision Engine cannot consume stale Portfolio Intelligence.
3. Portfolio-only holdings contract: decide whether portfolio-only holdings become Decision Engine rows or whether Reporting receives a separate communication-only portfolio source summary.
4. Boundary pass-through contract: decide whether entry, stop, target, breakout, pullback, or support fields should be passed through to Reporting and Telegram as source-supported communication fields.

## 7. Backlog Impact Assessment

New backlog items identified and added to `docs/sprints/project_backlog.md`:

- BL-0011 — Define and repair authoritative active portfolio source
- BL-0012 — Govern full pipeline freshness before Decision Engine execution
- BL-0013 — Decide portfolio-only holdings contract
- BL-0014 — Define governed boundary and trigger pass-through for reporting

Backlog impact assessment:
- New backlog items identified and added to project_backlog.md

## 8. Sprint Status Recommendation

Operational Sprint 3 should remain open and blocked by contract/data decisions.

Recommended tracker state:

- Overall Status: `BLOCKED / CONTRACT DECISION REQUIRED`
- Current Phase: `IMPLEMENTATION INVESTIGATION`
- Governance Status: `LEVEL 2 FOLLOW-UP REQUIRED`
- Current Next Action: `Resolve portfolio source, pipeline freshness, portfolio-row contract, and boundary pass-through decisions before OS3 implementation can proceed`

This does not mark Operational Sprint 3 as implemented, closed, or certified complete.

## 9. Recommended Next Step

Do not continue styling Telegram output until the portfolio source, pipeline freshness, portfolio-only holdings, and boundary pass-through decisions are resolved.

The next implementation work should be proposed as a governed Level 2 design and implementation plan, unless a proposed solution introduces allocation authority, new decision semantics, upstream tradeability, hidden filtering, or reporting-based decision modification. In that case, the work must escalate to Level 3 and requires explicit approval before implementation.
