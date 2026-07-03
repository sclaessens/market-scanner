# ME-RM04 - The Governor and Dispatch Station Roadmap Audit

Owner roles: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Job family: ME-RM - Roadmap Governance

Status: DOCS-ONLY AUDIT

## Audit summary

ME-RM04 records a roadmap and backlog direction change, not a runtime change.

The update introduces the Market Engine product architecture language:

```text
Boiler
  -> Refinery
  -> Analyzer
  -> The Governor
  -> Dispatch Station
```

It preserves the active ME-SA14 sequence and adds a governed path toward user-facing investment evaluation reports.

## Files added

```text
docs/market_engine/roadmap/me_rm04_governor_dispatch_station_roadmap_update.md
docs/market_engine/backlog/me_rm04_governor_dispatch_station_backlog_entry.md
docs/market_engine/audits/me_rm04_governor_dispatch_station_roadmap_audit.md
```

## Governance checks

### Active story preservation

PASS. ME-SA14 remains the next logical sprint. The roadmap update explicitly forbids dropping or bypassing ME-SA14.

### No dangling started work

PASS. The generic coverage sequence remains intact:

```text
ME-SA12 -> ME-SA13 -> ME-SA14
```

The new Governor / Dispatch Station sequence is inserted after ME-SA14 and after a docs-only ME-RM04 alignment sprint.

### Deferred story preservation

PASS. Existing deferred candidates are preserved and repositioned:

* ME-DL03 and ME-OUT03 become Dispatch Station candidates.
* ME-PR03 remains a Portfolio / Governor input dependency.
* ME-DE03 remains downstream of Governor / Decision Engine authority boundaries.
* ME-CANDIDATE03 remains a Refinery / Analyzer quality candidate.

### Runtime behavior

PASS. No runtime files are changed.

### Tests

Not applicable. This is documentation-only roadmap governance.

### Authority boundary

PASS. The update does not authorize scoring, ranking, BUY / SELL / HOLD action semantics, allocation, position sizing, target prices, buy-zone generation, order generation, broker behavior, Telegram sending, portfolio mutation, watchlist mutation, or Decision Engine decisions.

## Result

PASS. ME-RM04 safely adds the new product direction while keeping the current generic coverage / readiness work intact.
