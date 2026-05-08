# Sprint 0 Governance Cleanup Patch Notes

## Post-Sprint-0 Document Status

Status: HISTORICAL PATCH NOTES

Sprint 0 has been certified COMPLETE. This document preserves interim cleanup notes and may describe states or terms that were later removed during certification. Current governance authority is held by `AGENTS.md`, `docs/sprints/sprint_0_governance_status.md`, and `docs/audits/sprint_0_final_governance_audit.md`.

Certified doctrine:

- classification upstream
- allocation downstream
- Decision Engine = ONLY allocation authority

## Scope
This patch removes the remaining governance leakage identified after the first Sprint 0 migration.

## Files patched
- `scripts/core/build_context_backfill.py`
- `scripts/watchlist/parse_watchlist_commands.py`
- `scripts/watchlist/evaluate_watchlist.py`
- `scripts/watchlist/update_watchlist_actions.py`
- `scripts/run_full_pipeline.py`

## Removed governance violations
- Removed `context_tradeable` from historical context backfill output.
- Removed `context_tradeable_reason` from historical context backfill output.
- Removed `tradeable_count` from context backfill logging.
- Removed watchlist BUY action vocabulary from watchlist parsing and evaluation.
- Replaced watchlist action vocabulary with timing-state vocabulary.
- Removed upstream `urgency` field from watchlist evaluation output.
- Removed legacy watchlist decision-engine updater behavior from full pipeline orchestration.

## New watchlist semantics
The watchlist now expresses timing state, not allocation action.

Examples:
- `READY`
- `WAIT`
- `PULLBACK_PENDING`
- `BREAKOUT_PENDING`
- `STALE`
- `REJECTED`
- `EXPIRED`

## Governance validation expectation
After applying this patch and removing `__pycache__`, these checks should return only explicitly allowed manual transaction vocabulary or no results:

```bash
grep -R "BUY" scripts/ | grep -v decision_engine.py
grep -R "SELL" scripts/ | grep -v decision_engine.py
grep -R "tradeable" scripts/ | grep -v decision_engine.py
grep -R "urgency" scripts/ | grep -v decision_engine.py
```

## Allowed exceptions
Manual portfolio transaction modules may still use BUY/SELL vocabulary because they parse user-entered executed transactions. They are operational transaction records, not system allocation logic.

## Notes
No new edge logic, filters, thresholds, or allocation logic were introduced.
