# ME-SA14 - Staging Validation to Generic Coverage Adapter Backlog Entry

Sprint ID: ME-SA14
Status: COMPLETED BY ME-SA14
Job family: ME-SA / Source Acquisition and Source Coverage
Date: 2026-07-03
Architecture layer: Refinery

## Result

ME-SA14 implements the deterministic, fail-closed bridge from cached-source
staging-validation evidence to the generic ME-SA13 coverage classifier input.

```text
staging validation
  -> generic coverage input adapter
  -> ME-SA13 classifier
```

The adapter maps package availability, source support, manifest validity,
provenance, freshness, consumability, completeness, validation issues, and
evidence references without using ticker-specific behavior.

Accepted `company_profile` evidence remains descriptive only.
`sec_companyfacts` evidence remains partial until analytical completeness is
proved by an approved downstream contract.

## Boundaries

The implementation remains Refinery-only. It adds no provider calls,
acquisition, expanded run, Governor behavior, Dispatch Station behavior,
Recommendation Review execution, delivery, portfolio/watchlist writes,
production writes, or Decision Engine authority.

Reserved actionable and Decision Engine-ready states remain unreachable.

## Completed Acceptance Criteria

* Existing staging-validation payloads can be adapted in memory.
* Single-entry and batch APIs are public.
* Batch input order is preserved.
* Ticker and optional market values remain data only.
* Missing, malformed, stale, unsupported, unprovenanced, non-consumable, and
  incomplete evidence fails closed.
* No ticker allowlist or ticker-specific branch exists.
* Staging-validator semantics are unchanged.

## Next Backlog Item

```text
ME-RUN29 - Run expanded generic coverage classification from staging-validation evidence
```

ME-RUN29 is an execution/evidence sprint. It must not add Governor, Dispatch
Station, allocation, recommendation, delivery, or mutation behavior.
