# ME-RUN29 - Expanded Generic Coverage Classification Backlog Entry

Sprint ID: ME-RUN29
Status: COMPLETED BY ME-RUN29
Job family: ME-RUN / Run and orchestration
Date: 2026-07-03
Architecture layer: Refinery / RUN evidence

## Result

ME-RUN29 executes:

```text
staging-validation fixture evidence
  -> ME-SA14 adapter
  -> ME-SA13 classifier
  -> deterministic JSON and Markdown evidence
```

The run covers seven deterministic evidence rows and reports coverage,
readiness, source-family results, blockers, and reserved-state counts.

Because no committed real staging artifact root exists, the run uses the
committed fixture:

```text
tests/fixtures/market_engine/run/me_run29_staging_validation_evidence.json
```

Generated local output:

```text
artifacts/market_engine/me-run29-expanded-generic-coverage-classification-20260703T000000Z/coverage_classification_summary.json
artifacts/market_engine/me-run29-expanded-generic-coverage-classification-20260703T000000Z/coverage_classification_report.md
```

## Outcome

* accepted company-profile evidence remains descriptive-only;
* accepted SEC CompanyFacts evidence remains partial at the family gate;
* rejected, stale, unprovenanced, invalid-manifest, and unsupported evidence
  fails closed;
* no row is Recommendation Review eligible;
* actionable, actionable-review, decision-ready, and DE-ready counts are zero;
* no ticker-specific runtime branch exists.

## Boundaries

ME-RUN29 adds no provider/live access, acquisition, import, Governor, Dispatch
Station, delivery, portfolio/watchlist mutation, broker behavior, scoring,
ranking, recommendation semantics, allocation, execution, or Decision Engine
authority.

## Next Backlog Item

```text
ME-GV01 - Define The Governor investment evaluation contract
```
