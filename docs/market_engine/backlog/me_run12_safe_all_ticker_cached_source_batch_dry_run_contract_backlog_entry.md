# ME-RUN12 — Safe all-ticker cached-source batch dry-run contract backlog entry

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN12

## Goal

Define the contract for future all-ticker or broader cached-source batch dry-run behavior.

## Scope

Documentation-only contract sprint.

ME-RUN12 defines how a future batch runner may safely orchestrate already-existing cached SEC CompanyFacts source snapshots through the approved cached-source dry-run path.

ME-RUN12 does not implement runtime code, tests, fixtures, provider calls, source refresh behavior, live data behavior, artifacts, delivery, portfolio writes, watchlist writes, scheduler behavior, UI behavior, production report generation, Decision Engine decisions, or action/allocation authority.

## Defined contract

ME-RUN12 defines:

* future batch-level contract family: `market-engine-cached-source-batch-dry-run-v1`;
* approved cached-source root boundary: `data/market_engine/source_snapshots/`;
* future explicit input mode direction: `cached_source_batch`;
* approved ticker universe sources;
* forbidden ticker universe sources;
* deterministic cached-source discovery rules;
* cached-source ambiguity handling rules;
* per-ticker execution as the unit of work;
* per-ticker output preservation as `market-engine-end-to-end-dry-run-v1`;
* per-ticker failure isolation;
* batch summary shape and minimum counts;
* artifact default-off behavior;
* opt-in batch/per-ticker artifact expectations;
* operator visibility requirements;
* missing-data, stale-data, blocked-state, numeric-zero, and provenance requirements;
* fail-closed batch and ticker-level behavior;
* explicit non-goals;
* ME-RUN13 implementation requirements.

## Implemented documentation

```text
docs/market_engine/run/me_run12_safe_all_ticker_cached_source_batch_dry_run_contract.md
docs/market_engine/audits/me_run12_safe_all_ticker_cached_source_batch_dry_run_contract_audit.md
docs/market_engine/backlog/me_run12_safe_all_ticker_cached_source_batch_dry_run_contract_backlog_entry.md
docs/market_engine/roadmap/me_run12_safe_all_ticker_cached_source_batch_dry_run_contract_roadmap_entry.md
```

## Outcome

ME-RUN12 defines the safe future batch contract for local cached-source dry-runs over broader ticker sets.

The future batch wrapper may summarize execution state but may not introduce ranking, scoring, BUY / SELL / HOLD, conviction, urgency, tradeability, target prices, target weights, allocation advice, position sizing, order generation, execution advice, recommendation lists, trading queues, delivery behavior, portfolio mutation, watchlist mutation, production report generation, scheduler behavior, UI behavior, broker behavior, provider refresh, or live data access.

Each ticker remains an isolated execution unit and the final per-ticker output remains `market-engine-end-to-end-dry-run-v1`.

Artifact writing remains opt-in only.

## Next sprint candidate

Recommended next sprint after ME-RUN12:

```text
ME-RUN13 — Implement safe cached-source batch dry-run path
```

ME-RUN13 must preserve the ME-RUN12 contract boundaries and remain local, deterministic, cached-source-only, provider-free, delivery-free, portfolio/write-free, watchlist/write-free, scheduler-free, UI-free, and non-actionable.
