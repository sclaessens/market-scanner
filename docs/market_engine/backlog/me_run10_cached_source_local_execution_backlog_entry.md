# ME-RUN10 - Cached-source local execution backlog entry

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-RUN - Run / orchestration jobs

Status: COMPLETED BY ME-RUN10

## Goal

Implement the cached-source end-to-end local execution path defined by ME-RUN09.

## Scope

Local cached-source input loading, `cached_source_snapshot` command input mode, fail-closed validation, downstream contract construction through approved builders, local tests, documentation, and audit only.

## Implemented

ME-RUN10 added:

* `cached_source_snapshot` dry-run input mode;
* `market-engine-cached-source-local-execution-input-v1` wrapper support;
* cached SEC CompanyFacts snapshot path containment validation;
* cached snapshot to Source Context construction;
* downstream contract construction through Fundamental Observations, Derived Observations, Setup Detection, Analysis Review, Recommendation Review, Portfolio Review, Decision Engine handoff, Delivery / Reporting, and end-to-end dry-run summary;
* explicit local portfolio-context input support;
* opt-in local dry-run artifact support through the existing `--write-local-artifact` flag;
* local synthetic tests for success, failure, provenance, numeric-zero, artifact, and side-effect boundaries.

Implemented runtime:

```text
src/market_engine/run/cached_source_execution.py
src/market_engine/run/end_to_end_dry_run.py
src/market_engine/run/end_to_end_dry_run_command.py
src/market_engine/run/local_dry_run_artifacts.py
src/market_engine/run/__init__.py
```

Implemented tests:

```text
tests/market_engine/run/test_me_run10_cached_source_local_execution.py
```

Implemented documentation:

```text
docs/market_engine/run/me_run10_cached_source_local_execution_implementation.md
docs/market_engine/audits/me_run10_cached_source_local_execution_implementation_audit.md
docs/market_engine/roadmap/me_run10_cached_source_local_execution_roadmap_entry.md
```

## Outcome

ME-RUN10 proves Market Engine can run the local dry-run chain from an already-existing cached SEC CompanyFacts source snapshot and explicitly supplied local portfolio context without live provider calls or production side effects.

The final output remains:

```text
market-engine-end-to-end-dry-run-v1
```

Artifact writing remains opt-in and continues to use:

```text
market-engine-local-dry-run-artifact-v1
market-engine-local-dry-run-artifact-manifest-v1
```

## Boundaries

ME-RUN10 did not introduce provider calls, SEC/EDGAR live calls, yfinance calls, live market data calls, broker calls, Telegram/email delivery, production report generation, portfolio writes, watchlist writes, scheduler behavior, UI behavior, all-ticker production runs, automatic cache refresh, automatic cache cleanup, Decision Engine decisions, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, execution advice, ranking, scoring, urgency, conviction, or tradeability authority.

## Next logical sprint

```text
ME-RUN11 - Run cached-source local execution against a broader deterministic ticker bundle
```

ME-RUN11 should remain local, deterministic, cached-source-only, provider-free, broker-free, delivery-free, portfolio/write-free, watchlist/write-free, scheduler-free, UI-free, and non-actionable.
