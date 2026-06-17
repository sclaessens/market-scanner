# ME-RUN10 - Cached-source local execution implementation

## Status

COMPLETED BY ME-RUN10

## Sprint

ME-RUN10 - Implement cached-source end-to-end local execution path

## Job family

ME-RUN - Run / orchestration jobs

## Purpose

ME-RUN10 implements the cached-source local execution contract defined by ME-RUN09.

The implementation lets the existing local Market Engine dry-run command build a full `market-engine-end-to-end-dry-run-v1` payload from an already-existing cached SEC CompanyFacts source snapshot and an explicitly supplied local portfolio context.

The path is local, deterministic, non-production, provider-free, broker-free, delivery-free, portfolio/write-free, watchlist/write-free, scheduler-free, UI-free, and non-actionable.

## Implemented runtime

```text
src/market_engine/run/cached_source_execution.py
src/market_engine/run/end_to_end_dry_run.py
src/market_engine/run/end_to_end_dry_run_command.py
src/market_engine/run/local_dry_run_artifacts.py
src/market_engine/run/__init__.py
```

## Implemented input mode

ME-RUN10 adds the approved input mode:

```text
cached_source_snapshot
```

The final dry-run output remains:

```text
market-engine-end-to-end-dry-run-v1
```

Optional local artifact persistence remains:

```text
market-engine-local-dry-run-artifact-v1
market-engine-local-dry-run-artifact-manifest-v1
```

## Cached-source input behavior

The command may consume a direct cached source snapshot path:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m market_engine.run.end_to_end_dry_run_command \
  --input-mode cached_source_snapshot \
  --source-snapshot-json data/market_engine/source_snapshots/<run_id>/raw/<snapshot_id>.json \
  --source-snapshot-root data/market_engine/source_snapshots \
  --portfolio-context-json path/to/local_portfolio_context.json \
  --dry-run-id cached-source-local-run-001 \
  --generated-at 2026-06-17T15:00:00Z
```

The command may also consume a local wrapper file using:

```text
market-engine-cached-source-local-execution-input-v1
```

with `--stage-payloads-json`.

Required wrapper fields:

* `cached_source_local_execution_input_format_version`;
* `input_mode`;
* `non_production_local_execution`;
* `source_snapshot_path`;
* `source_snapshot_root`;
* optional `portfolio_context`.

## Portfolio context input

The cached-source path accepts an explicitly supplied local portfolio context JSON object with:

```text
market-engine-portfolio-context-v1
```

If portfolio context is omitted, downstream Portfolio Review remains responsible for preserving the missing-context limitation and the dry-run remains blocked where the approved contracts require it.

ME-RUN10 does not read production portfolio files and does not mutate portfolio state.

## Execution chain

ME-RUN10 constructs stage payloads through existing approved builders:

```text
cached SEC CompanyFacts snapshot
-> Source Context
-> Fundamental Observations
-> Derived Observations
-> Setup Detection
-> Analysis Review
-> Recommendation Review
-> Portfolio Review
-> Decision Engine handoff
-> Delivery / Reporting
-> End-to-end dry-run summary
```

The command then passes the generated stage payloads through the existing end-to-end dry-run inspector.

## Provenance behavior

Generated stage payloads include cached-source provenance:

* input mode;
* source snapshot path;
* source snapshot reference relative to the configured root when possible;
* configured source snapshot root;
* cached-source local execution input format version.

The final dry-run payload preserves this provenance in the dry-run provenance summary.

## Fail-closed behavior

ME-RUN10 fails closed for:

* missing cached source snapshot path;
* cached source snapshot path outside the configured source root;
* invalid cached source JSON;
* unsupported cached source metadata;
* missing required cached source metadata;
* malformed wrapper input;
* unsupported wrapper input contract;
* missing required portfolio context identity fields when portfolio context is supplied;
* downstream contract construction errors;
* unsafe local artifact persistence errors.

## Artifact behavior

Artifact writing remains disabled by default.

Local artifact writing requires:

```text
--write-local-artifact
```

Example:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m market_engine.run.end_to_end_dry_run_command \
  --input-mode cached_source_snapshot \
  --source-snapshot-json data/market_engine/source_snapshots/<run_id>/raw/<snapshot_id>.json \
  --source-snapshot-root data/market_engine/source_snapshots \
  --portfolio-context-json path/to/local_portfolio_context.json \
  --dry-run-id cached-source-local-run-001 \
  --generated-at 2026-06-17T15:00:00Z \
  --write-local-artifact \
  --artifact-output-root artifacts/market_engine/dry_runs \
  --artifact-created-at 2026-06-17T15:01:00Z
```

Expected artifact path:

```text
artifacts/market_engine/dry_runs/<dry_run_id>/manifest.json
artifacts/market_engine/dry_runs/<dry_run_id>/artifacts/market_engine_dry_run_<dry_run_id>_<date>.json
```

Generated artifacts are local non-production review evidence only.

## Tests added

```text
tests/market_engine/run/test_me_run10_cached_source_local_execution.py
```

Tests cover:

* successful cached-source local execution;
* missing cached source fail-closed behavior;
* malformed cached source fail-closed behavior;
* live provider input mode rejection;
* artifact writing disabled by default;
* artifact writing enabled only through explicit flag;
* cached-source provenance in final payload and artifact;
* wrapper input contract support;
* source path containment validation;
* numeric-zero preservation;
* import guardrails against side-effect dependencies.

## Safety boundaries

ME-RUN10 does not introduce provider calls, SEC/EDGAR live calls, yfinance calls, live market data calls, broker calls, Telegram/email delivery, production report generation, portfolio writes, watchlist writes, scheduler behavior, UI behavior, automatic cache refresh, automatic cache cleanup, all-ticker production runs, Decision Engine decisions, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, execution advice, ranking, scoring, urgency, conviction, or tradeability.

## Limitations

ME-RUN10 implements single cached SEC CompanyFacts snapshot execution with explicitly supplied local portfolio context.

ME-RUN10 does not implement:

* broad ticker bundles;
* all-ticker cached-source execution;
* source refresh jobs;
* automatic source snapshot discovery across a universe;
* automatic portfolio context sourcing;
* terminal progress reporting;
* production persistence.

## Recommended next sprint

Recommended next sprint:

```text
ME-RUN11 - Run cached-source local execution against a broader deterministic ticker bundle
```

Rationale: ME-RUN10 proves the cached-source execution path for one deterministic cached source snapshot. The next safe step is to validate the same path against a small committed deterministic bundle before any broader cached-source or operator-facing workflow is approved.
