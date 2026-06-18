# ME-RUN13 - Safe all-ticker cached-source batch dry-run implementation

## Status

COMPLETED BY ME-RUN13

## Sprint

ME-RUN13 - Implement safe all-ticker cached-source batch dry-run behavior

## Job family

ME-RUN - Run / orchestration jobs

## Purpose

ME-RUN13 implements the ME-RUN12 safe cached-source batch dry-run contract.

The implementation adds a local batch wrapper around the existing ME-RUN10 per-ticker cached-source dry-run path. It runs already-existing local cached SEC CompanyFacts snapshots through the approved per-ticker chain and emits a batch-level summary using:

```text
market-engine-cached-source-batch-dry-run-v1
```

Each successful per-ticker output remains:

```text
market-engine-end-to-end-dry-run-v1
```

## Implemented runtime

```text
src/market_engine/run/cached_source_batch_execution.py
src/market_engine/run/__init__.py
```

## Implemented behavior

ME-RUN13 implements:

* explicit cached-source batch builder;
* deterministic local snapshot discovery under a supplied source snapshot root;
* explicit requested ticker support;
* explicit cached-ticker discovery mode;
* deterministic ticker ordering;
* duplicate requested ticker rejection;
* missing root rejection;
* per-ticker failure isolation;
* missing cached-source ticker blocking;
* invalid cached-source ticker blocking;
* ambiguous cached-source ticker blocking;
* per-ticker execution through the existing cached-source local dry-run builders;
* per-ticker `market-engine-end-to-end-dry-run-v1` preservation;
* batch-level counts and per-ticker result summaries;
* numeric-zero evidence preservation;
* opt-in local batch artifact writing;
* overwrite protection for batch artifact directories.

## Batch output

The batch payload includes:

* `contract_version`;
* `batch_id`;
* `generated_at`;
* `input_mode`;
* `source_mode`;
* `source_snapshot_root`;
* ticker universe metadata;
* `batch_execution_state`;
* `batch_counts`;
* `per_ticker_results`;
* batch warnings and blocked reasons;
* optional artifact manifest reference;
* forbidden side-effect confirmation;
* authority-boundary confirmation;
* provenance;
* explicit `live_provider_call_made=false`;
* explicit non-production marker.

## Artifact behavior

Artifact writing is disabled by default.

When explicitly enabled, artifacts are written only under the supplied local artifact root:

```text
<artifact_output_root>/<batch_id>/batch_manifest.json
<artifact_output_root>/<batch_id>/<ticker>/dry_run.json
<artifact_output_root>/<batch_id>/<ticker>/manifest.json
```

Per-ticker artifact payloads preserve the local dry-run artifact contract identity:

```text
market-engine-local-dry-run-artifact-v1
```

Per-ticker manifests preserve:

```text
market-engine-local-dry-run-artifact-manifest-v1
```

No artifacts are written unless artifact writing is explicitly requested.

## Failure isolation

Ticker-level failures are recorded without stopping the whole batch.

Implemented ticker states include:

* `completed`;
* `completed_with_limitations`;
* `blocked_missing_cached_source`;
* `blocked_ambiguous_cached_source`;
* `blocked_unsupported_cached_source`;
* `blocked_invalid_cached_source`;
* `blocked_downstream_contract_failure`;
* `failed_unexpected_local_error`.

Every blocked or failed ticker records deterministic blocked reasons and does not trigger live provider fallback.

## Tests added

```text
tests/market_engine/run/test_me_run13_cached_source_batch_dry_run.py
```

Tests cover:

* successful batch run over multiple cached tickers;
* deterministic requested ticker ordering;
* deterministic discovery ordering;
* per-ticker failure isolation;
* missing cached-source blocking;
* invalid cached-source blocking;
* ambiguous cached-source blocking;
* batch summary counts;
* batch contract version;
* per-ticker output contract preservation;
* artifact writing disabled by default;
* artifact writing enabled only when explicitly requested;
* artifact overwrite protection;
* numeric-zero evidence preservation;
* forbidden side-effect dependency guardrails.

## Safety boundaries

ME-RUN13 remains cached-source/local-only, deterministic, non-production, provider-free, broker-free, delivery-free, portfolio/write-free, watchlist/write-free, scheduler-free, UI-free, and non-actionable.

ME-RUN13 does not introduce live provider calls, SEC/EDGAR fetches, yfinance calls, live market data calls, broker calls, Telegram/email delivery, production report generation, portfolio writes, watchlist writes, scheduler behavior, UI behavior, automatic cache refresh, automatic cache cleanup, Decision Engine decisions, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, execution advice, ranking, scoring, urgency, conviction, or tradeability authority.

## Limitations

ME-RUN13 implements a runtime function and local artifact writer. It does not add a scheduler, UI, production operator command, source refresh runner, automatic cache cleanup, or production artifact workflow.

## Recommended next sprint

Recommended next sprint:

```text
ME-RUN14 - Add cached-source batch dry-run command interface
```

Rationale: ME-RUN13 implements the safe batch behavior as runtime code. A separate sprint should define and implement a narrow operator-facing command interface if the project wants command-line batch execution beyond direct Python invocation.
