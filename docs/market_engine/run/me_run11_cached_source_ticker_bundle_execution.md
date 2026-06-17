# ME-RUN11 - Cached-source ticker bundle execution validation

## Status

COMPLETED BY ME-RUN11

## Sprint

ME-RUN11 - Run cached-source local execution against a broader deterministic ticker bundle

## Job family

ME-RUN - Run / orchestration jobs

## Purpose

ME-RUN11 validates the ME-RUN10 cached-source local execution path against a small deterministic ticker bundle.

The sprint does not add a broad batch runner or production execution path. It proves that the existing `cached_source_snapshot` command path can be invoked ticker-by-ticker for multiple local cached SEC CompanyFacts-like snapshots while preserving the existing dry-run output contract.

## Validated path

ME-RUN11 reuses:

```text
src/market_engine/run/cached_source_execution.py
src/market_engine/run/end_to_end_dry_run_command.py
```

The final per-ticker output remains:

```text
market-engine-end-to-end-dry-run-v1
```

The approved input mode remains:

```text
cached_source_snapshot
```

## Deterministic ticker bundle

The local synthetic test bundle uses:

```text
NVDA
MSFT
AMD
```

Each ticker uses a small deterministic SEC CompanyFacts-like cached source snapshot generated inside the test temporary directory. No live source data is fetched, refreshed, or written to production paths.

The bundle covers:

* a normal complete cached-source path;
* an alternate successful ticker path;
* numeric-zero source evidence and portfolio-context evidence;
* per-ticker cached-source provenance;
* artifact writing disabled by default;
* artifact writing enabled only for one explicitly selected ticker;
* malformed cached-source fail-closed behavior.

## Command shape

The tested command path remains the ME-RUN10 single-ticker command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m market_engine.run.end_to_end_dry_run_command \
  --input-mode cached_source_snapshot \
  --source-snapshot-json <local_snapshot_root>/<run_id>/raw/<ticker>_companyfacts.json \
  --source-snapshot-root <local_snapshot_root> \
  --portfolio-context-json <local_portfolio_context_json> \
  --dry-run-id <dry_run_id> \
  --generated-at 2026-06-17T16:00:00Z
```

Artifact writing remains opt-in:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m market_engine.run.end_to_end_dry_run_command \
  --input-mode cached_source_snapshot \
  --source-snapshot-json <local_snapshot_root>/<run_id>/raw/MSFT_companyfacts.json \
  --source-snapshot-root <local_snapshot_root> \
  --portfolio-context-json <local_portfolio_context_json> \
  --dry-run-id run11-msft-artifact \
  --generated-at 2026-06-17T16:00:00Z \
  --write-local-artifact \
  --artifact-output-root <local_artifact_root> \
  --artifact-created-at 2026-06-17T16:01:00Z
```

Expected local artifact path when explicitly requested:

```text
<local_artifact_root>/run11-msft-artifact/manifest.json
<local_artifact_root>/run11-msft-artifact/artifacts/market_engine_dry_run_run11-msft-artifact_2026-06-17.json
```

## Validation behavior

ME-RUN11 validates that every successful per-ticker payload preserves:

* `dry_run_format_version`;
* `input_mode`;
* ticker and CIK identity;
* completed run state;
* cached-source reference provenance;
* source refresh snapshot ID provenance;
* numeric-zero evidence where present;
* non-actionable dry-run authority boundary.

The malformed snapshot case returns a controlled command failure and does not trigger provider refresh, live provider fallback, or production writes.

## Boundaries preserved

ME-RUN11 remains local, deterministic, non-production, provider-free, broker-free, delivery-free, portfolio/write-free, watchlist/write-free, scheduler-free, UI-free, and non-actionable.

ME-RUN11 does not add provider refresh, SEC/EDGAR live calls, yfinance calls, live market data calls, broker calls, Telegram/email delivery, production reports, portfolio writes, watchlist writes, scheduler behavior, UI behavior, all-ticker production execution, automatic cache refresh, automatic cache cleanup, new financial logic, Decision Engine decisions, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, execution advice, ranking, scoring, urgency, conviction, or tradeability authority.

## Tests added

```text
tests/market_engine/run/test_me_run11_cached_source_ticker_bundle_execution.py
```

The tests cover:

* deterministic three-ticker cached-source execution;
* per-ticker output contract preservation;
* per-ticker cached-source provenance;
* numeric-zero evidence preservation;
* artifact writing disabled by default;
* artifact writing enabled only through `--write-local-artifact`;
* malformed cached-source fail-closed behavior;
* import guardrails against side-effect dependencies.

## Limitations

ME-RUN11 does not add a new batch output contract, all-ticker execution contract, source refresh runner, production cache discovery workflow, operator progress display, or generated production reports.

## Recommended next sprint

Recommended next sprint:

```text
ME-RUN12 - Define safe all-ticker cached-source batch dry-run contract
```

Rationale: ME-RUN11 validates a small deterministic ticker bundle by invoking the approved per-ticker cached-source path. Any broader cached-source batch behavior should be defined as a separate contract before implementation.
