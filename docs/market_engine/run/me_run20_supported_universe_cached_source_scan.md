# ME-RUN20 - Supported-universe cached-source scan

Sprint: ME-RUN20 - Execute clean supported-universe cached-source scan

Branch: `me-run20-execute-clean-supported-universe-cached-source-scan`

Status: Completed

## Purpose

ME-RUN20 executes a clean local cached-source Market Engine scan over the currently supported Professional Swing Universe subset identified by ME-SR05.

The run uses existing cached-source execution and local artifact-writing behavior. It does not introduce runtime code, provider calls, source refresh, delivery, portfolio writes, trading authority, allocation authority, ranking, scoring, or execution behavior.

## Source-Support Input

ME-SR05 classification input:

```text
data/market_engine/ticker_universe/professional_swing_universe/professional_swing_universe.csv
data/market_engine/source_snapshots
```

ME-SR05 classification result:

```text
format_version: market-engine-professional-swing-source-support-v1
entries: 53
supported_count: 12
missing_snapshot_count: 38
unsupported_count: 0
missing_required_source_field_count: 0
malformed_or_unreadable_count: 0
ambiguous_identity_count: 0
manual_review_only_count: 3
excluded_count: 0
```

Supported cached-source tickers:

```text
NVDA, AMD, ASML, META, MSFT, VRT, CLS, CRDO, IREN, COST, AVGO, TSM
```

The current cached-source batch command does not yet have a dedicated `--supported-universe` flag. ME-RUN20 therefore used the existing explicit `--tickers` command path with the ME-SR05-supported subset. This preserves the existing runtime contract and avoids introducing new CLI scope in an execution sprint.

## Command Used

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.run.cached_source_batch_dry_run_command \
  --source-snapshot-root data/market_engine/source_snapshots \
  --tickers NVDA,AMD,ASML,META,MSFT,VRT,CLS,CRDO,IREN,COST,AVGO,TSM \
  --portfolio-context data/market_engine/portfolio_contexts/local_portfolio_context.json \
  --batch-id me-run20-supported-universe-20260623T120000Z \
  --generated-at 2026-06-23T12:00:00Z \
  --write-local-artifacts \
  --artifact-output-root artifacts/market_engine \
  --emit-json
```

## Run Result

Batch contract:

```text
market-engine-cached-source-batch-dry-run-v1
```

Per-ticker output contract:

```text
market-engine-end-to-end-dry-run-v1
```

Counts:

```text
requested: 12
discovered_cached_source: 12
executed: 12
completed: 12
completed_with_limitations: 0
blocked: 0
failed: 0
skipped: 0
missing_cached_source: 0
ambiguous_cached_source: 0
unsupported_cached_source: 0
stale_source: 0
```

All supported cached-source tickers completed.

No blocked, failed, missing, stale, malformed, or ambiguous ticker was observed inside the ME-SR05-supported subset.

## Artifact Paths

Local artifact root:

```text
artifacts/market_engine/me-run20-supported-universe-20260623T120000Z/
```

Batch manifest:

```text
artifacts/market_engine/me-run20-supported-universe-20260623T120000Z/batch_manifest.json
```

Per-ticker dry-run artifacts were written for:

```text
AMD
ASML
AVGO
CLS
COST
CRDO
IREN
META
MSFT
NVDA
TSM
VRT
```

Each ticker directory contains:

```text
dry_run.json
manifest.json
```

Generated artifacts remain local and are not committed by default.

## Artifact Shape Verification

The batch manifest reported:

```text
manifest_contract: market-engine-cached-source-batch-dry-run-v1
batch_execution_state: completed
```

Sample ticker artifacts reported:

```text
NVDA artifact_format_version: market-engine-local-dry-run-artifact-v1
NVDA source_dry_run_format_version: market-engine-end-to-end-dry-run-v1
NVDA source_run_state: dry_run_completed

TSM artifact_format_version: market-engine-local-dry-run-artifact-v1
TSM source_dry_run_format_version: market-engine-end-to-end-dry-run-v1
TSM source_run_state: dry_run_completed
```

## Safety Boundary

The run output confirmed:

```text
No provider, live market data, broker, message-delivery, scheduler, UI, portfolio, watchlist, production-report, or execution side effects are performed.
Decision Engine remains the only future action/allocation authority; the batch dry-run summarizes cached-source local execution state only.
```

ME-RUN20 did not introduce provider calls, SEC or EDGAR live calls, yfinance calls, source refresh, Telegram/email delivery, production reports, portfolio writes, watchlist writes, scheduler behavior, UI behavior, Decision Engine behavior, BUY / SELL / HOLD semantics, allocation advice, target prices, ranking, scoring, urgency, conviction, tradeability, position sizing, order generation, or execution advice.

## Validation

Validation commands:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/run tests/market_engine/source_support -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q
git diff --check
```

Results are recorded in the ME-RUN20 audit and final sprint report.

## Next Sprint

ME-OUT01 - Define readable operator report from dry-run artifacts.
