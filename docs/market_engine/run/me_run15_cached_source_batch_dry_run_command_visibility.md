# ME-RUN15 - Cached-source batch dry-run command visibility implementation

Status: IMPLEMENTED

## Purpose

ME-RUN15 implements the operator-facing command visibility layer defined by ME-RUN14 for the first real cached-source batch dry-run.

The implementation keeps the existing batch runtime contract:

```text
market-engine-cached-source-batch-dry-run-v1
```

It adds the operator visibility contract:

```text
market-engine-real-cached-source-batch-dry-run-visibility-v1
```

## Command

```bash
market-engine-cached-source-batch-dry-run \
  --source-snapshot-root data/market_engine/source_snapshots \
  --discover-cached-tickers
```

Recommended operator capture command:

```bash
market-engine-cached-source-batch-dry-run \
  --source-snapshot-root data/market_engine/source_snapshots \
  --discover-cached-tickers \
  | tee /dev/tty | pbcopy
```

## Supported ticker input modes

The command supports exactly one ticker input mode per run:

```bash
--tickers NVDA,MSFT,AMD
```

```bash
--ticker-file tickers.txt
```

```bash
--discover-cached-tickers
```

## Terminal visibility sections

The command renders these required human-readable sections:

```text
RUN CONTEXT
INPUT DISCOVERY
SELECTED TICKERS
EXECUTION PROGRESS
BATCH SUMMARY
BLOCKED / FAILED TICKERS
ARTIFACTS
FORBIDDEN SIDE-EFFECT CONFIRMATION
NEXT REVIEW ACTIONS
```

## Artifact policy

Artifact writing remains opt-in only.

By default, the command renders human-readable output and does not write local artifacts.

When the operator passes:

```bash
--write-local-artifacts --artifact-output-root artifacts/market_engine
```

then the underlying batch runtime writes local artifacts through the existing ME-RUN13 artifact path.

Generated artifacts are not committed by default.

## Boundary

ME-RUN15 adds command visibility only. It does not add provider refresh, production delivery, portfolio mutation, watchlist mutation, scheduler behavior, UI behavior, ranking, scoring, action labels, allocation, or execution authority.

## Validation

Relevant test target:

```bash
python -m pytest tests/market_engine/run -q
```
