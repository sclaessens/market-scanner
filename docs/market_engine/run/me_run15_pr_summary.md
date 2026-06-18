# ME-RUN15 PR summary

## What changed

ME-RUN15 adds a human-readable command layer for the cached-source batch dry-run runtime.

## Main files

* `src/market_engine/run/cached_source_batch_dry_run_command.py`
* `tests/market_engine/run/test_cached_source_batch_dry_run_command.py`
* `pyproject.toml`

## Command

```bash
market-engine-cached-source-batch-dry-run --source-snapshot-root data/market_engine/source_snapshots --discover-cached-tickers
```

## Validation

Run locally after pulling this branch:

```bash
python -m pytest tests/market_engine/run -q
```
