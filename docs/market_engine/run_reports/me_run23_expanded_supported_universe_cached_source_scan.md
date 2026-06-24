# ME-RUN23 — Expanded supported-universe cached-source scan

## Status

Implemented as a local-only, deterministic run layer.

ME-RUN23 introduces `market-engine-expanded-supported-universe-cached-source-scan-v1`.

## Purpose

ME-RUN23 connects the scale-first path:

1. ME-UNI09 builds an expanded/proposed Professional Swing Universe from candidate-classification output.
2. ME-SR06 classifies source support for every expanded/proposed universe row.
3. ME-RUN23 selects only `supported_cached` rows and passes those tickers into the existing cached-source batch dry-run.

Rows that are not `supported_cached` stay visible in the ME-RUN23 output as non-supported entries. They are not passed into cached-source batch processing.

## Runtime entry points

Python builder:

```python
from market_engine.run import build_expanded_supported_universe_cached_source_scan
```

CLI:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.run.expanded_supported_universe_cached_source_scan_command \
  --candidate-classification-summary artifacts/market_engine/<candidate-run>/candidate_classification_summary.json \
  --professional-swing-universe data/market_engine/ticker_universe/professional_swing_universe.csv \
  --source-snapshot-root data/market_engine/source_snapshots \
  --batch-id me-run23-expanded-supported-universe-20260624T120000Z \
  --generated-at 2026-06-24T12:00:00+00:00 \
  --write-local-artifacts \
  --emit-json
```

Generated artifacts remain local artifacts and must not be committed.

## Output

The output contains:

- expanded universe count;
- ME-SR06 source-support summary counts;
- supported cached tickers selected for cached-source batch processing;
- non-supported entries with ticker, status, origin, and reason;
- existing cached-source batch dry-run payload when at least one ticker is `supported_cached`;
- blocked state when no expanded/proposed rows are `supported_cached`.

## Safety boundaries

ME-RUN23 is local cached-source only. It does not perform live provider calls, SEC/EDGAR fetches, yfinance calls, broker calls, Telegram delivery, portfolio mutation, watchlist mutation, or Decision Engine changes.

ME-RUN23 does not add recommendations, BUY/SELL/HOLD language, target prices, ranking, urgency, conviction, tradeability, allocation guidance, or broker-ready instructions.

## Local validation requirement

Because ME-RUN23 consumes local cached source snapshots and local candidate-classification artifacts, the real scan must be run by Steven in the local repository. The PR can validate the runtime contract and tests, but it cannot inspect Steven's untracked local `artifacts/` directory from GitHub.
