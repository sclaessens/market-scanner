# ME-RUN24 — Non-production portfolio-context fixture for expanded scans

## Status

Implemented as explicit non-production fixture support for expanded cached-source scans.

## Purpose

ME-RUN23 proved source support and cached-source scan selection for the expanded/proposed Professional Swing Universe:

- contract: `market-engine-expanded-supported-universe-cached-source-scan-v1`
- expanded entries: 53
- `supported_cached`: 12
- `missing_snapshot`: 38
- `manual_review_only`: 3
- supported cached tickers: NVDA, AMD, ASML, META, MSFT, VRT, CLS, CRDO, IREN, COST, AVGO, TSM

The ME-RUN23 local run reached Portfolio Review. The next blocker was missing `portfolio_context`, not source support.

ME-RUN24 only addresses that Portfolio Review blocker by allowing an explicitly supplied non-production portfolio-context fixture to flow into the expanded cached-source scan path.

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
  --batch-id me-run24-expanded-supported-universe-20260624T120000Z \
  --generated-at 2026-06-24T12:00:00+00:00 \
  --non-production-portfolio-context-fixture data/market_engine/portfolio_contexts/local_portfolio_context.json \
  --emit-json
```

The fixture argument is explicit. Default ME-RUN23 behavior remains unchanged when the argument is omitted.

## Reused contract

ME-RUN24 reuses the existing local portfolio-context batch contract:

```text
market-engine-local-portfolio-context-batch-v1
```

It does not create a parallel portfolio-context contract and does not duplicate Portfolio Review logic.

## Output and provenance

The expanded scan payload now reports `portfolio_context`.

When absent:

```text
portfolio_context_source: absent
```

When supplied through the explicit fixture path:

```text
portfolio_context_source: non_production_fixture
source_path: <fixture path>
no_broker_or_live_portfolio_access: true
no_portfolio_or_watchlist_mutation: true
```

## Safety boundaries

ME-RUN24 is non-production fixture support only.

ME-RUN24 does not canonicalize the universe, change source-support classification, introduce provider calls, use broker APIs, access live portfolio data, mutate portfolio state, mutate watchlist state, add Telegram or delivery side effects, add Decision Engine action authority, or add trading/action semantics.

ME-RUN24 does not add target prices, ranking, scoring, urgency, conviction, tradeability, allocation, order, or execution semantics.

All outputs remain non-actionable.

## Local validation

Exact test commands:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/run/test_expanded_supported_universe_cached_source_scan.py tests/market_engine/run/test_cached_source_batch_dry_run_command.py -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q
```
