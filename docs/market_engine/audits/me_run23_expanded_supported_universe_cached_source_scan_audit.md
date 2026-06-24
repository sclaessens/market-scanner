# ME-RUN23 — Expanded supported-universe cached-source scan audit

## Scope audited

ME-RUN23 adds a local-only run layer for the expanded/proposed Professional Swing Universe.

Files covered:

- `src/market_engine/run/expanded_supported_universe_cached_source_scan.py`
- `src/market_engine/run/expanded_supported_universe_cached_source_scan_command.py`
- `src/market_engine/run/__init__.py`
- `tests/market_engine/run/test_expanded_supported_universe_cached_source_scan.py`
- `docs/market_engine/run_reports/me_run23_expanded_supported_universe_cached_source_scan.md`

## Contract chain

ME-RUN23 uses the existing chain:

1. `market-engine-professional-swing-universe-expansion-v1`
2. `market-engine-expanded-professional-swing-source-support-v1`
3. `market-engine-cached-source-batch-dry-run-v1`

The sprint does not define a new universe-entry schema. It wraps existing contracts into `market-engine-expanded-supported-universe-cached-source-scan-v1`.

## Determinism

The implementation preserves deterministic order by using the ME-UNI09 final universe order, then filtering supported cached entries in that order. Optional `ticker_limit` is applied only after the `supported_cached` filter.

## Source-support behavior

All expanded/proposed rows are source-support classified before batch processing. Only rows with `supported_cached` are passed to cached-source batch processing. Non-supported rows are retained in output with explicit status and reason.

Covered non-supported states include:

- `missing_snapshot`
- `unsupported_sec_companyfacts`
- `missing_required_source_field`
- `malformed_or_unreadable_source_artifact`
- `ambiguous_identity`
- `manual_review_only`
- `excluded`

## Fail-closed behavior

The builder raises `ExpandedSupportedUniverseCachedSourceScanError` when ME-UNI09 expansion or ME-SR06 source-support classification fails. When the source-support pass succeeds but produces no `supported_cached` entries, ME-RUN23 returns `blocked_no_supported_cached_entries` without calling the cached-source batch layer.

## Safety audit

The implementation imports no provider/network modules such as `requests`, `urllib`, `socket`, `subprocess`, `yfinance`, `telegram`, or legacy `market_scanner` runtime.

The sprint adds no live data refresh, SEC/EDGAR fetch, broker integration, Telegram delivery, portfolio/write mutation, watchlist mutation, or Decision Engine integration.

The sprint adds no recommendation language, BUY/SELL/HOLD output, target prices, ranking, urgency, conviction, tradeability, allocation guidance, or broker-ready instructions.

## Test coverage added

Tests cover:

1. supported cached entries only are passed into cached-source batch processing;
2. missing snapshots remain visible but are not processed;
3. no-supported-cached case returns a blocked state;
4. deterministic supported ticker ordering;
5. `ticker_limit` applies after source-support filtering;
6. plain payload auditability;
7. CLI human-visible output;
8. no provider/network/action dependencies.

## Local validation requirement

GitHub changes cannot run against Steven's local `.venv` or untracked local `artifacts/` directory. Steven must run the full local test suite and then run the ME-RUN23 command against the current local candidate-classification summary and source snapshots.
