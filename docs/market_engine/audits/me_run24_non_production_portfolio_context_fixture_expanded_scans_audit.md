# ME-RUN24 — Non-production portfolio-context fixture for expanded scans audit

## Scope audited

ME-RUN24 adds explicit non-production portfolio-context fixture support to the expanded supported-universe cached-source scan path.

Files covered:

- `src/market_engine/run/local_portfolio_context_fixture.py`
- `src/market_engine/run/expanded_supported_universe_cached_source_scan.py`
- `src/market_engine/run/expanded_supported_universe_cached_source_scan_command.py`
- `src/market_engine/run/cached_source_batch_dry_run_command.py`
- `tests/market_engine/run/test_expanded_supported_universe_cached_source_scan.py`
- `tests/market_engine/run/test_cached_source_batch_dry_run_command.py`
- `docs/market_engine/run_reports/me_run24_non_production_portfolio_context_fixture_expanded_scans.md`

## Finding

ME-RUN23 proved that the expanded/proposed Professional Swing Universe can be source-support classified and that the `supported_cached` entries can reach cached-source batch execution. The local ME-RUN23 blocker was Portfolio Review missing `portfolio_context`.

ME-RUN24 is limited to that next blocker.

## Contract reuse

The implementation reuses `market-engine-local-portfolio-context-batch-v1`, previously used by the cached-source batch command path. No incompatible fixture contract was introduced.

The shared loader validates:

- existing JSON file;
- JSON object root;
- `portfolio_context_batch_format_version=market-engine-local-portfolio-context-batch-v1`;
- `non_production_local_context=true`;
- no portfolio write authority;
- matching per-ticker context.

Malformed, missing, or unsupported fixture input fails closed.

## Default behavior

Without `--non-production-portfolio-context-fixture`, expanded scans still call cached-source batch execution without portfolio context. The output records `portfolio_context_source=absent`.

## Fixture behavior

With `--non-production-portfolio-context-fixture <path>`, the expanded scan loads fixture contexts for the selected `supported_cached` tickers and passes them to the existing cached-source batch runtime boundary.

The output records:

- `portfolio_context_source=non_production_fixture`;
- fixture path;
- fixture contract;
- context ticker count;
- no broker or live portfolio access;
- no portfolio or watchlist mutation.

## Safety audit

The ME-RUN24 path imports no provider/network modules such as `requests`, `urllib`, `socket`, `subprocess`, `yfinance`, `telegram`, broker integrations, or legacy `market_scanner` runtime.

The sprint adds no provider calls, live market-data calls, broker calls, Telegram delivery, production reporting side effects, portfolio writes, watchlist writes, or Decision Engine action-authority changes.

ME-RUN24 does not canonicalize the universe. It does not add trading/action semantics, target prices, ranking, scoring, urgency, conviction, tradeability, allocation, order, or execution semantics.

## Test coverage added

Tests cover:

1. unchanged default behavior without fixture;
2. explicit non-production fixture acceptance and pass-through;
3. fixture provenance in human-visible output;
4. non-production boundary and no broker/live access confirmations;
5. missing, malformed, and unsupported fixture fail-closed behavior;
6. no provider/network/broker/delivery imports in the fixture path;
7. no forbidden action-authority language in expanded command fixture output;
8. deterministic output for the same input and fixture;
9. reuse of the existing approved local portfolio-context batch contract.

## Local validation required

Exact commands:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/run/test_expanded_supported_universe_cached_source_scan.py tests/market_engine/run/test_cached_source_batch_dry_run_command.py -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q
```
