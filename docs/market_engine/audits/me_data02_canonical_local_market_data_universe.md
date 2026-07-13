# ME-DATA02 - Canonical Local Market-data Universe

## Goal

Build a canonical, layered local market-data universe and inspect which instruments have reusable local price-history snapshots.

## ME-EVAL02 Trigger

ME-EVAL02 found 12 unresolved outcomes:

- `insufficient_forward_data`: 8
- `missing_price_history`: 4

The explicit missing local price-history tickers are:

- `CLS`
- `CRDO`
- `IREN`
- `VRT`

## Scope

ME-DATA02 is a datasprint only. It defines a canonical local universe, validates local price-history coverage, provides an incremental operator command, and writes deterministic local data-run artifacts.

## Non-goals

- no advice generation
- no outcome-rule changes
- no recommendation-rule changes
- no machine learning or backtesting framework
- no broker, order, portfolio, watchlist, Telegram, scheduler, daemon, queue, or production deployment
- no historical indexmembership reconstruction
- no fabricated S&P 500, Nasdaq-100, MidCap 400, STOXX, or European membership

## Existing Dataflows Inspected

- `data/market_engine/ticker_universe/ticker_universe.csv`
- `data/market_engine/ticker_universe/professional_swing_universe/professional_swing_universe.csv`
- `data/processed/*.csv`
- ME-DATA01 setup/price context
- ME-EVAL01 outcome evaluation
- ME-EVAL02 refresh flow
- cached-source snapshot import/staging validators

## Universe Architecture

The source of truth is:

```text
config/market_engine/universes/canonical_universe.json
```

Runtime:

- `src/market_engine/data/local_market_data_universe.py`
- `src/market_engine/data/supported_universe_price_history_command.py`

Generated run artifacts:

```text
artifacts/market_engine/data_runs/<run_id>/
```

## Universe Layers

- `local_price_history_covered`: symbols with existing `data/processed/<symbol>.csv`
- `explicit_supplemental_watch`: project, portfolio/watch-relevant, and unresolved outcome tickers
- `europe_large_cap_defined`: controlled European project/large-cap selection
- `etf_context`: compact ETF context set
- `market_context`: compact market-context membership

Blocked layers:

- `sp500`: no reproducible local constituent source present
- `nasdaq100_supplement`: no reproducible local constituent source present
- `sp_midcap400`: no reproducible local constituent source present
- `full_stoxx_europe600`: no reliable local membership or symbol mapping source present

## Counts Per Layer

From `me-data02-full-coverage-report-only-20260712T142000Z`:

- total canonical instruments: 308
- `local_price_history_covered`: 294
- `explicit_supplemental_watch`: 12
- `europe_large_cap_defined`: 4
- `etf_context`: 9
- `market_context`: 3
- unique equities: 299
- ETFs: 9
- context instruments: 3

## Deduplication

Deduplication is by stable `instrument_id`, formatted as:

```text
<asset_type>:<canonical-symbol-lowercase>
```

Multiple memberships are preserved on one canonical entry. Ambiguous source-symbol mappings fail closed.

## Universe Contract

Each canonical entry records:

- `instrument_id`
- `symbol`
- `asset_type`
- `name`
- `exchange`
- `country`
- `currency`
- `sector`
- `industry`
- `universe_memberships`
- `analysis_eligible`
- `advice_eligible`
- `context_only`
- `active`
- `source_symbol`
- `source_notes`
- `source_mapping_status`

## Symbol Mapping Strategy

Canonical symbol and source/acquisition symbol are separate. Overrides are central in `canonical_universe.json`.

Implemented examples:

- `BRK.B` canonical maps to local/source `BRK-B`
- `ASML` remains explicit because the existing local and project artifacts use `ASML`
- unsupported European mappings such as `RHM` and `ADYEN` are blocked instead of guessed

## European Mapping

The sprint does not import full STOXX Europe 600 membership. European support is currently a controlled explicit set with known limitations:

- `ASML`
- `HO`
- `RHM`
- `ADYEN`

`RHM` and `ADYEN` remain unsupported until source mapping is approved.

## ETF And Context Classification

The compact ETF set includes:

- `SPY`
- `QQQ`
- `IWM`
- `DIA`
- `XLK`
- `XLF`
- `XLE`
- `SMH`
- `SOXX`

Market context membership includes `SPY`, `QQQ`, and `GOLD`. Context membership does not create advice authority.

## Point-in-time Limitation

The universe snapshot is current/local and is not a historical point-in-time index constituent database. Survivorship-bias correction and historical membership reconstruction remain out of scope.

## Price-history Contract

ME-DATA02 uses the existing `data/processed/<source_symbol>.csv` layout. Required columns are:

- `Date`
- `Open`
- `High`
- `Low`
- `Close`

Adjusted close is accepted when present but not required by this data-run validator because existing local CSVs include both adjusted and close data inconsistently across sources.

## Minimum History

The minimum history threshold is 252 rows. This matches the existing setup/price context need for moving averages and gives a one-year daily baseline. Forward readiness is measured against the ME-EVAL advice date requirement, configured as `2026-07-11`.

## Acquisition / Import Flow

The command supports operator-supplied local CSV import through `--import-root`, but the real-world ME-DATA02 runs were `--report-only` because no approved operator-supplied new CSV snapshots were present.

Flow:

```text
canonical universe
-> resolve source symbol
-> inspect existing local snapshot
-> classify missing/stale/insufficient/valid/unsupported
-> optionally import operator-supplied local CSV
-> validate
-> write run artifacts
```

## Incremental Behavior

Statuses include:

- `valid_current_snapshot`
- `missing_price_history`
- `insufficient_history`
- `insufficient_forward_data`
- `validation_failed`
- `unsupported_symbol_mapping`
- `acquisition_failed`
- `imported`
- `refreshed`
- `skipped`

## Fail-closed Behavior

- unknown/unsupported mapping blocks
- duplicate canonical symbol blocks
- duplicate source symbol across entities blocks
- invalid CSV becomes `validation_failed`
- empty CSV fails validation
- insufficient history is explicit
- no broker, portfolio, watchlist, advice, or provider fallback is invoked

## Operator Command

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.data.supported_universe_price_history_command \
  --universe config/market_engine/universes/canonical_universe.json \
  --output-root data/processed \
  --artifact-root artifacts/market_engine/data_runs \
  --run-id <run-id> \
  --report-only
```

Optional controls:

- `--tickers`
- `--layer`
- `--limit`
- `--import-root`
- `--skip-valid`
- `--force-refresh`

## Critical Unresolved Outcome Run

Run:

```text
me-data02-critical-unresolved-20260712T140000Z
```

Results:

- selected instruments: 12
- valid: 0
- imported: 0
- refreshed: 0
- missing: 4
- insufficient: 8
- invalid: 0
- unsupported: 0
- failed: 0

Per critical ticker:

- `AMD`: `insufficient_forward_data`, end date `2026-04-30`
- `ASML`: `insufficient_forward_data`, end date `2026-04-30`
- `AVGO`: `insufficient_forward_data`, end date `2026-04-30`
- `CLS`: `missing_price_history`
- `COST`: `insufficient_forward_data`, end date `2026-04-30`
- `CRDO`: `missing_price_history`
- `IREN`: `missing_price_history`
- `META`: `insufficient_forward_data`, end date `2026-04-30`
- `MSFT`: `insufficient_forward_data`, end date `2026-04-30`
- `NVDA`: `insufficient_forward_data`, end date `2026-04-30`
- `TSM`: `insufficient_forward_data`, end date `2026-04-30`
- `VRT`: `missing_price_history`

## Representative Sample Run

Run:

```text
me-data02-representative-sample-20260712T141000Z
```

Results:

- selected instruments: 8
- insufficient: 5
- missing: 2
- unsupported: 1

The sample covered US large/project equity, European explicit mapping, ETFs, context membership, unsupported European mapping, and supplemental unresolved tickers.

## Full Universe Coverage Run

Run:

```text
me-data02-full-coverage-report-only-20260712T142000Z
```

Results:

- total canonical instruments: 308
- selected instruments: 308
- valid: 0
- imported: 0
- refreshed: 0
- skipped: 0
- missing: 12
- insufficient: 293
- invalid: 1
- unsupported: 2
- failed: 0
- completion status: `completed_with_blockers`

## Missing / Unsupported Tickers

Missing local price history:

- `CLS`
- `CRDO`
- `HO`
- `IREN`
- `VRT`
- `DIA`
- `IWM`
- `SMH`
- `SOXX`
- `XLE`
- `XLF`
- `XLK`

Unsupported mappings:

- `ADYEN`
- `RHM`

## ME-EVAL02 Rerun Result

Run:

```text
me-data02-post-data-refresh-check-20260712T143000Z
```

Result:

- selected outcomes: 12
- resolved: 0
- still unresolved: 12
- `insufficient_forward_data`: 8
- `missing_price_history`: 4
- missing tickers: `CLS`, `CRDO`, `IREN`, `VRT`

No new snapshots were imported in ME-DATA02, so ME-EVAL02 outcomes did not change.

## Tests

- py_compile:
  - `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile src/market_engine/data/__init__.py src/market_engine/data/local_market_data_universe.py src/market_engine/data/supported_universe_price_history_command.py`
  - passed
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/data -q`
  - 16 passed
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/evaluation -q`
  - 34 passed
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q`
  - 1650 passed
- `git diff --check`
  - passed

## Governance Checks

- no advice generation
- no provider invocation
- no live acquisition
- no broker/order execution
- no portfolio/watchlist mutation
- no Telegram
- no scheduler, queue, daemon, cron, worker, machine learning, model training, or recommendation-threshold logic
- forbidden runtime grep hits are limited to negative manifest guardrail fields:
  - `provider_invocation_allowed: false`
  - `broker_order_execution_performed: false`

## Known Limitations

- No reproducible local S&P 500, Nasdaq-100, S&P MidCap 400, or STOXX Europe membership source was present.
- Full acquisition/import for approximately 1,000-1,300 instruments was not performed.
- Existing local price-history files generally end at `2026-04-30`, before the ME-EVAL advice date.
- Missing operator-supplied snapshots prevent resolving `CLS`, `CRDO`, `IREN`, and `VRT`.
- One existing local CSV is invalid under the OHLC contract.

## Recommended Next Sprint

`ME-DATA03 - Operator-supplied local price snapshot import for ME-EVAL blockers`

This sprint should provide approved local CSV snapshots for the 12 critical unresolved outcomes, then rerun ME-EVAL02 without changing evaluation code.
