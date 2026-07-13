# ME-DATA04 - Complete Canonical Local Market Dataset

## Executive Summary

ME-DATA04 built a materially larger and current local market dataset using real external source data.

End status:

```text
operational_dataset_partial
```

The operational dataset is now large and mostly valid:

- canonical universe increased from 314 to 952 instruments;
- valid local histories increased from 0 to 946;
- valid local history coverage increased from 0.00% to 99.37%;
- missing price history decreased from 13 to 0;
- invalid histories decreased from 1 to 0;
- unsupported mappings decreased from 7 to 0;
- `CLS`, `CRDO`, `IREN`, and `VRT` now have valid local price histories through `2026-07-10`.

The sprint is not `operational_dataset_complete` because ME-EVAL02 still has `resolved: 0`. The blocker changed from missing data to forward-time availability: the latest fully available trading date from the selected price source was `2026-07-10`, while the existing evaluation candidates still require future horizon data after the advice date.

## Source Decision

### Universe Membership

Primary membership sources:

- Wikipedia S&P 500 constituent table;
- Wikipedia S&P MidCap 400 constituent table.

Supplemental sources:

- explicit ME-DATA04 project/evaluation supplement for critical tickers such as `CLS`, `CRDO`, `IREN`, `VRT`, `ASML`, and `TSM`;
- compact ETF context set.

Selection rules:

- include equities from the reproducible table sources;
- include compact ETFs as `asset_type=etf`;
- preserve project/evaluation supplemental instruments;
- normalize provider class-share syntax such as `BRK.B` -> `BRK-B` for acquisition while preserving canonical identity through run-local symbol overrides;
- exclude unsupported rows only when source mapping or acquisition fails closed.

The Nasdaq-100 Wikipedia page was inspected but not used as an operational source because it did not expose a parseable constituent ticker table during this run.

### Price History

Primary price-history source:

```text
Yahoo Finance daily OHLCV via the existing yfinance dependency
```

Acquisition settings:

- daily granularity;
- `auto_adjust=False`;
- `Adj Close` retained when returned;
- start date: `2025-01-01`;
- cutoff date: `2026-07-10`;
- storage: `data/processed/<source_symbol>.csv`;
- validation: existing local CSV validator plus checksum.

The cutoff date `2026-07-10` was selected because it was the latest fully available daily row returned by yfinance during the run on Monday `2026-07-13`. No partial intraday candle was used.

## Licensing And Operational Limits

Wikipedia source tables are public reproducible sources, but they are not official licensed index-provider feeds. Yahoo Finance data is accessed through `yfinance`; Yahoo Finance terms and redistribution limits apply. The dataset is appropriate as a local operational dataset for this repository, not as a claim of licensed commercial index redistribution.

Network access is required for rebuild/update. Unit tests remain offline.

## Baseline

Baseline artifacts:

```text
artifacts/market_engine/data_runs/me-data04-baseline-coverage-20260713T130000Z/
artifacts/market_engine/evaluation_refresh_runs/me-data04-baseline-eval02-refresh-20260713T131000Z/
```

Baseline coverage:

- canonical instruments: 314;
- selected instruments: 314;
- valid histories: 0;
- insufficient histories: 293;
- missing histories: 13;
- invalid histories: 1;
- unsupported mappings: 7;
- valid coverage: 0.00%.

Baseline ME-EVAL02:

- selected outcomes: 12;
- resolved: 0;
- remaining: 12;
- insufficient forward data: 8;
- missing price history: 4.

Baseline missing tickers:

- `CLS`;
- `CRDO`;
- `IREN`;
- `VRT`.

## Canonical Universe Results

ME-DATA04 generated a canonical universe with:

- total instruments: 952;
- unique equities: 943;
- ETFs: 9.

Layer counts:

- `local_price_history_covered`: 952;
- `sp500`: 503;
- `sp400`: 400;
- `explicit_supplemental_watch`: 6;
- `etf_context`: 9.

Universe artifact:

```text
artifacts/market_engine/universe_runs/me-data04-complete-dataset-20260713T133000Z-universe/
```

## Acquisition And Update Flow

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m market_engine.data.complete_local_market_dataset \
  --run-id me-data04-complete-dataset-20260713T133000Z
```

The command:

1. fetches reproducible membership tables;
2. writes run-local membership source files;
3. writes run-local symbol overrides for provider syntax;
4. rebuilds the canonical universe;
5. downloads yfinance daily OHLCV in batches;
6. writes local CSVs under `data/processed`;
7. validates local CSVs;
8. writes acquisition and validation artifacts;
9. runs coverage after acquisition.

Acquisition result:

- requested: 952;
- valid current snapshots from acquisition: 950;
- stale snapshots: 2;
- validation failures: 0;
- acquisition failures: 0.

## Validation Results

Coverage after acquisition:

```text
artifacts/market_engine/data_runs/me-data04-complete-dataset-20260713T133000Z-coverage-after/
```

Final coverage:

- canonical instruments: 952;
- valid histories: 946;
- insufficient histories: 6;
- missing histories: 0;
- invalid histories: 0;
- unsupported mappings: 0;
- valid coverage: 99.37%.

Remaining insufficient histories:

- `BLD`: end date `2026-07-02`;
- `JHG`: end date `2026-07-02`;
- `FDXF`: insufficient history, 31 rows;
- `HONA`: insufficient history, 18 rows;
- `Q`: insufficient history, 176 rows;
- `SOLS`: insufficient history, 181 rows.

## Coverage Before And After

Machine-readable comparison:

```text
artifacts/market_engine/data_runs/me-data04-complete-dataset-20260713T133000Z/before_after_comparison.json
```

| Metric | Before | After | Acceptance | Result |
|---|---:|---:|---:|---|
| Canonical universe | 314 | 952 | >900 | pass |
| Valid local history | 0 | 946 | >90% | pass |
| Current history coverage | 0 | 946 valid / 6 insufficient | materially current | partial |
| ME-EVAL02 resolved | 0 | 0 | >0 | fail |
| CLS resolved | missing | history valid, eval unresolved | investigated/resolved | partial |
| CRDO resolved | missing | history valid, eval unresolved | investigated/resolved | partial |
| IREN resolved | missing | history valid, eval unresolved | investigated/resolved | partial |
| VRT resolved | missing | history valid, eval unresolved | investigated/resolved | partial |

## Critical Tickers

### CLS

- canonical identity: `equity:cls`;
- exchange: NYSE;
- source symbol: `CLS`;
- chosen source: Yahoo Finance via yfinance;
- dataset status: valid local CSV;
- latest price date: `2026-07-10`;
- validation status: valid;
- remaining blocker: ME-EVAL02 still needs future horizon data.

### CRDO

- canonical identity: `equity:crdo`;
- exchange: NASDAQ;
- source symbol: `CRDO`;
- chosen source: Yahoo Finance via yfinance;
- dataset status: valid local CSV;
- latest price date: `2026-07-10`;
- validation status: valid;
- remaining blocker: ME-EVAL02 still needs future horizon data.

### IREN

- canonical identity: `equity:iren`;
- exchange: NASDAQ;
- source symbol: `IREN`;
- chosen source: Yahoo Finance via yfinance;
- dataset status: valid local CSV;
- latest price date: `2026-07-10`;
- validation status: valid;
- remaining blocker: ME-EVAL02 still needs future horizon data.

### VRT

- canonical identity: `equity:vrt`;
- exchange: NYSE;
- source symbol: `VRT`;
- chosen source: Yahoo Finance via yfinance;
- dataset status: valid local CSV;
- latest price date: `2026-07-10`;
- validation status: valid;
- remaining blocker: ME-EVAL02 still needs future horizon data.

## ME-EVAL02 Results

Post-dataset ME-EVAL02 artifact:

```text
artifacts/market_engine/evaluation_refresh_runs/me-data04-post-dataset-eval02-refresh-20260713T134000Z/
```

Before:

- selected outcomes: 12;
- resolved: 0;
- remaining: 12;
- insufficient forward data: 8;
- missing price history: 4.

After:

- selected outcomes: 12;
- resolved: 0;
- remaining: 12;
- insufficient forward data: 12;
- missing price history: 0.

Newly resolved outcomes:

```text
0
```

ME-DATA04 converted all four missing-price-history blockers into available local snapshots, but it did not satisfy the future-horizon requirement for outcome resolution.

## Failures And Exceptions

The run did not meet these hard acceptance criteria:

- ME-EVAL02 `resolved > 0`;
- at least one previously unresolved outcome newly resolved;
- current history coverage has 6 insufficient instruments.

Root cause:

- latest complete trading date available during the run was `2026-07-10`;
- ME-EVAL02 candidates still require forward horizon data after the advice date;
- six instruments had insufficient rows or stale end dates due to source/listing availability.

## Operator Commands

Baseline:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m market_engine.data.supported_universe_price_history_command \
  --universe config/market_engine/universes/canonical_universe.json \
  --output-root data/processed \
  --artifact-root artifacts/market_engine/data_runs \
  --run-id me-data04-baseline-coverage-20260713T130000Z \
  --report-only
```

Dataset build:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m market_engine.data.complete_local_market_dataset \
  --run-id me-data04-complete-dataset-20260713T133000Z
```

Evaluation refresh:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m market_engine.evaluation.advice_outcome_refresh_command \
  --evaluation-artifact artifacts/market_engine/evaluation_runs/me-eval01-advice-outcomes-20260712T120000Z/advice_outcome_index.json \
  --price-history-root data/processed \
  --output-root artifacts/market_engine/evaluation_refresh_runs \
  --run-id me-data04-post-dataset-eval02-refresh-20260713T134000Z
```

## Tests

Tests run:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine/data -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine/evaluation -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q
git diff --check
```

## Changed Files

Primary code:

- `src/market_engine/data/complete_local_market_dataset.py`;
- `tests/market_engine/data/test_complete_local_market_dataset.py`.

Primary generated artifacts:

- `artifacts/market_engine/data_runs/me-data04-baseline-coverage-20260713T130000Z/`;
- `artifacts/market_engine/data_runs/me-data04-complete-dataset-20260713T133000Z/`;
- `artifacts/market_engine/data_runs/me-data04-complete-dataset-20260713T133000Z-coverage-after/`;
- `artifacts/market_engine/evaluation_refresh_runs/me-data04-baseline-eval02-refresh-20260713T131000Z/`;
- `artifacts/market_engine/evaluation_refresh_runs/me-data04-post-dataset-eval02-refresh-20260713T134000Z/`;
- `artifacts/market_engine/universe_runs/me-data04-complete-dataset-20260713T133000Z-universe/`.

## Acceptance Criteria

| Criterion | Result | Status |
|---|---|---|
| canonical universe >900 | 952 | pass |
| valid local history >90% | 99.37% | pass |
| general dataset no longer ends on 2026-04-30 | latest critical and most histories end on 2026-07-10 | pass |
| missing histories for CLS/CRDO/IREN/VRT resolved | all four have valid local CSVs | pass |
| ME-EVAL02 automatically rerun | run completed | pass |
| ME-EVAL02 resolved >0 | resolved remains 0 | fail |
| no synthetic operational data | only yfinance/Wikipedia operational sources used | pass |
| no production side effects | no broker, portfolio, watchlist, Telegram, or orders | pass |

## Remaining Risks

- Wikipedia membership sources are reproducible but not licensed official index feeds.
- yfinance/Yahoo Finance availability, terms, and rate limits can change.
- ME-EVAL02 cannot resolve horizons until future trading days exist after the advice date.
- Six instruments remain insufficient due to short or stale source histories.

## Recommended Next Sprint

Recommended next sprint:

```text
ME-DATA05 - Post-cutoff forward outcome refresh after future trading days become available
```

This sprint should rerun the ME-DATA04 command after enough full trading days are available beyond `2026-07-10`, then rerun ME-EVAL02. It should not rebuild data infrastructure unless the source changes or yfinance access fails.
