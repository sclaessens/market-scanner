# ME-BOOT03 - Authoritative Universe And Local Price-history Bootstrap

## Goal

ME-BOOT03 combines the intended ME-UNIV03 and ME-DATA03 responsibilities into one controlled bootstrap sprint:

- establish versioned canonical universe membership sources;
- rebuild the canonical universe from those sources;
- validate local price-history coverage for the rebuilt universe;
- rerun ME-EVAL02 against the existing advice outcome evaluation artifact.

This is a bootstrap and data sprint only.

## Scope

Implemented runtime scope:

- local source snapshot ingestion under `config/market_engine/universes/sources/`;
- central symbol overrides under `config/market_engine/universes/symbol_overrides.json`;
- deterministic canonical universe rebuild through `market_engine.data.canonical_universe_bootstrap_command`;
- universe-run artifacts under `artifacts/market_engine/universe_runs/<run_id>/`;
- extended local price-history run artifacts for imported, refreshed, insufficient, and invalid snapshots;
- checksum and basic OHLC validation for local CSVs;
- ME-EVAL02 refresh rerun from the existing ME-EVAL01 artifact.

Non-goals preserved:

- no new recommendation, advice, setup, evaluation-threshold, ranking, conviction, or allocation logic;
- no advice regeneration;
- no machine learning, model training, optimizer, or backtesting framework;
- no broker, order, Telegram, scheduler, worker, queue, daemon, portfolio, or watchlist mutation;
- no uncontrolled provider architecture, yfinance fallback, broker data, or hidden network fallback;
- no historical index membership reconstruction or survivorship-bias correction.

## Existing Foundation

ME-DATA02 provided the reusable foundation:

- `config/market_engine/universes/canonical_universe.json`;
- `src/market_engine/data/local_market_data_universe.py`;
- `src/market_engine/data/supported_universe_price_history_command.py`;
- data-run artifacts under `artifacts/market_engine/data_runs/`;
- ME-EVAL02 refresh artifacts under `artifacts/market_engine/evaluation_refresh_runs/`.

ME-BOOT03 reuses that foundation and does not introduce a second universe architecture.

## Membership Sources

ME-BOOT03 adds local source snapshots with explicit snapshot dates, retrieval dates, provenance, status, and known limitations:

- `sp500`;
- `nasdaq100`;
- `sp400`;
- `aex`;
- `bel20`;
- `cac40`;
- `dax40`;
- `europe_additional_large_caps`;
- `etf_context`;
- `market_context`;
- `explicit_supplemental_watch`.

The source inventory is written to:

```text
artifacts/market_engine/universe_runs/me-boot03-membership-build-20260713T120000Z/source_inventory.json
```

## Provenance And Limitations

The index layers are controlled local partial snapshots because no complete official S&P 500, Nasdaq-100, S&P MidCap 400, AEX, BEL 20, CAC 40, or DAX 40 constituent files are present in the repository.

Therefore ME-BOOT03 marks implementation complete but price-history and authoritative membership coverage partial. It does not claim full 1,000+ canonical universe coverage.

## Universe Layers And Counts

Membership build run:

```text
artifacts/market_engine/universe_runs/me-boot03-membership-build-20260713T120000Z/
```

Canonical result:

- source snapshots: 11;
- total canonical instruments: 314;
- unique equities: 305;
- ETFs: 9;
- context instruments: 3.

Layer counts:

- `local_price_history_covered`: 294;
- `sp500`: 10;
- `nasdaq100`: 10;
- `sp400`: 6;
- `aex`: 2;
- `bel20`: 2;
- `cac40`: 2;
- `dax40`: 2;
- `europe_additional_large_caps`: 2;
- `etf_context`: 9;
- `market_context`: 3;
- `explicit_supplemental_watch`: 12.

Overlap is retained as multiple memberships on one canonical entry. The overlap report is written to:

```text
artifacts/market_engine/universe_runs/me-boot03-membership-build-20260713T120000Z/overlap_report.json
```

## Deduplication And Instrument Contract

Canonical deduplication remains based on stable `instrument_id`. Each canonical entry keeps:

- canonical symbol;
- source/acquisition symbol;
- asset type;
- exchange;
- country;
- currency;
- sector and industry where available;
- universe memberships;
- analysis and advice eligibility;
- context-only status;
- active status;
- source mapping status;
- source provenance.

Serialization is deterministic and sorted.

## Symbol Mapping

Symbol mapping is centralized in:

```text
config/market_engine/universes/symbol_overrides.json
```

Implemented examples:

- `BRK.B` -> `BRK-B`;
- `BF.B` -> `BF-B`;
- `ASML` retained as `ASML`;
- `ADYEN` -> `ADYEN.AS` but unsupported until operator validation;
- `RHM` -> `RHM.DE` but unsupported until operator validation.

Unsupported mappings in the universe run:

- `ABI`;
- `ADYEN`;
- `MC`;
- `RHM`;
- `RR`;
- `SAP`;
- `UCB`.

Ambiguous or duplicate source symbols fail closed.

## Point-in-time Limitation

The rebuilt universe is a current local snapshot as of `2026-07-13`. It is not a historical index membership database and does not address survivorship bias. A future point-in-time universe sprint is still required before historical backtests can make membership-accurate claims.

## Price-history Contract

ME-BOOT03 continues to use the existing local CSV layout:

```text
data/processed/<source_symbol>.csv
```

Required fields:

- `Date`;
- `Open`;
- `High`;
- `Low`;
- `Close`.

Validation checks:

- parseable dates;
- monotonic and unique dates;
- non-empty rows;
- required OHLC columns and values;
- basic high/low consistency;
- row count;
- forward-data end date;
- file checksum.

Adjusted close remains accepted when present but not required.

## History And Freshness

Existing thresholds are preserved:

- `minimum_analysis_history`: 252 rows;
- `minimum_evaluation_forward_history`: local price history must reach `2026-07-11`;
- freshness and evaluation readiness are reported through existing statuses:
  `valid_current_snapshot`, `missing_price_history`, `insufficient_history`, `insufficient_forward_data`, `validation_failed`, and `unsupported_symbol_mapping`.

## Input Modes

Supported modes:

- operator-supplied local snapshots through the existing `--import-root` command flow;
- report-only scans when no local import root is supplied.

No approved local operator import root was present for ME-BOOT03. No provider fallback was used.

## Import Flow And Incremental Behavior

The existing data command continues to support:

- full universe scans;
- layer filter;
- ticker filter;
- limit;
- resume flag surface;
- skip-valid;
- force-refresh;
- report-only;
- explicit run ID;
- explicit output and artifact roots.

ME-BOOT03 adds complete artifact files for imported, refreshed, insufficient, and invalid snapshots.

## Fail-closed Behavior

Fail-closed behavior is implemented for:

- missing source provenance;
- missing source snapshot date;
- duplicate raw entries;
- duplicate acquisition symbols;
- ambiguous source mappings;
- missing source symbols;
- missing local CSVs;
- invalid local CSV schema;
- duplicate or non-monotonic dates;
- unsupported primary-listing mappings;
- missing operator import root.

No global success status is emitted when critical blockers remain.

## Operator Commands

Membership build:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m market_engine.data.canonical_universe_bootstrap_command \
  --source-root config/market_engine/universes/sources \
  --base-config config/market_engine/universes/canonical_universe.json \
  --price-history-root data/processed \
  --artifact-root artifacts/market_engine/universe_runs \
  --run-id me-boot03-membership-build-20260713T120000Z \
  --symbol-overrides config/market_engine/universes/symbol_overrides.json \
  --write-canonical-config config/market_engine/universes/canonical_universe.json
```

Critical blocker run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m market_engine.data.supported_universe_price_history_command \
  --universe config/market_engine/universes/canonical_universe.json \
  --output-root data/processed \
  --artifact-root artifacts/market_engine/data_runs \
  --run-id me-boot03-critical-blockers-20260713T121000Z \
  --tickers AMD,ASML,AVGO,CLS,COST,CRDO,IREN,META,MSFT,NVDA,TSM,VRT \
  --report-only
```

Full coverage scan:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m market_engine.data.supported_universe_price_history_command \
  --universe config/market_engine/universes/canonical_universe.json \
  --output-root data/processed \
  --artifact-root artifacts/market_engine/data_runs \
  --run-id me-boot03-full-coverage-report-only-20260713T123000Z \
  --report-only
```

ME-EVAL02 refresh:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m market_engine.evaluation.advice_outcome_refresh_command \
  --evaluation-artifact artifacts/market_engine/evaluation_runs/me-eval01-advice-outcomes-20260712T120000Z/advice_outcome_index.json \
  --price-history-root data/processed \
  --output-root artifacts/market_engine/evaluation_refresh_runs \
  --run-id me-boot03-post-bootstrap-eval02-refresh-20260713T124000Z
```

## Critical Blocker Run

Run:

```text
artifacts/market_engine/data_runs/me-boot03-critical-blockers-20260713T121000Z/
```

Result:

- selected instruments: 12;
- valid: 0;
- imported: 0;
- refreshed: 0;
- missing: 4;
- insufficient: 8;
- invalid: 0;
- unsupported: 0.

Missing price history:

- `CLS`;
- `CRDO`;
- `IREN`;
- `VRT`.

Insufficient forward data remains for:

- `AMD`;
- `ASML`;
- `AVGO`;
- `COST`;
- `META`;
- `MSFT`;
- `NVDA`;
- `TSM`.

## Representative Expanded Run

Run:

```text
artifacts/market_engine/data_runs/me-boot03-representative-expanded-20260713T122000Z/
```

Result:

- selected instruments: 10;
- valid: 0;
- missing: 2;
- insufficient: 5;
- unsupported: 3.

The sample covers U.S. large-cap, Nasdaq, MidCap/project, AEX, BEL 20, CAC 40, DAX 40, ETF, market-context, and supplemental memberships.

## Full Coverage Result

Run:

```text
artifacts/market_engine/data_runs/me-boot03-full-coverage-report-only-20260713T123000Z/
```

Coverage:

- total instruments: 314;
- selected instruments: 314;
- valid: 0;
- imported: 0;
- refreshed: 0;
- skipped: 0;
- missing: 13;
- insufficient: 293;
- stale: 0;
- invalid: 1;
- unsupported: 7;
- failed: 0.

The invalid snapshot is a real local schema blocker. Unsupported mappings are European primary-listing mappings that require operator validation.

## ME-EVAL02 Rerun

Run:

```text
artifacts/market_engine/evaluation_refresh_runs/me-boot03-post-bootstrap-eval02-refresh-20260713T124000Z/
```

Result:

- selected outcomes: 12;
- resolved: 0;
- still unresolved: 12;
- insufficient forward data: 8;
- missing price history: 4;
- other blockers: 0.

Missing price-history tickers remain:

- `CLS`;
- `CRDO`;
- `IREN`;
- `VRT`.

The eight existing local snapshots still end on `2026-04-30`, so they remain insufficient for the `2026-07-11` evaluation-forward requirement.

## Tests

Validation performed:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/market_engine/data -q
```

Result:

```text
19 passed
```

Additional full-suite validation was run before PR creation and recorded in the PR summary.

## Governance Checks

ME-BOOT03 preserved:

- classification upstream, allocation downstream;
- Decision Engine as the only allocation authority;
- no portfolio/watchlist mutation;
- no broker or order actions;
- no hidden filtering or recommendation logic changes;
- no advice regeneration;
- deterministic local artifacts.

## Remaining Blockers

Remaining blockers are data availability, not evaluation logic:

- no operator-supplied local price CSVs for `CLS`, `CRDO`, `IREN`, and `VRT`;
- no forward local snapshots past `2026-04-30` for the eight insufficient-forward tickers;
- no complete official constituent source files for full S&P 500, Nasdaq-100, S&P MidCap 400, and European index coverage;
- seven unsupported European primary-listing source mappings.

## Recommended Next Sprint

Recommended next sprint:

```text
ME-DATA04 - Operator-supplied forward price-history import for ME-EVAL blockers
```

Acceptance should require a local `data/import/price_history` package or equivalent operator-supplied root containing validated CSVs for:

- `CLS`;
- `CRDO`;
- `IREN`;
- `VRT`;
- `AMD`;
- `ASML`;
- `AVGO`;
- `COST`;
- `META`;
- `MSFT`;
- `NVDA`;
- `TSM`.

That sprint should rerun the existing ME-BOOT03 import flow without provider fallback and then rerun ME-EVAL02.
