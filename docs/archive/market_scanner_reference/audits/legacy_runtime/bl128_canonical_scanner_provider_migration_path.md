# BL128 — Define canonical scanner/provider migration path

## Sprint Type

Documentation / governance / migration-planning only.

## Sprint Status

Completed.

## Scope

- `scripts/core/data_fetcher.py`
- `scripts/core/scanner.py`
- canonical scanner/provider boundary documentation
- migration path documentation

## Explicit Out of Scope

- Python code changes
- test changes
- archive moves
- fail-close changes
- runtime behavior changes
- provider execution
- yfinance execution
- SEC/EDGAR calls
- production data writes
- report generation
- Telegram delivery
- portfolio state changes
- watchlist state changes
- Decision Engine changes
- `scripts/core/decision_engine.py`

## BL128 Approval Boundary

BL128 approves a canonical migration path only.

BL128 does not approve:

- implementation;
- archive;
- fail-close;
- provider execution;
- yfinance execution;
- source-access implementation;
- production writes;
- tests;
- Python code changes.

## Active `scripts/core` Inventory

- `scripts/core/data_fetcher.py`
- `scripts/core/decision_engine.py`
- `scripts/core/scanner.py`

`scripts/core/decision_engine.py` remains P0 Decision Engine authority and is explicitly out of BL128 scope.

## Canonical Scanner Files Currently Present

- `src/market_scanner/scanner/__init__.py`
- `src/market_scanner/scanner/scanner_boundary.py`
- `src/market_scanner/scanner/scanner_contracts.py`

## Canonical Scanner Boundary Status

`src/market_scanner/scanner/scanner_boundary.py` is side-effect-free and planning-only.

It exposes:

- `build_universe_selection_plan()`
- `build_scanner_plan()`

It defines two stages:

- `universe_selection`
- `candidate_construction`

It sets:

- `provider_calls_allowed=False`
- `data_writes_allowed=False`
- `portfolio_watchlist_mutation_allowed=False`
- `reports_allowed=False`
- `telegram_delivery_allowed=False`

It still lists legacy scanner authorities:

- `archive/legacy_runtime/scripts/run_scan.py`
- `scripts/core/data_fetcher.py`
- `scripts/core/scanner.py`

It has:

- `migration_status="canonical_scanner_boundary_established"`

## Canonical App Evidence

- `src/market_scanner/app.py` imports `build_scanner_plan` from the canonical scanner boundary.
- The canonical app remains dry-run only.
- `run_canonical_app(dry_run=False)` fails closed.
- The app side-effect guarantees state:
  - `provider_calls_made=False`
  - `production_data_writes=False`
  - `reports_generated=False`
  - `telegram_artifacts_created=False`
  - `portfolio_or_watchlist_updates=False`
  - `legacy_runners_invoked=False`

## Existing Runtime-Boundary Documentation

`docs/audits/runtime_boundary/v2_scanner_runtime_boundary_migration.md` states that the canonical scanner boundary is deterministic and dry-run/planning-only.

It explicitly says the scanner boundary does not:

- execute a real scan;
- fetch market data;
- fetch fundamentals;
- read or write production CSVs;
- mutate portfolio/watchlist;
- generate reports;
- send Telegram;
- trigger Decision Engine behavior;
- produce investment recommendations.

It recommends later migrating scanner/universe logic incrementally from `scripts/core/scanner.py` and `scripts/core/data_fetcher.py` into canonical scanner modules with tests.

It says live provider/data access must remain disconnected until an explicit source-access sprint approves it.

## Prior Review Evidence

### BL121

`docs/audits/legacy_runtime/bl121_scanner_provider_boundary_review_remaining_core_scanner_modules.md` classified:

- `scripts/core/data_fetcher.py` as `SCANNER_PROVIDER_BLOCKED`;
- `scripts/core/scanner.py` as `SCANNER_PROVIDER_AND_SCORING_BLOCKED`.

BL121 found:

- no active import references for `scripts.core.data_fetcher` or `scripts.core.scanner`;
- canonical scanner boundary metadata still statically lists both files;
- yfinance provider markers in both files.

BL121 did not approve scanner/provider archive or fail-close.

### BL127

`docs/audits/legacy_runtime/bl127_review_remaining_scanner_provider_core_modules.md` classified:

- `scripts/core/data_fetcher.py` as `SCANNER_PROVIDER_BLOCKED / NOT_ARCHIVE_READY`;
- `scripts/core/scanner.py` as `SCANNER_PROVIDER_AND_SCORING_BLOCKED / CANONICAL_MIGRATION_REQUIRED_BEFORE_ARCHIVE / NOT_ARCHIVE_READY`.

BL127 confirmed:

- no active positive imports were found for the two scoped files;
- no direct runtime entrypoint/main guard was found in either scoped file;
- provider/source-access behavior remains in both scoped files;
- archive and fail-close were not approved for either file.

BL127 recommended BL128 as the scanner/provider source-access governance and canonical migration decision.

## Static Grep Evidence

### `scripts/core/data_fetcher.py`

Observed markers:

- `import yfinance as yf`
- `from config.settings import TICKERS_FILE`
- `load_tickers()`
- `TICKERS_FILE.open(...)`
- `_normalize_columns(...)`
- `fetch_ohlcv_data(...)`
- `yf.download(...)`
- `yf.Ticker(ticker)`
- `ticker_obj.history(...)`

### `scripts/core/scanner.py`

Observed markers:

- `import yfinance as yf`
- `get_sector(...)`
- `yf.Ticker(ticker).info`
- `_unclassified_row(...)`
- `is_liquid_leader(...)`
- `_score_common_components(...)`
- `detect_vcp(...)`
- `build_tradeplan(...)`
- `scan_ticker(...)`
- `_assign_relative_grades(...)`
- `rank_setups(...)`
- `score_trend`
- `score_momentum`
- `score_position`
- `score_relative_strength`
- setup / `primary_setup`
- `entry` / `stop` / `target` / `rr`
- `grade`

## Migration Lanes

### Lane A — Provider/source-access lane for `scripts/core/data_fetcher.py`

- Separate ticker/universe loading from OHLCV provider fetching.
- Ticker loading may become a future side-effect-free canonical universe input contract.
- Direct yfinance execution must not be migrated as implicit canonical runtime behavior.
- A future canonical market-data/source-access boundary is required before any provider behavior is migrated.
- Any live provider access must remain disabled unless a future approved provider/source-access sprint explicitly authorizes it.
- Archive remains blocked until canonical replacement or formal retirement is documented.

### Lane B — Scanner semantics lane for `scripts/core/scanner.py`

- Extract only pure scanner classification/scoring/setup/ranking logic after parity review.
- yfinance sector lookup must not be migrated as an implicit dependency.
- Sector metadata must come from injected input or a governed source in future canonical logic.
- setup classification, score components, liquidity state, discovery state, `rank_setups`, A/B/C grading, and trade-plan fields require contract tests before archive.
- Trade-plan semantics such as `entry`, `stop`, `target`, and `rr` require explicit review because they may affect downstream operator interpretation.

### Lane C — Canonical scanner boundary lane

- The existing `src/market_scanner/scanner/` boundary is metadata/dry-run oriented.
- BL128 does not change it.
- Future work should add side-effect-free contracts first, then migrate pure logic, then separately govern source access.
- Canonical scanner must remain disconnected from provider calls, production writes, reporting, Telegram, portfolio/watchlist mutation, and Decision Engine behavior until separately approved.

### Lane D — Archive-readiness lane

- `scripts/core/data_fetcher.py` is not archive-ready.
- `scripts/core/scanner.py` is not archive-ready.
- Neither should be fail-closed in BL128.
- Neither should be archived in BL128.
- A future archive sprint may only happen after:
  - no active imports;
  - canonical parity or formal retirement is documented;
  - provider/source access is removed, injected, or governed;
  - scanner semantics are contract-tested or formally retired;
  - full suite passes.

## BL128 Decisions

- Approve canonical migration path only.
- Do not approve provider execution.
- Do not approve yfinance execution.
- Do not approve source-access implementation.
- Do not approve archiving.
- Do not approve fail-closing.
- Keep `scripts/core/decision_engine.py` out of scope.
- Confirm active `scripts/core` inventory remains:
  - `scripts/core/data_fetcher.py`
  - `scripts/core/decision_engine.py`
  - `scripts/core/scanner.py`

## Safety Statement

- No live provider calls were run.
- No yfinance calls were run.
- No SEC/EDGAR calls were run.
- No production data writes were performed.
- No report generation was performed.
- No Telegram delivery was performed.
- No portfolio state was changed.
- No watchlist state was changed.
- No Decision Engine behavior was changed.

## Next Recommended Sprint

`BL129 — Establish canonical scanner semantics contracts before script-era scanner migration`

BL129 should remain documentation/test-contract planning only unless later explicitly approved. It should identify pure scanner semantics to contract-test before any migration, fail-close, or archive sprint.
