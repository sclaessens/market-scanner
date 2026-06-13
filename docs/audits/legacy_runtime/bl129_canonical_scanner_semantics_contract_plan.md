# BL129 — Establish canonical scanner semantics contracts before script-era scanner migration

## Sprint Type

Documentation / test-contract planning only.

## Sprint Status

Completed.

## Scope

- `scripts/core/scanner.py` static semantics inventory
- existing canonical scanner boundary documentation
- existing canonical scanner tests review
- scanner semantics contract planning

## Explicit Out of Scope

- Python code changes
- test code changes
- creating new test files
- changing existing tests
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
- `scripts/core/data_fetcher.py` implementation changes
- `scripts/core/decision_engine.py`

## BL128 Decision Context

`docs/audits/legacy_runtime/bl128_canonical_scanner_provider_migration_path.md` approved a migration path only.

BL128 did not approve:

- implementation;
- archive;
- fail-close;
- provider execution;
- yfinance execution;
- source-access implementation;
- production writes;
- tests;
- Python code changes.

BL128 identified that `scripts/core/scanner.py` requires scanner semantics contracts before migration or archive.

## Existing Canonical Scanner Contracts

`src/market_scanner/scanner/scanner_contracts.py` currently contains only:

- `ScannerStage`
- `ScannerPlan`

These are deterministic boundary/planning records. They are not scanner classification, scoring, or trade-plan semantics contracts.

## Existing Canonical Scanner Boundary Status

`src/market_scanner/scanner/scanner_boundary.py` is side-effect-free and planning-only.

It exposes:

- `build_universe_selection_plan()`
- `build_scanner_plan()`

It defines stages:

- `universe_selection`
- `candidate_construction`

It forbids:

- provider calls
- data writes
- portfolio/watchlist mutation
- reports
- Telegram delivery

It still lists legacy scanner authorities:

- `archive/legacy_runtime/scripts/run_scan.py`
- `scripts/core/data_fetcher.py`
- `scripts/core/scanner.py`

It has:

- `migration_status="canonical_scanner_boundary_established"`

Conclusion:

- The canonical scanner boundary is planning-only.
- It is not yet a semantic scanner implementation.

## Existing Canonical Scanner Tests

`tests/unit/test_v2_canonical_scanner.py` proves:

- scanner plan is deterministic;
- canonical owner and stage order are exposed;
- side effects are forbidden by default;
- scanner boundary import and plan construction create no files;
- scanner boundary does not import legacy scripts;
- scanner plan contains no investment behavior;
- archived legacy runners do not import canonical scanner/app modules.

These tests prove boundary and side-effect guarantees only. They do not prove scanner classification, scoring, ranking, or trade-plan parity.

## Existing Contract Test Convention

`tests/contract/` contains contract-style tests for v2 provider, portfolio, persistence, validation, reporting, Telegram, and other boundaries.

There is no canonical scanner semantics contract test yet.

BL129 plans future scanner semantics contract families only. It does not implement tests.

## Legacy Scanner Semantics Inventory

### A. Sector Metadata

Observed semantics:

- `get_sector(...)`
- yfinance / `yf.Ticker(ticker).info`

Contract planning notes:

- Future canonical logic must not use implicit yfinance sector lookup.
- Sector metadata must become injected input or governed input.

### B. Unclassified Output

Observed semantics:

- `_unclassified_row(...)`
- setup `NONE`
- `primary_setup` blank
- score fields zeroed
- `entry`, `stop`, `target`, and `rr` set to `None`
- `discovery_state`
- `discovery_reason`

Observed fallback reasons:

- `missing_required_columns`
- `insufficient_history`
- `missing_required_metrics`
- `no_setup_detected`

### C. Liquidity

Observed semantics:

- `is_liquid_leader(...)`
- `liquidity_state`
- `LIQUID`
- `ILLIQUID`
- `UNKNOWN`

### D. Scoring

Observed semantics:

- `_score_common_components(...)`
- `score_trend`
- `score_momentum`
- `score_position`
- `score_relative_strength`
- `raw_score`
- `score`
- deterministic rounding rules

### E. Setup Classification

Observed semantics:

- `detect_vcp(...)`
- `scan_ticker(...)`
- `BREAKOUT`
- `PULLBACK`
- `VCP`
- `NONE`
- `primary_setup`
- setup string composition

### F. Trade-Plan Semantics

Observed semantics:

- `build_tradeplan(...)`
- `entry`
- `stop`
- `target`
- `rr`
- null behavior when no valid setup/tradeplan exists

### G. Ranking/Grading

Observed semantics:

- `rank_setups(...)`
- `_assign_relative_grades(...)`
- setup priority sort order
- raw score ranking
- `rr` tie-breaker
- ticker tie-breaker
- A/B/C relative grade assignment

### H. Returned Output Fields

Observed output fields:

- `ticker`
- `date`
- `sector`
- `setup`
- `primary_setup`
- `raw_score`
- `score`
- `score_trend`
- `score_momentum`
- `score_position`
- `score_relative_strength`
- `trend_ok`
- `momentum_ok`
- `liquidity_state`
- `discovery_state`
- `discovery_reason`
- `regime_state`
- `close`
- `ma20`
- `ma50`
- `ma200`
- `atr14`
- `high_20d`
- `low_20d`
- `avg_vol_20`
- `volume_ratio`
- `breakout_strength`
- `extension_atr`
- `ret_20d_pct`
- `rs_20d_pct`
- `atr_pct`
- `entry`
- `stop`
- `target`
- `rr`

## Planned Future Contract Families

### A. Scanner Input Contract

Planned coverage:

- governed OHLCV/indicator-shaped input;
- required columns;
- insufficient-history behavior;
- missing-required-column behavior;
- QQQ/reference return input as explicit injected value;
- sector metadata as injected/governed input, not yfinance.

### B. Scanner Classification Contract

Planned coverage:

- `NONE`, `VCP`, `PULLBACK`, and `BREAKOUT` cases;
- `primary_setup` selection priority;
- setup string composition;
- unclassified fallback reasons.

### C. Score Component Contract

Planned coverage:

- trend, momentum, position, and relative-strength components;
- raw score composition;
- deterministic rounding;
- no Decision Engine recommendation semantics.

### D. Trade-Plan Contract

Planned coverage:

- `entry`, `stop`, `target`, and `rr` field presence;
- null behavior;
- setup-specific formulas requiring explicit future review;
- trade-plan fields as scanner/operator-facing fields, not BUY/SELL/HOLD recommendations.

### E. Ranking/Grading Contract

Planned coverage:

- `rank_setups` ordering;
- setup priority;
- raw score ordering;
- `rr` tie-breaker;
- ticker tie-breaker;
- A/B/C relative grading.

### F. Side-Effect Safety Contract

Planned coverage:

- no yfinance/provider calls;
- no file reads/writes;
- no reports;
- no Telegram;
- no portfolio/watchlist mutation;
- no Decision Engine invocation;
- no investment recommendations.

## Migration Gate

`scripts/core/scanner.py` cannot be migrated, archived, or fail-closed until:

- scanner semantics contract plan is approved;
- future contract tests are implemented and passing;
- yfinance sector lookup is removed or replaced with injected/governed sector input;
- provider/source access remains outside scanner semantics;
- canonical parity or formal retirement is documented;
- full suite passes.

## BL129 Decisions

- Approve scanner semantics contract plan only.
- Do not approve implementation.
- Do not approve tests in BL129.
- Do not approve Python code changes.
- Do not approve yfinance/provider execution.
- Do not approve archive.
- Do not approve fail-close.
- Keep `scripts/core/data_fetcher.py` provider/source-access lane separate.
- Keep `scripts/core/decision_engine.py` out of scope.

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
- No Python code was changed.
- No tests were changed.

## Next Recommended Sprint

`BL130 — Implement canonical scanner semantics contract tests with synthetic inputs`

BL130 should require explicit approval before implementation. It should use synthetic inputs only, prohibit provider execution and yfinance, prohibit archive/fail-close/runtime behavior changes unless separately approved, and keep Decision Engine out of scope.
