# BL127 — Review remaining scanner/provider core modules after logging validation archive

## Sprint Type

Review-only documentation/audit sprint.

## Sprint Status

Completed.

## Scope

- `scripts/core/data_fetcher.py`
- `scripts/core/scanner.py`

## Explicit Out of Scope

- `scripts/core/decision_engine.py`
- archive moves
- fail-close code changes
- runtime behavior changes
- provider execution
- yfinance execution
- SEC/EDGAR calls
- production data writes
- report generation
- Telegram delivery
- portfolio state changes
- watchlist state changes
- Decision Engine behavior changes
- Python code changes
- test changes

## Active `scripts/core` Inventory After BL126

- `scripts/core/data_fetcher.py`
- `scripts/core/decision_engine.py`
- `scripts/core/scanner.py`

`scripts/core/decision_engine.py` is P0 Decision Engine authority and is explicitly out of BL127 scope.

## Active Import Search

Observed grep evidence:

- Search for `scripts.core.data_fetcher` across `src`, `tests`, `scripts`, and `.github` returned no matches.
- Search for `scripts.core.scanner` across `src`, `tests`, `scripts`, and `.github` returned no matches.

Conclusion:

- No active positive imports were found for the two scoped files.

## Runtime Entrypoint/Main-Guard Search

Observed grep evidence:

- Search for `__main__`, `if __name__`, `argparse`, `click`, `typer`, `fire`, and `main(` in `scripts/core/data_fetcher.py` returned no matches.
- Search for `__main__`, `if __name__`, `argparse`, `click`, `typer`, `fire`, and `main(` in `scripts/core/scanner.py` returned no matches.

Conclusion:

- No direct runtime entrypoint was found in either scoped file.

## Provider/Network/Yfinance Markers

### `scripts/core/data_fetcher.py`

Observed markers:

- `import yfinance as yf`
- `def fetch_ohlcv_data(...)`
- `yf.download(...)`
- `yf.Ticker(ticker).history(...)`

### `scripts/core/scanner.py`

Observed markers:

- `import yfinance as yf`
- `yf.Ticker(ticker).info` inside `get_sector(...)`

Conclusion:

- Provider/source-access behavior remains in both scoped files.
- BL127 did not execute providers or yfinance.

## Write-Risk Markers

Observed evidence:

- `scripts/core/data_fetcher.py` reads `TICKERS_FILE` using `open(...)`.
- No scoped `to_csv`, `to_json`, report, Telegram, portfolio, or watchlist write markers were found.

Conclusion:

- No scoped production write markers were identified, but provider/source-access risk remains.

## Scanner/Scoring/Trade-Plan Semantics

Observed `scripts/core/scanner.py` semantics:

- `scan_ticker(...)`
- `detect_vcp(...)`
- `build_tradeplan(...)`
- `rank_setups(...)`
- `_assign_relative_grades(...)`
- `score_trend`
- `score_momentum`
- `score_position`
- `score_relative_strength`
- setup classification
- `primary_setup`
- `entry`
- `stop`
- `target`
- `rr`
- `grade`

Additional tail review:

- Lines 641-720 confirm returned scanner output fields and ranking logic.
- `rank_setups` sorts by setup type, raw score, risk/reward, and ticker.
- `_assign_relative_grades` assigns A/B/C grades.

Conclusion:

- `scripts/core/scanner.py` still contains scanner/scoring/setup/ranking/trade-plan semantics.
- It requires canonical scanner migration/parity or a formal retirement decision before archive.

## Boundary/Reference Documents Observed

- `docs/audits/runtime_boundary/v2_scanner_runtime_boundary_migration.md`
- `docs/audits/legacy_runtime/bl121_scanner_provider_boundary_review_remaining_core_scanner_modules.md`
- `docs/audits/legacy_runtime/bl122_archive_readiness_review_script_era_indicators_helper.md`
- `docs/audits/legacy_runtime/bl126_archive_fail_closed_logging_validation_core_helpers.md`

## Classifications

### `scripts/core/data_fetcher.py`

Classification:

`SCANNER_PROVIDER_BLOCKED / NOT_ARCHIVE_READY`

Rationale:

- yfinance download/history provider access remains.
- ticker-file dependency remains.
- no active imports were found, but provider/source behavior must not be archived or fail-closed casually without a canonical scanner source-access migration decision.

### `scripts/core/scanner.py`

Classification:

`SCANNER_PROVIDER_AND_SCORING_BLOCKED / CANONICAL_MIGRATION_REQUIRED_BEFORE_ARCHIVE / NOT_ARCHIVE_READY`

Rationale:

- yfinance sector lookup remains.
- scanner/scoring/setup/ranking/trade-plan semantics remain.
- canonical scanner migration/parity or a formal retirement decision is required before archive.

## Validation

Full pytest suite:

```text
667 passed in 1.15s
```

Git status:

```text
On branch bl127-review-remaining-scanner-provider-core-modules
nothing to commit, working tree clean
```

## Operator Terminal Noise

During review, the operator accidentally pasted file paths as shell commands:

```text
scripts/core/data_fetcher.py
scripts/core/decision_engine.py
scripts/core/scanner.py
```

This produced shell parse errors such as:

```text
from: command not found
```

This was non-impacting terminal noise. It did not execute Python, did not call providers, and did not write project data.

## Safety Statement

- No live provider calls were intentionally run.
- No yfinance calls were intentionally run.
- No SEC/EDGAR calls were run.
- No production data writes were performed.
- No report generation was performed.
- No Telegram delivery was performed.
- No portfolio state was changed.
- No watchlist state was changed.
- No Decision Engine behavior was changed.

## BL127 Decision

- Do not archive either scoped file in BL127.
- Do not fail-close either scoped file in BL127.
- Keep both scoped files active until a dedicated canonical scanner/provider migration or source-access governance sprint is approved.
- BL127 confirms the remaining active `scripts/core` inventory is:
  - `scripts/core/data_fetcher.py`
  - `scripts/core/decision_engine.py`
  - `scripts/core/scanner.py`
- Decision Engine remains out of scope.

## Next Recommended Sprint

`BL128 — Scanner/provider source-access governance and canonical migration decision`

BL128 should be review-only or governance-only unless explicitly approved otherwise. It should decide whether `scripts/core/data_fetcher.py` and `scripts/core/scanner.py` require canonical migration/parity work, formal retirement, or another controlled cleanup path.
