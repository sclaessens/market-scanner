# ME-EVAL02 - Scheduled/Future Outcome Refresh Using Local Snapshots

## Goal

Implement a controlled local refresh flow that re-evaluates existing unresolved advice outcomes when newer local price-history snapshots are available.

## Scope

ME-EVAL02 adds a manual, deterministic, offline operator command that refreshes unresolved outcomes from an existing ME-EVAL01 `advice_outcome_index.json`.

The word `scheduled` means the command can be run later when local snapshots have changed. ME-EVAL02 does not implement a scheduler, daemon, queue, cron integration, background worker, cloud scheduler, provider acquisition flow, or live data collection.

## Non-goals

- no live `yfinance`
- no live price acquisition
- no providers or external API calls
- no SEC/EDGAR refresh
- no broker integration or orders
- no portfolio or watchlist mutation
- no Telegram or delivery side effects
- no production writes
- no scheduler platform
- no new advice rules
- no recommendation logic changes
- no Decision Engine changes

## Existing Blockers

The ME-EVAL01 sample evaluation reported:

- `insufficient_forward_data`: 8
- `missing_price_history`: 4

The missing local price-history tickers are:

- `CLS`
- `CRDO`
- `IREN`
- `VRT`

## Existing Contracts And Artifacts Inspected

- `src/market_engine/evaluation/advice_outcomes.py`
- `src/market_engine/evaluation/advice_outcome_command.py`
- `tests/market_engine/evaluation/test_advice_outcomes.py`
- `tests/market_engine/evaluation/test_advice_outcome_command.py`
- `artifacts/market_engine/evaluation_runs/me-eval01-advice-outcomes-20260712T120000Z/advice_outcome_index.json`
- `artifacts/market_engine/evaluation_runs/me-eval01-advice-outcomes-20260712T120000Z/unresolved_outcomes.json`
- `artifacts/market_engine/advice_batches/me-data01-setup-price-market-context-20260711T140000Z/advice_index.json`
- local price-history CSVs under `data/processed`

## Chosen Refresh Architecture

ME-EVAL02 reuses the ME-EVAL01 evaluation logic rather than adding a parallel evaluation engine.

Implementation:

- `src/market_engine/evaluation/advice_outcome_refresh.py`
- `src/market_engine/evaluation/advice_outcome_refresh_command.py`
- small public helpers added to `src/market_engine/evaluation/advice_outcomes.py`

The refresh flow:

1. loads an existing `market-engine-advice-outcome-index`;
2. selects ticker rows with unresolved horizon outcomes;
3. loads the original advice index path recorded by ME-EVAL01;
4. re-evaluates only the selected tickers against the supplied local price-history root;
5. compares previous status/blocker to refreshed status/blocker;
6. writes a local refresh run artifact.

## Input Contract

Primary input:

```text
artifacts/market_engine/evaluation_runs/<run_id>/advice_outcome_index.json
```

Required fields:

- `run_id`
- `input.advice_index_path`
- `horizons`
- `tickers[].ticker`
- `tickers[].advice`
- `tickers[].outcomes`

The original advice is not regenerated. The recorded `input.advice_index_path` is used only to recover the original advice rows and advice-date anchor for outcome evaluation.

## Selection Of Unresolved Outcomes

An outcome row is selected when any horizon in `tickers[].outcomes` has:

```json
{"status": "unresolved"}
```

The operator may restrict selection with `--tickers`. The filter is deterministic, case-insensitive, and does not change advice semantics.

## Snapshot Selection

For each selected ticker, the refresh flow resolves local price-history snapshots with the existing ME-EVAL01 resolver:

- `data/processed/<ticker>.csv`
- `data/processed/<ticker upper>.csv`
- `data/processed/<ticker lower>.csv`
- `data/processed/<ticker>/*.csv`
- recursive CSV match by ticker stem

The selected snapshot reference and last available local price date are recorded per ticker.

Adjusted close is preferred. Close is used only when adjusted close is unavailable.

## Refresh Semantics

The refresh is retrospective outcome evaluation only:

- the original advice label remains unchanged;
- advice classification is not rerun;
- recommendation logic is not called;
- thresholds and horizons are taken from the existing evaluation artifact;
- newer local price data is only used to determine what happened after the original advice date.

## Fail-closed Behavior

- no local CSV: `missing_price_history`
- local CSV exists but forward rows remain insufficient: `insufficient_forward_data`
- invalid local CSV: explicit blocker such as `invalid_price_history`
- missing original advice context: `missing_evaluation_context`
- ticker selected from unresolved outcomes but absent from the original advice index: `unknown_ticker`
- no live fallback is attempted
- partial runs report per-ticker outcomes and global counts

## Result Contract

Refresh runs are written under:

```text
artifacts/market_engine/evaluation_refresh_runs/<run_id>/
```

Output files:

- `manifest.json`
- `refresh_outcome_index.json`
- `refresh_report.md`
- `missing_price_history.json`

Run-level fields include:

- run type
- run id
- input mode
- previous evaluation artifact
- price-history root
- selected outcome count
- resolved count
- still unresolved count
- `insufficient_forward_data` count
- `missing_price_history` count
- other blocker count
- missing ticker list
- snapshot references

Ticker-level fields include:

- ticker
- advice
- previous evaluation identifier
- previous status
- previous blocker
- used snapshot
- snapshot status
- snapshot last available date
- new status
- new blocker
- resolved boolean
- outcome metrics
- human-readable explanation

## CLI / Operator Flow

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.evaluation.advice_outcome_refresh_command \
  --evaluation-artifact artifacts/market_engine/evaluation_runs/me-eval01-advice-outcomes-20260712T120000Z/advice_outcome_index.json \
  --price-history-root data/processed \
  --output-root artifacts/market_engine/evaluation_refresh_runs \
  --run-id me-eval02-refresh-local-snapshots-20260712T130000Z
```

Optional:

```text
--tickers AMD,NVDA
--allow-overwrite
```

## Real-world Run

- run_id: `me-eval02-refresh-local-snapshots-20260712T130000Z`
- input evaluation artifact: `artifacts/market_engine/evaluation_runs/me-eval01-advice-outcomes-20260712T120000Z/advice_outcome_index.json`
- price-history root: `data/processed`
- output directory: `artifacts/market_engine/evaluation_refresh_runs/me-eval02-refresh-local-snapshots-20260712T130000Z`

Summary:

- selected outcomes: 12
- resolved: 0
- still unresolved: 12
- `insufficient_forward_data`: 8
- `missing_price_history`: 4
- other blockers: 0

## Results Per Ticker / Outcome

| Ticker | Advice | Previous blocker | Snapshot | Last local price date | New blocker |
|---|---|---|---|---|---|
| AMD | wait_for_price | insufficient_forward_data | `data/processed/AMD.csv` | 2026-04-30 | insufficient_forward_data |
| ASML | buy_candidate | insufficient_forward_data | `data/processed/ASML.csv` | 2026-04-30 | insufficient_forward_data |
| AVGO | wait_for_price | insufficient_forward_data | `data/processed/AVGO.csv` | 2026-04-30 | insufficient_forward_data |
| CLS | watchlist | missing_price_history | none |  | missing_price_history |
| COST | buy_candidate | insufficient_forward_data | `data/processed/COST.csv` | 2026-04-30 | insufficient_forward_data |
| CRDO | watchlist | missing_price_history | none |  | missing_price_history |
| IREN | watchlist | missing_price_history | none |  | missing_price_history |
| META | avoid_for_now | insufficient_forward_data | `data/processed/META.csv` | 2026-04-30 | insufficient_forward_data |
| MSFT | watchlist | insufficient_forward_data | `data/processed/MSFT.csv` | 2026-04-30 | insufficient_forward_data |
| NVDA | buy_candidate | insufficient_forward_data | `data/processed/NVDA.csv` | 2026-04-30 | insufficient_forward_data |
| TSM | buy_candidate | insufficient_forward_data | `data/processed/TSM.csv` | 2026-04-30 | insufficient_forward_data |
| VRT | watchlist | missing_price_history | none |  | missing_price_history |

## Explicit Missing-price-history List

- `CLS`
- `CRDO`
- `IREN`
- `VRT`

## Test Results

- py_compile:
  - `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile src/market_engine/evaluation/advice_outcomes.py src/market_engine/evaluation/advice_outcome_refresh.py src/market_engine/evaluation/advice_outcome_refresh_command.py`
  - passed
- ME-EVAL02 tests:
  - `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/evaluation/test_advice_outcome_refresh.py -q`
  - 14 passed
- evaluation tests:
  - `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/evaluation -q`
  - 34 passed
- broader tests:
  - `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q`
  - 1634 passed
- diff check:
  - `git diff --check`
  - passed

## Governance Checks

- no live/provider fetch is implemented
- no `yfinance`, `requests`, `httpx`, provider adapter, SEC/EDGAR, broker, order, Telegram, portfolio write, watchlist write, cron, scheduler, Celery, or APScheduler runtime dependency is introduced
- `scheduler_implemented` is explicitly false in the refresh manifest
- all artifact writes are explicit local evaluation refresh artifacts
- forbidden runtime grep:
  - hits are limited to negative manifest guardrail fields such as `provider_invocation_allowed: false`, `broker_order_execution_performed: false`, and `scheduler_implemented: false`

## Known Limitations

- The real-world run cannot resolve any outcome until local price histories extend beyond the original advice date.
- Four tickers still lack local price-history CSVs.
- ME-EVAL02 is manual/operator-run only and intentionally does not automate periodic execution.

## Recommendation For Next Sprint

`ME-DATA02 - Import missing and forward local price snapshots for unresolved outcomes`

The refresh mechanism now exists and is deterministic. The remaining blocker is data availability, not evaluation refresh mechanics.
