# ME-EVAL01 - Advice Outcome Tracking and Feedback Loop Audit

## Objective

Create the first deterministic feedback loop for advice labels by tracking later local price outcomes.

## Board-level product objective

Move from producing advice labels to measuring whether advice labels are useful.

## Input

- advice_index: `artifacts/market_engine/advice_batches/me-data01-setup-price-market-context-20260711T140000Z/advice_index.json`
- advice run id: `me-data01-setup-price-market-context-20260711T140000Z`
- advice generated at: `2026-07-11T14:00:00Z`
- price_data_root: `data/processed`

## Guardrails

- OpenAI API required: no
- Provider invocation: no
- API key required: no
- Live source acquisition: no
- Broker/order execution: no
- Portfolio/watchlist mutation: no
- Telegram/delivery side effects: no
- Decision Engine allocation authority changes: no

## Implementation

- modules:
  - `src/market_engine/evaluation/__init__.py`
  - `src/market_engine/evaluation/advice_outcomes.py`
  - `src/market_engine/evaluation/advice_outcome_command.py`
- CLI:
  - `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.evaluation.advice_outcome_command --advice-index <advice_index.json> --price-data-root data/processed --output-root artifacts/market_engine/evaluation_runs --run-id <run_id>`
- input schema:
  - `market-engine-advice-batch-index-v1`
- output schemas:
  - `market-engine-evaluation-run-manifest-v1`
  - `market-engine-advice-outcome-index-v1`
  - `market-engine-label-performance-summary-v1`
  - `market-engine-unresolved-advice-outcomes-v1`
- horizons:
  - `1w`: 5 trading days
  - `1m`: 21 trading days
  - `3m`: 63 trading days
- price column selection:
  - adjusted close is preferred when present
  - close is used only when adjusted close is unavailable
- unresolved handling:
  - missing CSV becomes `missing_price_history`
  - missing date column becomes `invalid_price_history`
  - missing close/adjusted close becomes `missing_close_price`
  - insufficient local forward rows becomes `insufficient_forward_data`
  - missing advice date becomes `missing_advice_date`

## Sample evaluation run

- run_id: `me-eval01-advice-outcomes-20260712T120000Z`
- output directory: `artifacts/market_engine/evaluation_runs/me-eval01-advice-outcomes-20260712T120000Z`
- tickers_total: 12
- resolved outcomes: 0
- unresolved outcomes: 12
- resolved_by_horizon:
  - `1w`: 0
  - `1m`: 0
  - `3m`: 0
- unresolved_reasons:
  - `insufficient_forward_data`: 8
  - `missing_price_history`: 4
- label_counts:
  - `avoid_for_now`: 1
  - `buy_candidate`: 4
  - `wait_for_price`: 2
  - `watchlist`: 5
- preliminary outcome counts:
  - `unresolved`: 12

## Product result

- Did this create a feedback loop?
  - Yes. ME-EVAL01 records advice labels, advice dates, local entry-price anchors, horizon outcome status, label-level summaries, unresolved reasons, and rule feedback artifacts for each advice batch.
- Could outcomes be resolved with current local data?
  - No. The sample advice date is `2026-07-11`, while the available local price histories for the eight covered tickers do not provide enough later rows for even the 1-week horizon. Four sample tickers have no local price-history CSV under `data/processed`.
- What blocks stronger evaluation?
  - Forward local price snapshots after the advice date are needed for the eight tickers with existing price files.
  - Local price-history CSVs are needed for `CLS`, `CRDO`, `IREN`, and `VRT`.

## Rule feedback

- buy_candidate:
  - 4 labels, 0 resolved, 4 unresolved. Need future/local price history before rule quality can be judged.
- wait_for_price:
  - 2 labels, 0 resolved, 2 unresolved. Need future/local price history before rule quality can be judged.
- avoid_for_now:
  - 1 label, 0 resolved, 1 unresolved. Need future/local price history before rule quality can be judged.
- watchlist:
  - 5 labels, 0 resolved, 5 unresolved. Need future/local price history before rule quality can be judged.

## Tests

- py_compile:
  - `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile src/market_engine/evaluation/__init__.py src/market_engine/evaluation/advice_outcomes.py src/market_engine/evaluation/advice_outcome_command.py`
  - passed
- evaluation tests:
  - `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/evaluation -q`
  - 20 passed
- no-API/no-network tests:
  - included in `tests/market_engine/evaluation/test_advice_outcomes.py`
- targeted no-API/advice/evaluation tests:
  - `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/advisory/test_grounded_advisory_no_api_baseline.py tests/market_engine/advice tests/market_engine/evaluation -q`
  - 60 passed
- broader tests:
  - `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q`
  - 1620 passed
- diff check:
  - `git diff --check`
  - passed

## Governance scans

- provider/API/source-refresh:
  - no runtime provider invocation, OpenAI API usage, source refresh, live acquisition, or download behavior is introduced
  - test references to `urllib.request.urlopen` only assert that network calls are not made
- broker/order/side-effect:
  - no broker/order execution, portfolio mutation, watchlist mutation, Telegram, delivery, or side-effect behavior is introduced
- evaluation output:
  - unresolved outcomes are persisted as product evidence rather than treated as runtime failure
- mandatory scripts scans:
  - the required `scripts/` BUY/SELL/tradeable scans still report pre-existing legacy `scripts/portfolio` and ignored `__pycache__` hits
  - ME-EVAL01 does not modify `scripts/` and does not introduce new script-layer allocation, tradeability, BUY, SELL, or REMOVE behavior

## Remaining blockers

- No ME-EVAL01 implementation blocker remains.
- Current local data cannot resolve the sample horizons yet.
- Four sample tickers lack local price-history CSVs: `CLS`, `CRDO`, `IREN`, and `VRT`.

## Next sprint

`ME-EVAL02 - Scheduled/future outcome refresh using local snapshots`

The primary sample blocker is insufficient forward data after a recent advice date. ME-EVAL02 should rerun the deterministic evaluation against later local snapshots without live acquisition. `ME-DATA02 - Import missing local price history for unresolved outcomes` remains a parallel or follow-up candidate for the four tickers with missing local price-history CSVs.
