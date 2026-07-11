# ME-ADV02 - 500-Ticker Advice Batch Output Audit

## Objective

Produce broad deterministic advice batch output from ME-ADV01 across the widest available ticker/status universe, reporting coverage against a 500-ticker target.

## Board-level product objective

The project must produce visible advice output at scale and expose what prevents broader or more diverse advice output.

## Baseline sequence

```text
ME-GH02 -> ME-ADV01 -> ME-ADV02 -> ME-DATA01 -> ME-EVAL01 -> ME-APP01
```

ME-DATA01 is inserted because ME-ADV02 produced visible batch advice output, but the sample output was entirely `watchlist`. The next step must close the specific data gaps that block `buy_candidate`, `wait_for_price`, `avoid_for_now`, `hold_existing`, and `take_loss_review` diversity.

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
  - `src/market_engine/advice/advice_batch.py`
  - `src/market_engine/advice/advice_batch_command.py`
  - `src/market_engine/advice/__init__.py`
- CLI:
  - `python -m market_engine.advice.advice_batch_command`
- input:
  - ME-GH02 `ticker_status_index.json`
  - schema version `market-engine-ticker-status-index-v1`
- outputs:
  - `manifest.json`
  - `advice_index.json`
  - `advice_index.md`
  - `advice_summary.json`
  - `buy_candidates.md`
  - `wait_for_price.md`
  - `watchlist.md`
  - `avoid_for_now.md`
  - `unable_to_advise.md`
  - `missing_data_report.md`
  - `coverage_report.md`
- schema versions:
  - `market-engine-advice-batch-manifest-v1`
  - `market-engine-advice-batch-index-v1`
  - `market-engine-advice-batch-summary-v1`

The batch module reuses ME-ADV01 `build_advice_index` and `render_advice_markdown` instead of duplicating advice decision logic.

## Sample batch run

- input ticker status index: `artifacts/market_engine/batch_status/me-gh02-sample-status-index-20260711T120000Z/ticker_status_index.json`
- target size: 500
- run_id: `me-adv02-advice-batch-20260711T130000Z`
- output directory: `artifacts/market_engine/advice_batches/me-adv02-advice-batch-20260711T130000Z`
- tickers in status index: 12
- tickers with advice: 12
- tickers missing artifact/status versus target: 488
- coverage percentage: 2.40%
- advice counts:
  - `buy_candidate`: 0
  - `wait_for_price`: 0
  - `watchlist`: 12
  - `avoid_for_now`: 0
  - `hold_existing`: 0
  - `take_loss_review`: 0
  - `unable_to_advise`: 0
- confidence counts:
  - `low`: 12
  - `medium`: 0
  - `high`: 0
- top missing inputs:
  - `portfolio_context`: 12
  - `setup_price_market_context`: 12
- evaluation readiness:
  - ready for outcome tracking: false
  - reason: only watchlist labels were produced; price/setup or portfolio context is needed before outcome tracking can evaluate advice quality.
- recommended next sprint:
  - `ME-DATA01 - Close highest-impact advice data coverage gaps`

## Product outcome

- Did this produce broad advice output?
  - Yes. ME-ADV02 produced deterministic batch advice output for every ticker in the widest available ME-GH02 status index.
- Did it expose coverage gaps?
  - Yes. The sample has 12 advice-covered tickers against a 500-ticker target, leaving 488 target tickers without status/advice coverage.
- Did it identify whether evaluation is possible next?
  - Yes. Outcome tracking is not yet useful because every advised ticker is `watchlist`. The batch identified `portfolio_context` and `setup_price_market_context` as the highest-impact blockers for advice diversity.

## Validation

- py_compile:
  - passed for `src/market_engine/advice/__init__.py`, `deterministic_advice.py`, `advice_engine_command.py`, `advice_batch.py`, and `advice_batch_command.py`
- advice tests:
  - `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/advice -q`
  - 26 passed
- batch tests:
  - included in `tests/market_engine/advice/test_advice_batch_output.py`
- no-API/network tests:
  - included in advice batch tests
- diff check:
  - run before commit

## Governance scans

- provider/API/source-refresh:
  - no runtime provider/API/source-refresh calls are introduced in `src/market_engine/advice`
  - expected test strings only guard against forbidden environment or network behavior
- broker/order/side-effect:
  - no broker/order execution, portfolio mutation, watchlist mutation, Telegram, or delivery side effects are introduced
  - guardrail fields explicitly record false side-effect flags
- advice labels:
  - the allowed advice label set remains exactly:
    - `buy_candidate`
    - `wait_for_price`
    - `watchlist`
    - `avoid_for_now`
    - `hold_existing`
    - `take_loss_review`
    - `unable_to_advise`

## Remaining blockers

No ME-ADV02 implementation blocker remains.

The product blocker exposed by the batch output is data coverage and advice diversity:

- only 12 of the 500 target tickers have status/advice coverage;
- all covered tickers are `watchlist`;
- `portfolio_context` and `setup_price_market_context` are missing for all 12 covered tickers.

## Next sprint

`ME-DATA01 - Close highest-impact advice data coverage gaps`

This sprint should be narrowly targeted at the missing data proven by ME-ADV02:

- setup/price/market context needed to move from `watchlist` toward `buy_candidate` or `wait_for_price`;
- portfolio context needed to support `hold_existing` or `take_loss_review`;
- status/advice coverage beyond the current 12-ticker ME-GH02 sample.

ME-EVAL01 should follow once ME-DATA01 produces at least one outcome-trackable advice label such as `buy_candidate`, `wait_for_price`, `avoid_for_now`, `hold_existing`, or `take_loss_review`.
