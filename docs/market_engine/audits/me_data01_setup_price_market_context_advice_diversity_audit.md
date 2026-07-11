# ME-DATA01 - Setup/Price/Market Context for Advice Diversity Audit

## Objective

Close the highest-impact advice data coverage gap exposed by ME-ADV02: missing setup/price/market context.

## Roadmap alignment

This sprint is the concrete implementation of:

```text
ME-DATA01 - Close highest-impact advice data coverage gaps
```

The highest-impact gap is narrowed to:

```text
setup_price_market_context
```

This is not a generic data layer, readiness layer, provider path, or review queue.

## Board-level product objective

Break watchlist-only output by enabling `buy_candidate`, `wait_for_price`, and `avoid_for_now` where evidence supports those labels.

## Guardrails

- OpenAI API required: no
- Provider invocation: no
- API key required: no
- Live source acquisition: no
- Broker/order execution: no
- Portfolio/watchlist mutation: no
- Telegram/delivery side effects: no
- Decision Engine allocation authority changes: no

## Existing data inspection

- dry_run artifacts inspected:
  - 12 ME-GH02 sample dry-run artifacts under `artifacts/market_engine/me-run28-expanded-supported-universe-acquisition-dry-run-classification-20260702T115652Z/`
- setup/price/market fields found:
  - no embedded `setup_price_market_context` or setup-detection payloads were present in the sample dry-run artifacts
  - local price-history CSVs were present under `data/processed/` for 8 of the 12 sample tickers: `AMD`, `ASML`, `AVGO`, `COST`, `META`, `MSFT`, `NVDA`, and `TSM`
- fields missing:
  - local price history was missing for `CLS`, `CRDO`, `IREN`, and `VRT`
  - broad market context remains missing for the 8 tickers with local price history
  - portfolio context remains missing for all 12 sample tickers
- local artifacts reused:
  - existing dry-run artifacts
  - existing local `data/processed/<ticker>.csv` price-history files
- no live acquisition performed:
  - no provider calls
  - no yfinance download
  - no SEC refresh
  - no OpenAI/API call

## Implementation

- modules:
  - `src/market_engine/advice/setup_price_market_context.py`
  - `src/market_engine/advice/deterministic_advice.py`
  - `src/market_engine/advice/advice_batch.py`
  - `src/market_engine/advice/__init__.py`
- extractor:
  - `extract_setup_price_market_context(ticker_status_row, dry_run_payload)`
  - prefers embedded artifact context when present
  - otherwise derives a deterministic partial context from existing local price history
  - fails closed as `missing` or `invalid` when evidence is absent or malformed
- schema:
  - `market-engine-setup-price-market-context-v1`
- advice rule changes:
  - `uptrend` plus constructive setup and `near_entry_zone` or `fair_zone` can produce `buy_candidate`
  - `uptrend` plus constructive setup and `above_preferred_entry` can produce `wait_for_price`
  - `downtrend`, `weak_setup`, `below_support_or_breakdown`, or elevated/high risk can produce `avoid_for_now`
  - missing or inconclusive setup context remains `watchlist` for valid non-stale partial artifacts
  - invalid setup context produces `unable_to_advise`
- batch output changes:
  - advice rows include `setup_price_market_context`
  - markdown reports include `Setup`, `Trend`, `Price position`, and `Risk`
  - `advice_summary.json` includes `setup_price_market_context_counts`
  - `coverage_report.md` includes setup/price/market context distribution
  - `missing_data_report.md` counts setup/price/market context gaps

## Before ME-DATA01

- input run: `me-adv02-advice-batch-20260711T130000Z`
- advice counts:
  - `buy_candidate`: 0
  - `wait_for_price`: 0
  - `watchlist`: 12
  - `avoid_for_now`: 0
  - `hold_existing`: 0
  - `take_loss_review`: 0
  - `unable_to_advise`: 0
- top missing:
  - `portfolio_context`: 12
  - `setup_price_market_context`: 12
- evaluation readiness:
  - false
  - only watchlist labels were produced

## After ME-DATA01

- output run: `me-data01-setup-price-market-context-20260711T140000Z`
- advice counts:
  - `buy_candidate`: 4
  - `wait_for_price`: 2
  - `watchlist`: 5
  - `avoid_for_now`: 1
  - `hold_existing`: 0
  - `take_loss_review`: 0
  - `unable_to_advise`: 0
- confidence counts:
  - `low`: 5
  - `medium`: 7
  - `high`: 0
- top missing:
  - `portfolio_context`: 12
  - `market_context`: 8
  - `local_price_history`: 4
  - `price_level_context`: 4
  - `setup_detection`: 4
  - `setup_price_market_context`: 4
- setup_price_market_context available/partial/missing:
  - `available`: 0
  - `partial`: 8
  - `missing`: 4
  - `invalid`: 0
- evaluation readiness:
  - true
  - at least one outcome-trackable advice label was produced

## Product result

- Did advice diversity improve?
  - Yes. The sample moved from 12 `watchlist` labels to 4 `buy_candidate`, 2 `wait_for_price`, 1 `avoid_for_now`, and 5 `watchlist`.
- Did buy_candidate/wait_for_price/avoid_for_now appear?
  - Yes.
- If not, what exact data remains missing?
  - Not applicable for advice diversity. Remaining data gaps are still explicit: portfolio context for all 12 tickers, broad market context for the 8 tickers with local price history, and local price/setup evidence for `CLS`, `CRDO`, `IREN`, and `VRT`.

## Tests

- py_compile:
  - passed for advice modules including `setup_price_market_context.py`
- advice tests:
  - `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/advice -q`
  - 39 passed
- setup/price/context tests:
  - `tests/market_engine/advice/test_setup_price_market_context.py`
  - `tests/market_engine/advice/test_advice_with_setup_price_market_context.py`
- no-API/no-network tests:
  - included in setup/price/context tests
- diff check:
  - run before commit

## Governance scans

- provider/API/source-refresh:
  - no OpenAI/API/provider/source-refresh behavior is introduced
  - no live acquisition or download is introduced
- broker/order/side-effect:
  - no broker/order execution, portfolio mutation, watchlist mutation, Telegram, or delivery side effects are introduced
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

No ME-DATA01 implementation blocker remains.

Remaining product blockers:

- portfolio context is still missing for all 12 sample tickers;
- broad market context is still missing for the 8 tickers with local price history;
- local price/setup evidence is missing for `CLS`, `CRDO`, `IREN`, and `VRT`;
- status/advice coverage remains 12 of 500 target tickers.

## Next sprint

`ME-EVAL01 - Advice outcome tracking and feedback loop`

ME-DATA01 produced advice diversity and at least one outcome-trackable label. ME-EVAL01 is now useful because `buy_candidate`, `wait_for_price`, and `avoid_for_now` labels exist in the sample output.
