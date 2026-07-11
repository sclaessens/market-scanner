# ME-ADV01 - Minimal Deterministic Advice Engine v1 Audit

## Objective

Produce the first deterministic advice labels from existing ticker status index and dry-run artifacts.

## Board-level product objective

The project must now produce visible investment output, not only readiness/blocker infrastructure. ME-ADV01 turns the ME-GH02 ticker status index and linked dry-run artifacts into one deterministic advice label per ticker.

## Baseline sequence

```text
ME-GH02 -> ME-ADV01 -> ME-ADV02 -> ME-EVAL01 -> ME-APP01
```

## Guardrails

- OpenAI API required: no.
- Provider invocation: no.
- API key required: no.
- Source acquisition: no.
- Broker/order execution: no.
- Portfolio/watchlist mutation: no.
- Telegram/delivery side effects: no.

## Advice labels

- `buy_candidate`: buy candidate for manual review.
- `wait_for_price`: wait for better price or timing context.
- `watchlist`: follow manually, not yet a buy candidate.
- `avoid_for_now`: avoid for now because deterministic blockers or conflict flags are present.
- `hold_existing`: existing position can remain under manual review.
- `take_loss_review`: existing position needs manual loss/risk review.
- `unable_to_advise`: deterministic advice is not possible from available artifacts.

## Implementation

- Modules:
  - `src/market_engine/advice/__init__.py`
  - `src/market_engine/advice/deterministic_advice.py`
  - `src/market_engine/advice/advice_engine_command.py`
- CLI: `python -m market_engine.advice.advice_engine_command`.
- Input: ME-GH02 `ticker_status_index.json`.
- Outputs:
  - `manifest.json`
  - `advice_index.json`
  - `advice_index.md`
  - `advice_summary.json`
  - `unable_to_advise.json`
- Schema versions:
  - `market-engine-advice-index-v1`
  - `market-engine-advice-summary-v1`
  - `market-engine-unable-to-advise-v1`
  - `market-engine-advice-run-manifest-v1`

## Deterministic rules

Rule precedence:

1. Invalid or missing artifact -> `unable_to_advise`.
2. Missing fundamental context -> `unable_to_advise`.
3. Serious unsupported/conflict signal -> `avoid_for_now`.
4. Existing position with loss or elevated risk -> `take_loss_review`.
5. Existing position without loss or elevated-risk flag -> `hold_existing`.
6. Stale context with important missing data -> `unable_to_advise`.
7. Stale context with explicit blockers but usable artifact -> `watchlist`.
8. Actionable or Decision Engine-ready context with setup/price context and no missing data -> `buy_candidate`.
9. Actionable or Decision Engine-ready context with setup/price context but unresolved uncertainty -> `wait_for_price`.
10. Missing setup/price context with valid non-stale artifact -> `watchlist`.
11. Blocked partial analysis with known blockers -> `watchlist`.
12. Valid fallback with incomplete evidence -> `watchlist`.

The rules are deterministic and do not use hidden scoring, model output, external data, or provider calls.

## Sample run

- Input ticker status index: `artifacts/market_engine/batch_status/me-gh02-sample-status-index-20260711T120000Z/ticker_status_index.json`.
- Run ID: `me-adv01-sample-advice-20260711T120000Z`.
- Output directory: `artifacts/market_engine/advice_runs/me-adv01-sample-advice-20260711T120000Z`.
- Tickers total: 12.
- Advice counts:
  - `buy_candidate`: 0
  - `wait_for_price`: 0
  - `watchlist`: 12
  - `avoid_for_now`: 0
  - `hold_existing`: 0
  - `take_loss_review`: 0
  - `unable_to_advise`: 0
- Confidence counts:
  - `low`: 12
  - `medium`: 0
  - `high`: 0
- Unable to advise count: 0.
- Watchlist or better count: 12.
- Buy candidate count: 0.

## Product outcome

ME-ADV01 produced concrete visible advice labels. The first ME-GH02 sample run did not collapse into `unable_to_advise`; all 12 valid, non-stale, partial-analysis tickers were labeled `watchlist` with explicit missing setup/price and portfolio context.

## Tests

- py_compile: passed.
- Advice tests: `tests/market_engine/advice -q` passed, 14 tests.
- No-API/network tests: included in the advice test suite and combined no-API regression.
- Diff check: passed.

## Governance scans

- Provider/API scan: reviewed; no runtime provider/API/source-refresh path added.
- Broker/order/side-effect scan: reviewed; no broker/order, portfolio/watchlist mutation, Telegram, or delivery side effects added.
- Source acquisition scan: reviewed; no source acquisition or live data refresh added.

## Remaining blockers

None for ME-ADV01. The advice labels are intentionally minimal and low-confidence for the sample because setup/price and portfolio context are missing.

## Next sprint

`ME-ADV02 - 500-ticker advice batch output`
