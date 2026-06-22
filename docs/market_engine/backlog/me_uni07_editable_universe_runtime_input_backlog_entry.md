# ME-UNI07 - Editable universe runtime input backlog entry

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI07

## Goal

Wire the editable Professional Swing Universe into local Market Engine runtime input.

## Scope

* Consume the ME-UNI06 Professional Swing Universe loader.
* Select runtime tickers using the approved editable-universe predicate.
* Preserve excluded ticker metadata.
* Preserve explicit no-provider, no-canonical-promotion, and no-execution-authority boundaries.
* Provide argv construction for the existing local cached-source batch command path.
* Add focused tests and sprint documentation.

## Outcome

ME-UNI07 added `src/market_engine/run/editable_universe_runtime_input.py`.

The new module converts validated editable-universe rows into explicit cached-source batch runtime ticker input while preserving governance boundaries.

## Validation

Local validation is required because this connector environment could not access the macOS checkout or run the project `.venv`.

Recommended commands:

```bash
git fetch origin
git checkout me-uni07-wire-editable-universe-runtime-input-connector
git pull origin me-uni07-wire-editable-universe-runtime-input-connector
git diff --check | tee /dev/tty | pbcopy
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/run/test_editable_universe_runtime_input.py tests/market_engine/ticker_universe/test_professional_swing_universe.py -q | tee /dev/tty | pbcopy
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q | tee /dev/tty | pbcopy
```

## Next sprint

Recommended:

```text
ME-SR05 - Classify source support for Professional Swing Universe
```

Optional hardening candidate:

```text
ME-UNI08 - Add first-class professional-swing-universe CLI flag
```

## Non-goals

No provider refresh, SEC/EDGAR live calls, yfinance calls, live market data calls, broker calls, cached-source refresh, reporting delivery, Telegram/email delivery, production writes, portfolio writes, watchlist writes, scheduler behavior, UI behavior, canonical-universe promotion, source-support authority, Decision Engine action semantics, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability, or execution advice.
