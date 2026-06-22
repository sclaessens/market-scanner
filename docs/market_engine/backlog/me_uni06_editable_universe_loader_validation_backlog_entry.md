# ME-UNI06 - Editable universe loader and validation backlog entry

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI06

## Goal

Implement the editable Professional Swing Universe loader and validation behavior defined by ME-UNI04 and seeded by ME-UNI05.

## Scope

* Add a runtime loader for `data/market_engine/ticker_universe/professional_swing_universe/professional_swing_universe.csv`.
* Validate required columns, allowed domains, duplicate `(ticker, market)` rows, booleans, integer operator priority, ticker format, and default selection semantics.
* Preserve optional metadata columns.
* Add targeted tests for valid loading, failure modes, default filtering, deterministic ordering, and current seed CSV validation.
* Document implementation and audit results.

## Outcome

ME-UNI06 implemented `load_professional_swing_universe` and `validate_professional_swing_universe` under `src/market_engine/ticker_universe/professional_swing.py`.

The loader preserves the editable universe as a candidate-management source only. It does not promote rows to the canonical execution universe or source-supported universe.

## Validation

Targeted isolated tests passed in this execution environment:

```text
29 passed in 0.16s
```

Full repository validation remains recommended before merge in the local checkout.

## Next sprint

```text
ME-SR05 - Classify source support for Professional Swing Universe
```

## Non-goals

No provider refresh, SEC/EDGAR live calls, yfinance calls, live market data calls, broker calls, cached-source execution, reporting, Telegram/email delivery, portfolio writes, watchlist writes, scheduler behavior, UI behavior, canonical-universe promotion, source-support authority, Decision Engine behavior, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability, or execution advice.
