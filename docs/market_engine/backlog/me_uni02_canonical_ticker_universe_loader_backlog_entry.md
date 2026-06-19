# ME-UNI02 - Canonical ticker universe loader backlog entry

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI02

## Goal

Implement the canonical ticker universe loader and validation layer defined by ME-UNI01.

## Scope

ME-UNI02 implements:

* canonical ticker universe CSV loading;
* explicit path override support;
* required-column validation;
* required-value validation;
* allowed-value validation;
* duplicate ticker and market rejection;
* default active cached-source selection;
* deterministic ordering;
* normalized typed entries and result metadata;
* local synthetic tests;
* implementation documentation and audit.

## Implemented runtime

```text
src/market_engine/ticker_universe/
```

## Implemented tests

```text
tests/market_engine/ticker_universe/test_canonical_ticker_universe.py
```

## Implemented documentation

```text
docs/market_engine/ticker_universe/me_uni02_canonical_ticker_universe_loader_implementation.md
docs/market_engine/audits/me_uni02_canonical_ticker_universe_loader_audit.md
docs/market_engine/backlog/me_uni02_canonical_ticker_universe_loader_backlog_entry.md
docs/market_engine/roadmap/me_uni02_canonical_ticker_universe_loader_roadmap_entry.md
```

## Outcome

Market Engine can now validate and load `market-engine-canonical-ticker-universe-v1` CSV files from:

```text
data/market_engine/ticker_universe/ticker_universe.csv
```

The loader fails closed on missing files, malformed CSV input, missing required columns, empty required values, invalid allowed values, invalid priority values, invalid ticker formats and duplicate normalized ticker and market rows.

## Non-goals

ME-UNI02 does not implement ME-RUN16, provider refresh, SEC/EDGAR live calls, yfinance calls, live market data calls, internet calls, broker calls, Telegram/email delivery, production reports, portfolio writes, watchlist writes, scheduler behavior, UI behavior, automatic cache refresh, automatic cache cleanup, generated artifact commits, Decision Engine action labels, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Recommended next sprint

```text
ME-RUN16 - Execute first real cached-source batch dry-run using canonical ticker universe
```

ME-RUN16 should consume the ME-UNI02 loader in the RUN job family while preserving cached-source/local-only boundaries.
