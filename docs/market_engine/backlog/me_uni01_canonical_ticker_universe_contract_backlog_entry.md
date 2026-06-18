# ME-UNI01 - Canonical ticker universe contract backlog entry

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI01

## Goal

Define the canonical ticker universe contract for Market Engine.

## Rationale

The ME-UNI01 sequence document established the Ticker Universe job family and the need for a canonical source of truth after ME-RUN15.

This contract sprint completes that definition by specifying the exact canonical CSV path, required fields, optional metadata, value domains, validation rules, active/default selection behavior, source-policy semantics, downstream RUN integration requirements and ME-UNI02 implementation boundary.

## Scope

Documentation-only sprint.

ME-UNI01 defines:

* canonical ticker universe contract identity;
* reserved canonical CSV path;
* UTF-8 CSV file format;
* required columns;
* optional metadata columns;
* per-field semantics;
* allowed value domains;
* normalization behavior;
* validation and fail-closed rules;
* duplicate detection rules;
* active/inactive behavior;
* priority ordering behavior;
* source-policy behavior;
* downstream RUN visibility requirements;
* Telegram sequencing gates;
* ME-UNI02 implementation requirements;
* forbidden behavior.

## Implemented documentation

```text
docs/market_engine/ticker_universe/me_uni01_canonical_ticker_universe_contract.md
docs/market_engine/audits/me_uni01_canonical_ticker_universe_contract_audit.md
docs/market_engine/backlog/me_uni01_canonical_ticker_universe_contract_backlog_entry.md
docs/market_engine/roadmap/me_uni01_canonical_ticker_universe_contract_roadmap_entry.md
```

## Defined contract

```text
market-engine-canonical-ticker-universe-v1
```

## Canonical file path

```text
data/market_engine/ticker_universe/ticker_universe.csv
```

## Required columns

```text
ticker
name
market
asset_type
active
priority
source_policy
portfolio_relevant
telegram_preview_eligible
telegram_delivery_eligible
notes
```

## Outcome

ME-UNI01 is complete as the contract-definition sprint.

ME-RUN16 remains blocked until ME-UNI02 implements canonical ticker universe loading and validation.

ME-TG01 remains blocked until ME-UNI02 and initial canonical-universe RUN validation are completed.

## Non-goals

ME-UNI01 does not introduce runtime code, tests, CSV data, fixtures, provider refresh, SEC/EDGAR live calls, yfinance calls, live market data calls, internet calls, broker calls, Telegram/email delivery, production reports, portfolio writes, watchlist writes, scheduler behavior, UI behavior, automatic cache refresh, automatic cache cleanup, generated artifact commits, Decision Engine action labels, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Recommended next sprint

```text
ME-UNI02 - Implement canonical ticker universe loading and validation
```

ME-UNI02 should implement only the loader, validator, normalized record output, active/default selection, deterministic ordering, operator-readable errors, tests and documentation required by `market-engine-canonical-ticker-universe-v1`.
