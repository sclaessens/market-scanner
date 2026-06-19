# ME-UNI02 - Canonical ticker universe loader implementation

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI02

## Purpose

ME-UNI02 implements the canonical ticker universe loader and validator required by the ME-UNI01 contract.

The implementation gives Market Engine a deterministic, fail-closed way to load the approved ticker universe CSV before later RUN sprints consume it. It does not execute batch runs, refresh providers, create artifacts, send messages, mutate portfolios, mutate watchlists or introduce action authority.

## Contract implemented

```text
market-engine-canonical-ticker-universe-v1
```

Canonical default path:

```text
data/market_engine/ticker_universe/ticker_universe.csv
```

The loader also accepts an explicit path override for tests and future command integration.

## Runtime API

Implemented package:

```text
src/market_engine/ticker_universe/
```

Public API:

```text
load_canonical_ticker_universe(...)
validate_canonical_ticker_universe(...)
CanonicalTickerUniverseEntry
CanonicalTickerUniverseResult
CanonicalTickerUniverseValidationError
```

`load_canonical_ticker_universe` returns a `CanonicalTickerUniverseResult` containing validated entries and run-visible counts.

By default, entries include only rows where:

```text
active=true
source_policy in cached_source_only,cached_source_required
```

Callers may request all valid rows, including inactive, blocked and manual-review-only rows, with `include_inactive=True`.

## Validation behavior

The implementation validates:

* file existence and readable UTF-8 CSV input;
* header presence;
* unnamed column rejection;
* duplicate column rejection after trimming;
* required column presence;
* required value presence, except `notes` may be blank;
* ticker normalization by trimming and uppercasing only;
* ticker character set;
* allowed `market` values;
* allowed `asset_type` values;
* allowed boolean values;
* integer `priority >= 1`;
* allowed `source_policy` values;
* duplicate normalized ticker and market rows;
* Telegram delivery eligibility cannot override preview ineligibility.

Validation errors fail closed with operator-readable messages that include row, field and invalid value when available.

## Ordering

Validated entries are ordered deterministically by:

```text
priority ascending
ticker ascending
market ascending
```

Priority remains an operator batching order only. It is not a score, ranking, urgency, conviction or tradeability field.

## Metadata

Required fields are normalized into `CanonicalTickerUniverseEntry`.

Unknown optional columns are preserved as string metadata. Metadata is not interpreted as provider authority, delivery authority, portfolio authority, watchlist authority, ranking, score, allocation or execution guidance.

## Intentionally excluded behavior

ME-UNI02 does not implement:

* ME-RUN16 canonical-universe batch execution;
* provider refresh;
* live SEC/EDGAR calls;
* yfinance calls;
* external network calls;
* source snapshot creation;
* Telegram rendering or delivery;
* email delivery;
* broker integration;
* portfolio writes;
* watchlist writes;
* scheduler behavior;
* UI behavior;
* production reports;
* Decision Engine behavior;
* BUY / SELL / HOLD semantics;
* allocation advice;
* target prices;
* position sizing;
* order generation;
* ranking;
* scoring;
* urgency;
* conviction;
* tradeability;
* execution advice.

## Relationship to ME-RUN16

ME-RUN16 remains a downstream implementation sprint. It may consume the canonical ticker universe loader, but ME-UNI02 itself does not connect the loader into batch dry-run execution.

ME-RUN16 must still make the canonical universe path, contract version, loaded row count, selected ticker count, excluded counts, validation state and selected ticker order visible before broader RUN behavior is approved.
