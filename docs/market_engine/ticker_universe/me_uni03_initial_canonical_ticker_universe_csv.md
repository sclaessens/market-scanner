# ME-UNI03 - Initial canonical ticker universe CSV

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI03

## Purpose

ME-UNI03 creates the first approved canonical ticker universe CSV for Market Engine.

The sprint populates the reserved canonical path defined by ME-UNI01 and validated by the ME-UNI02 loader:

```text
data/market_engine/ticker_universe/ticker_universe.csv
```

The file is input governance only. It does not grant analysis authority, portfolio authority, watchlist authority, delivery authority, trade authority, BUY / SELL / HOLD semantics, allocation authority, target-price authority, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Contract used

```text
market-engine-canonical-ticker-universe-v1
```

## CSV shape

The CSV contains all required v1 columns:

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

It also includes optional metadata columns preserved by the ME-UNI02 loader:

```text
sector
theme
risk_bucket
operator_group
```

These optional columns are descriptive metadata only. They are not ranking, scoring, conviction, urgency, allocation, position sizing, target price, tradeability or execution fields.

## Initial universe contents

ME-UNI03 introduces 14 initial rows:

```text
NVDA
AMD
ASML
META
MSFT
VRT
CLS
CRDO
IREN
COST
HO
SMCI
AVGO
TSM
```

Default active cached-source selection contains 13 rows because `SMCI` is intentionally marked `manual_review_only`.

Default selected order is governed only by `priority`, then `ticker`, then `market`:

```text
NVDA
AMD
ASML
META
MSFT
VRT
CLS
CRDO
IREN
COST
HO
AVGO
TSM
```

## Selection rationale

The initial universe balances:

* current portfolio context: `ASML`, `COST`, `HO`;
* earlier short-term watchlist context: `AMD`, `NVDA`, `META`, `SMCI`;
* recent operator-requested candidates: `CLS`, `VRT`, `CRDO`, `IREN`;
* AI infrastructure and semiconductor context anchors: `MSFT`, `AVGO`, `TSM`.

`SMCI` is included for traceability but excluded from default RUN selection through `source_policy=manual_review_only` because it requires additional governance caution before default batch use.

## Validation performed

Connector/runtime constraints prevented running the repository pytest suite locally in this session because the macOS checkout is not mounted and the container has no GitHub network access.

A contract-level CSV validation was performed against the ME-UNI02 rules before committing the CSV:

* required columns present;
* required values present;
* allowed market values only;
* allowed asset type values only;
* allowed source policy values only;
* boolean values are `true` or `false`;
* priorities are integers greater than or equal to 1;
* normalized ticker values match the v1 ticker pattern;
* no duplicate normalized ticker and market pairs;
* `telegram_delivery_eligible=false` for every row.

Validation result:

```text
loaded rows: 14
active cached-source selected rows: 13
manual-review-only rows: 1
blocked rows: 0
inactive rows: 0
```

## Files changed

```text
data/market_engine/ticker_universe/ticker_universe.csv
docs/market_engine/ticker_universe/me_uni03_initial_canonical_ticker_universe_csv.md
docs/market_engine/audits/me_uni03_initial_canonical_ticker_universe_csv_audit.md
docs/market_engine/backlog/me_uni03_initial_canonical_ticker_universe_csv_backlog_entry.md
docs/market_engine/roadmap/me_uni03_initial_canonical_ticker_universe_csv_roadmap_entry.md
```

## Non-goals

ME-UNI03 does not implement provider refresh, SEC/EDGAR live calls, yfinance calls, live market data calls, internet calls, broker calls, Telegram/email delivery, production reports, portfolio writes, watchlist writes, scheduler behavior, UI behavior, automatic cache refresh, automatic cache cleanup, generated runtime artifacts, Decision Engine action labels, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Recommended next sprint

```text
ME-RUN16 - Execute first real cached-source batch dry-run using canonical ticker universe
```

ME-RUN16 may consume the canonical CSV and must surface path, contract version, loaded row count, selected row count, excluded counts, validation state, selected ticker order and source policy for each selected ticker.
