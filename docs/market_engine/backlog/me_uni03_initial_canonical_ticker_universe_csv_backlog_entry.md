# ME-UNI03 - Initial canonical ticker universe CSV backlog entry

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI03

## Goal

Create the first canonical ticker universe CSV at the approved Market Engine path:

```text
data/market_engine/ticker_universe/ticker_universe.csv
```

## Scope

ME-UNI03 implements:

* initial version-controlled canonical ticker universe CSV;
* required ME-UNI01/ME-UNI02 column coverage;
* optional metadata columns for operator visibility;
* active cached-source default selection metadata;
* manual-review-only exclusion for `SMCI`;
* documentation and audit.

## Initial CSV contents

The initial universe contains 14 rows:

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

Default selected active cached-source rows: 13.

Manual-review-only rows: 1.

## Implemented data file

```text
data/market_engine/ticker_universe/ticker_universe.csv
```

## Implemented documentation

```text
docs/market_engine/ticker_universe/me_uni03_initial_canonical_ticker_universe_csv.md
docs/market_engine/audits/me_uni03_initial_canonical_ticker_universe_csv_audit.md
docs/market_engine/backlog/me_uni03_initial_canonical_ticker_universe_csv_backlog_entry.md
docs/market_engine/roadmap/me_uni03_initial_canonical_ticker_universe_csv_roadmap_entry.md
```

## Outcome

Market Engine now has a first canonical universe file that the ME-UNI02 loader can validate and that ME-RUN16 can consume in a cached-source-only downstream sprint.

## Non-goals

ME-UNI03 does not implement provider refresh, SEC/EDGAR live calls, yfinance calls, live market data calls, internet calls, broker calls, Telegram/email delivery, production reports, portfolio writes, watchlist writes, scheduler behavior, UI behavior, automatic cache refresh, automatic cache cleanup, generated runtime artifacts, Decision Engine action labels, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Recommended next sprint

```text
ME-RUN16 - Execute first real cached-source batch dry-run using canonical ticker universe
```
