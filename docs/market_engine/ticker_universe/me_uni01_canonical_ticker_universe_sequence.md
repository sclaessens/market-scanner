# ME-UNI01 — Canonical ticker universe sequence

Job family: ME-UNI — Ticker Universe

Status: PLANNED

## Purpose

ME-UNI introduces a dedicated job family for the canonical list of tickers that Market Engine may analyse.

After ME-RUN15, cached-source batch dry-runs have operator-facing command visibility, but the project still lacks one approved, editable and extensible source of truth for the analysis scope.

The ticker universe must become that source of truth before broader cached-source batch runs, Telegram previews, Telegram delivery, production reporting, portfolio automation or watchlist mutation are introduced.

## Planned sequence

```text
ME-UNI01 — Define canonical ticker universe contract
ME-UNI02 — Implement canonical ticker universe loading and validation
ME-RUN16 — Execute first real cached-source batch dry-run using canonical ticker universe
ME-RUN17 — Broader cached-source batch review using canonical ticker universe
ME-TG01 — Define Telegram preview contract
ME-TG02 — Implement Telegram render-only preview
ME-TG03 — Implement gated Telegram delivery
```

## Canonical path

Recommended canonical path:

```text
data/market_engine/ticker_universe/ticker_universe.csv
```

## Recommended required fields

```text
ticker
name
market
asset_type
active
priority
notes
```

## Recommended optional fields

```text
sector
theme
source_policy
risk_bucket
portfolio_relevant
telegram_preview_eligible
telegram_delivery_eligible
```

## Dependency rule

ME-RUN16 is blocked until ME-UNI02 is completed.

ME-TG01 is blocked until ME-UNI02 and initial canonical-universe RUN validation are completed.

Telegram delivery remains blocked until render-only previews and explicit safe gates are validated.

## Non-goals

ME-UNI01 must not introduce runtime code, provider calls, live data calls, delivery behavior, portfolio writes, watchlist writes, scheduler behavior, UI behavior or trade/action authority.
