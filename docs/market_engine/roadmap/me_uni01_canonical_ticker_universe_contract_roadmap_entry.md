# ME-UNI01 - Canonical ticker universe contract roadmap entry

Owner roles: Product Owner / Scrum Master / Technical Architect / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI01

## Placement

ME-UNI01 follows the ME-UNI planning sequence that was inserted after ME-RUN15.

ME-RUN15 implemented operator-facing cached-source batch dry-run command visibility. ME-UNI01 now defines the canonical ticker universe contract required before broader canonical-universe batch execution can proceed.

## Roadmap purpose

Market Engine needs one approved ticker universe before broader real cached-source batch analysis, Telegram preview rendering or Telegram delivery are allowed.

ME-UNI01 prevents analysis scope from being driven by ad hoc ticker lists, hidden watchlists, broker state, provider discovery, external APIs or implicit all-market universes.

## Contract defined

```text
market-engine-canonical-ticker-universe-v1
```

## Canonical path

```text
data/market_engine/ticker_universe/ticker_universe.csv
```

## Documentation added

```text
docs/market_engine/ticker_universe/me_uni01_canonical_ticker_universe_contract.md
docs/market_engine/audits/me_uni01_canonical_ticker_universe_contract_audit.md
docs/market_engine/backlog/me_uni01_canonical_ticker_universe_contract_backlog_entry.md
docs/market_engine/roadmap/me_uni01_canonical_ticker_universe_contract_roadmap_entry.md
```

## Roadmap update

ME-UNI01 is complete as a documentation-only contract sprint.

Recommended next implementation sprint:

```text
ME-UNI02 - Implement canonical ticker universe loading and validation
```

ME-UNI02 must implement the loader and validation layer before ME-RUN16 may consume the canonical ticker universe for the first real cached-source batch dry-run.

## Blocking rules preserved

ME-RUN16 is blocked until ME-UNI02 is completed.

ME-TG01 is blocked until ME-UNI02 and initial canonical-universe cached-source RUN validation are completed.

Telegram delivery remains blocked until render-only previews and explicit safe gates are validated.

## Forbidden roadmap expansion

ME-UNI01 does not authorize provider refresh, live data calls, broker calls, Telegram delivery, email delivery, production reports, portfolio writes, watchlist writes, scheduler behavior, UI behavior, automatic cache refresh, generated artifact commits, Decision Engine action semantics, BUY / SELL / HOLD labels, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability or execution advice.
