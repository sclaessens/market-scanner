# ME-UNI05 Backlog Entry - Professional Swing Universe seed import

Owner roles: Product Owner / Operator / Technical Architect / Data Steward / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI05

## Goal

Import and manually normalize the first Professional Swing Universe seed list under the ME-UNI04 approved editable universe path category.

## Outcome

ME-UNI05 created the first editable seed CSV:

```text
data/market_engine/ticker_universe/professional_swing_universe/professional_swing_universe.csv
```

The seed file contains 53 normalized rows using the required ME-UNI04 columns.

## Scope completed

ME-UNI05 completed:

* seed-list import documentation;
* creation of the initial Professional Swing Universe CSV;
* manual normalization into the ME-UNI04 CSV shape;
* contract-level validation notes;
* backlog and roadmap entry synchronization;
* audit documentation.

## Files changed

```text
data/market_engine/ticker_universe/professional_swing_universe/professional_swing_universe.csv
docs/market_engine/ticker_universe/me_uni05_professional_swing_universe_seed_import.md
docs/market_engine/audits/me_uni05_professional_swing_universe_seed_import_audit.md
docs/market_engine/backlog/me_uni05_professional_swing_universe_seed_import_backlog_entry.md
docs/market_engine/roadmap/me_uni05_professional_swing_universe_seed_import_roadmap_entry.md
```

## Boundaries

ME-UNI05 does not introduce runtime code, tests, loader behavior, provider calls, source refresh, cached-source execution, reporting, Telegram/email delivery, portfolio writes, watchlist writes, scheduler behavior, UI behavior, Decision Engine behavior, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Next sprint

### ME-UNI06 - Implement editable universe loader and validation

Status: RECOMMENDED NEXT AFTER ME-UNI05

Job family: ME-UNI - Ticker Universe

Goal: implement a fail-closed loader and validation layer for the editable Professional Swing Universe CSV defined by ME-UNI04 and seeded by ME-UNI05.

Scope: CSV loading, required-column validation, allowed-value validation, duplicate `(ticker, market)` rejection, normalized record creation, explicit validation errors, default-selection behavior, tests, implementation documentation, audit, and backlog/roadmap synchronization.

Non-goals: no source-support classification authority, provider calls, source refresh, cached-source execution, output/reporting behavior, delivery behavior, scheduler behavior, portfolio/watchlist writes, Decision Engine action semantics, BUY / SELL / HOLD semantics, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Planned sequence preserved

```text
ME-UNI06 - Implement editable universe loader and validation
ME-SR05 - Classify source support for Professional Swing Universe
ME-RUN20 - Execute clean supported-universe cached-source scan
ME-OUT01 - Define readable operator report from dry-run artifacts
ME-CANDIDATE01 - Define non-actionable candidate classification contract
```
