# ME-UNI04 Backlog Entry - Editable Professional Swing Universe contract

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI04

## Goal

Define the editable Professional Swing Universe contract before importing a broader professional swing candidate list or implementing loader, source-support, RUN, output, or candidate-classification behavior.

## Outcome

ME-UNI04 defines:

```text
market-engine-editable-professional-swing-universe-v1
```

The contract establishes:

* approved path category;
* future seed CSV path;
* relationship to canonical ticker universe;
* required CSV columns;
* optional metadata columns;
* allowed values;
* validation rules;
* default selection semantics;
* normalized record shape;
* downstream sequencing;
* ME-UNI05 and ME-UNI06 boundaries;
* forbidden behavior.

## Approved path category

```text
data/market_engine/ticker_universe/professional_swing_universe/
```

Future default seed path:

```text
data/market_engine/ticker_universe/professional_swing_universe/professional_swing_universe.csv
```

ME-UNI04 does not create the CSV.

## Scope

Documentation and contract only.

ME-UNI04 does not introduce runtime code, test code, CSV data, fixtures, provider calls, source refresh, cached-source execution, reporting, Telegram/email delivery, portfolio writes, watchlist writes, scheduler behavior, UI behavior, Decision Engine behavior, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Files changed

```text
docs/market_engine/ticker_universe/me_uni04_editable_professional_swing_universe_contract.md
docs/market_engine/audits/me_uni04_editable_professional_swing_universe_contract_audit.md
docs/market_engine/backlog/me_uni04_editable_professional_swing_universe_contract_backlog_entry.md
docs/market_engine/roadmap/me_uni04_editable_professional_swing_universe_contract_roadmap_entry.md
```

## Next sprint

### ME-UNI05 - Import and normalize Professional Swing Universe seed list

Status: RECOMMENDED NEXT AFTER ME-UNI04

Job family: ME-UNI - Ticker Universe

Goal: create the first editable Professional Swing Universe seed CSV under the approved ME-UNI04 path category and normalize supplied candidate rows into the ME-UNI04 CSV shape.

Scope: seed-list import, manual normalization, contract-level validation notes, documentation, roadmap/backlog synchronization and audit only.

Non-goals: no loader implementation, provider calls, source refresh, cached-source execution, output/reporting behavior, delivery behavior, scheduler behavior, portfolio/watchlist writes, Decision Engine action semantics, BUY / SELL / HOLD semantics, allocation advice, target prices, position sizing, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Planned sequence preserved

```text
ME-UNI05 - Import and normalize Professional Swing Universe seed list
ME-UNI06 - Implement editable universe loader and validation
ME-SR05 - Classify source support for Professional Swing Universe
ME-RUN20 - Execute clean supported-universe cached-source scan
ME-OUT01 - Define readable operator report from dry-run artifacts
ME-CANDIDATE01 - Define non-actionable candidate classification contract
```
