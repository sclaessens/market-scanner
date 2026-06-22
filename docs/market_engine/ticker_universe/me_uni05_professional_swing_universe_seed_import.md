# ME-UNI05 - Professional Swing Universe seed import

Owner roles: Product Owner / Operator / Technical Architect / Data Steward / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI05

## Purpose

ME-UNI05 imports and manually normalizes the first editable Professional Swing Universe seed list into the ME-UNI04 CSV shape.

The seed list is an operator-maintained candidate universe. It is intended for later loader validation, source-support classification, clean cached-source scanning, readable operator reporting, and non-actionable candidate classification.

ME-UNI05 does not make the Professional Swing Universe canonical. It does not grant source-support authority, provider-refresh authority, cached-source execution authority, delivery authority, portfolio/watchlist authority, Decision Engine authority, BUY / SELL / HOLD semantics, allocation authority, target-price authority, ranking, scoring, urgency, conviction, tradeability, position-sizing authority, order authority, or execution advice.

## Contract consumed

ME-UNI05 consumes the ME-UNI04 editable Professional Swing Universe contract:

```text
market-engine-editable-professional-swing-universe-v1
```

## Seed file created

```text
data/market_engine/ticker_universe/professional_swing_universe/professional_swing_universe.csv
```

The seed file contains 53 normalized rows.

## Required CSV shape

The seed file uses the ME-UNI04 required columns exactly:

```text
ticker
name
market
asset_type
active
universe_status
source_policy_hint
operator_priority
swing_profile
liquidity_profile
volatility_profile
market_cap_profile
theme
sector
notes
```

No optional columns were introduced in ME-UNI05. This keeps the first seed import narrow and makes ME-UNI06 loader validation easier to implement.

## Normalization performed

Manual normalization performed by ME-UNI05:

* tickers are uppercase and contain only approved ticker characters;
* markets are normalized to approved ME-UNI04 market values;
* asset types are normalized to lowercase ME-UNI04 values;
* `active` values are normalized to `true`;
* universe statuses are normalized to ME-UNI04 allowed values;
* source-policy hints are normalized to ME-UNI04 allowed values;
* operator priorities are integer values from 1 through 53;
* swing, liquidity, volatility and market-cap profiles use ME-UNI04 allowed values;
* themes and sectors use lowercase snake_case style where practical;
* notes are descriptive and do not contain trade instructions.

## Seed-list composition

The seed contains:

* overlap rows from the existing canonical ticker universe;
* additional USA-listed professional swing candidates;
* selected non-USA listings that require future source mapping;
* manual-review-only rows where current governance should prevent default automated selection.

The seed deliberately keeps canonical-overlap rows explicit. A row appearing in this file does not imply canonical universe membership and does not promote a ticker to canonical execution eligibility.

## Default-selection implication

Based on ME-UNI04 default selection semantics, only rows with this combination are future default-selection candidates:

```text
active=true
universe_status in candidate,watching
source_policy_hint in cached_source_candidate,unknown
```

Rows marked `needs_source_mapping`, `manual_review_only`, `blocked`, `rejected`, or `research_required` must remain excluded from default automated selection unless a later approved contract changes that behavior.

ME-UNI05 does not implement this selection behavior. ME-UNI06 may implement loader and validation behavior in a later sprint.

## Contract-level validation notes

Manual validation checked:

* required header presence;
* required field presence, except that `notes` may be blank under ME-UNI04;
* approved market values;
* approved asset-type values;
* approved active values;
* approved universe-status values;
* approved source-policy-hint values;
* integer operator priorities greater than or equal to 1;
* approved swing-profile values;
* approved liquidity-profile values;
* approved volatility-profile values;
* approved market-cap-profile values;
* duplicate `(ticker, market)` rows are absent;
* no runtime or provider behavior was introduced.

## Files changed

```text
data/market_engine/ticker_universe/professional_swing_universe/professional_swing_universe.csv
docs/market_engine/ticker_universe/me_uni05_professional_swing_universe_seed_import.md
docs/market_engine/audits/me_uni05_professional_swing_universe_seed_import_audit.md
docs/market_engine/backlog/me_uni05_professional_swing_universe_seed_import_backlog_entry.md
docs/market_engine/roadmap/me_uni05_professional_swing_universe_seed_import_roadmap_entry.md
```

## Downstream sequence preserved

```text
ME-UNI06 - Implement editable universe loader and validation
ME-SR05 - Classify source support for Professional Swing Universe
ME-RUN20 - Execute clean supported-universe cached-source scan
ME-OUT01 - Define readable operator report from dry-run artifacts
ME-CANDIDATE01 - Define non-actionable candidate classification contract
```

## Forbidden behavior preserved

ME-UNI05 does not introduce Python runtime code, tests, provider calls, source refresh, cached-source execution, reporting, Telegram delivery, email delivery, portfolio writes, watchlist writes, scheduler behavior, UI behavior, Decision Engine behavior, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Next sprint

```text
ME-UNI06 - Implement editable universe loader and validation
```

ME-UNI06 may implement fail-closed loader and validation behavior for the editable Professional Swing Universe. It must not classify source support as authoritative unless a later approved source-support contract grants that authority.
