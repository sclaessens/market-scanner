# ME-UNI06 - Editable universe loader and validation implementation

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI06

## Purpose

ME-UNI06 implements the editable Professional Swing Universe loader and fail-closed validation behavior defined by ME-UNI04 and seeded by ME-UNI05.

The implementation exposes normalized records for the editable Professional Swing Universe without promoting those rows into the canonical ticker universe, source support, reporting, recommendation, portfolio, or execution authority.

## Contract consumed

```text
market-engine-editable-professional-swing-universe-v1
```

## Runtime implemented

```text
src/market_engine/ticker_universe/professional_swing.py
src/market_engine/ticker_universe/__init__.py
```

Public API:

```text
load_professional_swing_universe
validate_professional_swing_universe
ProfessionalSwingUniverseEntry
ProfessionalSwingUniverseResult
ProfessionalSwingUniverseValidationError
PROFESSIONAL_SWING_UNIVERSE_PATH
REQUIRED_PROFESSIONAL_SWING_UNIVERSE_COLUMNS
EDITABLE_PROFESSIONAL_SWING_UNIVERSE_CONTRACT_VERSION
```

## Implemented validation behavior

The loader fails closed when:

* the CSV file is missing, unreadable, empty or malformed;
* the header is missing;
* required columns are missing;
* columns are unnamed or duplicated after trimming;
* a non-blank row has more values than the header;
* required values are empty, except `notes`;
* ticker normalization produces a value outside the approved ticker character set;
* market, asset type, universe status, source-policy hint, swing profile, liquidity profile, volatility profile or market-cap profile is outside the approved domain;
* `active` is not `true` or `false`;
* `operator_priority` is not an integer greater than or equal to `1`;
* duplicate `(normalized_ticker, market)` rows exist.

## Default selection behavior

Default selection returns only rows where:

```text
active=true
universe_status in candidate,watching
source_policy_hint in cached_source_candidate,unknown
```

Rows are ordered deterministically by:

```text
operator_priority ascending
ticker ascending
market ascending
```

`include_inactive=True` returns all valid rows, still sorted deterministically.

## Current seed outcome

Against the ME-UNI05 seed CSV:

```text
loaded_row_count=53
selected_row_count=45
```

The default selection excludes source-mapping and manual-review-only rows such as `ASML`, `MSTR`, and `HO`.

## Tests implemented

```text
tests/market_engine/ticker_universe/test_professional_swing_universe.py
```

Coverage includes:

* valid minimal loading;
* optional metadata preservation;
* default exclusion rules;
* deterministic ordering;
* missing file failure;
* missing required-column failure;
* empty required-value failure;
* blank notes allowance;
* duplicate ticker/market rejection;
* allowed-domain validation;
* active boolean validation;
* operator-priority validation;
* ticker-format validation;
* dependency-boundary checks;
* current ME-UNI05 seed CSV validation.

## Boundaries preserved

ME-UNI06 does not introduce provider calls, SEC/EDGAR live calls, yfinance calls, live market data calls, broker calls, source refresh, cached-source execution, Telegram/email delivery, production reporting, portfolio writes, watchlist writes, scheduler behavior, UI behavior, canonical-universe promotion, source-support authority, Decision Engine behavior, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability, or execution advice.
