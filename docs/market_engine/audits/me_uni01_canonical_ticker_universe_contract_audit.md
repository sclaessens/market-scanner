# ME-UNI01 - Canonical ticker universe contract audit

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI01

## Audit purpose

This audit verifies that ME-UNI01 defines the canonical ticker universe contract without introducing runtime code, tests, provider calls, source refresh, production execution, Telegram delivery, portfolio mutation, watchlist mutation, scheduler behavior, UI behavior, or action/allocation authority.

## Scope audited

Audited contract document:

```text
docs/market_engine/ticker_universe/me_uni01_canonical_ticker_universe_contract.md
```

Related planning document from the prior sequence update:

```text
docs/market_engine/ticker_universe/me_uni01_canonical_ticker_universe_sequence.md
```

## Contract identity check

ME-UNI01 defines:

```text
market-engine-canonical-ticker-universe-v1
```

The audit confirms that this is an input-governance contract only.

## Canonical path check

ME-UNI01 defines the reserved canonical path:

```text
data/market_engine/ticker_universe/ticker_universe.csv
```

The audit confirms that ME-UNI01 does not create this CSV and does not introduce ticker data. File creation and loader behavior are deferred to ME-UNI02.

## File format check

ME-UNI01 defines a deterministic UTF-8 CSV contract with one header row, comma separators, one ticker record per row, no spreadsheet formatting dependencies, no hidden formulas, no duplicate headers and no unnamed columns.

## Required fields check

ME-UNI01 defines these required fields:

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

The audit confirms that ME-UNI02 must reject the canonical CSV when any required field is missing.

## Value-domain check

ME-UNI01 defines explicit allowed values for:

* market;
* asset_type;
* boolean fields;
* source_policy;
* priority.

The audit confirms that invalid values must fail closed with operator-readable validation errors.

## Selection semantics check

ME-UNI01 defines default canonical RUN selection as:

```text
active=true
source_policy in cached_source_only,cached_source_required
```

The audit confirms deterministic ordering:

```text
priority ascending
ticker ascending
market ascending
```

Priority is explicitly not conviction, score, ranking, urgency or tradeability.

## Exclusion semantics check

The audit confirms that inactive, blocked and manual-review-only records remain part of loaded universe metadata but must be excluded from default RUN selection.

## Duplicate validation check

ME-UNI01 requires duplicate active records and duplicate rows for the same normalized ticker and market to be rejected unless a later contract introduces effective dates.

## Downstream RUN check

ME-UNI01 blocks ME-RUN16 until ME-UNI02 implements canonical ticker universe loading and validation.

ME-RUN16 must surface canonical universe path, contract version, row counts, selected ticker count, excluded counts, validation state, selected order and source policy for each selected ticker.

## Telegram sequencing check

ME-UNI01 preserves Telegram preview and delivery gates:

* ME-TG01 remains blocked until ME-UNI02 and initial canonical-universe RUN validation are completed;
* `telegram_preview_eligible` remains render-preview governance only;
* `telegram_delivery_eligible` remains dormant until render-only previews and delivery gates are separately validated.

## Implementation boundary check

ME-UNI02 may implement loading, validation, duplicate detection, active/default selection, deterministic ordering, normalized records, operator-readable errors, tests and fixtures.

ME-UNI02 must not introduce provider calls, source refresh, delivery behavior, portfolio writes, watchlist writes, scheduler behavior, UI behavior, Decision Engine action semantics, scoring, ranking, conviction, urgency, target prices, allocation or execution advice.

## Forbidden behavior check

ME-UNI01 does not introduce or approve:

* runtime code;
* tests;
* CSV data;
* fixtures;
* provider refresh;
* SEC/EDGAR live calls;
* yfinance calls;
* live market data calls;
* internet calls;
* broker calls;
* Telegram delivery;
* email delivery;
* production reports;
* portfolio writes;
* watchlist writes;
* scheduler behavior;
* UI behavior;
* automatic cache refresh;
* automatic cache cleanup;
* generated artifact commits;
* Decision Engine action labels;
* BUY / SELL / HOLD semantics;
* allocation advice;
* target prices;
* target weights;
* position sizing;
* order generation;
* ranking;
* scoring;
* urgency;
* conviction;
* tradeability;
* execution advice.

## Audit conclusion

ME-UNI01 satisfies its documentation-only acceptance criteria.

It defines the canonical ticker universe contract, including path, file format, required fields, optional metadata, value domains, active/inactive semantics, priority semantics, source-policy semantics, validation rules, default selection rules, downstream RUN requirements, Telegram sequencing gates and ME-UNI02 implementation boundaries.
