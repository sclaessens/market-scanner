# ME-UNI01 - Canonical ticker universe contract

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI01

## Purpose

ME-UNI01 defines the canonical ticker universe contract for Market Engine.

The canonical ticker universe is the approved, editable, version-controlled source of truth for which instruments Market Engine may include in cached-source batch dry-runs, later preview rendering, and later gated delivery workflows.

It is input governance only. It does not grant analysis authority, delivery authority, portfolio authority, watchlist authority, trade authority, allocation authority, execution authority, BUY / SELL / HOLD semantics, ranking, scoring, urgency, conviction, or target-price authority.

## Contract identity

ME-UNI01 defines this contract:

```text
market-engine-canonical-ticker-universe-v1
```

The contract governs the canonical CSV shape, allowed values, validation behavior, ordering semantics, downstream RUN consumption, and implementation requirements for ME-UNI02.

## Canonical repository path

The canonical file path is:

```text
data/market_engine/ticker_universe/ticker_universe.csv
```

This path is reserved for the approved ticker universe. ME-UNI01 does not create the CSV itself; ME-UNI02 may introduce it together with loader and validation behavior.

## File format

The canonical ticker universe file must be a UTF-8 CSV file with:

* one header row;
* comma separators;
* one ticker record per row;
* no hidden workbook formulas;
* no spreadsheet-only formatting dependencies;
* deterministic parsing independent of Excel, Numbers or Google Sheets;
* no comments outside explicit columns;
* no duplicate header names;
* no unnamed columns.

Blank lines must be ignored only if they contain no values in any column.

Whitespace around field values must be trimmed before validation.

## Required columns

ME-UNI02 must reject the file when any required column is missing.

Required columns:

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

### ticker

Canonical display ticker used by Market Engine operators.

Rules:

* required;
* trimmed;
* uppercased by the loader for normalized comparison;
* must contain only uppercase letters, digits, dots or hyphens after normalization;
* must not contain spaces;
* must not contain provider-specific suffixes unless the suffix is part of the approved canonical ticker;
* must not be inferred from company name.

Examples:

```text
NVDA
MSFT
ASML
BRK.B
```

### name

Human-readable company or instrument name.

Rules:

* required;
* trimmed;
* used for operator visibility only;
* must not be used as the runtime key;
* must not be used to infer tickers automatically.

### market

Approved market or listing region indicator.

Allowed values for v1:

```text
USA
EURONEXT
LSE
XETRA
OTHER
```

Rules:

* required;
* trimmed;
* uppercased by the loader;
* used for disambiguation and reporting context;
* does not by itself authorize provider calls or live exchange access.

### asset_type

Approved asset class category.

Allowed values for v1:

```text
equity
fund
etf
index
crypto
other
```

Rules:

* required;
* lowercased by the loader;
* used for eligibility and context;
* does not authorize provider calls, broker actions or trade actions.

### active

Whether the ticker is currently eligible for canonical-universe selection.

Allowed values:

```text
true
false
```

Rules:

* required;
* lowercased by the loader;
* `true` records are eligible for active selection;
* `false` records must remain visible in the file but excluded from default RUN selection;
* inactive records must not be silently deleted by loaders.

### priority

Operator-maintained ordering and batching priority.

Rules:

* required;
* integer;
* `1` is highest priority;
* values must be greater than or equal to `1`;
* duplicate priorities are allowed;
* deterministic selection order must be `priority` ascending, then `ticker` ascending, then `market` ascending unless an explicit downstream contract overrides it.

Priority is not conviction, score, ranking, urgency or tradeability.

### source_policy

Approved source-use policy for the ticker.

Allowed values for v1:

```text
cached_source_only
cached_source_required
manual_review_only
blocked
```

Rules:

* required;
* lowercased by the loader;
* `cached_source_only` means RUN jobs may use already-existing cached source snapshots only;
* `cached_source_required` means the ticker is active only when a suitable cached source snapshot exists;
* `manual_review_only` means the ticker can remain in the universe but must not be included in default automated batch execution;
* `blocked` means the ticker must be excluded from all default RUN, preview and delivery workflows.

No source policy authorizes live provider refresh, yfinance calls, SEC/EDGAR calls, internet calls, broker calls or automatic backfilling.

### portfolio_relevant

Whether the ticker is relevant for portfolio-context reporting.

Allowed values:

```text
true
false
```

Rules:

* required;
* lowercased by the loader;
* informational only in ME-UNI01;
* does not create or mutate portfolio state;
* does not imply the user owns the instrument;
* does not authorize allocation, target weights or position sizing.

### telegram_preview_eligible

Whether the ticker may be considered for future render-only Telegram previews after the Telegram preview contract exists.

Allowed values:

```text
true
false
```

Rules:

* required;
* lowercased by the loader;
* informational until ME-TG01 defines preview behavior;
* does not authorize Telegram sending;
* must be ignored by delivery code until ME-TG contracts explicitly consume it.

### telegram_delivery_eligible

Whether the ticker may be considered for future gated Telegram delivery after render-only preview and delivery gates are validated.

Allowed values:

```text
true
false
```

Rules:

* required;
* lowercased by the loader;
* must not be interpreted as send authorization in ME-UNI01 or ME-UNI02;
* can only be consumed after ME-TG03 or a later approved delivery contract defines safe delivery gates.

### notes

Operator-maintained free-text context.

Rules:

* required column;
* value may be blank;
* must not be parsed for tickers, decisions, priorities or source policy;
* must not be used to smuggle execution instructions.

## Optional columns

Optional columns may be present and must be preserved in normalized records when ME-UNI02 chooses to expose metadata.

Recommended optional columns:

```text
sector
theme
risk_bucket
exchange
currency
provider_ticker
watchlist_group
review_cadence
last_manual_reviewed_at
```

Optional columns must not weaken required-column validation. Unknown optional columns may be allowed if they do not conflict with reserved required names and are preserved as metadata only.

## Normalized ticker universe record

ME-UNI02 must expose each validated row as a normalized record with at least:

```text
contract_version
source_path
row_number
ticker
normalized_ticker
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
metadata
validation_state
```

`metadata` may include optional columns. It must not contain hidden authority flags beyond the approved contract fields.

## Validation rules

ME-UNI02 must fail closed when the canonical CSV is missing, unreadable, malformed or invalid.

Required validation:

* file exists at the configured canonical path;
* file is parseable as UTF-8 CSV;
* header is present;
* required columns are present exactly once;
* no duplicate column names after trimming;
* each non-blank row has a ticker;
* ticker normalization produces a non-empty value;
* normalized ticker matches the approved ticker character set;
* market is one of the allowed values;
* asset_type is one of the allowed values;
* boolean fields contain only `true` or `false`;
* priority is an integer greater than or equal to `1`;
* source_policy is one of the allowed values;
* duplicate active records for the same normalized ticker and market are rejected;
* duplicate rows with the same normalized ticker and market are rejected unless a later contract explicitly introduces effective dates;
* `telegram_delivery_eligible=true` must not override `telegram_preview_eligible=false`;
* `source_policy=blocked` records must not be selected for default execution even when `active=true`;
* `source_policy=manual_review_only` records must not be selected for default execution;
* active default selection must contain only records with `active=true` and source_policy in `cached_source_only` or `cached_source_required`.

Validation errors must be explicit and operator-readable. They must include the row number, field name, invalid value, and failure reason where possible.

## Selection semantics

Default canonical RUN selection must include only records where:

```text
active=true
source_policy in cached_source_only,cached_source_required
```

Default ordering must be deterministic:

```text
priority ascending
ticker ascending
market ascending
```

Inactive, blocked and manual-review-only records must remain part of the loaded universe metadata but excluded from default execution selection.

A later RUN contract may define explicit override behavior, but ME-UNI01 does not approve override flags.

## Downstream RUN integration requirements

ME-RUN16 is blocked until ME-UNI02 implements canonical ticker universe loading and validation.

ME-RUN16 must consume canonical ticker selection instead of relying only on ad hoc explicit tickers, local ticker text files or cached-source discovery.

ME-RUN16 must make visible:

* canonical universe path;
* contract version;
* loaded row count;
* selected active ticker count;
* excluded inactive count;
* excluded blocked count;
* excluded manual-review-only count;
* validation state;
* selected ticker order;
* source policy for each selected ticker.

ME-RUN16 must fail closed when the canonical ticker universe is invalid.

ME-RUN16 must not use the ticker universe to refresh sources, fetch providers, infer new tickers, enrich records, mutate portfolios, mutate watchlists, send Telegram messages, schedule runs or generate production reports.

## Telegram sequencing requirements

ME-TG01 remains blocked until ME-UNI02 and initial canonical-universe RUN validation are completed.

Telegram preview eligibility fields are governance inputs only. They do not create render behavior, send behavior or delivery authority.

Telegram delivery eligibility fields are dormant until render-only previews and safe delivery gates are separately implemented and validated.

## ME-UNI02 implementation requirements

ME-UNI02 may implement only:

* canonical CSV loading;
* required-column validation;
* value validation;
* duplicate detection;
* active/default selection;
* deterministic ordering;
* normalized record construction;
* operator-readable validation errors;
* tests and fixtures for the loader;
* documentation and audit updates for implementation.

ME-UNI02 must not introduce provider calls, source refresh, batch execution changes beyond consuming validated ticker selections, delivery behavior, portfolio writes, watchlist writes, scheduler behavior, UI behavior, Decision Engine action semantics, scoring, ranking, conviction, urgency, target prices, allocation or execution advice.

## Acceptance criteria

ME-UNI01 is complete when documentation defines:

* canonical path;
* CSV format;
* contract identity;
* required columns;
* optional columns;
* value domains;
* active/inactive semantics;
* priority semantics;
* source-policy semantics;
* duplicate and validation rules;
* default selection rules;
* downstream RUN requirements;
* Telegram sequencing gates;
* ME-UNI02 implementation boundary;
* forbidden behavior.

## Forbidden behavior

ME-UNI01 does not introduce or approve:

* runtime code;
* test code;
* CSV data creation;
* fixture creation;
* provider refresh;
* SEC/EDGAR live calls;
* yfinance calls;
* live market data calls;
* internet calls;
* broker calls;
* Telegram delivery;
* email delivery;
* production report generation;
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

## Next sprint

The next implementation sprint is:

```text
ME-UNI02 - Implement canonical ticker universe loading and validation
```

ME-UNI02 must implement the loader and validation behavior defined here before ME-RUN16 can execute the first cached-source batch dry-run using the canonical ticker universe.
