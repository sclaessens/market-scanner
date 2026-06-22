# ME-UNI04 - Editable Professional Swing Universe contract

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-UNI - Ticker Universe

Status: COMPLETED BY ME-UNI04

## Purpose

ME-UNI04 defines the editable Professional Swing Universe contract for Market Engine.

The Professional Swing Universe is an operator-maintained candidate universe for professional swing-trading style review workflows. It is intended to support later import, normalization, source-support classification, cached-source scanning, readable operator reporting, and non-actionable candidate classification.

This contract is input governance only. It does not grant provider-refresh authority, live-data authority, delivery authority, portfolio authority, watchlist authority, trade authority, allocation authority, execution authority, BUY / SELL / HOLD semantics, ranking, scoring, urgency, conviction, tradeability, target-price authority, target-weight authority, or position-sizing authority.

## Contract identity

ME-UNI04 defines this contract:

```text
market-engine-editable-professional-swing-universe-v1
```

The contract governs the editable universe source shape, required columns, optional metadata, validation expectations, canonical-universe relationship, downstream sequencing, and implementation boundaries for ME-UNI05 and ME-UNI06.

## Approved repository path category

The approved path category for the editable Professional Swing Universe is:

```text
data/market_engine/ticker_universe/professional_swing_universe/
```

The default future seed path is:

```text
data/market_engine/ticker_universe/professional_swing_universe/professional_swing_universe.csv
```

ME-UNI04 does not create the CSV file. ME-UNI05 may import and normalize the first seed list under this path.

## Relationship to canonical ticker universe

The existing canonical ticker universe remains governed by:

```text
market-engine-canonical-ticker-universe-v1
```

and remains located at:

```text
data/market_engine/ticker_universe/ticker_universe.csv
```

The Professional Swing Universe must not silently replace the canonical ticker universe.

Rules:

* canonical universe = currently approved execution universe for cached-source RUN paths;
* Professional Swing Universe = editable candidate source for future universe expansion and review;
* overlap is allowed and must be explicit;
* a ticker present in the Professional Swing Universe is not automatically canonical;
* a ticker present in the Professional Swing Universe is not automatically source-supported;
* a ticker present in the Professional Swing Universe is not automatically preview-eligible or delivery-eligible;
* promotion from Professional Swing Universe to canonical execution universe requires a later approved governance step.

## File format

The editable Professional Swing Universe must be a UTF-8 CSV file with:

* one header row;
* comma separators;
* one instrument record per row;
* deterministic parsing independent of Excel, Numbers or Google Sheets;
* no hidden workbook formulas;
* no spreadsheet-only formatting dependencies;
* no duplicate header names after trimming;
* no unnamed columns;
* no comment rows outside explicit columns.

Blank lines may be ignored only if they contain no values in any column.

Whitespace around field values must be trimmed before validation.

## Required columns

ME-UNI05 and ME-UNI06 must treat the following columns as required:

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

### ticker

Operator-maintained display ticker.

Rules:

* required;
* trimmed;
* normalized to uppercase for comparison;
* must contain only uppercase letters, digits, dots or hyphens after normalization;
* must not contain spaces;
* must not be inferred from company name;
* provider-specific ticker variants must be stored only in optional provider metadata fields.

### name

Human-readable company or instrument name.

Rules:

* required;
* trimmed;
* operator visibility only;
* must not be used as runtime key;
* must not be used to infer missing tickers.

### market

Operator-maintained market or listing region indicator.

Allowed values for v1:

```text
USA
EURONEXT
LSE
XETRA
TSX
ASX
TSE
HKEX
OTHER
UNKNOWN
```

Rules:

* required;
* trimmed;
* normalized to uppercase;
* used for disambiguation and source-support classification;
* does not authorize exchange access, provider calls, broker access or live market data.

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
unknown
```

Rules:

* required;
* normalized to lowercase;
* used for eligibility and source-support triage;
* does not authorize trading, provider calls or broker actions.

### active

Whether the row remains active inside the editable universe.

Allowed values:

```text
true
false
```

Rules:

* required;
* normalized to lowercase;
* inactive rows remain visible for auditability;
* inactive rows must not be imported into default candidate-selection flows unless a later contract explicitly defines inclusion behavior.

### universe_status

Operator-maintained state of the row inside the editable universe.

Allowed values for v1:

```text
candidate
watching
research_required
needs_source_mapping
manual_review_only
blocked
rejected
```

Rules:

* required;
* normalized to lowercase;
* `candidate` and `watching` may be considered for future source-support classification;
* `research_required` and `needs_source_mapping` require manual review before source-supported execution;
* `manual_review_only` must not enter default automated execution;
* `blocked` and `rejected` must remain excluded from default downstream selection.

### source_policy_hint

Operator hint for future source-support handling.

Allowed values for v1:

```text
cached_source_candidate
source_mapping_required
manual_review_only
unsupported
unknown
```

Rules:

* required;
* normalized to lowercase;
* hint only;
* does not authorize source refresh;
* does not prove cached-source support;
* does not override actual source-support classification in ME-SR05 or later.

### operator_priority

Operator-maintained review ordering.

Rules:

* required;
* integer;
* values must be greater than or equal to `1`;
* duplicate values are allowed;
* default deterministic order is `operator_priority` ascending, then `ticker` ascending, then `market` ascending.

Operator priority is not score, ranking, conviction, urgency, tradeability, expected return, allocation priority or target weight.

### swing_profile

Operator-maintained style bucket for future reporting and review grouping.

Allowed values for v1:

```text
breakout
pullback
trend_continuation
mean_reversion
relative_strength
earnings_momentum
thematic_momentum
quality_compounder
turnaround
speculative_growth
unknown
```

Rules:

* required;
* normalized to lowercase;
* descriptive only;
* must not be interpreted as a trade setup or entry signal until a later approved non-actionable classification contract consumes it.

### liquidity_profile

Operator-maintained liquidity bucket.

Allowed values for v1:

```text
high
medium
low
unknown
```

Rules:

* required;
* normalized to lowercase;
* descriptive only;
* must not be converted into position sizing, execution advice or tradeability.

### volatility_profile

Operator-maintained volatility bucket.

Allowed values for v1:

```text
low
medium
high
extreme
unknown
```

Rules:

* required;
* normalized to lowercase;
* descriptive only;
* must not be converted into risk score, conviction, urgency, allocation or execution guidance.

### market_cap_profile

Operator-maintained company-size bucket.

Allowed values for v1:

```text
mega_cap
large_cap
mid_cap
small_cap
micro_cap
unknown
```

Rules:

* required;
* normalized to lowercase;
* descriptive only.

### theme

Operator-maintained thematic grouping.

Rules:

* required;
* trimmed;
* lowercase snake_case is recommended;
* may be `unknown`;
* descriptive only;
* must not be parsed for tickers, source policy, delivery, ranking or trade authority.

### sector

Operator-maintained sector grouping.

Rules:

* required;
* trimmed;
* lowercase snake_case is recommended;
* may be `unknown`;
* descriptive only.

### notes

Operator-maintained free-text context.

Rules:

* required column;
* value may be blank;
* must not be parsed for tickers, source policy, trade instructions, entry levels, stop levels, target prices, allocation or execution instructions.

## Optional columns

Recommended optional columns:

```text
exchange
currency
provider_ticker
sec_cik_hint
isin
figi
country
watchlist_group
source_support_status
source_support_reason
last_operator_reviewed_at
created_at
updated_at
```

Optional columns are metadata only unless a later contract explicitly promotes them to required governed fields.

Optional `source_support_status` may exist as imported metadata, but ME-UNI04 does not authorize it as final source truth. ME-SR05 must define the authoritative source-support classification behavior.

## Normalized Professional Swing Universe record

ME-UNI06 must expose each validated row as a normalized record with at least:

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
metadata
validation_state
```

`metadata` may include optional columns. It must not contain hidden authority flags beyond approved contract fields.

## Validation rules

ME-UNI06 must fail closed when the editable CSV is missing, unreadable, malformed or invalid.

Required validation:

* file exists at the configured path;
* file is parseable as UTF-8 CSV;
* header is present;
* required columns are present exactly once;
* no duplicate column names after trimming;
* no unnamed columns;
* each non-blank row has required values except `notes` may be blank;
* ticker normalization produces a non-empty value;
* normalized ticker matches the approved ticker character set;
* market is one of the allowed values;
* asset_type is one of the allowed values;
* active is `true` or `false`;
* universe_status is one of the allowed values;
* source_policy_hint is one of the allowed values;
* operator_priority is an integer greater than or equal to `1`;
* swing_profile is one of the allowed values;
* liquidity_profile is one of the allowed values;
* volatility_profile is one of the allowed values;
* market_cap_profile is one of the allowed values;
* duplicate rows with the same normalized ticker and market are rejected unless a later contract introduces effective dates;
* active default selection must exclude `manual_review_only`, `blocked`, `rejected`, `research_required`, and `needs_source_mapping` rows.

Validation errors must be explicit and operator-readable. They must include row number, field name, invalid value and failure reason where possible.

## Default selection semantics

Default future editable-universe selection may include only rows where:

```text
active=true
universe_status in candidate,watching
source_policy_hint in cached_source_candidate,unknown
```

Default ordering must be deterministic:

```text
operator_priority ascending
ticker ascending
market ascending
```

Default selection does not mean the ticker is source-supported or canonical. It only means the row is eligible for future import normalization and source-support classification.

## Downstream sequencing

The approved sequence after ME-UNI04 is:

```text
ME-UNI05 - Import and normalize Professional Swing Universe seed list
ME-UNI06 - Implement editable universe loader and validation
ME-SR05 - Classify source support for Professional Swing Universe
ME-RUN20 - Execute clean supported-universe cached-source scan
ME-OUT01 - Define readable operator report from dry-run artifacts
ME-CANDIDATE01 - Define non-actionable candidate classification contract
```

ME-UNI05 may create the initial seed CSV and documentation. It must not implement loader/runtime behavior unless explicitly re-scoped.

ME-UNI06 may implement loader and validation behavior. It must not classify source support as authoritative unless explicitly approved.

ME-SR05 may classify source support. It must not convert support classification into trade/action semantics.

ME-RUN20 may execute only clean supported-universe cached-source scans after ME-SR05 defines support eligibility.

ME-OUT01 may define readable operator reporting from dry-run artifacts. It must remain non-actionable unless a later approved contract changes that boundary.

ME-CANDIDATE01 may define candidate classification, but it must remain non-actionable and must not introduce BUY / SELL / HOLD, allocation, target price, position sizing, execution, ranking, scoring, urgency or conviction authority.

## ME-UNI05 implementation requirements

ME-UNI05 may implement only:

* seed-list import documentation;
* creation of the initial Professional Swing Universe CSV;
* manual normalization of supplied rows into the ME-UNI04 CSV shape;
* contract-level validation notes;
* backlog and roadmap synchronization;
* audit documentation.

ME-UNI05 must not introduce Python runtime code, tests, provider calls, source refresh, cached-source execution, reporting, Telegram delivery, portfolio writes, watchlist writes, scheduler behavior, UI behavior, Decision Engine behavior, BUY / SELL / HOLD semantics, allocation advice, target prices, target weights, position sizing, order generation, ranking, scoring, urgency, conviction, tradeability or execution advice.

## Acceptance criteria

ME-UNI04 is complete when documentation defines:

* contract identity;
* approved path category;
* relationship to canonical ticker universe;
* file format;
* required columns;
* optional columns;
* allowed values;
* validation rules;
* default selection semantics;
* normalized record shape;
* downstream sequencing;
* ME-UNI05 and ME-UNI06 boundaries;
* forbidden behavior.

## Forbidden behavior

ME-UNI04 does not introduce or approve:

* runtime code;
* test code;
* CSV data creation;
* fixture creation;
* provider refresh;
* SEC/EDGAR live calls;
* yfinance calls;
* live market data calls;
* external API calls;
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
* source-support authority;
* canonical-universe promotion;
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

The next sprint is:

```text
ME-UNI05 - Import and normalize Professional Swing Universe seed list
```

ME-UNI05 must create the initial editable Professional Swing Universe seed file under the approved path category without implementing loader/runtime behavior or downstream action authority.
