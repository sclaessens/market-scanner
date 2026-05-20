# Operational Sprint 4 Fundamental Data Source Developer Specification

## 1. Status

Status: DEVELOPER SPECIFICATION

This document defines the future staged MVP implementation scope for the Fundamental data source and quality classification contract.

This document is documentation-only. It does not implement code, tests, generated files, CSV artifacts, workflow changes, provider integration, Reporting changes, Telegram changes, portfolio changes, or Decision Engine changes.

Operational Sprint 4 is not closed, not certified complete, and not implemented by this document.

## 2. Scope Statement

This developer specification authorizes only a future staged MVP implementation plan for the Fundamental Layer.

The future implementation may add support for a manually maintained raw fundamentals artifact and descriptive quality classification rules.

The future implementation must not include provider/API integration.

The future implementation must preserve Fundamental Layer classification-only semantics.

The future implementation must not loosen the Decision Engine or permit allocation decisions without approved fundamental quality data.

## 3. Future Implementation Files That May Be Changed

The future implementation sprint may modify only files necessary for the staged MVP.

Permitted future implementation files:

- `scripts/core/build_fundamental_layer.py`
- focused Fundamental Layer tests under `tests/`
- documentation only if implementation findings require a governance-safe clarification

This developer specification itself does not modify those files.

## 4. Files and Areas That Must Not Be Changed Without Separate Authorization

The future implementation must not change the following unless separately authorized by a new governance decision:

- Decision Engine semantics
- Reporting logic
- Telegram logic
- portfolio logic
- scanner logic
- validation logic
- context logic
- timing logic
- portfolio intelligence logic
- provider/API integration
- generated CSV files
- generated processed artifacts
- manual edits to processed artifacts
- workflow files under `.github/workflows/`
- runtime orchestration beyond what is already governed

Explicitly forbidden future changes under this MVP:

- changing buy/sell/review decision semantics
- changing allocation authority
- changing Reporting or Telegram to infer decisions from fundamental quality
- changing portfolio handling
- adding API calls
- adding credentials or provider configuration
- committing generated `data/processed/*` outputs
- manually editing `data/processed/fundamental_quality.csv`

## 5. Raw Artifact Contract

### 5.1 Artifact Path

The staged MVP raw artifact path is:

```text
data/raw/fundamentals.csv
```

This file is a manually maintained source artifact.

It is optional at runtime.

If it is missing, the existing source-missing fallback behavior must remain unchanged.

### 5.2 Artifact Authority

`data/raw/fundamentals.csv` is source input only.

It does not create allocation authority, ranking authority, scoring authority, tradeability semantics, urgency semantics, conviction semantics, or hidden filtering.

The Fundamental Layer may read it only to produce descriptive quality classifications and source metadata.

### 5.3 Required Row Identity

Required raw artifact row identity:

- `ticker`
- `as_of_date`

The combination of `ticker` and `as_of_date` must be unique.

Duplicate rows for the same `ticker` and `as_of_date` must fail fast in the future implementation to prevent ambiguous source authority.

### 5.4 Required Columns

The MVP raw artifact must include at least:

- `ticker`
- `as_of_date`
- `source_name`
- `source_last_updated`
- `report_period`
- `currency`
- `revenue_growth_yoy`
- `eps_growth_yoy`
- `gross_margin`
- `operating_margin`
- `debt_to_equity`
- `free_cash_flow_positive`

### 5.5 Optional Columns

Optional fields may include:

- `net_income_growth_yoy`
- `return_on_equity`
- `current_ratio`
- `interest_coverage`
- `free_cash_flow_margin`
- `analyst_revision_trend`
- `notes`

Optional fields must remain descriptive only.

They must not become hidden ranking, scoring, tradeability, urgency, conviction, allocation, or filtering mechanisms.

### 5.6 Source Metadata Columns

Required source metadata columns:

- `source_name`
- `source_last_updated`
- `report_period`
- `currency`

The future implementation should propagate source metadata into Fundamental Layer output metadata where possible.

### 5.7 Freshness Metadata Expectations

The future implementation must calculate or expose:

- `source_last_updated`
- `source_freshness_days`

The MVP stale threshold is:

```text
120 calendar days from source_last_updated to the upstream opportunity date.
```

If the upstream opportunity date is unavailable or malformed, the implementation must fail fast or classify the row as `INSUFFICIENT_DATA` with invalid date metadata according to the existing Fundamental Layer error-handling style.

### 5.8 Duplicate Handling

Duplicate raw source rows for the same `ticker` and `as_of_date` must fail fast.

Raw-only tickers must not create output rows.

Raw-only rows may be ignored by the Fundamental Layer output. A future logging enhancement may record unused raw rows, but this MVP does not require that logging enhancement.

## 6. Fundamental Layer Behavior

The future implementation must follow these behaviors exactly.

### 6.1 Upstream Input

The Fundamental Layer must continue to read:

```text
data/processed/context_strength.csv
```

This remains the authoritative upstream input for the Fundamental Layer.

### 6.2 Row Preservation

The Fundamental Layer must preserve:

- upstream row count
- upstream row identity
- upstream ticker/date identity where present
- one output row per upstream row

The implementation must never remove upstream rows because of missing, partial, stale, invalid, or absent fundamental data.

### 6.3 Raw-Only Tickers

Rows present only in `data/raw/fundamentals.csv` must not be added to the Fundamental Layer output.

The raw artifact is not an opportunity source and must not expand the upstream universe.

### 6.4 Missing Raw Artifact

If `data/raw/fundamentals.csv` is missing, the Fundamental Layer must emit the existing fallback behavior:

- preserve all upstream rows
- emit `quality_state = INSUFFICIENT_DATA`
- emit `quality_metadata_status = source_missing`
- emit `source_data_status = source_missing`

Existing missing-source behavior must remain backward compatible.

### 6.5 Raw Artifact Present

If `data/raw/fundamentals.csv` exists, the Fundamental Layer must:

1. validate required columns
2. validate duplicate row identity
3. match source rows by ticker
4. use deterministic as-of rules
5. classify each upstream row into one descriptive quality state
6. output source metadata
7. preserve all upstream rows

### 6.6 Deterministic As-Of Rules

For each upstream row:

1. identify the upstream `ticker`
2. identify the upstream opportunity date from the existing date field used by the Fundamental Layer contract
3. select raw rows with matching `ticker`
4. select the latest `as_of_date` on or before the upstream opportunity date
5. if no raw row exists on or before the upstream opportunity date, classify as `INSUFFICIENT_DATA` with `source_data_status = row_missing`
6. calculate `source_freshness_days` from selected `source_last_updated` to upstream opportunity date
7. apply stale, invalid, partial, or sufficient classification rules

The implementation must not use future-dated raw rows for past opportunity dates.

### 6.7 Source Status Metadata

Every output row must include metadata explaining source status.

At minimum, metadata must explain whether the row is:

- source missing
- row missing
- source available
- partial
- stale
- invalid
- duplicate blocked by fail-fast behavior

### 6.8 Forbidden Runtime Semantics

The future implementation must never:

- filter rows
- rank rows
- score tradeability
- create allocation semantics
- create urgency semantics
- create conviction semantics
- create action semantics
- create buy/sell/hold/trim decisions
- bypass the Decision Engine
- alter Reporting or Telegram decision semantics

## 7. Quality States

### 7.1 `SUFFICIENT_DATA`

`SUFFICIENT_DATA` means a source row exists, source metadata is valid, all required MVP fields are present and valid, and `source_freshness_days` is less than or equal to 120 calendar days.

This state is descriptive only. It does not mean buy, sell, hold, rank, priority, conviction, urgency, or tradeability.

### 7.2 `PARTIAL_DATA`

`PARTIAL_DATA` means a source row exists and source metadata is usable, but one or more required MVP fields are missing or blank while at least one required data field remains valid.

This state is descriptive only. It does not permit filtering, ranking, allocation, or trade action.

### 7.3 `STALE_DATA`

`STALE_DATA` means a source row exists and is structurally usable, but `source_freshness_days` is greater than 120 calendar days.

Stale rows must not be promoted to `SUFFICIENT_DATA`.

### 7.4 `INSUFFICIENT_DATA`

`INSUFFICIENT_DATA` means the raw source is missing, no matching row exists, source metadata is invalid, required fields are materially absent, values are invalid enough to prevent descriptive classification, or date handling cannot be performed safely.

This state preserves the current safe fallback posture.

## 8. Metadata Fields

The future Fundamental Layer output must include or preserve expected metadata fields including:

- `quality_state`
- `quality_metadata_status`
- `source_data_status`
- `source_name`
- `source_last_updated`
- `source_freshness_days`
- `missing_required_fields`
- `partial_data_reason`
- `stale_data_reason`
- `invalid_data_reason`

### 8.1 `quality_metadata_status`

Expected values:

- `complete`
- `partial`
- `stale`
- `source_missing`
- `row_missing`
- `invalid`

### 8.2 `source_data_status`

Expected values:

- `source_available`
- `source_missing`
- `row_missing`
- `partial_data`
- `stale_data`
- `invalid_data`

Duplicate source rows should fail fast before output generation rather than appear as a normal row-level status.

### 8.3 Missing and Reason Fields

`missing_required_fields` should contain a deterministic delimiter-separated list of missing required columns for the affected row, or an empty value if none are missing.

`partial_data_reason`, `stale_data_reason`, and `invalid_data_reason` should be deterministic, concise, English-only, and descriptive.

## 9. Deterministic Classification Rules

### 9.1 Source Missing

If `data/raw/fundamentals.csv` does not exist:

- preserve all upstream rows
- classify every row as `INSUFFICIENT_DATA`
- set `quality_metadata_status = source_missing`
- set `source_data_status = source_missing`

### 9.2 Row Missing

If the raw artifact exists but no matching row is available for the upstream ticker using the deterministic as-of rule:

- classify as `INSUFFICIENT_DATA`
- set `quality_metadata_status = row_missing`
- set `source_data_status = row_missing`

### 9.3 Valid Source Row

If a matching raw row exists, source metadata is valid, all required fields are present and valid, and the row is fresh:

- classify as `SUFFICIENT_DATA`
- set `quality_metadata_status = complete`
- set `source_data_status = source_available`

### 9.4 Partial Source Row

If a matching raw row exists and source metadata is usable, but one or more required MVP fields are missing or blank:

- classify as `PARTIAL_DATA` when at least one required data field remains valid
- set `quality_metadata_status = partial`
- set `source_data_status = partial_data`
- populate `missing_required_fields`
- populate `partial_data_reason`

If materially all required data fields are missing or invalid, classify as `INSUFFICIENT_DATA`.

### 9.5 Stale Source Row

If a matching row exists but `source_freshness_days > 120`:

- classify as `STALE_DATA`
- set `quality_metadata_status = stale`
- set `source_data_status = stale_data`
- populate `stale_data_reason`

Stale classification should take precedence over otherwise sufficient data.

### 9.6 Invalid Numeric Value

Required numeric fields must parse deterministically as numbers.

Invalid numeric values include:

- non-numeric text
- malformed percentages
- empty strings in required numeric fields
- values that cannot be converted by the approved parser

Invalid numeric values must not be coerced silently.

Rows with invalid numeric values must classify as `PARTIAL_DATA` or `INSUFFICIENT_DATA` depending on severity and must populate `invalid_data_reason`.

### 9.7 Invalid Boolean Value

`free_cash_flow_positive` must use an approved deterministic boolean representation.

Approved values:

- `true`
- `false`
- `TRUE`
- `FALSE`
- `1`
- `0`

Other values are invalid and must populate `invalid_data_reason`.

### 9.8 Duplicate Source Rows

If duplicate raw rows exist for the same `ticker` and `as_of_date`, the implementation must fail fast before producing output.

This prevents ambiguous source authority.

### 9.9 Unknown Raw-Only Ticker

If the raw artifact contains tickers not present in the upstream context artifact, those tickers must not create output rows.

The Fundamental Layer output must remain based on upstream rows only.

### 9.10 Date or As-Of Mismatch

If no `as_of_date` exists on or before the upstream opportunity date, classify the upstream row as `INSUFFICIENT_DATA` with row-missing metadata.

If date fields are malformed, classify affected rows as `INSUFFICIENT_DATA` with invalid metadata or fail fast if malformed dates prevent deterministic processing.

Future implementation should prefer fail-fast behavior for malformed source-level dates that make global contract interpretation unsafe.

## 10. Future Test Requirements

Codex must add focused tests during the future implementation sprint.

### 10.1 Missing Raw Artifact

Test that missing `data/raw/fundamentals.csv` preserves existing fallback behavior.

Expected result:

- all upstream rows preserved
- `quality_state = INSUFFICIENT_DATA`
- `quality_metadata_status = source_missing`
- `source_data_status = source_missing`

### 10.2 Valid Raw Data

Test that valid raw data produces `SUFFICIENT_DATA`.

Expected result:

- matching upstream rows classify as `SUFFICIENT_DATA`
- source metadata is propagated
- row count is preserved
- no raw-only rows are added

### 10.3 Missing Required Fields

Test that missing required fields produce `PARTIAL_DATA` or `INSUFFICIENT_DATA` according to the rules.

Expected result:

- missing fields appear in `missing_required_fields`
- partial reason metadata is populated
- rows are preserved

### 10.4 Stale Source Rows

Test that stale source rows produce `STALE_DATA`.

Expected result:

- `source_freshness_days > 120`
- `quality_state = STALE_DATA`
- stale reason metadata is populated

### 10.5 Invalid Numeric Values

Test malformed numeric values.

Expected result:

- invalid values are not silently coerced
- affected rows classify deterministically
- invalid reason metadata is populated

### 10.6 Invalid Boolean Values

Test malformed `free_cash_flow_positive` values.

Expected result:

- invalid boolean values classify deterministically
- invalid reason metadata is populated

### 10.7 Duplicate Raw Source Rows

Test duplicate raw rows with the same `ticker` and `as_of_date`.

Expected result:

- implementation fails fast before output generation
- failure message is deterministic and English-only

### 10.8 Raw-Only Tickers

Test that raw-only tickers do not create Fundamental Layer output rows.

Expected result:

- output row count equals upstream row count
- raw-only ticker is absent from output if absent upstream

### 10.9 Upstream Row Count Preservation

Test that output row count always equals upstream input row count across valid, missing, partial, stale, and invalid source states.

### 10.10 Upstream Row Identity Preservation

Test that upstream ticker/date identity is preserved and not replaced by raw artifact dates.

### 10.11 Forbidden Semantics

Test that forbidden semantics do not appear in Fundamental Layer output fields or runtime behavior.

Forbidden examples include:

- ranking
- score authority
- tradeability
- allocation
- urgency
- conviction
- hidden filtering
- buy/sell/hold/trim decisions
- Decision Engine bypass

### 10.12 Decision Engine Not Modified

Test or verify that the implementation does not require Decision Engine changes.

The future implementation PR should not modify Decision Engine files.

### 10.13 Reporting and Telegram Not Modified

Test or verify that Reporting and Telegram files are not modified.

The future implementation PR should not modify Reporting or Telegram files.

## 11. Acceptance Criteria

The future staged MVP implementation is acceptable only if all criteria below are satisfied:

- implementation remains limited to the Fundamental Layer and focused tests
- no provider/API integration is introduced
- no Decision Engine changes are made
- no Reporting changes are made
- no Telegram changes are made
- no portfolio logic changes are made
- no scanner, validation, context, timing, or portfolio intelligence logic changes are made
- missing raw artifact fallback behavior remains intact
- `data/raw/fundamentals.csv` is optional at runtime
- valid raw data can produce `SUFFICIENT_DATA`
- partial raw data can produce `PARTIAL_DATA` or `INSUFFICIENT_DATA`
- stale raw data can produce `STALE_DATA`
- invalid data is handled deterministically
- duplicate raw source rows fail fast
- raw-only tickers do not create output rows
- upstream row count is preserved
- upstream row identity is preserved
- all new tests pass
- full test suite passes if run
- no generated CSV/data files are committed
- governance forbidden semantics remain absent
- repository content remains English-only

## 12. Implementation Non-Goals

This MVP does not:

- select a provider
- call an API
- automate data ingestion
- add credentials
- add provider configuration
- tune Decision Engine decisions
- change buy/sell/review logic
- add ranking
- add scoring authority
- add tradeability semantics
- add urgency semantics
- add conviction semantics
- change Telegram UX
- change Reporting schema
- change Reporting logic
- change portfolio handling
- change scanner behavior
- change validation behavior
- change context behavior
- change timing behavior
- change portfolio intelligence behavior

## 13. Backlog Impact Assessment

BL-0015 remains sufficient for this staged MVP implementation path:

- `BL-0015 — Define and implement approved Fundamental data source and quality classification contract`

No new backlog item is required because this developer specification remains within the scope already captured by BL-0015.

Backlog impact assessment:
- No new backlog items identified.

## 14. Recommended Next Step

After this developer specification is reviewed and merged, create a separate implementation prompt for Codex.

The implementation prompt must instruct Codex to implement only the staged MVP described here, with focused tests, no provider/API integration, no Decision Engine changes, no Reporting or Telegram changes, no generated artifact commits, and no runtime behavior outside the Fundamental Layer.