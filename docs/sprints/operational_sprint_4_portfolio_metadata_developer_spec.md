# Operational Sprint 4 Portfolio Metadata Developer Specification

## 1. Status

Status: DEVELOPER SPECIFICATION

This document defines the future staged MVP implementation scope for Portfolio Metadata source integration.

This document is documentation-only. It does not implement code, tests, generated files, CSV artifacts, workflow changes, provider integration, Reporting changes, Telegram changes, Decision Engine changes, or runtime orchestration changes.

Operational Sprint 4 is not closed, not certified complete, and not implemented by this document.

## 2. Scope Statement

This developer specification authorizes only a future staged MVP implementation plan for the separate portfolio metadata artifact:

```text
data/portfolio/portfolio_metadata.csv
```

The future implementation must preserve Portfolio Intelligence as descriptive and classification-only.

The future implementation must not loosen the Decision Engine.

The future implementation must not authorize Reporting or Telegram changes.

The future implementation must not authorize provider/API integration.

The future implementation must not introduce allocation authority, ranking authority, urgency, conviction, tradeability, hidden filtering, Reporting-based decision logic, or Decision Engine bypass.

## 3. Future Implementation Files That May Be Changed

The future implementation sprint may modify only files necessary for the Portfolio Metadata MVP.

Permitted future implementation files:

- `scripts/core/build_portfolio_intelligence.py`
- focused Portfolio Intelligence tests under `tests/`
- documentation only if implementation findings require a governance-safe clarification

This developer specification itself does not modify those files.

## 4. Files and Areas That Must Not Be Changed Without Separate Authorization

The future implementation must not change the following unless separately authorized by a new governance decision:

- Decision Engine semantics
- Reporting logic
- Telegram logic
- scanner logic
- validation logic
- context logic
- fundamental logic
- timing logic
- portfolio positions repair
- provider/API integration
- generated CSV files
- generated processed artifacts
- manual edits to processed artifacts
- runtime orchestration
- GitHub workflow files

Explicitly forbidden future changes under this MVP:

- changing buy/sell/review decision semantics
- changing allocation authority
- changing Reporting or Telegram to infer decisions from portfolio metadata
- repairing or rewriting active portfolio positions as part of this MVP
- adding API calls
- adding credentials or provider configuration
- committing generated `data/processed/*` outputs
- manually editing processed Portfolio Intelligence or Decision Engine artifacts

## 5. Portfolio Metadata Artifact Contract

### 5.1 Artifact Path

The staged MVP portfolio metadata artifact path is:

```text
data/portfolio/portfolio_metadata.csv
```

This file is a manually maintained source artifact.

It is optional at runtime.

If it is missing, existing Portfolio Intelligence partial metadata behavior must remain unchanged.

### 5.2 Artifact Authority

`data/portfolio/portfolio_metadata.csv` is descriptive metadata input only.

It does not create allocation authority, ranking authority, scoring authority, tradeability semantics, urgency semantics, conviction semantics, action semantics, Reporting semantics, or Decision Engine bypass.

Portfolio Intelligence may read it only to produce descriptive metadata completeness and sector exposure classifications.

### 5.3 Required Row Identity

Required row identity:

- `ticker`

Each normalized ticker may appear at most once.

Duplicate rows for the same normalized ticker must fail fast in the future implementation to prevent ambiguous metadata authority.

Ticker matching must be normalized by trimming whitespace and using uppercase matching.

### 5.4 Required Columns

The MVP metadata artifact must include at least:

- `ticker`
- `sector`
- `industry`
- `asset_class`
- `currency`
- `metadata_source`
- `metadata_last_updated`

### 5.5 Optional Columns

Optional fields may include:

- `country`
- `region`
- `exchange`
- `sector_taxonomy`
- `industry_group`
- `notes`

Optional fields must remain descriptive only.

They must not become hidden ranking, scoring, tradeability, urgency, conviction, allocation, or filtering mechanisms.

### 5.6 Metadata Source Columns

Required source metadata columns:

- `metadata_source`
- `metadata_last_updated`

Recommended optional provenance columns:

- `sector_taxonomy`
- `notes`

### 5.7 Freshness Metadata Expectations

The MVP stale threshold is approved as:

```text
365 calendar days from metadata_last_updated to the upstream opportunity date or run date used by Portfolio Intelligence.
```

`metadata_last_updated` must be parseable as an ISO date in `YYYY-MM-DD` form.

If Portfolio Intelligence cannot safely determine an opportunity date or comparable run date, it must classify the metadata as incomplete or invalid rather than treating it as complete.

### 5.8 Duplicate Handling

Duplicate metadata rows for the same normalized `ticker` must fail fast.

### 5.9 Metadata-Only Ticker Handling

Rows present in `portfolio_metadata.csv` but absent from the upstream Portfolio Intelligence universe must not create output rows.

Metadata-only tickers may be ignored by output generation. A future logging enhancement may record unused metadata rows, but that is not required in this MVP.

### 5.10 Missing Metadata Handling

If `portfolio_metadata.csv` is missing, the implementation must preserve existing Portfolio Intelligence behavior and keep metadata incomplete.

If the artifact exists but no matching row exists for a ticker, the affected output row must be preserved and classified as incomplete with a deterministic reason.

No row may be removed because metadata is missing, partial, stale, invalid, or absent.

## 6. Proposed Taxonomy

### 6.1 Asset Class Taxonomy

Accepted `asset_class` values for the MVP:

- `Equity`
- `ETF`
- `Cash`
- `Other`

Values outside this set are invalid.

### 6.2 Sector Taxonomy

The safer staged MVP approach is controlled free-text sector validation with source taxonomy metadata.

For the MVP:

- `sector` must be non-empty
- `sector` must be treated as descriptive text
- `sector_taxonomy` is optional but recommended
- if `sector_taxonomy` is present, it should identify the taxonomy source, for example `GICS`, `ICB`, `manual`, or another descriptive source label

The MVP does not require hard-coding a full GICS taxonomy list.

Rationale: a controlled non-empty sector field with provenance avoids premature taxonomy overengineering while still preventing silent missing-sector metadata.

### 6.3 Industry Field

`industry` must be non-empty for complete metadata.

It remains descriptive and must not be interpreted as ranking, score, or allocation authority.

### 6.4 Currency Field

`currency` must be non-empty for complete metadata.

Recommended values should use ISO-style currency codes such as `USD`, `EUR`, or `GBP`, but the MVP only requires deterministic non-empty validation unless a stricter taxonomy is introduced in the future developer implementation.

## 7. Portfolio Intelligence Behavior

The future implementation must follow these behaviors exactly.

### 7.1 Existing Inputs

Portfolio Intelligence must continue reading its existing authoritative upstream inputs.

The portfolio metadata artifact is an optional descriptive enrichment source only.

### 7.2 Optional Metadata Read

If `data/portfolio/portfolio_metadata.csv` exists, Portfolio Intelligence may read it and use it to classify metadata completeness descriptively.

If the file is missing, existing partial metadata behavior must remain intact.

### 7.3 Row Preservation

Portfolio Intelligence output must preserve:

- upstream row count
- upstream row identity
- one output row per upstream row
- upstream opportunity universe

The implementation must never remove rows because of missing, partial, stale, invalid, or absent metadata.

### 7.4 Metadata-Only Tickers

Rows present only in `portfolio_metadata.csv` must not be added to Portfolio Intelligence output.

The metadata artifact is not an opportunity source and must not expand the row universe.

### 7.5 Matching Rule

Metadata rows must match output rows by normalized ticker.

Normalization rule:

- trim whitespace
- uppercase ticker

### 7.6 Metadata Completeness Classification

For each output row, Portfolio Intelligence must classify metadata completeness descriptively using the metadata states defined in this specification.

### 7.7 Source Metadata Propagation

Where appropriate, Portfolio Intelligence should propagate or use:

- `sector`
- `industry`
- `asset_class`
- `currency`
- `metadata_source`
- `metadata_last_updated`
- optional `sector_taxonomy`

The exact output schema changes must be limited to descriptive metadata fields and must not create decision/action semantics.

### 7.8 Duplicate Metadata Rows

Duplicate normalized ticker rows must fail fast before output generation.

### 7.9 Forbidden Runtime Semantics

The future implementation must never:

- filter rows
- rank rows
- score tradeability
- create allocation semantics
- create urgency semantics
- create conviction semantics
- create action semantics
- create buy/sell/hold/trim decisions
- create Reporting or Telegram decision semantics
- bypass the Decision Engine

## 8. Metadata States and Field Mapping

### 8.1 `COMPLETE`

`COMPLETE` means a matching metadata row exists, all required metadata fields are present and valid, `metadata_last_updated` is valid, metadata freshness is less than or equal to 365 calendar days, and no duplicate metadata ambiguity exists.

`COMPLETE` may support:

- `portfolio_metadata_status = COMPLETE`
- `portfolio_metadata_reason` describing complete source-supported metadata
- descriptive sector exposure enrichment

It does not mean buy, sell, hold, rank, priority, conviction, urgency, or tradeability.

### 8.2 `PARTIAL`

`PARTIAL` means a matching metadata row exists, but one or more required metadata fields are missing while enough metadata exists to identify the row as partially source-supported.

`PARTIAL` must keep portfolio metadata incomplete.

### 8.3 `MISSING`

`MISSING` means the metadata artifact is missing or no matching metadata row exists for the ticker.

`MISSING` must keep portfolio metadata incomplete.

### 8.4 `STALE`

`STALE` means a matching metadata row exists and may otherwise be structurally valid, but `metadata_last_updated` is older than the 365-day threshold.

`STALE` must keep portfolio metadata incomplete.

### 8.5 `INVALID`

`INVALID` means metadata exists but contains invalid taxonomy, invalid asset class, malformed dates, missing source metadata, or other invalid required fields.

`INVALID` must keep portfolio metadata incomplete.

### 8.6 Mapping to Existing Portfolio Intelligence Fields

Metadata state should inform the following fields descriptively:

- `portfolio_metadata_status`
- `portfolio_metadata_reason`
- `sector_exposure_state`
- `diversification_state`
- `concentration_state`
- `overlap_state`

Recommended MVP mapping:

| Metadata State | `portfolio_metadata_status` | Expected Decision Metadata Effect |
|---|---|---|
| `COMPLETE` | `COMPLETE` | Metadata completeness blocker may be cleared for that row if all other Portfolio Intelligence requirements are also satisfied. |
| `PARTIAL` | `PARTIAL` | Metadata remains incomplete. |
| `MISSING` | `PARTIAL` | Metadata remains incomplete. |
| `STALE` | `PARTIAL` | Metadata remains incomplete. |
| `INVALID` | `PARTIAL` | Metadata remains incomplete. |

This mapping is descriptive and does not authorize allocation decisions.

### 8.7 Sector Exposure and Related States

When metadata is `COMPLETE`, Portfolio Intelligence may classify sector exposure descriptively.

The MVP should avoid complex concentration or diversification changes unless already supported by existing Portfolio Intelligence behavior.

If required concentration/diversification logic needs additional design, implementation must stop and request governance clarification rather than inventing new allocation semantics.

## 9. Deterministic Classification Rules

### 9.1 Metadata Source Missing

If `data/portfolio/portfolio_metadata.csv` does not exist:

- preserve all output rows
- preserve existing partial metadata behavior
- do not fail the build solely because the optional artifact is missing

### 9.2 Metadata Row Missing

If the artifact exists but no matching normalized ticker row exists:

- preserve the output row
- classify metadata as `MISSING`
- keep `portfolio_metadata_status` incomplete
- populate a deterministic metadata reason

### 9.3 Valid Complete Metadata

If a matching row exists, required fields are present, asset class is valid, dates are valid, and freshness is within 365 days:

- classify metadata as `COMPLETE`
- support `portfolio_metadata_status = COMPLETE` where no other metadata blocker remains
- propagate descriptive metadata where appropriate

### 9.4 Partial Metadata

If a matching row exists but one or more required fields are missing:

- classify metadata as `PARTIAL`
- keep `portfolio_metadata_status` incomplete
- populate a deterministic metadata reason identifying missing fields

### 9.5 Stale Metadata

If `metadata_last_updated` is older than 365 calendar days:

- classify metadata as `STALE`
- keep `portfolio_metadata_status` incomplete
- populate a deterministic stale metadata reason

### 9.6 Invalid Sector Value

For the staged MVP, invalid sector values are:

- empty sector values
- values that cannot be represented as non-empty descriptive text

If `sector_taxonomy` is present but malformed or blank while relied upon, the row may be classified as `PARTIAL` or `INVALID` according to implementation detail, but it must not be treated as `COMPLETE` if the metadata contract is ambiguous.

### 9.7 Invalid Asset Class Value

If `asset_class` is not one of the approved values, classify metadata as `INVALID` and keep metadata incomplete.

Approved values:

- `Equity`
- `ETF`
- `Cash`
- `Other`

### 9.8 Invalid or Malformed `metadata_last_updated`

If `metadata_last_updated` is missing or cannot be parsed as an ISO date, classify metadata as `INVALID` and keep metadata incomplete.

### 9.9 Duplicate Metadata Rows

Duplicate normalized ticker rows must fail fast before output generation.

### 9.10 Metadata-Only Ticker

Metadata rows for tickers absent from the Portfolio Intelligence upstream universe must not create output rows.

### 9.11 Missing Required Fields

Required metadata fields missing from the artifact schema must fail fast because the artifact cannot be interpreted safely.

Required field values missing for a row should be classified at row level as `PARTIAL`, `MISSING`, or `INVALID` according to the rules above.

## 10. Future Test Requirements

Codex must add focused Portfolio Intelligence tests during the future implementation sprint.

### 10.1 Missing Metadata Artifact

Test that missing `data/portfolio/portfolio_metadata.csv` preserves existing partial behavior.

Expected result:

- output row count preserved
- existing portfolio metadata status behavior remains unchanged
- no build failure solely due to missing optional artifact

### 10.2 Valid Complete Metadata

Test that valid complete metadata can produce complete metadata status where appropriate.

Expected result:

- metadata source row matches by normalized ticker
- `portfolio_metadata_status` can become `COMPLETE` for rows with complete metadata and no other metadata blocker
- descriptive metadata is propagated where appropriate

### 10.3 Missing Required Field Values

Test that missing required metadata values produce partial or missing metadata status.

Expected result:

- row preserved
- incomplete metadata status retained
- deterministic reason identifies missing fields

### 10.4 Stale Metadata

Test that stale metadata remains incomplete.

Expected result:

- metadata older than 365 days does not become complete
- deterministic stale reason is populated

### 10.5 Invalid Sector Values

Test empty or invalid sector values.

Expected result:

- row preserved
- metadata remains incomplete
- deterministic invalid or partial reason is populated

### 10.6 Invalid Asset Class Values

Test unsupported `asset_class` values.

Expected result:

- metadata classified as invalid
- metadata status remains incomplete
- invalid reason is deterministic

### 10.7 Duplicate Metadata Rows

Test duplicate normalized ticker rows.

Expected result:

- implementation fails fast before output generation
- failure message is deterministic and English-only

### 10.8 Metadata-Only Tickers

Test metadata-only tickers.

Expected result:

- no additional output rows
- row count equals upstream row count

### 10.9 Row Count Preservation

Test that output row count is preserved across missing, complete, partial, stale, invalid, and duplicate-free metadata states.

### 10.10 Row Identity Preservation

Test that upstream ticker/date identity is preserved and not replaced by metadata artifact values.

### 10.11 Metadata Status Transitions

Test deterministic transitions between partial and complete metadata states.

### 10.12 Sector Exposure Classification Remains Descriptive

Test that sector exposure classification does not create allocation, ranking, urgency, conviction, tradeability, or action semantics.

### 10.13 Decision Engine Not Modified

Verify that the future implementation PR does not modify Decision Engine files.

### 10.14 Reporting and Telegram Not Modified

Verify that the future implementation PR does not modify Reporting or Telegram files.

### 10.15 Forbidden Semantics

Verify no forbidden semantics are introduced in Portfolio Intelligence output fields or runtime behavior.

Forbidden examples include:

- allocation outside the Decision Engine
- ranking authority
- score authority
- tradeability
- urgency
- conviction
- hidden filtering
- buy/sell/hold/trim decisions
- Decision Engine bypass
- Reporting or Telegram decision semantics

## 11. Acceptance Criteria

The future staged MVP implementation is acceptable only if all criteria below are satisfied:

- implementation remains limited to Portfolio Intelligence and focused tests
- no provider/API integration is introduced
- no Decision Engine changes are made
- no Reporting changes are made
- no Telegram changes are made
- no scanner, validation, context, fundamental, or timing changes are made
- no portfolio position source repair is performed
- no runtime orchestration changes are made
- no GitHub workflow changes are made
- no generated CSV/data files are committed
- missing metadata artifact fallback remains intact
- metadata-only tickers do not create rows
- complete metadata can produce complete metadata status where no other metadata blocker remains
- partial metadata remains incomplete
- stale metadata remains incomplete
- invalid metadata remains incomplete
- missing metadata remains incomplete
- upstream row count is preserved
- upstream row identity is preserved
- all new tests pass
- full test suite passes if run
- governance forbidden semantics remain absent
- repository content remains English-only

## 12. Implementation Non-Goals

This MVP does not:

- select a provider
- call an API
- automate metadata ingestion
- repair portfolio positions
- tune Decision Engine decisions
- change buy/sell/review logic
- add allocation logic
- add ranking
- add scoring authority
- add tradeability semantics
- add urgency semantics
- add conviction semantics
- change Telegram UX
- change Reporting schema
- change Reporting logic
- change portfolio position source semantics
- change scanner sector source semantics
- change context sector source semantics
- implement hybrid metadata precedence
- implement provider-assisted metadata refresh

## 13. Backlog Impact Assessment

`BL-0016 — Define approved Portfolio Metadata and Sector Exposure contract` remains sufficient for this staged MVP implementation path.

No new backlog item is required because this developer specification remains within the scope already captured by BL-0016.

Backlog impact assessment:
- No new backlog items identified.

## 14. Recommended Next Step

After this developer specification is reviewed and merged, create a separate implementation prompt for Codex.

The implementation prompt must instruct Codex to implement only the Portfolio Metadata MVP described here, with focused tests, no Decision Engine changes, no Reporting or Telegram changes, no provider/API integration, no generated artifact commits, and no runtime behavior outside Portfolio Intelligence metadata handling.