# Operational Sprint 4 Portfolio Metadata Design

## 1. Status

Status: LEVEL 2 DESIGN

This document defines a governed Level 2 design for Portfolio Metadata and Sector Exposure source integration.

This is a documentation-only design. It does not authorize implementation, runtime code changes, tests, generated artifact changes, CSV edits, provider integration, Reporting changes, Telegram changes, or Decision Engine semantic changes.

No sprint is closed or certified complete by this document.

## 2. Problem Statement

The staged Fundamental data source MVP has been implemented and merged. Operational verification confirmed that `SUFFICIENT_DATA` now reaches Decision Engine inputs for tested rows.

Fundamental data is therefore no longer the immediate blocker for those tested rows.

However, all 291 final decisions remained `REVIEW` because Portfolio Intelligence and the Decision Engine still receive incomplete portfolio metadata.

The observed blocking state is:

- `portfolio_metadata_status = PARTIAL`
- `portfolio_decision_state = PORTFOLIO_METADATA_REVIEW_REQUIRED`
- `opportunity_decision_state = INSUFFICIENT_DECISION_METADATA`
- `arbitration_state = MISSING_METADATA`

The root cause is missing portfolio sector metadata. Current portfolio source files do not provide explicit sector, industry, asset class, metadata source, or metadata freshness fields.

The Decision Engine must not be loosened. Missing metadata must be resolved by defining an approved descriptive metadata source contract, not by bypassing required metadata checks.

## 3. Current-State Contract

### 3.1 Current Portfolio Source Files

Current portfolio source files include:

```text
data/portfolio/portfolio_positions.csv
data/portfolio/portfolio_review.csv
```

`portfolio_positions.csv` currently contains holdings and position fields such as ticker, quantity, average cost, last price, market value, unrealized P/L, status, last action, and last action timestamp.

`portfolio_review.csv` currently contains review and risk-observation fields such as ticker, quantity, average cost, last price, P/L percentage, moving averages, exposure state, drawdown state, risk state, portfolio reason, and review timestamp.

Neither file currently contains a governed portfolio metadata contract for sector exposure.

### 3.2 Current Portfolio Intelligence Behavior

Portfolio Intelligence consumes upstream opportunity rows and portfolio source data to enrich rows descriptively.

Current output includes fields such as:

- `in_portfolio`
- `portfolio_position_state`
- `exposure_state`
- `diversification_state`
- `concentration_state`
- `overlap_state`
- `sector_exposure_state`
- `portfolio_metadata_status`
- `portfolio_metadata_reason`

The current investigation found `portfolio_metadata_status = PARTIAL` for all 291 rows.

### 3.3 Current Decision Engine Behavior

The Decision Engine correctly keeps final actions at `REVIEW` when portfolio metadata is incomplete.

It reports missing metadata through fields such as:

- `allocation_decision = REVIEW_REQUIRED`
- `execution_decision = REVIEW_REQUIRED`
- `portfolio_decision_state = PORTFOLIO_METADATA_REVIEW_REQUIRED`
- `opportunity_decision_state = INSUFFICIENT_DECISION_METADATA`
- `arbitration_state = MISSING_METADATA`

This behavior is governance-safe and must not be loosened by this design.

### 3.4 Current Portfolio Metadata Limitations

Current limitations:

- no approved portfolio metadata artifact
- no explicit `sector` field
- no explicit `industry` field
- no explicit `asset_class` field
- no metadata source provenance
- no metadata freshness timestamp
- no sector taxonomy contract
- no duplicate metadata handling
- no complete metadata criteria
- no deterministic missing metadata rules

## 4. Approved Portfolio Metadata Source Options

This section compares possible approaches without implementing them.

### 4.1 Option A — Add Sector Fields Directly to `portfolio_positions.csv`

Description: extend `data/portfolio/portfolio_positions.csv` with sector, industry, asset class, and metadata source fields.

Governance fit: moderate. The option is simple, but it mixes holdings state with descriptive classification metadata.

Separation of concerns: weak to moderate. Position quantity and metadata would be stored together, increasing the risk that metadata maintenance appears to alter position state.

Operational complexity: low. Only one file would need to be maintained.

Freshness and auditability: moderate. Metadata updates would be auditable, but intermingled with holdings changes.

Reliability: moderate. The approach is straightforward but risks accidental edits to position-critical fields.

Testing impact: manageable. Tests would need to validate expanded portfolio position schema and missing metadata behavior.

Risks:

- mixes holdings and metadata responsibilities
- increases accidental edit risk in position files
- can blur whether a file change is a position update or metadata update
- may complicate future provider-assisted metadata refresh

Recommendation: not preferred for the first governed implementation unless repository constraints require a single portfolio source file.

### 4.2 Option B — Create Separate `data/portfolio/portfolio_metadata.csv`

Description: create a dedicated manually maintained descriptive metadata artifact.

Proposed artifact:

```text
data/portfolio/portfolio_metadata.csv
```

Governance fit: strong. The file cleanly separates holdings state from descriptive metadata.

Separation of concerns: strong. Holdings remain in portfolio position files; metadata lives in a dedicated source file.

Operational complexity: moderate. Operators must maintain one additional file, but the responsibilities are clear.

Freshness and auditability: strong. Metadata can have its own source and freshness fields independent of holdings updates.

Reliability: strong. Validation can be focused on metadata completeness, taxonomy, and freshness without touching position quantities.

Testing impact: favorable. Missing file, missing rows, duplicate rows, stale metadata, invalid taxonomy, and row-preservation scenarios can be tested independently.

Risks:

- introduces an additional source artifact
- requires deterministic ticker matching
- requires rules for metadata-only tickers
- requires clear completeness criteria

Recommendation: preferred Level 2 approach.

### 4.3 Option C — Reuse Scanner or Context Sector Metadata If Available

Description: reuse sector information from scanner, context, or another upstream opportunity artifact if those fields exist and are source-supported.

Governance fit: conditional. This can be acceptable only if the upstream sector metadata has explicit provenance and freshness semantics.

Separation of concerns: moderate to weak. Portfolio metadata would depend on opportunity-universe metadata, which may not cover portfolio-only holdings.

Operational complexity: low if stable fields already exist, but higher if provenance must be reconstructed.

Freshness and auditability: uncertain unless source fields and timestamps are already governed.

Reliability: uncertain because scanner or context outputs may change independently of portfolio metadata needs.

Testing impact: more complex. Tests would need to prove upstream sector fields are stable, source-supported, and complete enough for portfolio metadata.

Risks:

- hidden dependency on opportunity scan universe
- portfolio-only holdings may remain uncovered
- unclear metadata provenance
- stale or missing sector values may be hard to audit

Recommendation: not recommended as the primary first implementation path without stronger repository evidence.

### 4.4 Option D — Hybrid Approach

Description: use `portfolio_metadata.csv` as the authoritative source while optionally allowing scanner/context fields to fill non-portfolio opportunity metadata in future governed work.

Governance fit: strong if precedence rules are explicit.

Separation of concerns: strong if portfolio metadata remains authoritative for portfolio decisions.

Operational complexity: medium to high due to precedence and source-freshness handling.

Freshness and auditability: strong if each source has explicit provenance and freshness metadata.

Reliability: strong after design, but too complex for the first implementation step.

Testing impact: larger test surface due to precedence and fallback scenarios.

Risks:

- source precedence ambiguity
- hidden inference if not strictly governed
- broader scope than required to resolve the current blocker

Recommendation: defer until the dedicated portfolio metadata artifact contract is implemented and validated.

## 5. Recommended Level 2 Approach

Recommended approach: create a separate portfolio metadata artifact:

```text
data/portfolio/portfolio_metadata.csv
```

This is the preferred first governed implementation path because it provides the cleanest separation of concerns, strongest auditability, and smallest deterministic contract.

The first implementation should be limited to manually maintained metadata. Provider/API integration should remain out of scope until the artifact contract is stable.

## 6. Proposed Portfolio Metadata Artifact Contract

### 6.1 Artifact Purpose

`data/portfolio/portfolio_metadata.csv` is a descriptive source artifact for portfolio metadata and sector exposure classification.

It is not an allocation artifact. It does not create trading decisions, rankings, urgency, conviction, tradeability, or hidden filtering.

### 6.2 Row Identity

Required row identity:

- `ticker`

Each ticker may appear at most once in the metadata artifact.

### 6.3 Required Columns

Required columns:

- `ticker`
- `sector`
- `industry`
- `asset_class`
- `currency`
- `metadata_source`
- `metadata_last_updated`

### 6.4 Optional Columns

Optional columns may include:

- `country`
- `region`
- `exchange`
- `sector_taxonomy`
- `industry_group`
- `notes`

Optional columns must remain descriptive and must not introduce ranking, scoring authority, tradeability, urgency, conviction, allocation, or filtering semantics.

### 6.5 Source Metadata Fields

Required source metadata fields:

- `metadata_source`
- `metadata_last_updated`

Recommended optional source metadata fields:

- `sector_taxonomy`
- `notes`

### 6.6 Freshness Metadata Expectations

`metadata_last_updated` must be parseable as an ISO date.

A future developer specification should define the stale threshold explicitly. Initial design recommendation: 365 calendar days for descriptive sector metadata, because sector and asset-class metadata changes less frequently than price, timing, or fundamental data.

This design does not authorize the threshold; it must be approved in the developer specification before implementation.

### 6.7 Duplicate Handling

Duplicate metadata rows for the same normalized ticker must fail fast.

This avoids ambiguous sector or asset-class authority.

### 6.8 Metadata-Only Row Handling

Rows present in `portfolio_metadata.csv` but absent from the portfolio/opportunity universe must not create new Portfolio Intelligence output rows.

They may be ignored or logged in future implementation, but they must not expand the upstream row universe.

### 6.9 Missing Metadata Handling

If `portfolio_metadata.csv` is missing, Portfolio Intelligence should preserve existing row output and classify portfolio metadata as incomplete.

If the artifact exists but no row exists for a required ticker, Portfolio Intelligence should preserve the row and classify the row's metadata as `MISSING` or `PARTIAL`, depending on the future implementation contract.

No row may be removed because of missing metadata.

## 7. Portfolio Metadata Classification Contract

### 7.1 Descriptive Metadata States

Proposed metadata states:

| State | Meaning |
|---|---|
| `COMPLETE` | Required metadata fields are present, valid, source-supported, and fresh under the approved freshness rule. |
| `PARTIAL` | A metadata row exists, but one or more required fields are missing while enough metadata exists to identify the row as partially source-supported. |
| `MISSING` | No metadata source exists or no matching metadata row exists for the ticker. |
| `STALE` | Metadata exists but exceeds the approved freshness threshold. |
| `INVALID` | Metadata exists but contains invalid taxonomy, invalid date, invalid asset class, or duplicate ambiguity. |

These states are descriptive only.

### 7.2 Mapping to Portfolio Intelligence Fields

`portfolio_metadata_status` should be informed by the metadata state:

- `COMPLETE` may support `portfolio_metadata_status = COMPLETE`
- `PARTIAL`, `MISSING`, `STALE`, or `INVALID` should support incomplete status such as `PARTIAL` or a future explicitly governed status value

`portfolio_metadata_reason` should describe the source-supported reason for the status.

`sector_exposure_state`, `diversification_state`, `concentration_state`, and `overlap_state` may use complete metadata descriptively, but must not create allocation authority.

### 7.3 Classification-Only Boundary

Metadata classification must remain descriptive and classification-only.

It must not produce:

- allocation decisions
- ranking
- scoring authority
- tradeability
- urgency
- conviction
- hidden filtering
- Reporting or Telegram decision semantics

## 8. Deterministic Rules

### 8.1 Metadata Source Missing

If `data/portfolio/portfolio_metadata.csv` is missing, preserve all Portfolio Intelligence output rows and classify metadata as incomplete.

The Decision Engine must remain conservative if required metadata is incomplete.

### 8.2 Metadata Row Missing for Ticker

If the metadata artifact exists but no row exists for a ticker requiring metadata, preserve the row and classify metadata as `MISSING` or incomplete.

### 8.3 Required Fields Missing

If a matching row exists but required fields are missing, classify as `PARTIAL` if some required metadata is present, otherwise `MISSING` or `INVALID` according to the future developer specification.

### 8.4 Stale Metadata

If `metadata_last_updated` exceeds the approved freshness threshold, classify metadata as `STALE`.

Stale metadata must not be treated as complete.

### 8.5 Invalid Sector or Asset Class Values

Invalid values include blank required taxonomy fields, unsupported asset-class values, malformed taxonomy strings, or values outside the approved taxonomy.

Initial recommended asset-class taxonomy:

- `Equity`
- `ETF`
- `Cash`
- `Other`

The future developer specification must define the exact taxonomy before implementation.

### 8.6 Duplicate Metadata Rows

Duplicate normalized ticker rows must fail fast before output generation.

### 8.7 Metadata Rows for Tickers Not Present in Universe

Metadata-only tickers must not create output rows.

They may be ignored or logged in a future implementation, but the upstream opportunity and portfolio universe must remain authoritative.

### 8.8 Date and Freshness Handling

`metadata_last_updated` must be parseable as an ISO date.

Malformed dates should classify affected metadata as `INVALID` or fail fast, depending on whether the future implementation treats metadata date validity as row-level or source-level critical.

### 8.9 Future Hybrid Source Precedence

If a future hybrid model is approved, precedence should be:

1. `data/portfolio/portfolio_metadata.csv` for portfolio metadata
2. explicitly governed scanner/context metadata only for non-portfolio opportunity enrichment if approved
3. no silent inference from unrelated fields

Hybrid behavior is not authorized by this design.

## 9. Portfolio Intelligence Interaction

Portfolio Intelligence may consume portfolio metadata to:

- enrich rows descriptively
- classify metadata completeness
- classify sector exposure descriptively
- preserve row count
- preserve row identity
- preserve the upstream opportunity universe
- propagate metadata status and reasons to downstream artifacts

Portfolio Intelligence must not:

- allocate
- rank
- score tradeability
- infer buy/sell/hold/trim decisions
- filter rows
- bypass the Decision Engine
- create Reporting or Telegram decision semantics
- expand rows from metadata-only tickers

## 10. Decision Engine Interaction

Complete portfolio metadata should be passed to the Decision Engine only through governed Portfolio Intelligence outputs.

Decision Engine remains the only allocation authority.

This design does not authorize changing allocation logic.

This design does not authorize loosening missing-metadata checks.

Future implementation may only allow Portfolio Intelligence to provide complete metadata when the portfolio metadata source contract is satisfied.

If metadata remains partial, missing, stale, or invalid, the Decision Engine should remain conservative under existing governance.

## 11. Reporting and Telegram Boundary

Reporting and Telegram must remain communication-only.

They may display source-supported metadata only if already present in Decision Engine or Reporting artifacts.

They must not infer sector metadata.

They must not compensate for missing Decision Engine metadata.

They must not create decisions, priorities, rankings, urgency, conviction, tradeability, or allocation semantics from metadata fields.

## 12. Future Test Plan

No tests are created by this design document. Future implementation should include tests for:

1. missing `portfolio_metadata.csv`
2. valid complete metadata
3. partial metadata
4. stale metadata
5. invalid sector values
6. invalid asset class values
7. duplicate metadata rows
8. metadata-only tickers do not create output rows
9. row count preservation
10. row identity preservation
11. portfolio metadata status transitions
12. sector exposure descriptive classification
13. Decision Engine remains unchanged unless separately authorized
14. Reporting remains unchanged unless separately authorized
15. Telegram remains unchanged unless separately authorized
16. forbidden semantics remain absent
17. generated CSV/data files are not committed

## 13. Implementation Authorization Boundary

This document does not authorize implementation.

Future implementation requires a separate developer specification.

Any Decision Engine semantic change requires escalation.

Any provider/API integration requires separate approval.

No generated CSV/data files may be committed by this design.

No runtime files, tests, scripts, reports, workflows, or generated artifacts may be changed by this design.

No sprint is closed or certified complete by this document.

## 14. Backlog Impact Assessment

`BL-0016 — Define approved Portfolio Metadata and Sector Exposure contract` remains sufficient for the identified work.

No new backlog item is required at this stage because this design remains within BL-0016 scope.

Backlog impact assessment:
- No new backlog items identified.

## 15. Recommended Next Step

Create a governed developer specification for the first implementation sprint after this Level 2 design is reviewed and merged.

The developer specification should remain limited to the separate `data/portfolio/portfolio_metadata.csv` artifact contract and Portfolio Intelligence descriptive metadata completeness handling.

It must not authorize Decision Engine loosening, provider/API integration, Reporting changes, Telegram changes, generated artifact commits, or allocation logic changes.