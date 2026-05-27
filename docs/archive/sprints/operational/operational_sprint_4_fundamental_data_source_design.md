# Operational Sprint 4 Fundamental Data Source Design

## 1. Status

Status: LEVEL 2 DESIGN

This document prepares a governed Level 2 design for an approved Fundamental data source and quality classification contract.

This is not an implementation document. It does not authorize implementation, provider integration, runtime code changes, tests, generated files, CSV edits, workflow changes, reporting changes, Telegram changes, or Decision Engine semantic changes.

Operational Sprint 4 is not closed, not certified complete, and not authorized for implementation by this document.

## 2. Problem Statement

After OS3A, BL-0011, and OS3B, the operator pipeline now runs fresh end-to-end through the governed sequence:

```text
scanner -> validation -> context -> fundamental -> timing -> portfolio state -> portfolio review -> portfolio intelligence -> final decisions -> reporting -> Telegram delivery
```

The flow is technically correct. The remaining blocker is not orchestration, Reporting, Telegram delivery, or path freshness.

The current system is operationally incomplete because the Fundamental Layer has no approved real fundamental data source. It therefore emits source-missing fallback classifications, and the Decision Engine correctly keeps all affected rows in `REVIEW`.

All-`REVIEW` output is expected until approved fundamentals exist because allocation decisions must not be authorized from missing quality metadata. The correct governance response is to define a descriptive Fundamental data source and quality classification contract, not to loosen the Decision Engine.

## 3. Current-State Contract

### 3.1 Current Fundamental Layer Input

The current Fundamental Layer reads the governed upstream context artifact:

```text
data/processed/context_strength.csv
```

The current input universe is therefore the upstream opportunity universe produced before the Fundamental Layer.

### 3.2 Current Fallback Behavior

The current Fundamental Layer intentionally emits fallback rows when approved real fundamental data is missing.

The OS3C investigation recorded that `data/processed/fundamental_quality.csv` contained 291 rows and that all rows had:

- `quality_state = INSUFFICIENT_DATA`
- `quality_metadata_status = source_missing`
- `source_data_status = source_missing`

The investigation also recorded:

- `missing_fundamentals_count = 291`
- `partial_data_count = 0`
- `stale_data_count = 0`

This behavior is expected and governance-safe.

### 3.3 Current Output Semantics

The current Fundamental Layer output is descriptive. It communicates that fundamental quality data is unavailable or insufficient.

It does not rank, score, prioritize, filter, allocate, or create actionable trading semantics.

### 3.4 Current Governance Limitations

The current contract has the following limitations:

- no configured raw fundamentals artifact
- no approved provider integration
- no governed real-data classification contract
- no provider freshness contract
- no partial-data classification contract
- no stale-data classification contract
- no real-data fixtures or tests
- no approved source metadata contract that can support downstream quality interpretation

These limitations prevent meaningful Decision Engine decisions that depend on fundamental quality metadata.

## 4. Approved Data Source Options

This section evaluates possible source approaches without implementing any of them.

### 4.1 Option A — Manually Maintained Raw Fundamentals CSV

Description: create a governed raw fundamentals CSV maintained by the operator or data steward.

Likely artifact:

```text
data/raw/fundamentals.csv
```

Governance fit: strong. A manually maintained artifact is explicit, auditable, reviewable in Git, and easy to constrain to descriptive input fields.

Operational complexity: low to medium. It requires manual updates and clear source discipline, but avoids provider authentication, API variability, rate limits, and external runtime failure modes.

Freshness and auditability: strong auditability, medium freshness. Freshness must be encoded through source metadata such as `source_last_updated`, `report_period`, and `source_freshness_days`.

Reliability: strong for deterministic local builds; dependent on manual data quality.

Testing impact: favorable. Static fixtures can be created from the same contract and used to prove valid, partial, stale, invalid, duplicate, and missing-source behavior.

Risks:

- manual data may become stale
- values may be entered incorrectly
- source provenance may be incomplete
- update cadence may be inconsistent

Recommendation: recommended as the first governed implementation approach because it is the smallest auditable path to real-data classification without provider complexity or allocation-authority drift.

### 4.2 Option B — Provider/API Integration

Description: integrate an external fundamentals provider or API directly into the pipeline.

Governance fit: medium. Provider integration can be governed, but it introduces external data availability, API contract volatility, authentication, rate limits, and nondeterministic runtime failure modes.

Operational complexity: high. It requires provider selection, credentials, error handling, retries, caching, rate-limit behavior, source normalization, and reproducibility controls.

Freshness and auditability: potentially strong freshness, weaker auditability unless responses are persisted into a raw artifact before classification.

Reliability: dependent on provider uptime, API stability, request limits, and network availability.

Testing impact: more complex. Tests must isolate provider calls, use fixtures, and prove deterministic behavior independent of live API availability.

Risks:

- provider response changes could break contracts
- live calls could make pipeline output nondeterministic
- missing credentials could block operation
- provider metrics could be misinterpreted as ranking or scoring authority

Recommendation: not recommended as the first implementation step unless routed through a persisted raw artifact and explicitly approved by Level 2 design. Escalate if provider behavior changes runtime determinism or authority boundaries.

### 4.3 Option C — Hybrid Approach

Description: use a provider or external source to populate a governed raw fundamentals artifact, then have the Fundamental Layer read only that persisted artifact.

Governance fit: strong if the provider ingestion process is separated from classification and if the Fundamental Layer reads the artifact only.

Operational complexity: medium to high. It separates ingestion from classification but still requires provider governance and artifact refresh procedures.

Freshness and auditability: strong if provider responses are persisted with source timestamps and provenance metadata.

Reliability: better than direct live provider reads because the classification build can remain deterministic from local artifacts.

Testing impact: favorable for the classification layer, more complex for future ingestion tooling.

Risks:

- ingestion process may become hidden runtime dependency
- provider fields may drift
- operator may confuse ingestion freshness with classification authority
- raw artifact governance must be strict

Recommendation: recommended as a later evolution after a manual raw CSV contract is proven. It should reuse the same raw artifact contract.

### 4.4 Option D — Staged MVP Approach

Description: implement the smallest manual raw artifact first, then later evaluate provider-assisted population after the classification contract is stable.

Governance fit: strongest. It separates contract design from provider integration and prevents implementation scope creep.

Operational complexity: low for initial MVP, with a clean path to later automation.

Freshness and auditability: strong auditability from the start; freshness rules are explicit even if updates are manual.

Reliability: strong for deterministic local operation.

Testing impact: strongest. The contract can be tested with static fixtures before any provider is introduced.

Risks:

- manual process may not scale
- initial coverage may be incomplete
- operator discipline is required

Recommendation: recommended Level 2 approach.

## 5. Recommended Level 2 Approach

The recommended approach for the first governed implementation sprint is the staged MVP approach:

1. define a manually maintained raw fundamentals artifact
2. define the descriptive quality classification contract
3. update the Fundamental Layer in a future implementation sprint to read the raw artifact if present
4. preserve existing source-missing fallback behavior if the raw artifact is absent
5. add tests for real-data, partial-data, stale-data, invalid-data, duplicate, and source-missing cases
6. defer provider/API integration until the raw artifact contract and classification behavior are stable

This is the smallest auditable approach that enables real-data classification without introducing allocation authority.

## 6. Proposed Raw Fundamentals Artifact Contract

### 6.1 Proposed Artifact

Proposed path:

```text
data/raw/fundamentals.csv
```

The file should be treated as a raw source artifact, not a processed classification output.

### 6.2 Row Identity

Required row identity fields:

- `ticker`
- `as_of_date`

The combination of `ticker` and `as_of_date` must be unique in the raw artifact.

The Fundamental Layer should match the upstream opportunity row by `ticker` and determine the applicable raw source row using deterministic freshness rules.

### 6.3 Required Source Metadata Columns

Required source metadata columns:

- `ticker`
- `as_of_date`
- `source_name`
- `source_last_updated`
- `report_period`
- `currency`

### 6.4 Required Fundamental Data Columns

Recommended required data columns for MVP classification:

- `revenue_growth_yoy`
- `eps_growth_yoy`
- `gross_margin`
- `operating_margin`
- `debt_to_equity`
- `free_cash_flow_positive`

These fields are proposed only as descriptive input data. They do not create ranking authority, scoring authority, tradeability, urgency, conviction, allocation, or filtering authority.

### 6.5 Optional Fundamental Data Columns

Optional columns may include:

- `net_income_growth_yoy`
- `return_on_equity`
- `current_ratio`
- `interest_coverage`
- `free_cash_flow_margin`
- `analyst_revision_trend`
- `notes`

Optional fields may support richer future classification but must not become hidden scoring or ranking semantics.

### 6.6 Freshness Metadata

Freshness should be calculated or recorded through:

- `source_last_updated`
- `as_of_date`
- `report_period`
- derived `source_freshness_days`

A future developer specification should define the maximum permitted age for `SUFFICIENT_DATA` classification. For initial planning, a default stale threshold of 120 calendar days after `source_last_updated` may be considered, but this value is not authorized by this design and must be explicitly approved in the developer specification.

### 6.7 Missing and Partial Data Handling

The raw artifact may contain incomplete rows. The classification contract must handle them descriptively.

Missing required fields should not remove rows. Missing or partial data should classify the row as `PARTIAL_DATA` or `INSUFFICIENT_DATA`, depending on severity and required-field coverage.

## 7. Proposed Fundamental Quality Classification Contract

### 7.1 Quality States

The proposed descriptive quality states are:

| State | Meaning |
|---|---|
| `SUFFICIENT_DATA` | Required fundamental fields are present, valid, source-supported, and fresh enough under the approved freshness rule. |
| `PARTIAL_DATA` | A raw source row exists, but one or more required fields are missing or incomplete while enough data remains to identify the row as source-supported but incomplete. |
| `STALE_DATA` | A raw source row exists and may otherwise be structurally valid, but source freshness exceeds the approved stale threshold. |
| `INSUFFICIENT_DATA` | No usable source row exists, the source is missing, required fields are materially absent, values are invalid, or source metadata is insufficient. |

These states are descriptive classifications only.

### 7.2 Required Output Metadata Fields

The future Fundamental Layer output should include metadata fields such as:

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

### 7.3 Metadata Status Values

Proposed `quality_metadata_status` values:

- `complete`
- `partial`
- `stale`
- `source_missing`
- `invalid`

Proposed `source_data_status` values:

- `source_available`
- `source_missing`
- `row_missing`
- `partial_data`
- `stale_data`
- `invalid_data`
- `duplicate_source_rows`

These values should remain descriptive and auditable.

## 8. Classification Rules

The future implementation should use deterministic classification rules.

### 8.1 Source Missing

If `data/raw/fundamentals.csv` does not exist, every upstream row must be preserved and classified as:

- `quality_state = INSUFFICIENT_DATA`
- `quality_metadata_status = source_missing`
- `source_data_status = source_missing`

### 8.2 Row Missing for Ticker

If the raw artifact exists but no source row exists for the upstream `ticker`, preserve the upstream row and classify it as:

- `quality_state = INSUFFICIENT_DATA`
- `source_data_status = row_missing`

### 8.3 Required Fields Missing

If a source row exists but required fields are missing, classify as `PARTIAL_DATA` when partial source-supported classification remains meaningful, otherwise `INSUFFICIENT_DATA`.

The output must list missing fields in `missing_required_fields`.

### 8.4 Partial Data

A row should be `PARTIAL_DATA` when:

- source metadata is present
- at least one required fundamental field is valid
- one or more required fields are missing
- the row is not stale under the approved freshness rule

### 8.5 Stale Data

A row should be `STALE_DATA` when:

- a source row exists
- the source metadata is structurally valid
- the source freshness exceeds the approved stale threshold

Stale data must not be promoted to `SUFFICIENT_DATA`.

### 8.6 Sufficient Data

A row should be `SUFFICIENT_DATA` only when:

- source metadata is present
- required fields are present
- required numeric values are valid
- source freshness is within the approved threshold
- no duplicate source row ambiguity exists

### 8.7 Invalid Numeric Values

Invalid numeric values include non-numeric text in numeric fields, impossible ratios where the approved contract forbids them, malformed percentages, and boolean fields outside the approved boolean representation.

Invalid values should classify the row as `INSUFFICIENT_DATA` or `PARTIAL_DATA`, depending on severity.

### 8.8 Duplicate Source Rows

Duplicate raw rows for the same row identity must fail fast or classify affected rows as invalid, depending on the future developer specification.

Recommended direction: fail fast on duplicate `ticker` and `as_of_date` in the raw artifact to avoid ambiguous source authority.

### 8.9 Unknown Tickers

Raw rows for tickers not present in the upstream opportunity universe should not create new processed output rows in the Fundamental Layer.

They may be logged in the future as unused source rows, but they must not alter the upstream row universe.

### 8.10 Date Mismatch

If raw source rows do not match the upstream opportunity date exactly, the implementation must use deterministic as-of selection rules.

Recommended direction: select the latest `as_of_date` on or before the upstream opportunity `date`, then apply freshness rules.

This must be explicitly approved in the future developer specification.

## 9. Test Plan for Future Implementation

No tests are created by this design document. The following tests should be required in the future implementation sprint.

### 9.1 Source Missing Fallback

Verify that missing `data/raw/fundamentals.csv` preserves all upstream rows and emits `INSUFFICIENT_DATA` with source-missing metadata.

### 9.2 Valid Source Data

Verify that valid raw source rows classify matching upstream rows as `SUFFICIENT_DATA` without changing row count, row identity, or ordering semantics.

### 9.3 Partial Data

Verify that rows with missing required fields classify as `PARTIAL_DATA` or `INSUFFICIENT_DATA` according to the approved contract and expose missing-field metadata.

### 9.4 Stale Data

Verify that old source rows classify as `STALE_DATA` and are not promoted to `SUFFICIENT_DATA`.

### 9.5 Duplicate Rows

Verify duplicate raw rows for the same source row identity are handled deterministically, preferably fail-fast.

### 9.6 Invalid Values

Verify malformed numeric and boolean values classify as invalid or insufficient and do not produce hidden scoring or ranking.

### 9.7 Row Preservation

Verify the Fundamental Layer output preserves the upstream opportunity universe and does not add, suppress, filter, rank, or reorder opportunities based on quality data.

### 9.8 Forbidden Semantics

Verify no forbidden fields or terms are introduced, including ranking, tradeability, allocation, urgency, conviction, hidden filtering, buy/sell semantics, or Decision Engine bypass.

### 9.9 Decision Engine Remains Unchanged

Verify the future Fundamental data implementation does not require Decision Engine loosening and does not change Decision Engine authority.

### 9.10 Reporting and Telegram Remain Communication-Only

Verify Reporting and Telegram continue to communicate source-supported Decision Engine outputs and do not infer decisions from Fundamental quality metadata.

## 10. Governance Boundary

The Fundamental Layer must remain classification and enrichment only.

It must not introduce:

- ranking
- scoring authority
- tradeability
- allocation
- urgency
- conviction
- hidden filtering
- Decision Engine bypass
- Reporting-based decision logic

Fundamental quality metadata may inform downstream interpretation only through approved contracts. Allocation authority remains exclusively with the Decision Engine.

## 11. Implementation Authorization Boundary

This document does not authorize implementation.

Future implementation requires a separate developer specification and explicit implementation authorization.

Any provider/API integration must be explicitly approved before implementation.

Any Decision Engine semantic change requires escalation beyond this Level 2 design.

Any change that introduces allocation authority, tradeability semantics, ranking authority, scoring authority, hidden filtering, or reporting-based decision logic requires governance escalation before implementation.

## 12. Backlog Impact Assessment

BL-0015 remains sufficient for the identified future work:

- `BL-0015 — Define and implement approved Fundamental data source and quality classification contract`

No new backlog item is required at this stage because provider evaluation, raw artifact definition, classification rules, and future implementation planning are covered by BL-0015.

Backlog impact assessment:
- No new backlog items identified.

## 13. Recommended Next Step

Create a separate developer specification for the staged MVP approach only after this Level 2 design is reviewed and approved.

The developer specification should remain implementation-focused but must preserve this design's governance boundaries, especially row preservation, descriptive classification-only semantics, and exclusive Decision Engine allocation authority.