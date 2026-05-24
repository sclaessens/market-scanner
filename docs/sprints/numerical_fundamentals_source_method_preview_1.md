# Numerical Fundamentals Source Method Preview 1

## 1. Status and Scope

This document is a documentation-only governance preview.

It defines a proposed method for future numerical fundamentals sourcing for the first governed batch of metadata-complete tickers.

This document does not:

- collect numerical fundamentals;
- approve numerical values;
- update `data/raw/fundamentals.csv`;
- update tracked source data;
- modify code;
- modify tests;
- modify generated outputs;
- run the pipeline;
- change runtime behavior;
- change Decision Engine logic;
- change Reporting logic;
- change Telegram logic;
- change scanner logic;
- change Fundamental Layer logic;
- change Portfolio Intelligence logic.

No numerical data is approved by this document.

No source-data update is authorized by this document.

## 2. Background

The project now has a governed source-data provisioning flow:

1. The Automated Source Data Steward protocol was documented.
2. Portfolio metadata expansion preview was completed.
3. Portfolio metadata source lookup preview was completed.
4. Fifteen approved metadata rows were added to `data/portfolio/portfolio_metadata.csv`.
5. Post-merge validation confirmed those 15 rows are metadata-complete.
6. Fundamentals expansion preview selected the same 15 metadata-complete tickers.
7. Provenance-only local ignored fundamentals rows were added locally to `data/raw/fundamentals.csv`.
8. Fundamental Layer validation confirmed those rows as:
   - `quality_state = INSUFFICIENT_DATA`
   - `quality_metadata_status = partial`
   - `source_data_status = partial_data`

The selected rows remain `INSUFFICIENT_DATA` because numerical metrics are blank.

The next bottleneck is governed numerical fundamentals sourcing.

`data/raw/fundamentals.csv` remains local ignored source data and must not be committed unless repository policy changes.

## 3. Doctrine Boundary

Numerical fundamentals sourcing must preserve:

- classification upstream;
- allocation downstream;
- Decision Engine = ONLY allocation authority;
- Fundamental Layer remains descriptive/classification-only;
- fundamentals data must not create buy/sell advice;
- no upstream tradeability;
- no hidden filtering;
- no ranking authority outside the Decision Engine;
- no scoring authority outside the Decision Engine;
- no allocation semantics outside the Decision Engine;
- no urgency or conviction semantics outside the Decision Engine;
- Reporting communicates only;
- Telegram communicates only;
- English-only repository content.

Numerical fundamentals may support descriptive quality classification only.

## 4. Selected Candidate Batch

Selected candidate batch for future numerical fundamentals work:

- `AMAT`
- `ANET`
- `ASML`
- `COST`
- `DELL`
- `ENPH`
- `EOG`
- `EQIX`
- `EW`
- `EXPD`
- `FDX`
- `FTNT`
- `HAL`
- `HLT`
- `HPE`

Expected current state for each ticker:

| ticker | metadata_status | local_raw_fundamentals_status | numerical_metrics_status | expected_quality_state |
|---|---|---|---|---|
| AMAT | metadata-complete | provenance-only row present locally | blank | INSUFFICIENT_DATA |
| ANET | metadata-complete | provenance-only row present locally | blank | INSUFFICIENT_DATA |
| ASML | metadata-complete | provenance-only row present locally | blank | INSUFFICIENT_DATA |
| COST | metadata-complete | provenance-only row present locally | blank | INSUFFICIENT_DATA |
| DELL | metadata-complete | provenance-only row present locally | blank | INSUFFICIENT_DATA |
| ENPH | metadata-complete | provenance-only row present locally | blank | INSUFFICIENT_DATA |
| EOG | metadata-complete | provenance-only row present locally | blank | INSUFFICIENT_DATA |
| EQIX | metadata-complete | provenance-only row present locally | blank | INSUFFICIENT_DATA |
| EW | metadata-complete | provenance-only row present locally | blank | INSUFFICIENT_DATA |
| EXPD | metadata-complete | provenance-only row present locally | blank | INSUFFICIENT_DATA |
| FDX | metadata-complete | provenance-only row present locally | blank | INSUFFICIENT_DATA |
| FTNT | metadata-complete | provenance-only row present locally | blank | INSUFFICIENT_DATA |
| HAL | metadata-complete | provenance-only row present locally | blank | INSUFFICIENT_DATA |
| HLT | metadata-complete | provenance-only row present locally | blank | INSUFFICIENT_DATA |
| HPE | metadata-complete | provenance-only row present locally | blank | INSUFFICIENT_DATA |

This is a governance preview. No pipeline run was performed for this document.

## 5. Proposed Numerical Fundamentals Metrics

For the first numerical fundamentals MVP, all percentage metrics should use decimal notation.

Examples:

- `0.125` means 12.5%.
- `-0.047` means negative 4.7%.

Ratios should be numeric and unitless unless a later governed source contract states otherwise.

Blank values are allowed when a value is not clearly sourced.

No value may be inferred from price action.

No value may be invented.

No scoring or ranking is introduced.

| metric | purpose | expected value type | allowed unit | expected period | blank allowed | calculated values allowed | review trigger |
|---|---|---|---|---|---|---|---|
| `revenue_growth_yoy` | Describes top-line growth. | Numeric decimal. | Percent as decimal. | Most recent completed fiscal year, or trailing twelve months only when explicitly sourced. | Yes | No, unless calculation method and source inputs are explicitly governed. | Missing period, conflicting annual/TTM values, unclear source definition, non-numeric value. |
| `eps_growth_yoy` | Describes earnings-per-share growth. | Numeric decimal. | Percent as decimal. | Most recent completed fiscal year, or trailing twelve months only when explicitly sourced. | Yes | No, unless calculation method and source inputs are explicitly governed. | Basic vs diluted ambiguity, adjusted vs GAAP ambiguity, negative base-year ambiguity, non-numeric value. |
| `gross_margin` | Describes gross profit efficiency. | Numeric decimal. | Percent as decimal. | Same period as the selected revenue metric where possible. | Yes | No, unless calculation method and source inputs are explicitly governed. | Source omits gross profit, period mismatch, non-comparable segment reporting, non-numeric value. |
| `operating_margin` | Describes operating income efficiency. | Numeric decimal. | Percent as decimal. | Same period as the selected revenue metric where possible. | Yes | No, unless calculation method and source inputs are explicitly governed. | Adjusted vs GAAP ambiguity, period mismatch, non-numeric value. |
| `net_margin` | Describes net income efficiency. | Numeric decimal. | Percent as decimal. | Same period as the selected revenue metric where possible. | Yes | No, unless calculation method and source inputs are explicitly governed. | Continuing vs total operations ambiguity, period mismatch, non-numeric value. |
| `debt_to_equity` | Describes balance-sheet leverage. | Numeric ratio. | Unitless ratio. | Most recent reported balance sheet period. | Yes | No, unless calculation method and source inputs are explicitly governed. | Negative equity, source-specific debt definition, unavailable equity, non-numeric value. |
| `return_on_equity` | Describes shareholder-equity returns. | Numeric decimal. | Percent as decimal. | Most recent completed fiscal year, or trailing twelve months only when explicitly sourced. | Yes | No, unless calculation method and source inputs are explicitly governed. | Negative equity, adjusted vs GAAP ambiguity, period mismatch, non-numeric value. |
| `free_cash_flow_margin` | Describes free-cash-flow generation relative to revenue. | Numeric decimal. | Percent as decimal. | Same period as the selected revenue metric where possible. | Yes | No, unless calculation method and source inputs are explicitly governed. | Free cash flow definition unclear, capital expenditure treatment unclear, period mismatch, non-numeric value. |

## 6. Proposed Source Method

### Manual Approved-Source Extraction

Description:

- Human or Codex-assisted extraction from explicitly approved public financial sources.
- Values are manually reviewed before any local update.
- Suitable for small batches such as the selected 15 tickers.

Requirements:

- source name;
- source reference;
- fiscal period;
- source freshness date;
- metric definition clarity;
- currency and unit clarity;
- no credentials;
- no scraping if prohibited;
- no prohibited source terms;
- no inferred or invented values.

Strengths:

- Keeps the first numerical batch auditable.
- Avoids credentials.
- Avoids provider integration.
- Allows metric-level review before local ignored source-data writes.
- Fits the current manual CSV MVP flow.

Limitations:

- Manual work can be slow.
- Definitions may vary across sources.
- Review discipline is required to avoid period or unit drift.

### Provider/API-Assisted Extraction

Description:

- Future automated or semi-automated provider/API use.
- Remains governed under `BL-0017`.
- Not authorized by this document.

Requirements before use:

- provider approval;
- API terms review;
- credential handling;
- rate-limit handling;
- source reliability review;
- metric mapping;
- validation tests;
- deterministic failure and retry behavior;
- explicit generated-output and source-data commit rules.

Strengths:

- More scalable after governance is complete.
- Better suited to repeated refreshes.

Limitations:

- Requires credential and terms governance.
- Requires source mapping and runtime test coverage.
- Increases operational complexity.

Recommended next immediate method:

- Manual approved-source extraction for a small batch preview.

Provider/API-assisted extraction should remain deferred unless separately approved under `BL-0017`.

## 7. Metric Definition Rules

The first numerical preview should use these strict interpretations:

- `revenue_growth_yoy`: year-over-year revenue growth for the most recent completed fiscal year, or trailing twelve months if explicitly sourced.
- `eps_growth_yoy`: year-over-year diluted EPS growth for the most recent completed fiscal year, or trailing twelve months if explicitly sourced.
- `gross_margin`: gross profit divided by revenue for the same reported period.
- `operating_margin`: operating income divided by revenue for the same reported period.
- `net_margin`: net income divided by revenue for the same reported period.
- `debt_to_equity`: total debt divided by total equity, as reported by the source.
- `return_on_equity`: net income divided by shareholders' equity, as reported by the source.
- `free_cash_flow_margin`: free cash flow divided by revenue for the same reported period.

Clarifications:

- If a source uses a different definition, mark `REVIEW_REQUIRED`.
- If periods differ across metrics, mark `REVIEW_REQUIRED`.
- If trailing twelve months and annual values conflict, mark `REVIEW_REQUIRED`.
- If a company has negative equity and ratio interpretation is unclear, mark `REVIEW_REQUIRED`.
- If a metric is unavailable, leave it blank rather than inventing.
- If a value requires calculation and the calculation method has not been governed, mark `REVIEW_REQUIRED`.
- If the source value is not parseable as a numeric value under the approved unit convention, mark `REVIEW_REQUIRED`.

## 8. Approval States for Numerical Fundamentals

### APPROVED

A numerical metric or row may be approved only when:

- ticker is in the approved batch;
- source method is approved;
- metric definition is clear;
- period is clear;
- value is numeric and parseable;
- unit convention is clear;
- source name is present;
- source reference is present;
- source freshness date is valid and not in the future;
- no value is inferred from price action;
- no value is invented;
- no credential or secret is present;
- no decision semantics are present.

### REVIEW_REQUIRED

A metric or row must be review-required when:

- source method is not yet approved;
- metric definition is unclear;
- source periods differ;
- source values conflict;
- value requires calculation not yet governed;
- currency or unit is unclear;
- fiscal period is unclear;
- value is not parseable;
- value appears stale;
- source requires credentials;
- source terms are unclear;
- manual judgment is required.

### REJECTED

A metric or row must be rejected when:

- ticker is outside approved scope;
- source is prohibited;
- value is invented;
- value is inferred from price action;
- source data includes credentials or secrets;
- value contains decision/action semantics;
- contradiction cannot be resolved;
- source violates governance restrictions.

## 9. Proposed Preview Table for Future Task

The next numerical source lookup preview must produce this table:

| ticker | metric_name | proposed_value | unit_convention | fiscal_period | source_name | source_reference | source_freshness_date | metric_definition_status | period_status | parse_status | steward_state | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TICKER | metric name | preview value or blank | decimal percent or unitless ratio | explicit period | source name | source reference | ISO date | CLEAR / REVIEW_REQUIRED | CLEAR / REVIEW_REQUIRED | PARSEABLE / REVIEW_REQUIRED | APPROVED / REVIEW_REQUIRED / REJECTED | source and review notes |

The next task may collect preview values, but must not update `data/raw/fundamentals.csv` unless a separate local ignored update task explicitly authorizes it.

## 10. Local Ignored Update Rules for Later Task

A later approved local ignored update may write to `data/raw/fundamentals.csv` only under these rules:

- create a local ignored backup first;
- update only approved batch rows;
- preserve schema exactly;
- fill only approved metrics;
- leave unclear metrics blank;
- do not overwrite unrelated rows;
- do not commit `data/raw/fundamentals.csv`;
- do not commit backups;
- run Fundamental Layer validation;
- restore generated outputs before committing documentation;
- document row count before and after;
- document selected tickers' `quality_state` changes;
- document any `PARTIAL_DATA`, `INSUFFICIENT_DATA`, `STALE_DATA`, or validation failures.

## 11. Human Review Triggers

Human review is required when:

- new source method is proposed;
- API/provider is proposed;
- credentials are involved;
- metric definition is unclear;
- periods conflict;
- trailing twelve months versus annual values conflict;
- source values conflict;
- calculated values are needed;
- negative denominators make ratios ambiguous;
- source freshness is questionable;
- ticker mapping is ambiguous;
- any row or metric is `REVIEW_REQUIRED`;
- generated artifacts would be committed;
- Decision Engine semantics could be affected.

## 12. Codex Execution Permissions for Future Numerical Tasks

Codex may, after explicit approval:

- perform a preview-only numerical source lookup;
- create a documentation artifact with candidate values;
- classify metrics or rows under the protocol;
- prepare a local ignored update only for approved values;
- run Fundamental Layer validation;
- report distributions and blockers.

Codex may not:

- bypass preview/approval;
- silently expand the batch scope;
- invent values;
- use unapproved providers;
- use credentials without governance;
- scrape prohibited sources;
- commit `data/raw/fundamentals.csv`;
- commit generated outputs;
- add scoring, ranking, allocation, or tradeability semantics;
- modify runtime logic as part of numerical sourcing.

## 13. Backlog Impact Assessment

Backlog impact assessment:
- No new backlog items identified.

Existing backlog coverage remains sufficient:

- `BL-0015 — Define and implement approved Fundamental data source and quality classification contract`
- `BL-0017 — Define governed automated data ingestion strategy for fundamentals and portfolio metadata`
- `BL-0016 — Define approved Portfolio Metadata and Sector Exposure contract`
- `BL-0011 — Define and repair authoritative active portfolio source`

## 14. Recommended Next Step

Perform a numerical fundamentals source lookup preview for the 15 selected tickers.

The next task should:

- use manual approved-source extraction unless a provider/API is separately approved;
- collect preview values only;
- classify each metric or row as `APPROVED`, `REVIEW_REQUIRED`, or `REJECTED`;
- use the table defined in this document;
- avoid updating `data/raw/fundamentals.csv` until a later explicit local ignored update task;
- avoid runtime, Decision Engine, Reporting, Telegram, scanner, Fundamental Layer, Portfolio Intelligence, generated output, and tracked source-data changes.

## 15. Validation Notes

Validation was documentation-only.

No runtime tests were run.

No pipeline was run.

No provider APIs were called.

No scraping was performed.

No source-data values were added, edited, collected, or approved.

No numerical fundamentals were collected.

No CSV files, raw fundamentals files, generated files, runtime files, code, tests, or backlog files were changed.
