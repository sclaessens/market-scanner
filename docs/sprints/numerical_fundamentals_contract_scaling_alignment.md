# Numerical Fundamentals Contract and Scaling Alignment

## 1. Status and Scope

This is a documentation-only governance and contract-alignment sprint.

The goal is to align the numerical fundamentals contract so future source-data creation can scale safely to larger controlled batches.

This document does not:

- collect data;
- approve numerical source values;
- update `data/raw/fundamentals.csv`;
- update tracked source data;
- change code;
- change tests;
- change runtime behavior;
- change generated outputs;
- change Decision Engine logic;
- change Reporting logic;
- change Telegram logic;
- change scanner logic;
- change Fundamental Layer logic;
- change Portfolio Intelligence logic.

No source-data values are approved or changed by this document.

## 2. Pilot 1 Learning Summary

`docs/sprints/numerical_fundamentals_pilot_1.md` proved that the optimized numerical fundamentals flow can work:

1. Select a narrow scope.
2. Use manual approved-source extraction.
3. Classify metrics at steward level.
4. Write only approved metrics to local ignored `data/raw/fundamentals.csv`.
5. Run Fundamental Layer and pipeline validation.
6. Commit documentation only.

Pilot 1 covered:

- tickers: `AMAT`, `ASML`, `COST`;
- attempted metrics: `revenue_growth_yoy`, `gross_margin`, `operating_margin`, `net_margin`;
- approved metrics: 9;
- review-required metrics: 3;
- rejected metrics: 0.

The approved metrics were written locally to ignored `data/raw/fundamentals.csv`.

`net_margin` was review-required because the current local raw fundamentals schema does not contain a `net_margin` column.

The pilot tickers validated as `PARTIAL_DATA`, not `SUFFICIENT_DATA`, because key writable MVP metrics were still missing, including `eps_growth_yoy` and `debt_to_equity`.

The pilot showed that the flow works, but the contract was not yet clear enough for 10-15 ticker batches. Scaling requires stricter metric, source, schema, calculation, and approval rules.

## 3. Current Bottlenecks

### Schema Mismatch

`net_margin` was proposed in earlier numerical fundamentals previews, but it is not writable in the current local raw fundamentals schema inspected during this sprint.

Current local raw fundamentals columns:

- `ticker`
- `as_of_date`
- `source_name`
- `source_reference`
- `source_freshness_date`
- `currency`
- `revenue_growth_yoy`
- `eps_growth_yoy`
- `gross_margin`
- `operating_margin`
- `debt_to_equity`
- `fundamental_notes`

### Sufficiency Ambiguity

Earlier documents did not fully separate:

- provenance-only rows;
- partial numerical rows;
- sufficient numerical rows.

Future prompts need an explicit list of writable MVP metrics required for `SUFFICIENT_DATA`.

### Calculation Ambiguity

Some metrics can be calculated from annual reports or filings, but future tasks need explicit formulas, period rules, and review triggers.

### Source-Method Ambiguity

Manual source extraction works, but source classes and source references must be repeatable before scaling.

### Scaling Risk

Large batches should not be attempted until the Automated Source Data Steward can classify metrics consistently without repeatedly stopping for the same schema and definition questions.

## 4. Writable MVP Metrics

The current raw fundamentals schema supports these writable numerical MVP metrics:

- `revenue_growth_yoy`
- `eps_growth_yoy`
- `gross_margin`
- `operating_margin`
- `debt_to_equity`

These are the official writable MVP metrics for the current scaling contract.

Additional existing non-numerical fields remain writable only as provenance or notes fields:

- `currency`
- `fundamental_notes`

### Candidate / Future Metrics

The following metrics remain candidate or future metrics until the raw schema and Fundamental Layer contract explicitly support them:

- `net_margin`
- `return_on_equity`
- `free_cash_flow_margin`

These metrics may be analytically useful, but future source-data tasks must not write them to `data/raw/fundamentals.csv` unless a later approved schema and implementation sprint authorizes them.

If local raw schemas in another environment already contain additional supported numeric columns, Codex must inspect that schema before writing and must still avoid writing candidate metrics unless the current task explicitly authorizes them.

## 5. Net Margin Decision

### Option A - Defer `net_margin`

Decision for immediate scaling:

- Keep `net_margin` as a candidate/future metric.
- Do not write it to `data/raw/fundamentals.csv`.
- Continue scaling using existing writable MVP metrics.

Advantages:

- Allows immediate scaling without schema or code changes.
- Avoids confusing source-data updates that cannot be represented in the current local raw artifact.
- Preserves the current Fundamental Layer contract.

Disadvantages:

- Loses one useful profitability metric during near-term scaling.
- Requires future follow-up if net margin should become part of the observed fundamentals profile.

### Option B - Add `net_margin` to raw fundamentals schema

This option is deferred.

It requires a developer implementation sprint.

The sprint must:

- update schema expectations;
- update Fundamental Layer parsing if needed;
- update focused tests;
- update contract documentation;
- preserve classification-only doctrine;
- avoid Decision Engine authority changes.

Advantages:

- Improves profitability coverage.
- Aligns future source-data rows with a commonly used profitability metric.

Disadvantages:

- Requires code, test, and schema changes.
- Should not be bundled into source-data expansion work.

Backlog item `BL-0019` captures this deferred option.

## 6. Fundamental Quality Sufficiency Contract

The numerical fundamentals sufficiency model is descriptive and classification-only.

### `SOURCE_MISSING`

No raw fundamentals row exists for the ticker/date context, or no governed source/provenance fields are available.

### `PROVENANCE_ONLY`

A raw row exists with required source/provenance fields, but no approved numerical metrics are present.

The current runtime may represent this as `INSUFFICIENT_DATA` with partial or source-missing metadata details.

### `PARTIAL_DATA`

A raw row exists and at least one approved numerical metric is present, but one or more required writable MVP metrics are missing or invalid.

### `SUFFICIENT_DATA`

A raw row exists and all required writable MVP metrics are present and valid.

Required writable MVP metrics for `SUFFICIENT_DATA`:

- `revenue_growth_yoy`
- `eps_growth_yoy`
- `gross_margin`
- `operating_margin`
- `debt_to_equity`

Clarifications:

- Margins alone are not sufficient.
- Provenance-only rows are not sufficient.
- Missing `eps_growth_yoy` keeps the row partial.
- Missing `debt_to_equity` keeps the row partial.
- Candidate/future metrics such as `net_margin`, `return_on_equity`, and `free_cash_flow_margin` must not block sufficiency unless a future approved schema and Fundamental Layer update promotes them into the writable MVP set.

## 7. Metric Definition Rules for Writable MVP Metrics

All percentage values use decimal convention.

Examples:

- 25 percent is written as `0.25`.
- Negative 8 percent is written as `-0.08`.

All approved values must be numeric and parseable.

### `revenue_growth_yoy`

Definition:

- Year-over-year revenue growth for the most recent completed fiscal year.

Expected period:

- Most recent completed fiscal year.

Allowed convention:

- Decimal percentage.

Direct reporting:

- Preferred.

Allowed calculation:

- `(current_period_revenue - prior_period_revenue) / prior_period_revenue`

Review triggers:

- Current or prior revenue is missing.
- Periods differ.
- Revenue definition changes.
- Continuing operations and total operations are unclear.
- Source reports non-comparable adjusted revenue.

### `eps_growth_yoy`

Definition:

- Year-over-year diluted EPS growth for the most recent completed fiscal year.

Expected period:

- Most recent completed fiscal year.

Allowed convention:

- Decimal percentage.

Direct reporting:

- Preferred.

Allowed calculation:

- `(current_period_diluted_eps - prior_period_diluted_eps) / abs(prior_period_diluted_eps)`

Review triggers:

- Prior EPS is zero.
- Prior EPS is negative.
- EPS is restated.
- EPS is non-comparable.
- Source uses adjusted EPS instead of GAAP or IFRS diluted EPS.
- Basic and diluted EPS are ambiguous.

### `gross_margin`

Definition:

- Gross profit divided by revenue for the same fiscal period.

Expected period:

- Same fiscal year as revenue.

Allowed convention:

- Decimal percentage.

Direct reporting:

- Preferred.

Allowed calculation:

- `gross_profit / revenue`

Review triggers:

- Gross profit is missing.
- Revenue and gross profit periods differ.
- Segment-only values are presented without consolidated values.
- The source uses adjusted gross profit without clear reconciliation.

### `operating_margin`

Definition:

- Operating income divided by revenue for the same fiscal period.

Expected period:

- Same fiscal year as revenue.

Allowed convention:

- Decimal percentage.

Direct reporting:

- Preferred.

Allowed calculation:

- `operating_income / revenue`

Review triggers:

- Operating income is adjusted rather than GAAP or IFRS.
- Revenue and operating income periods differ.
- Operating income is missing or presented only at segment level.

### `debt_to_equity`

Definition:

- Total debt divided by total shareholders' equity as of fiscal year end.

Expected period:

- Fiscal year-end balance sheet corresponding to the selected fiscal period.

Allowed convention:

- Unitless numeric ratio.

Direct reporting:

- Preferred when the source explicitly defines total debt and equity.

Allowed calculation:

- `total_debt / total_equity`

Review triggers:

- Equity is negative.
- Total debt definition is unclear.
- Lease treatment requires judgment.
- Source reports net debt instead of total debt.
- IFRS or US GAAP presentation requires manual interpretation.
- Non-controlling interests create denominator ambiguity.

## 8. Source Method Rules for Scaling

Allowed for manual extraction:

- company annual reports;
- official investor relations annual results;
- SEC 10-K filings;
- SEC 20-F filings;
- official company financial statements.

Allowed with human review:

- reputable public financial data pages, if terms permit manual viewing and no scraping or API use occurs.

Not allowed:

- paid or restricted APIs without approval;
- credentials;
- scraped data;
- social media;
- analyst opinions;
- price-action-derived values;
- untraceable values.

Every metric must include:

- source name;
- source reference;
- fiscal period;
- source freshness date;
- value origin, either `DIRECT_REPORTED` or `CALCULATED`;
- calculation inputs where applicable;
- calculation formula where applicable.

## 9. Automated Steward Approval Rules for Scaling

The Automated Source Data Steward may approve a metric without additional human review only when all of these conditions are true:

- ticker is in the approved batch;
- metric is writable under the current schema;
- source class is allowed;
- source reference is explicit;
- fiscal period is clear;
- value is numeric and parseable;
- value convention is clear;
- calculation formula is allowed;
- calculation inputs are visible and period-consistent;
- no manual judgment is required;
- no conflicting values are present;
- no credential or secret is involved;
- no decision semantics are present.

Human review is required when:

- ticker is outside the approved batch;
- metric is candidate/future only;
- source is only allowed with review;
- periods differ;
- definitions differ;
- calculated inputs are unclear;
- EPS is negative, zero, or non-comparable;
- equity is negative;
- debt definition is unclear;
- values conflict;
- source freshness is questionable;
- manual judgment is needed.

For these metrics, according to these source rules and calculation rules, the Automated Source Data Steward may process approved batches and locally add approved values to ignored `data/raw/fundamentals.csv`.

This authority is limited to:

- approved batch tickers;
- writable MVP metrics;
- approved source classes;
- metric-level approvals that satisfy every condition above;
- local ignored raw fundamentals updates only.

It does not authorize committing `data/raw/fundamentals.csv`.

It does not authorize automated ingestion, provider/API integration, runtime changes, or Decision Engine changes.

## 10. Scaling Batch Rules

### Pilot Batch

- Up to 3 tickers.
- Up to 5 writable MVP metrics.
- Used when testing a new source or metric combination.

### Standard Controlled Batch

- Up to 10 tickers.
- Writable MVP metrics only.
- No candidate/future metrics.
- Allowed after Pilot 1 learnings are incorporated.

### Expanded Controlled Batch

- Up to 15 tickers.
- Requires previous controlled batch success.
- Only for sources and metrics with stable approval behavior.

### Full-Universe Expansion

- Not authorized.
- Requires a separate implementation or operations plan under `BL-0017`.

Prioritization:

1. current portfolio holdings;
2. metadata-complete A-grade tickers;
3. metadata-complete B-grade tickers;
4. broader scanner universe only with explicit approval.

## 11. Future Data Creation Flow

Future numerical fundamentals batches should use this optimized flow:

1. Select approved batch.
2. Confirm metadata-complete status.
3. Confirm local provenance row exists.
4. Extract writable MVP metrics only.
5. Classify each metric.
6. Write locally only approved metrics.
7. Leave review-required metrics blank.
8. Run Fundamental Layer validation.
9. Run the full pipeline only when safe and useful.
10. Restore generated outputs.
11. Commit documentation only.
12. Keep `data/raw/fundamentals.csv` ignored and untracked.

Candidate/future metrics are not written.

Generated outputs are not committed.

Raw fundamentals are not committed.

Source-data values remain local unless repository policy changes.

## 12. Documentation Updates Required

Existing documents checked:

- `docs/sprints/numerical_fundamentals_source_method_preview_1.md`
- `docs/sprints/numerical_fundamentals_source_lookup_preview_1.md`
- `docs/sprints/numerical_fundamentals_pilot_1.md`
- `docs/sprints/operational_sprint_automated_source_data_steward_protocol.md`

Small alignment edits were made to earlier numerical fundamentals preview documents to clarify that the broader metric list was exploratory and that the current scaling contract limits writable metrics to the current raw schema.

The Automated Source Data Steward protocol remains directionally consistent and did not require an edit in this sprint.

## 13. Backlog Impact Assessment

A new backlog item was added because optional `net_margin` schema support requires a future developer implementation sprint and was not specifically captured by existing backlog items.

Added backlog item:

- `BL-0019 — Add optional net margin support to raw fundamentals schema and Fundamental Layer contract`

Existing backlog coverage remains sufficient for the broader source-data and ingestion strategy:

- `BL-0015 — Define and implement approved Fundamental data source and quality classification contract`
- `BL-0017 — Define governed automated data ingestion strategy for fundamentals and portfolio metadata`
- `BL-0016 — Define approved Portfolio Metadata and Sector Exposure contract`
- `BL-0011 — Define and repair authoritative active portfolio source`

## 14. Recommended Next Step

Review and merge this contract alignment.

After approval, the next data-steward prompt can process a standard controlled batch of up to 10 tickers using only writable MVP metrics:

- `revenue_growth_yoy`
- `eps_growth_yoy`
- `gross_margin`
- `operating_margin`
- `debt_to_equity`

The task should write only approved metric values locally to ignored `data/raw/fundamentals.csv`, run Fundamental Layer validation, restore generated outputs, and commit documentation only.
