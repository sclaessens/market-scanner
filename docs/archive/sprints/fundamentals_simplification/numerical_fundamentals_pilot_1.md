# Numerical Fundamentals Pilot 1

## Status and Scope

This document records the first governed Numerical Fundamentals Pilot.

The pilot combined:

- learning review from the broad numerical source lookup preview;
- a narrow manual approved-source pilot;
- metric-level steward classification;
- local ignored raw fundamentals update only for fully approved metrics;
- Fundamental Layer validation;
- documentation-only PR preparation.

This was not a coding sprint, runtime-logic change, automated ingestion implementation, provider/API integration task, or tracked source-data update.

No code, tests, tracked CSV source data, generated outputs, runtime behavior, Decision Engine logic, Reporting logic, Telegram logic, scanner logic, Fundamental Layer logic, Portfolio Intelligence logic, watchlist files, or portfolio source CSV values were intentionally modified.

`data/raw/fundamentals.csv` was updated locally because this task explicitly authorized local ignored updates for approved pilot metrics. It remains ignored and must not be committed.

## Protocol References

Governance references:

- `docs/sprints/operational_sprint_automated_source_data_steward_protocol.md`
- `docs/sprints/numerical_fundamentals_source_method_preview_1.md`
- `docs/sprints/numerical_fundamentals_source_lookup_preview_1.md`
- `docs/sprints/fundamentals_source_data_expansion_preview_1.md`
- `docs/sprints/fundamentals_provenance_only_update_1.md`
- `docs/sprints/project_backlog.md`

## Learning From Broad Preview

The broad numerical fundamentals source lookup preview covered 15 tickers and 8 metrics, for 120 metric candidates.

All 120 metrics were classified as `REVIEW_REQUIRED`.

The main blockers were procedural:

- the broad batch was too large for reliable first-pass metric approval;
- source method and source-term confidence had not yet been narrowed to official company or filing sources;
- metric period, unit, and definition checks needed to be documented per metric;
- calculations needed explicit inputs and formulas before approval;
- no numerical values were approved for local write.

No candidate values from the broad preview were treated as approved in this pilot.

The optimized pilot reduced scope to three tickers and four metrics, and used official company or filing sources only.

## Pilot Scope

Pilot tickers:

- `AMAT`
- `ASML`
- `COST`

Pilot metrics:

- `revenue_growth_yoy`
- `gross_margin`
- `operating_margin`
- `net_margin`

The selected metrics test revenue growth and margin consistency without introducing balance-sheet ratios.

The pilot did not expand beyond these three tickers and four metrics.

Scaling alignment note:

- `docs/sprints/numerical_fundamentals_contract_scaling_alignment.md` converts the Pilot 1 learning into the current scaling contract.
- `net_margin` remains candidate/future until schema support is implemented.

## Source Method Used

Manual approved-source extraction was used.

Approved source classes used in this pilot:

- official company investor-relations results page;
- official annual report filed through SEC-hosted company filing material.

No paid or restricted APIs were used.

No credentials or secrets were created.

No automated scraping or provider integration was introduced.

No value was inferred from price action.

No decision, allocation, tradeability, urgency, conviction, ranking, scoring, eligibility, or hidden filtering semantics were introduced.

## Source Lookup Preview Table

Percentage metrics use decimal convention.

| ticker | metric_name | proposed_value | unit_convention | fiscal_period | source_name | source_reference | source_freshness_date | value_origin | calculation_inputs | calculation_formula | metric_definition_status | period_status | parse_status | steward_state | notes |
|---|---:|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AMAT | revenue_growth_yoy | 0.04 | decimal_percent | FY2025 ended 2025-10-26 | Applied Materials FY2025 results | https://ir.appliedmaterials.com/news-releases/news-release-details/applied-materials-announces-fourth-quarter-and-fiscal-year-2025/ | 2026-05-24 | DIRECT_REPORTED | Not applicable | Not applicable | CLEAR | CLEAR | PARSEABLE | APPROVED | Official company results report annual revenue growth of 4 percent. |
| AMAT | gross_margin | 0.487 | decimal_percent | FY2025 ended 2025-10-26 | Applied Materials FY2025 results | https://ir.appliedmaterials.com/news-releases/news-release-details/applied-materials-announces-fourth-quarter-and-fiscal-year-2025/ | 2026-05-24 | DIRECT_REPORTED | Not applicable | Not applicable | CLEAR | CLEAR | PARSEABLE | APPROVED | Official company results report FY2025 GAAP gross margin of 48.7 percent. |
| AMAT | operating_margin | 0.292 | decimal_percent | FY2025 ended 2025-10-26 | Applied Materials FY2025 results | https://ir.appliedmaterials.com/news-releases/news-release-details/applied-materials-announces-fourth-quarter-and-fiscal-year-2025/ | 2026-05-24 | DIRECT_REPORTED | Not applicable | Not applicable | CLEAR | CLEAR | PARSEABLE | APPROVED | Official company results report FY2025 GAAP operating margin of 29.2 percent. |
| AMAT | net_margin |  | decimal_percent | FY2025 ended 2025-10-26 | Applied Materials FY2025 results | https://ir.appliedmaterials.com/news-releases/news-release-details/applied-materials-announces-fourth-quarter-and-fiscal-year-2025/ | 2026-05-24 | MISSING | Not applied | Not applied | CLEAR | CLEAR | MISSING | REVIEW_REQUIRED | Source values are available, but the current local raw fundamentals schema has no `net_margin` column, so this pilot did not approve a local write for this metric. |
| ASML | revenue_growth_yoy | 0.156 | decimal_percent | FY2025 ended 2025-12-31 | ASML Annual Report 2025 | https://www.sec.gov/Archives/edgar/data/937966/000162828026011377/asml-2025xannualxreportx.htm | 2026-05-24 | DIRECT_REPORTED | Not applicable | Not applicable | CLEAR | CLEAR | PARSEABLE | APPROVED | Official annual report states total net sales increased 15.6 percent year over year. |
| ASML | gross_margin | 0.518 | decimal_percent | FY2025 ended 2025-12-31 | ASML Annual Report 2025 | https://www.sec.gov/Archives/edgar/data/937966/000162828026011377/asml-2025xannualxreportx.htm | 2026-05-24 | DIRECT_REPORTED | Not applicable | Not applicable | CLEAR | CLEAR | PARSEABLE | APPROVED | Official annual report reports gross profit as 51.8 percent of total net sales. |
| ASML | operating_margin | 0.369 | decimal_percent | FY2025 ended 2025-12-31 | ASML Annual Report 2025 | https://www.sec.gov/Archives/edgar/data/937966/000162828026011377/asml-2025xannualxreportx.htm | 2026-05-24 | DIRECT_REPORTED | Not applicable | Not applicable | CLEAR | CLEAR | PARSEABLE | APPROVED | Official annual report reports operating income as 36.9 percent of total net sales. |
| ASML | net_margin |  | decimal_percent | FY2025 ended 2025-12-31 | ASML Annual Report 2025 | https://www.sec.gov/Archives/edgar/data/937966/000162828026011377/asml-2025xannualxreportx.htm | 2026-05-24 | MISSING | Not applied | Not applied | CLEAR | CLEAR | MISSING | REVIEW_REQUIRED | Source values are available, but the current local raw fundamentals schema has no `net_margin` column, so this pilot did not approve a local write for this metric. |
| COST | revenue_growth_yoy | 0.081 | decimal_percent | FY2025 ended 2025-08-31 | Costco FY2025 results | https://investor.costco.com/news/news-details/2025/Costco-Wholesale-Corporation-Reports-Fourth-Quarter-and-Fiscal-Year-2025-Operating-Results/ | 2026-05-24 | DIRECT_REPORTED | Not applicable | Not applicable | CLEAR | CLEAR | PARSEABLE | APPROVED | Official company results report fiscal-year net sales growth of 8.1 percent. |
| COST | gross_margin | 0.1112 | decimal_percent | FY2025 ended 2025-08-31 | Costco FY2025 results | https://investor.costco.com/news/news-details/2025/Costco-Wholesale-Corporation-Reports-Fourth-Quarter-and-Fiscal-Year-2025-Operating-Results/ | 2026-05-24 | CALCULATED | FY2025 net sales 269912; FY2025 merchandise costs 239886 | `(269912 - 239886) / 269912` | CLEAR | CLEAR | PARSEABLE | APPROVED | Calculation uses same-period official company inputs from the FY2025 results table. |
| COST | operating_margin | 0.037724 | decimal_percent | FY2025 ended 2025-08-31 | Costco FY2025 results | https://investor.costco.com/news/news-details/2025/Costco-Wholesale-Corporation-Reports-Fourth-Quarter-and-Fiscal-Year-2025-Operating-Results/ | 2026-05-24 | CALCULATED | FY2025 operating income 10383; FY2025 total revenue 275235 | `10383 / 275235` | CLEAR | CLEAR | PARSEABLE | APPROVED | Calculation uses same-period official company inputs from the FY2025 results table. |
| COST | net_margin |  | decimal_percent | FY2025 ended 2025-08-31 | Costco FY2025 results | https://investor.costco.com/news/news-details/2025/Costco-Wholesale-Corporation-Reports-Fourth-Quarter-and-Fiscal-Year-2025-Operating-Results/ | 2026-05-24 | MISSING | Not applied | Not applied | CLEAR | CLEAR | MISSING | REVIEW_REQUIRED | Source values are available, but the current local raw fundamentals schema has no `net_margin` column, so this pilot did not approve a local write for this metric. |

## Metric-Level Steward Classification

| steward_state | metric_count |
|---|---:|
| APPROVED | 9 |
| REVIEW_REQUIRED | 3 |
| REJECTED | 0 |

Approved metrics:

- `AMAT` `revenue_growth_yoy`
- `AMAT` `gross_margin`
- `AMAT` `operating_margin`
- `ASML` `revenue_growth_yoy`
- `ASML` `gross_margin`
- `ASML` `operating_margin`
- `COST` `revenue_growth_yoy`
- `COST` `gross_margin`
- `COST` `operating_margin`

Review-required metrics:

- `AMAT` `net_margin`
- `ASML` `net_margin`
- `COST` `net_margin`

Rejected metrics:

- None.

## Local Ignored Update Decision

At least one metric was fully approved, so the local ignored raw fundamentals artifact was updated.

Updated local ignored artifact:

- `data/raw/fundamentals.csv`

The update was limited to:

- pilot tickers only;
- approved metrics only;
- existing raw schema columns only.

The local update also refreshed row-level source provenance for the pilot rows because the previous provenance-only marker no longer accurately described the approved numerical values.

`net_margin` was not written because the current local raw fundamentals schema does not contain a `net_margin` column, and this task required preserving the existing schema exactly.

## Backup

Backup path:

- `data/raw/fundamentals_backup_before_numerical_pilot_1.csv`

Backup row count:

- 36

The backup is local ignored data and was not committed.

## Raw Fundamentals Row Counts

| state | row_count |
|---|---:|
| Before update | 36 |
| After update | 36 |

No rows were added.

No rows were removed.

No duplicate ticker rows were introduced.

## Local Update Table

| ticker | metric_name | approved_for_local_update | value_written_locally | source_reference | validation_state |
|---|---|---|---:|---|---|
| AMAT | revenue_growth_yoy | YES | 0.04 | https://ir.appliedmaterials.com/news-releases/news-release-details/applied-materials-announces-fourth-quarter-and-fiscal-year-2025/ | VALIDATED_NUMERICAL_PILOT |
| AMAT | gross_margin | YES | 0.487 | https://ir.appliedmaterials.com/news-releases/news-release-details/applied-materials-announces-fourth-quarter-and-fiscal-year-2025/ | VALIDATED_NUMERICAL_PILOT |
| AMAT | operating_margin | YES | 0.292 | https://ir.appliedmaterials.com/news-releases/news-release-details/applied-materials-announces-fourth-quarter-and-fiscal-year-2025/ | VALIDATED_NUMERICAL_PILOT |
| AMAT | net_margin | NO |  | https://ir.appliedmaterials.com/news-releases/news-release-details/applied-materials-announces-fourth-quarter-and-fiscal-year-2025/ | REVIEW_REQUIRED |
| ASML | revenue_growth_yoy | YES | 0.156 | https://www.sec.gov/Archives/edgar/data/937966/000162828026011377/asml-2025xannualxreportx.htm | VALIDATED_NUMERICAL_PILOT |
| ASML | gross_margin | YES | 0.518 | https://www.sec.gov/Archives/edgar/data/937966/000162828026011377/asml-2025xannualxreportx.htm | VALIDATED_NUMERICAL_PILOT |
| ASML | operating_margin | YES | 0.369 | https://www.sec.gov/Archives/edgar/data/937966/000162828026011377/asml-2025xannualxreportx.htm | VALIDATED_NUMERICAL_PILOT |
| ASML | net_margin | NO |  | https://www.sec.gov/Archives/edgar/data/937966/000162828026011377/asml-2025xannualxreportx.htm | REVIEW_REQUIRED |
| COST | revenue_growth_yoy | YES | 0.081 | https://investor.costco.com/news/news-details/2025/Costco-Wholesale-Corporation-Reports-Fourth-Quarter-and-Fiscal-Year-2025-Operating-Results/ | VALIDATED_NUMERICAL_PILOT |
| COST | gross_margin | YES | 0.1112 | https://investor.costco.com/news/news-details/2025/Costco-Wholesale-Corporation-Reports-Fourth-Quarter-and-Fiscal-Year-2025-Operating-Results/ | VALIDATED_NUMERICAL_PILOT |
| COST | operating_margin | YES | 0.037724 | https://investor.costco.com/news/news-details/2025/Costco-Wholesale-Corporation-Reports-Fourth-Quarter-and-Fiscal-Year-2025-Operating-Results/ | VALIDATED_NUMERICAL_PILOT |
| COST | net_margin | NO |  | https://investor.costco.com/news/news-details/2025/Costco-Wholesale-Corporation-Reports-Fourth-Quarter-and-Fiscal-Year-2025-Operating-Results/ | REVIEW_REQUIRED |

## Fundamental Layer Validation Results

Direct command run:

```bash
PYTHONPATH=. .venv/bin/python scripts/core/build_fundamental_layer.py
```

Result:

- Success.
- Output row count: 6.
- `quality_state` distribution: `PARTIAL_DATA=2`, `SUFFICIENT_DATA=4`.
- The current restored upstream `context_strength.csv` did not contain the pilot tickers, so the direct builder run did not provide pilot ticker statuses.

Because the direct builder could not validate the pilot tickers against the restored six-row upstream context, a full pipeline run was performed as useful optional validation.

Full pipeline command run:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_full_pipeline.py
```

Result:

- Success.
- The existing scanner step emitted external price-download warnings for `AZO` and `ORLY`, then completed with 289 scanner rows after duplicate removal.
- No new provider integration, credentials, source-data provider code, or runtime logic was introduced.

Post-pipeline `fundamental_quality.csv` observations:

- Row count: 289.
- `quality_state`: `INSUFFICIENT_DATA=280`, `PARTIAL_DATA=5`, `SUFFICIENT_DATA=4`.
- `quality_metadata_status`: `complete=4`, `partial=32`, `row_missing=253`.
- `source_data_status`: `partial_data=32`, `row_missing=253`, `source_available=4`.

Pilot ticker post-validation status:

| ticker | quality_state | quality_metadata_status | source_data_status | missing_required_fields | validation_state |
|---|---|---|---|---|---|
| AMAT | PARTIAL_DATA | partial | partial_data | `eps_growth_yoy`, `debt_to_equity` | VALIDATED_NUMERICAL_PILOT |
| ASML | PARTIAL_DATA | partial | partial_data | `eps_growth_yoy`, `debt_to_equity` | VALIDATED_NUMERICAL_PILOT |
| COST | PARTIAL_DATA | partial | partial_data | `eps_growth_yoy`, `debt_to_equity` | VALIDATED_NUMERICAL_PILOT |

The approved pilot metrics were recognized by the Fundamental Layer.

The pilot tickers moved away from pure `INSUFFICIENT_DATA` when validated through the refreshed full pipeline context.

They remain `PARTIAL_DATA` because the current local raw schema includes `eps_growth_yoy` and `debt_to_equity`, and those fields remain blank for the pilot tickers.

## Generated Artifact Handling

The direct builder and full pipeline created or updated generated/runtime artifacts.

Tracked runtime artifacts touched by the full pipeline were restored before commit:

- `data/logs/context_layer_log.csv`
- `data/portfolio/portfolio_positions.csv`
- `data/portfolio/portfolio_review.csv`
- `data/processed/context_strength.csv`
- `data/processed/scanner_ranked.csv`

Ignored generated validation artifacts were inspected but not committed.

Ignored local source artifacts were not committed:

- `data/raw/fundamentals.csv`
- `data/raw/fundamentals_backup_before_numerical_pilot_1.csv`

## Git Ignored And Untracked Confirmation

`data/raw/fundamentals.csv` remains ignored through `.gitignore`.

The local backup remains ignored.

Only this documentation artifact is intended for commit.

## Validation Limitations

The source lookup was intentionally limited to three tickers and four metrics.

Only metrics with clear source, period, unit convention, and parseable values were approved.

`net_margin` remains `REVIEW_REQUIRED` for this pilot because the local raw fundamentals file currently lacks a `net_margin` column and the task required preserving the schema exactly.

The full pipeline was run only because direct Fundamental Layer validation could not see the pilot tickers in the restored six-row upstream context. The full pipeline used the repository's existing scanner behavior and produced existing external download warnings for `AZO` and `ORLY`.

Generated outputs were not committed.

No runtime tests were run because this was a data-steward pilot, not a runtime-logic change.

## Backlog Impact Assessment

Backlog impact assessment:
- No new backlog items identified.

Existing backlog coverage remains sufficient:

- `BL-0015 — Define and implement approved Fundamental data source and quality classification contract`
- `BL-0017 — Define governed automated data ingestion strategy for fundamentals and portfolio metadata`
- `BL-0016 — Define approved Portfolio Metadata and Sector Exposure contract`
- `BL-0011 — Define and repair authoritative active portfolio source`

## Recommended Next Step

Review this pilot report and confirm whether the governance pattern is acceptable.

The next task should either:

- run a second narrow numerical pilot for `eps_growth_yoy` and `debt_to_equity` on the same three tickers; or
- define a local raw fundamentals schema-alignment decision for `net_margin` before attempting to approve and write net margin values.
