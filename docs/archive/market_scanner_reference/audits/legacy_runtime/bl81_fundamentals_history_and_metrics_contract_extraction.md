# BL81 — Extract fundamentals history and metrics contracts from script-era modules

Status: COMPLETED

## Purpose

BL81 extracts the stable fundamentals history and metrics contracts from the remaining script-era fundamentals modules.

This sprint is documentation-only. It does not archive, delete, move, refactor, or execute runtime Python files.

The goal is to separate reusable contracts from legacy script-era runtime behavior, so future migration can target canonical `src/market_scanner/fundamentals/` and `src/market_scanner/analysis/` modules without blindly preserving old script files.

## Scope

Reviewed script-era modules:

* `scripts/fundamentals/build_history_intake.py`
* `scripts/fundamentals/build_metrics.py`

Out of scope:

* `scripts/fundamentals/build_analysis.py`
* `scripts/fundamentals/build_quality.py`
* `scripts/fundamentals/sec_companyfacts_transform.py`
* `scripts/fundamentals/sec_ticker_cik_index.py`
* `scripts/fundamentals/run_sec_transformation_review.py`
* `scripts/fundamentals/sec_companyfacts_bulk_intake.py`
* Any runtime execution of script-era fundamentals modules.
* Any production data write.
* Any SEC/EDGAR provider call.

## Background

Recent cleanup sequence:

* BL74 removed active runtime imports from `scripts.fundamentals` in tests, source, and workflows.
* BL75 archived `scripts/fundamentals/__init__.py`.
* BL76 classified the remaining script-era Python files.
* BL78 reviewed the remaining `scripts/fundamentals/*.py` files and concluded that none should be archived blindly.
* BL79 archived the `scripts/core/build_fundamental_*` compatibility wrappers.
* BL80 classified the remaining fundamentals blockers and recommended pure contract extraction as the next governed lane.

BL81 implements the first extraction lane at documentation level:

* history schema and validation contract;
* metrics formula and missing-value contract.

No behavior is migrated in this sprint.

## Source modules inspected

### `scripts/fundamentals/build_history_intake.py`

This module defines:

* required raw fundamentals history columns;
* key columns;
* identity/source value columns;
* date columns;
* numeric columns;
* supported fiscal periods;
* forbidden investment-semantics columns;
* validation output shape;
* invalid row reporting policy;
* optional JSON report behavior.

The reusable contract is the schema and validation policy. The runtime CLI and report-writing behavior are not approved as canonical behavior by BL81.

### `scripts/fundamentals/build_metrics.py`

This module defines:

* identity columns passed through to metrics output;
* raw numeric input columns;
* derived metric columns;
* helper/status columns;
* ratio formulas;
* YoY growth formulas;
* prior-year lookup policy;
* missing-input policy;
* zero-denominator policy;
* optional CSV output behavior.

The reusable contract is the formula and missing-value policy. The runtime CLI and CSV-writing behavior are not approved as canonical behavior by BL81.

## Extracted contract 1 — Fundamentals history schema

A canonical fundamentals history record should contain the following required columns:

| Column                  | Contract role                                                                       |
| ----------------------- | ----------------------------------------------------------------------------------- |
| `ticker`                | Entity identifier. Required value. Normalized comparison should be uppercase.       |
| `fiscal_year`           | Fiscal year. Required value. Must parse as integer in accepted range.               |
| `fiscal_period`         | Fiscal period. Required value. Must be one of the supported period values.          |
| `period_end_date`       | End date of the fiscal period. Optional value, but if present must parse as a date. |
| `report_date`           | Filing/report date. Optional value, but if present must parse as a date.            |
| `currency`              | Reporting currency. Required value.                                                 |
| `revenue`               | Raw numeric source field. Optional value, but if present must parse as numeric.     |
| `gross_profit`          | Raw numeric source field. Optional value, but if present must parse as numeric.     |
| `operating_income`      | Raw numeric source field. Optional value, but if present must parse as numeric.     |
| `net_income`            | Raw numeric source field. Optional value, but if present must parse as numeric.     |
| `diluted_eps`           | Raw numeric source field. Optional value, but if present must parse as numeric.     |
| `total_debt`            | Raw numeric source field. Optional value, but if present must parse as numeric.     |
| `total_equity`          | Raw numeric source field. Optional value, but if present must parse as numeric.     |
| `free_cash_flow`        | Raw numeric source field. Optional value, but if present must parse as numeric.     |
| `source_name`           | Source/provenance identifier. Required value.                                       |
| `source_reference`      | Source reference or source-specific trace. Required value.                          |
| `source_freshness_date` | Source freshness date. Optional value, but if present must parse as a date.         |
| `extraction_date`       | Extraction date. Optional value, but if present must parse as a date.               |
| `notes`                 | Free text notes. Required column, value may be empty.                               |

## Extracted contract 2 — History key policy

The raw fundamentals history uniqueness key is:

```text
ticker + fiscal_year + fiscal_period
```

Canonical policy:

* `ticker` comparison should be normalized to uppercase.
* `fiscal_year` comparison should use cleaned text or integer-equivalent value.
* `fiscal_period` comparison should be normalized to uppercase.
* Duplicate keys should invalidate the input.
* Duplicate-key reporting should count all rows participating in duplicated keys.

## Extracted contract 3 — Fiscal period policy

Supported fiscal periods:

```text
FY
Q1
Q2
Q3
Q4
TTM
```

Canonical policy:

* Empty fiscal period values are invalid.
* Fiscal period comparison should be case-insensitive through uppercase normalization.
* Unsupported fiscal period values are invalid.

## Extracted contract 4 — Fiscal year policy

Canonical policy:

* `fiscal_year` must parse as an integer.
* Accepted range:

```text
1900 <= fiscal_year <= 2200
```

* Non-integer fiscal years are invalid.
* Out-of-range fiscal years are invalid.

## Extracted contract 5 — Date validation policy

Date columns:

```text
period_end_date
report_date
source_freshness_date
extraction_date
```

Canonical policy:

* Empty date values are allowed.
* Non-empty date values must parse as dates.
* Invalid non-empty date values should be reported per column.
* Row numbers in validation reports should use CSV-style row numbering, meaning data row index + 2 when the header is row 1.

## Extracted contract 6 — Numeric validation policy

Numeric input columns:

```text
revenue
gross_profit
operating_income
net_income
diluted_eps
total_debt
total_equity
free_cash_flow
```

Canonical policy:

* Empty numeric values are allowed.
* Non-empty numeric values must parse as floats.
* Invalid non-empty numeric values should be reported per column.
* Missing numeric values do not invalidate the history input by themselves.
* Missing numeric values may later produce partial metrics.

## Extracted contract 7 — Required value policy

The following columns require non-empty values:

```text
ticker
fiscal_year
fiscal_period
currency
source_name
source_reference
```

Canonical policy:

* Empty values in these columns invalidate the input.
* Missing required value rows should be reported by column.
* Required value checks are separate from required column checks.

## Extracted contract 8 — Forbidden investment-semantics policy

The history input must not contain columns that encode investment decision authority.

Forbidden column names include:

```text
buy
sell
action
final_action
decision
allocation
position_size
urgency
conviction
tradeability
eligible
eligibility
ranking
score
priority
entry
stop
target
```

Canonical policy:

* Column-name matching should be case-insensitive after trimming whitespace.
* Presence of any forbidden column invalidates the input.
* This preserves the separation between fundamentals/source-data evidence and Decision Engine authority.
* Fundamentals history and metrics must remain descriptive/evidence-only.

## Extracted contract 9 — Validation result shape

A canonical validation result should preserve at least the following fields:

```text
status
row_count
issue_count
missing_required_columns
forbidden_columns
duplicate_key_count
invalid_fiscal_year_rows
invalid_fiscal_period_rows
invalid_date_columns
invalid_numeric_columns
missing_required_value_columns
```

Canonical policy:

* `status` should be `VALID` or `INVALID`.
* `row_count` should reflect the number of input rows read.
* `issue_count` should be derived from all detected issue categories.
* The validation result should be machine-readable.
* The validation result should not create investment recommendations.

## Extracted contract 10 — Metrics identity columns

Metrics output should preserve the following identity/source columns from the raw history input:

```text
ticker
fiscal_year
fiscal_period
period_end_date
report_date
currency
source_name
source_reference
source_freshness_date
extraction_date
```

Canonical policy:

* Metrics output should be row-preserving relative to valid input history.
* Metrics should not remove or reorder source identity in a way that breaks provenance.
* Metrics output should not add Decision Engine authority fields.

## Extracted contract 11 — Metric columns

Derived metric columns:

```text
gross_margin
operating_margin
net_margin
free_cash_flow_margin
debt_to_equity
return_on_equity
revenue_yoy_growth
eps_yoy_growth
free_cash_flow_yoy_growth
```

Helper/status columns:

```text
metric_status
metric_missing_inputs
metric_warnings
```

Canonical policy:

* Metrics should be deterministic.
* Metrics should be descriptive only.
* Metrics must not imply buy/sell/hold, allocation, ranking, urgency, conviction, tradeability, or target-price authority.

## Extracted contract 12 — Ratio formulas

Canonical ratio formulas:

```text
gross_margin = gross_profit / revenue
operating_margin = operating_income / revenue
net_margin = net_income / revenue
free_cash_flow_margin = free_cash_flow / revenue
debt_to_equity = total_debt / total_equity
return_on_equity = net_income / total_equity
```

Canonical division policy:

* If numerator is missing, result is missing.
* If denominator is missing, result is missing.
* If denominator is zero, result is missing.
* Zero-denominator cases should be captured in missing-input or warning evidence rather than raising ungoverned runtime errors.
* The formulas are neutral descriptive calculations, not recommendation signals.

## Extracted contract 13 — YoY growth formulas

Canonical YoY formulas:

```text
revenue_yoy_growth = (current_revenue - prior_year_revenue) / abs(prior_year_revenue)
eps_yoy_growth = (current_diluted_eps - prior_year_diluted_eps) / abs(prior_year_diluted_eps)
free_cash_flow_yoy_growth = (current_free_cash_flow - prior_year_free_cash_flow) / abs(prior_year_free_cash_flow)
```

Canonical prior-year lookup key:

```text
ticker + (fiscal_year - 1) + fiscal_period
```

Canonical YoY policy:

* Prior-year lookup should use the same fiscal period.
* `ticker` comparison should be uppercase-normalized.
* If no prior-year row exists, YoY growth values are missing.
* Missing prior-year rows should produce a warning:

```text
yoy_growth:missing_prior_year
```

* If current or prior value is missing, YoY growth is missing.
* If the prior value is zero, YoY growth is missing.
* Prior-year denominator uses `abs(prior)`.

## Extracted contract 14 — Metric missing-input policy

Canonical missing-input issue format:

```text
<metric_name>:missing:<input_name>
```

For multiple missing inputs:

```text
<metric_name>:missing:<input_name>|<input_name>
```

Canonical zero-denominator issue format:

```text
<metric_name>:zero_denominator:<denominator_name>
```

Canonical policy:

* Base ratio missing inputs should be reported in `metric_missing_inputs`.
* YoY growth missing or zero-prior issues should be reported in `metric_warnings`.
* Missing metrics should not invalidate the entire metrics output if the history input itself is valid.
* Missing or partial metrics should be visible for downstream source-quality/readiness evaluation.

## Extracted contract 15 — Metric status policy

Canonical policy:

* `metric_status` should be `complete` only when all derived metric values are present.
* `metric_status` should be `partial` when one or more derived metric values are missing.
* `metric_missing_inputs` should be semicolon-delimited.
* `metric_warnings` should be semicolon-delimited.
* Empty `metric_missing_inputs` or `metric_warnings` should be represented as empty strings.

## Extracted contract 16 — Input validation before metrics

Canonical policy:

* Metrics generation requires valid fundamentals history input.
* The history validation contract should run before metrics calculation.
* If history validation fails, metrics generation must not silently proceed.
* Failure should expose the validation result in a machine-readable or inspectable way.
* Future canonical implementation should avoid production writes during validation failure handling unless explicitly approved.

## Extracted contract 17 — Output/write policy

Script-era behavior includes optional write behavior:

* `build_history_intake.py` can write an optional JSON validation report.
* `build_metrics.py` can write an optional CSV metrics output.

BL81 canonical decision:

* Output/write behavior is not automatically approved for canonical migration.
* Reusable contract is the schema/formula/validation policy, not the CLI write behavior.
* Canonical implementation should separate pure computation from persistence.
* Persistence should live only under approved canonical persistence boundaries.
* Documentation-only extraction does not authorize writes to `data/raw`, `data/processed`, `data/local`, reports, or logs.

## Migration target recommendation

Recommended canonical target areas:

* `src/market_scanner/fundamentals/fundamental_contracts.py`
* `src/market_scanner/fundamentals/fundamentals_normalization_contracts.py`
* `src/market_scanner/fundamentals/source_data_readiness.py`
* `src/market_scanner/analysis/analysis_contracts.py`

Recommended future test areas:

* canonical fundamentals history schema tests;
* canonical metrics formula tests;
* missing-input and zero-denominator tests;
* no-investment-authority/forbidden-column tests;
* row-preservation and source-provenance tests.

BL81 does not implement these changes. It only records the extracted contract requirements.

## Archive readiness decision

BL81 does not make either source file archive-ready.

Current status:

| File                                           | Status after BL81 | Reason                                                                     |
| ---------------------------------------------- | ----------------- | -------------------------------------------------------------------------- |
| `scripts/fundamentals/build_history_intake.py` | NOT_ARCHIVE_READY | Contract extracted, but canonical implementation/parity is not yet proven. |
| `scripts/fundamentals/build_metrics.py`        | NOT_ARCHIVE_READY | Contract extracted, but canonical implementation/parity is not yet proven. |

Future archive of these files requires:

1. canonical contract implementation or explicit retirement decision;
2. canonical tests covering the extracted schema and metric behavior;
3. no active runtime or static boundary dependency requiring the legacy file;
4. green full test suite.

## Recommended next sprint

Recommended next sprint:

```text
BL82 — Map extracted BL81 contracts to existing canonical fundamentals modules

BL82 should remain documentation-only. It should compare the BL81 extracted contracts against existing src/market_scanner/fundamentals/ and src/market_scanner/analysis/ files without implementing code.

A later code-aware Codex sprint can add canonical fundamentals history and metrics contract tests after the mapping confirms the correct canonical owners.

## Validation

No script-era runtime files were executed.

Full test suite should be run for repository safety.

Command:

```bash
pytest -q
```

Result:

```text
522 passed in 0.58s
```

## Guardrails

* No live SEC/EDGAR calls were run.
* No yfinance calls were run.
* No credentials were read.
* No production data was written.
* No production reports were generated.
* No Telegram messages were sent.
* No portfolio/watchlist production state was modified.
* No Decision Engine authority was changed.
* No script-era Python runtime files were executed.

## Final status

BL81 completed documentation-only extraction of the fundamentals history and metrics contracts from script-era modules.

The extracted contracts are now available for future canonical implementation, test migration, or retirement decisions.

No script-era fundamentals file is archived or declared archive-ready by BL81.
