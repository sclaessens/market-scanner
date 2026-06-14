# SEC-6D — Controlled Local SEC Transformation Review

Status: IMPLEMENTED
Backlog context: BL-0015 / BL-0017
Date: 2026-05-31

## Implemented Scope

SEC-6D implemented a controlled local review runner that composes existing SEC ticker/CIK coverage and Company Facts transformation utilities on explicit local inputs.

Implemented module:

```text
scripts/fundamentals/run_sec_transformation_review.py
```

The runner is standalone. It does not call SEC endpoints, download SEC data, read SEC cache directories automatically, integrate with the full pipeline, or modify existing metrics, quality, analysis, Decision Engine, Reporting, Telegram, portfolio, scanner, validation, context, timing, or portfolio intelligence behavior.

## Input Requirements

The runner requires explicit local inputs:

- project ticker list;
- ticker/CIK source file;
- local Company Facts-like JSON directory;
- source freshness date;
- extraction date;
- explicit output path unless `--validate-only` is supplied.

Supported CLI options:

```text
--project-tickers
--ticker-cik-source
--companyfacts-dir
--output
--source-freshness-date
--extraction-date
--validate-only
```

## Output Policy

Review output is written only when a caller provides an explicit output path.

The runner does not write by default to:

```text
data/processed/
data/reports/
reports/
```

SEC-6D committed no generated operational review output, no SEC data, no downloaded SEC files, no extracted SEC files, no reports, and no real operational logs.

## Review Output Shape

The review output preserves the internal raw fundamentals history fields:

```text
ticker
fiscal_year
fiscal_period
period_end_date
report_date
currency
revenue
gross_profit
operating_income
net_income
diluted_eps
total_debt
total_equity
free_cash_flow
source_name
source_reference
source_freshness_date
extraction_date
notes
```

It also adds descriptive review-only columns:

```text
mapping_status
transformation_status
review_required
review_reason
missing_fields
derived_fields_status
```

Review-only columns remain descriptive and do not create ranking, scoring, tradeability, urgency, conviction, eligibility, buy/sell, allocation, final-action, or hidden filtering semantics.

## Missing Data Behavior

The runner preserves requested ticker representation when:

- CIK mapping is missing;
- CIK mapping requires review;
- local Company Facts input is missing;
- local Company Facts input cannot be transformed;
- transformable facts are unavailable.

Missing values are not guessed, inferred, or treated as zero. Missing conditions are represented through review status, review reason, missing fields, and row-level notes.

## Derived Field Behavior

`total_debt` and `free_cash_flow` are populated only when the underlying SEC-6C transformer can derive them from approved local/fixture component conditions.

When components are missing, insufficient, ambiguous, or conflicting, derived fields remain blank and review notes are preserved.

## Tests Added

Focused tests:

```text
tests/fundamentals/test_run_sec_transformation_review.py
```

Tests cover:

- controlled review runner execution with explicit local fixture inputs;
- no network/SEC call on import;
- output written only to a provided temporary path;
- full fixture transformation output;
- missing CIK mapping row preservation;
- missing Company Facts file row preservation;
- direct field transformation;
- derived field transformation under approved fixture conditions;
- missing derived components preserved with review notes;
- review-only columns avoid allocation/trade/action semantics;
- no pipeline or downstream integration exposure;
- no default generated operational output;
- validate-only CLI behavior without output writes.

Tests are fixture/temp-dir based and do not require internet, call live SEC endpoints, download real SEC data, write to real `data/local/`, write to `data/processed/`, commit generated artifacts, or depend on current market data.

## No Runtime Downstream Change Confirmation

SEC-6D introduced no runtime downstream behavior changes.

It did not modify:

- Decision Engine logic;
- Reporting semantics;
- Telegram delivery or formatting;
- portfolio behavior;
- scanner behavior;
- validation/context/timing/portfolio intelligence behavior;
- existing fundamental metrics behavior;
- existing fundamental quality behavior;
- existing fundamental analysis behavior;
- full pipeline orchestration;
- GitHub workflow files;
- generated CSV/data files.

The runner remains standalone and does not feed data into:

```text
data/raw/fundamentals_history.csv
data/processed/fundamental_metrics.csv
data/processed/fundamental_quality.csv
data/processed/fundamental_analysis.csv
```

## Backlog Impact Assessment

Backlog impact assessment:

- No new backlog items identified.

SEC-6D remains within BL-0015 and the approved SEC source-data sprint sequence. Future governed automation remains covered by BL-0017.

## Recommended Next Sprint

Recommended next sprint:

```text
SEC-7 — Controlled Real Data Review Specification
```

Purpose:

- define explicit review tickers;
- confirm generated output location and commit policy;
- run review only from explicit local inputs;
- keep generated real review output uncommitted unless separately approved;
- review results as source-data quality, not investment advice or allocation.
