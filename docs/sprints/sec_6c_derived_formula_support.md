# SEC-6C — Fixture-Based Derived Formula Support

Status: IMPLEMENTED
Backlog context: BL-0015 / BL-0017
Date: 2026-05-31

## Implemented Scope

SEC-6C extended the standalone SEC Company Facts transformer with fixture/local-input derived formula support for:

```text
total_debt
free_cash_flow
```

Updated module:

```text
scripts/fundamentals/sec_companyfacts_transform.py
```

The implementation remains standalone. It does not call SEC endpoints, download SEC data, read SEC cache directories automatically, integrate with the pipeline, or modify existing metrics, quality, analysis, Decision Engine, Reporting, Telegram, portfolio, scanner, validation, context, timing, or portfolio intelligence behavior.

## total_debt Formula Support

`total_debt` may be derived only when fixture/local input provides clean, source-supported, non-overlapping components.

Supported deterministic paths:

- simple debt family: current debt plus noncurrent debt;
- lease-inclusive debt family: lease-inclusive current debt plus lease-inclusive noncurrent debt.

Blocked or review-required behavior:

- simple and lease-inclusive families are not mixed in the same period;
- short-term borrowings are not silently mixed into total debt;
- finance lease liabilities are not silently mixed into total debt;
- missing current or noncurrent components keep `total_debt` blank;
- possible component overlap keeps `total_debt` blank;
- conflicting facts fail clearly rather than selecting a silent winner.

## free_cash_flow Formula Support

`free_cash_flow` may be derived only when fixture/local input provides source-supported operating cash flow and capital expenditure components.

Supported deterministic paths:

- positive capex outflow: `operating_cash_flow - capex`;
- already signed negative capex: `operating_cash_flow + capex`.

Blocked or review-required behavior:

- missing operating cash flow keeps `free_cash_flow` blank;
- missing capex keeps `free_cash_flow` blank;
- unit conflicts fail clearly;
- conflicting facts fail clearly rather than selecting a silent winner.

## Evidence and Notes Behavior

Derived values preserve source evidence in the row-level `notes` JSON.

Evidence includes:

- formula name;
- formula version;
- component source tags;
- component units;
- component values;
- fiscal year and fiscal period;
- period end date;
- filed date when present;
- form when present;
- frame when present;
- accession when present;
- review notes for derived, missing, blocked, duplicate, or review-required component conditions.

## Tests Added

Updated focused tests:

```text
tests/fundamentals/test_sec_companyfacts_transform.py
```

Tests cover:

- existing SEC-6A direct-field behavior;
- clean simple `total_debt` derivation;
- clean lease-inclusive `total_debt` derivation;
- blocked overlapping simple and lease-inclusive debt families;
- insufficient debt components;
- short-term borrowings and finance lease liabilities not silently mixed into debt;
- positive-capex free cash flow derivation;
- negative-capex free cash flow normalization;
- missing capex behavior;
- missing operating cash flow behavior;
- derived source evidence and notes;
- missing components not treated as zero;
- no live SEC/network call on import;
- no pipeline or downstream integration exposure;
- generated output written only to a provided temporary path.

Tests are fixture/temp-dir based and do not require internet, call live SEC endpoints, download real SEC data, write to real `data/local/`, commit generated artifacts, or depend on current market data.

## No Runtime Downstream Change Confirmation

SEC-6C introduced no runtime downstream behavior changes.

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

The transformer remains standalone and does not feed data into:

```text
data/raw/fundamentals_history.csv
data/processed/fundamental_metrics.csv
data/processed/fundamental_quality.csv
data/processed/fundamental_analysis.csv
```

## Backlog Impact Assessment

Backlog impact assessment:

- No new backlog items identified.

SEC-6C remains within BL-0015 and the approved SEC source-data sprint sequence. Future governed automation remains covered by BL-0017.

## Recommended Next Sprint

Recommended next sprint:

```text
SEC-7 — Controlled Real Data Review Specification
```

Purpose:

- define explicit review tickers;
- confirm generated output policy before any real output is produced;
- keep real SEC data outputs uncommitted unless separately approved;
- review results as source-data quality, not investment advice or allocation.
