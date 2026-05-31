# SEC-6A — Direct SEC Fundamentals Transform

Status: IMPLEMENTED
Backlog context: BL-0015 / BL-0017
Date: 2026-05-31

## Implemented Scope

SEC-6A implemented a narrow standalone transformer for local SEC Company Facts-like JSON payloads into internal raw fundamentals history rows.

Implemented module:

```text
scripts/fundamentals/sec_companyfacts_transform.py
```

The transformer accepts an explicit local JSON path or already-loaded payload, plus explicit ticker, CIK, source freshness date, and extraction date context.

SEC-6A does not call SEC endpoints, download SEC data, read SEC cache directories automatically, integrate with the pipeline, or modify existing metrics, quality, analysis, Decision Engine, Reporting, Telegram, portfolio, scanner, validation, context, timing, or portfolio intelligence behavior.

## Allowed Direct Fields

SEC-6A maps only direct SEC-supported candidate fields from the SEC-4 investigation:

```text
revenue
gross_profit
operating_income
net_income
total_equity
diluted_eps
```

`diluted_eps` is preserved only when a per-share unit is present, with review notes retained in row-level evidence.

## Blocked Derived Fields

SEC-6A keeps these fields blank in transformed rows:

```text
total_debt
free_cash_flow
```

No total debt derivation, free cash flow derivation, debt-to-equity calculation, free cash flow margin calculation, or derived-metric implementation was added.

## Source Evidence Behavior

Each output row preserves source evidence in the `notes` field as structured JSON.

Evidence includes:

- source tag used;
- unit used;
- fiscal year and fiscal period;
- period end date;
- filed date when present;
- form when present;
- frame when present;
- accession when present;
- review notes for alternate tags, duplicate same-value facts, missing optional fields, and blocked derived fields.

The output also preserves `source_name`, `source_reference`, `source_freshness_date`, and `extraction_date`.

## Missing Value Behavior

Missing direct fields remain blank.

Missing `gross_profit` or `diluted_eps` does not drop the row. Missing values are not guessed, inferred, or treated as zero.

The transformer rejects unit conflicts and conflicting same-tag, same-period, same-unit values clearly instead of silently selecting a winner.

## Generated Output Policy

Generated CSV output is written only when an explicit `output_path` or CLI `--output` is supplied.

SEC-6A does not commit generated operational fundamentals CSV files, SEC downloads, extracted SEC files, reports, real operational logs, or generated SEC data.

## Tests Added

Focused tests:

```text
tests/fundamentals/test_sec_companyfacts_transform.py
```

Tests cover:

- minimal valid local Company Facts-like transformation;
- revenue candidate precedence;
- alternate revenue mapping;
- missing gross profit preservation;
- operating income and net income direct mapping;
- total equity instant fact mapping;
- diluted EPS per-share mapping and review note behavior;
- blocked `total_debt`;
- blocked `free_cash_flow`;
- missing values not treated as zero;
- deterministic output ordering;
- source evidence preservation;
- unit conflict failure;
- no live SEC/network call on import;
- no pipeline or downstream integration exposure;
- generated output written only to a provided temporary path;
- validate-only CLI behavior.

Tests are fixture/temp-dir based and do not require internet, call live SEC endpoints, download real SEC data, write to real `data/local/`, commit generated artifacts, or depend on current market data.

## No Runtime Downstream Change Confirmation

SEC-6A introduced no runtime downstream behavior changes.

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

The new transformer is standalone and does not feed data into:

```text
data/raw/fundamentals_history.csv
data/processed/fundamental_metrics.csv
data/processed/fundamental_quality.csv
data/processed/fundamental_analysis.csv
```

## Backlog Impact Assessment

Backlog impact assessment:

- No new backlog items identified.

SEC-6A remains within BL-0015 and the approved SEC source-data sprint sequence. Future governed automation remains covered by BL-0017.

## Recommended Next Sprint

Recommended next sprint:

```text
SEC-6B — Derived SEC Fundamentals Formula Specification
```

Purpose:

- approve deterministic derivation rules for `total_debt`;
- approve deterministic derivation rules for `free_cash_flow`;
- define duplicate/amended fact selection policy;
- define unit and period conflict behavior for derived fields;
- keep implementation separate from full pipeline integration until separately approved.
