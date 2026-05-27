# Technical Analysis Contract

Status: ACTIVE ANALYSIS CONTRACT

## Purpose

This document defines the technical architecture for the simplified fundamentals platform.

The Technical Analyst / Architect owns layer boundaries and contracts. Developer / Codex owns implementation only after explicit approval.

This document does not authorize code changes, tests, raw data edits, generated artifact updates, provider/API usage, scraping, pipeline runs, or runtime behavior changes.

## Target layer boundaries

The simplified fundamentals platform should use clear sequential layers:

```text
raw fundamentals history
-> calculated fundamental metrics
-> fundamental quality classification
-> fundamental analysis classification
-> downstream portfolio/decision layers
```

Each layer must have one primary responsibility.

| Layer | Target artifact | Responsibility | Must not do |
|---|---|---|---|
| Raw history | `data/raw/fundamentals_history.csv` | Store source-supported raw financial statement history. | Calculate ratios, classify quality, analyze business quality, allocate. |
| Metrics | `data/processed/fundamental_metrics.csv` | Calculate deterministic metrics from raw history. | Score, rank, filter, allocate, interpret business quality. |
| Quality | `data/processed/fundamental_quality.csv` | Classify completeness and reliability. | Decide business attractiveness or allocation. |
| Analysis | `data/processed/fundamental_analysis.csv` | Classify descriptive financial characteristics. | Produce buy/sell, eligibility, conviction, urgency, ranking, or allocation semantics. |

## Input and output artifacts

### Raw fundamentals history

Input:

- source-supported financial statements;
- approved local source evidence;
- manual extraction records or future approved provider output.

Output:

- `data/raw/fundamentals_history.csv`

Policy:

- local ignored unless repository policy changes;
- source truth for raw historical values;
- no metrics or analysis states.

### Fundamental metrics

Input:

- `data/raw/fundamentals_history.csv`

Output:

- `data/processed/fundamental_metrics.csv`

Policy:

- generated artifact;
- deterministic calculations only;
- no allocation authority.

### Fundamental quality

Input:

- raw history;
- calculated metrics;
- data quality helper calculations.

Output:

- `data/processed/fundamental_quality.csv`

Policy:

- descriptive data readiness only;
- no business interpretation.

### Fundamental analysis

Input:

- calculated metrics;
- quality classifications;
- approved financial analysis rules.

Output:

- `data/processed/fundamental_analysis.csv`

Policy:

- descriptive business/fundamental characteristics only;
- no final decision semantics.

## Proposed small algorithms

Future implementation should be split into small focused algorithms rather than one overloaded Fundamental Layer.

Suggested future scripts:

- `scripts/core/build_fundamentals_history_intake.py`
- `scripts/core/build_fundamental_metrics.py`
- `scripts/core/build_fundamental_quality.py`
- `scripts/core/build_fundamental_analysis.py`

These script names are proposals only. They do not authorize implementation.

## Proposed script contracts

### `scripts/core/build_fundamentals_history_intake.py`

| Field | Contract |
|---|---|
| Input | Approved manual intake template, provider export, or source-steward staging artifact defined in a future spec. |
| Output | `data/raw/fundamentals_history.csv` or a validated staging artifact, depending on repository policy. |
| Responsibility | Normalize raw statement values, source references, period metadata, date semantics, and currency fields. |
| Must not do | Calculate ratios, classify quality, interpret business characteristics, allocate, run Decision Engine logic. |
| Likely tests needed | Required columns, date parsing, source reference presence, duplicate period detection, raw-only forbidden fields. |

### `scripts/core/build_fundamental_metrics.py`

| Field | Contract |
|---|---|
| Input | `data/raw/fundamentals_history.csv`. |
| Output | `data/processed/fundamental_metrics.csv`. |
| Responsibility | Calculate deterministic single-period, year-over-year, multi-year, and helper metrics. |
| Must not do | Source lookup, manual inference, business interpretation, quality-state authority beyond helper flags, allocation. |
| Likely tests needed | Formula correctness, missing input behavior, zero denominator handling, sign-change review flags, period/currency consistency. |

### `scripts/core/build_fundamental_quality.py`

| Field | Contract |
|---|---|
| Input | Raw history, metrics, helper fields. |
| Output | `data/processed/fundamental_quality.csv`. |
| Responsibility | Classify data completeness, source readiness, metric readiness, and review-required conditions. |
| Must not do | Decide whether a company is attractive, rank opportunities, filter opportunities, allocate. |
| Likely tests needed | Source missing, raw history incomplete, raw history ready, metrics partial, metrics ready, review-required conditions. |

### `scripts/core/build_fundamental_analysis.py`

| Field | Contract |
|---|---|
| Input | `data/processed/fundamental_metrics.csv`; `data/processed/fundamental_quality.csv`; approved analysis rules. |
| Output | `data/processed/fundamental_analysis.csv`. |
| Responsibility | Classify growth, margins, profitability, debt, cash flow, consistency, and trend descriptively. |
| Must not do | Allocate, rank, score, create eligibility, urgency, conviction, tradeability, buy/sell, or final action semantics. |
| Likely tests needed | State mapping, review-required propagation, no forbidden fields, deterministic outputs, row traceability. |

## Runtime responsibility split

Future runtime should preserve this separation:

- Data Steward: source evidence and raw values.
- Financial Analyst: formula meaning and interpretation rules.
- Functional Analyst: workflow and acceptance criteria.
- Technical Analyst / Architect: data contracts and layer boundaries.
- Developer / Codex: implementation after approval.
- Decision Engine: only allocation authority.

## Validation strategy

Validation should be focused and layered:

1. Raw schema validation.
2. Source metadata validation.
3. Duplicate ticker-period validation.
4. Period and currency consistency validation.
5. Formula output validation.
6. Quality state validation.
7. Analysis state validation.
8. Forbidden semantics validation.
9. Downstream contract validation.

Full pipeline runs should not be required for documentation-only validation or isolated formula checks.

## Timeout-safe validation approach

Future implementation should prefer:

- focused unit tests for formulas;
- small fixture CSVs;
- isolated builder runs;
- no provider calls during validation unless explicitly approved;
- no scraping during validation;
- deterministic local data fixtures;
- clear logs for missing data and review-required states.

Provider/API calls, scraping, and full pipeline runs should remain explicit and controlled.

## Generated artifact policy

Generated files are evidence, not doctrine.

`data/processed/`, `data/logs/`, and `reports/` artifacts should not be modified in documentation-only sprints. Future implementation sprints may regenerate artifacts only when explicitly authorized and validated.

## Local ignored source-data policy

`data/raw/fundamentals_history.csv` should be local ignored unless repository policy changes.

Because local ignored data can drift from repository documentation, future validation must clearly distinguish:

- tracked active doctrine;
- local source data;
- generated artifacts;
- validation evidence;
- archived history.

## Migration strategy from `data/raw/fundamentals.csv`

The current `data/raw/fundamentals.csv` should be treated as a temporary MVP or compatibility artifact.

A future migration specification should define:

- whether existing raw fields map directly into `fundamentals_history.csv`;
- how `as_of_date` is split or translated;
- how source metadata is preserved;
- how current sufficient/partial/missing cases are reclassified;
- how generated outputs are refreshed;
- whether the old Fundamental Layer becomes a compatibility wrapper or is replaced.

No migration is authorized by this document.

## Compatibility wrapper or replacement path

Two future paths are possible:

| Path | Meaning | Requirement |
|---|---|---|
| Compatibility wrapper | Keep current Fundamental Layer interface while reading simplified downstream artifacts. | Must preserve existing pipeline contracts until replacement is approved. |
| Replacement | Replace old Fundamental Layer with metrics, quality, and analysis builders. | Requires explicit Governance v2 review and developer specification. |

Any path must preserve Decision Engine authority and avoid upstream allocation semantics.

## Forbidden technical semantics

Do not introduce upstream fields or logic that imply:

- buy;
- sell;
- hold as final action;
- tradeability;
- urgency;
- conviction;
- eligibility;
- priority;
- ranking;
- score authority;
- allocation;
- hidden filtering;
- reporting-based decision logic.

## Handoff to Developer / Codex

Developer / Codex may implement only after:

- active contracts are approved;
- implementation scope is explicit;
- target files are named;
- tests are specified;
- generated artifact policy is defined;
- source-data policy is defined;
- migration rules are approved;
- forbidden semantics are listed;
- validation commands are approved.

Until then, this document is architecture and analysis doctrine only.