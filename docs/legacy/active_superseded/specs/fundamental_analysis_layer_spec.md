# Fundamental Analysis Layer Specification

Status: ACTIVE SPECIFICATION
Backlog driver: BL-0015
Related sequence: Sprint E1, Sprint E2, Sprint E3

## 1. Purpose

This document specifies the future Fundamental Analysis Layer before implementation.

The Fundamental Analysis Layer will interpret validated raw fundamentals history, deterministic fundamental metrics, and Fundamental Quality metadata into descriptive business/fundamental states.

This layer must help the operator understand the character of a company's fundamentals without creating allocation, tradeability, urgency, conviction, ranking, scoring, final actions, or buy/sell decisions.

This document does not authorize implementation, code changes, test changes, data changes, generated artifact updates, provider/API usage, scraping, pipeline execution, allocation changes, Decision Engine changes, Reporting changes, Telegram changes, or ticker-category runtime logic.

## 2. Current Baseline

The project now has three controlled building blocks:

| Sprint | Runtime surface | Purpose | Status |
|---|---|---|---|
| E1 | `scripts/core/build_fundamentals_history_intake.py` | Validate raw fundamentals history schema and source evidence. | Implemented. |
| E2 | `scripts/core/build_fundamental_metrics.py` | Compute deterministic financial metrics from validated raw history. | Implemented. |
| E3 | `scripts/core/build_fundamental_layer.py` | Preserve the existing `fundamental_quality.csv` compatibility surface and optionally map raw-history/metrics evidence into quality metadata. | Implemented. |

The next step is not to wire everything into decisions. The next step is to define how descriptive analysis should work.

## 3. Layer Responsibility

The Fundamental Analysis Layer may:

- interpret metric patterns descriptively;
- classify financial profile states;
- classify margin profile states;
- classify growth profile states;
- classify leverage profile states;
- classify cash-flow profile states;
- surface data-quality limitations;
- identify review-needed conditions;
- preserve row identity and upstream universe;
- produce a row-preserving descriptive artifact for later governed consumption.

The Fundamental Analysis Layer may not:

- decide whether to buy, sell, hold, trim, add, enter, exit, or avoid;
- assign allocation;
- assign position size;
- assign urgency;
- assign conviction;
- assign tradeability;
- assign eligibility;
- rank tickers;
- score tickers for priority;
- filter rows;
- override the Decision Engine;
- create Reporting decision semantics;
- implement ticker-category runtime logic unless separately approved later.

## 4. Proposed Artifact

Future target artifact:

```text
data/processed/fundamental_analysis.csv
```

This artifact is proposed by this specification but not implemented here.

The artifact should be row-preserving relative to the selected upstream universe used by the future implementation.

Preferred upstream identity:

```text
ticker
date
```

If the implementation consumes `fundamental_quality.csv`, it should preserve that artifact's row identity and row count. If it consumes a different explicitly approved upstream artifact, the row-preservation rule must be restated in the implementation sprint.

## 5. Proposed Input Artifacts

Potential future inputs:

| Input artifact | Purpose | Required at first implementation? |
|---|---|---|
| `data/processed/fundamental_quality.csv` | Pipeline-facing data-readiness and compatibility metadata. | Yes. |
| `data/processed/fundamental_metrics.csv` | Deterministic financial metrics. | Recommended. |
| `data/raw/fundamentals_history.csv` | Raw source facts and source freshness evidence. | Optional if metrics/quality already expose enough evidence. |
| `data/reference/ticker_categories.csv` | Future category metadata. | No. Do not require in first implementation. |

The first implementation should avoid ticker-category dependency unless a separate category source-artifact sprint has already approved it.

## 6. Proposed Output Fields

Minimum proposed fields:

```text
ticker
date
fundamental_analysis_state
fundamental_analysis_reason
fundamental_profile_state
margin_profile_state
growth_profile_state
leverage_profile_state
cash_flow_profile_state
fundamental_review_flag
fundamental_review_reason
analysis_data_status
analysis_input_coverage
analysis_warnings
```

These field names are proposed. A future implementation sprint may refine them if it preserves governance boundaries and tests the resulting contract.

## 7. Allowed State Semantics

Allowed descriptive states may include:

| State family | Allowed examples | Meaning |
|---|---|---|
| Fundamental analysis state | `ANALYSIS_READY`, `LIMITED_ANALYSIS`, `INSUFFICIENT_DATA`, `REVIEW_REQUIRED` | Overall descriptive analysis readiness. |
| Fundamental profile state | `STABLE_PROFILE`, `IMPROVING_PROFILE`, `DETERIORATING_PROFILE`, `MIXED_PROFILE`, `UNKNOWN_PROFILE` | Broad descriptive fundamental profile. |
| Margin profile state | `MARGIN_STABLE`, `MARGIN_EXPANDING`, `MARGIN_COMPRESSING`, `MARGIN_NEGATIVE`, `MARGIN_UNKNOWN` | Descriptive margin behavior. |
| Growth profile state | `GROWTH_POSITIVE`, `GROWTH_NEGATIVE`, `GROWTH_MIXED`, `GROWTH_UNKNOWN` | Descriptive growth behavior. |
| Leverage profile state | `LEVERAGE_LOW`, `LEVERAGE_MODERATE`, `LEVERAGE_HIGH`, `LEVERAGE_UNKNOWN` | Descriptive balance-sheet leverage context. |
| Cash-flow profile state | `CASH_FLOW_POSITIVE`, `CASH_FLOW_NEGATIVE`, `CASH_FLOW_MIXED`, `CASH_FLOW_UNKNOWN` | Descriptive cash-flow context. |
| Review flag | `NO_REVIEW_FLAG`, `REVIEW_DATA_LIMITATION`, `REVIEW_METRIC_CONFLICT`, `REVIEW_STALE_SOURCE`, `REVIEW_EXTREME_VALUE` | Operator review context only. |

These states are descriptive and do not authorize decisions.

## 8. Forbidden Analysis Semantics

The Fundamental Analysis Layer must not output fields or values that imply:

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

It must not include hidden equivalents of these concepts under different names.

## 9. Metric Interpretation Guidance

The first implementation should keep interpretation simple and transparent.

Potential interpretation examples:

| Metric evidence | Descriptive interpretation only |
|---|---|
| Positive gross/operating/net margin values | Supports a more complete margin profile. |
| Negative net margin | May support `MARGIN_NEGATIVE` or review context. |
| Positive revenue YoY | May support `GROWTH_POSITIVE`. |
| Negative revenue YoY | May support `GROWTH_NEGATIVE`. |
| Positive free cash flow margin | May support `CASH_FLOW_POSITIVE`. |
| Negative free cash flow margin | May support `CASH_FLOW_NEGATIVE`. |
| Missing debt/equity data | Should produce unknown or limited leverage context. |
| High debt-to-equity | May produce `LEVERAGE_HIGH`, but thresholds require explicit approval. |
| Mixed metrics | Should produce `MIXED_PROFILE` or review context, not a decision. |

Thresholds must be explicit, simple, documented in tests, and descriptive only.

If no approved threshold exists, use only sign-based or presence-based interpretation at first implementation.

## 10. Quality and Analysis Boundary

Fundamental Quality answers:

```text
Do we have enough usable source/metric evidence to evaluate fundamentals?
```

Fundamental Analysis answers:

```text
What descriptive fundamental pattern does the available evidence show?
```

Decision Engine answers:

```text
What final action, if any, is authorized?
```

The first two are upstream and descriptive. The last one is downstream and authoritative.

## 11. Ticker-Category Boundary

Ticker-category logic is documented in:

```text
docs/active/logic/ticker_category_model.md
```

The first Fundamental Analysis implementation should not require ticker-category runtime input.

Future category-aware analysis may be useful, but it requires a separate approved sprint that defines:

- category source artifact;
- category validation rules;
- category assignment evidence;
- category-specific analysis mapping;
- tests proving category metadata does not create hidden allocation or filtering.

## 12. Proposed Implementation Sequence

Recommended future sequence:

1. Implement a simple Fundamental Analysis builder using `fundamental_quality.csv` and optional `fundamental_metrics.csv` fixture inputs.
2. Keep it unwired from the full pipeline initially.
3. Add focused tests for row preservation, allowed states, forbidden fields, and descriptive-only semantics.
4. Add a separate closeout after merge.
5. Only then decide whether pipeline orchestration should include `fundamental_analysis.csv`.
6. Only after orchestration decisions should downstream consumption be considered.

## 13. Proposed Builder Name

Recommended future script:

```text
scripts/core/build_fundamental_analysis.py
```

Recommended future tests:

```text
tests/core/test_build_fundamental_analysis.py
```

These names are proposed only. They do not authorize implementation in this sprint.

## 14. Required Future Tests

A future implementation sprint should add tests covering:

1. valid quality and metrics fixtures produce row-preserving analysis output;
2. insufficient quality data produces `INSUFFICIENT_DATA` or equivalent descriptive state;
3. partial metrics produce `LIMITED_ANALYSIS` or equivalent descriptive state;
4. complete metrics produce `ANALYSIS_READY` or equivalent descriptive state;
5. positive/negative margins map to descriptive margin states;
6. positive/negative growth metrics map to descriptive growth states;
7. missing leverage inputs produce unknown leverage state;
8. mixed metrics produce mixed/review state, not a decision;
9. stale or limited source data is surfaced descriptively;
10. forbidden semantic fields are not present;
11. forbidden decision-like values are not emitted;
12. row count is preserved;
13. ticker/date identity is preserved;
14. no ticker-category input is required;
15. no generated real files are required;
16. no full pipeline run is required.

## 15. Documentation Updates Needed After Future Implementation

After a future implementation sprint, update only if needed:

- `docs/active/logic/calculation_registry.md` to mark Fundamental Analysis as current;
- `docs/active/contracts/pipeline_contracts.md` if a new artifact contract becomes active;
- `docs/sprints/project_backlog.md` only if new backlog items are identified;
- a sprint closeout document after merge.

Do not rewrite active doctrine unless the implementation exposes a real conflict.

## 16. Backlog Impact Assessment

Backlog impact assessment:

- No new backlog items identified.

BL-0015 remains the broader fundamentals implementation driver. This specification prepares the Fundamental Analysis substep but does not complete BL-0015.

## 17. Validation

Documentation-only validation for this sprint should confirm:

- only documentation files changed;
- no code files changed;
- no tests changed;
- no CSV files changed;
- no raw data changed;
- no generated files changed;
- no workflow files changed;
- no provider APIs called;
- no scraping performed;
- no pipeline run;
- no tests run.

## 18. Sprint E4 Closeout Recommendation

Sprint E4 may be considered complete when this specification is reviewed and merged.

Recommended next sprint after E4:

```text
Sprint E5 — Fundamental Analysis Builder
```

Recommended type:

```text
implementation sprint, tightly scoped
```

Sprint E5 should implement only the descriptive Fundamental Analysis builder and should not combine pipeline orchestration, ticker-category runtime logic, Decision Engine changes, or Reporting changes.