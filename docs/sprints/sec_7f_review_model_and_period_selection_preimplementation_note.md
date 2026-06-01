# SEC-7F Review Model and Period Selection Pre-Implementation Note

Status: PRE-IMPLEMENTATION NOTE
Branch: docs/sec-7f-review-model-framing
Target future implementation sprint: SEC-7F

## Recommended Sprint Title

```text
SEC-7F — SEC Review Model and Period Selection Implementation
```

This replaces the earlier narrower framing:

```text
SEC-7F — Period Filtering and Review Summary Implementation
```

SEC-7F must be treated as a structural review-model implementation sprint, not as a cosmetic filtering or summarization workaround.

## Purpose

This note prepares the future Codex implementation sprint for SEC-7F.

It documents the deeper technical problem found after SEC-7C, SEC-7D, and SEC-7E: the SEC review path can now preserve transformable rows, but it still lacks a sufficiently explicit review model for separating accepted facts, rejected/skipped facts, period evidence, row evidence, ticker evidence, and run-level diagnostic summaries.

SEC-7F must address the root architecture issue before SEC output can move closer to governed fundamentals ingestion.

## Problem Statement

SEC-7C improved the isolated SEC Company Facts transformer so messy individual facts no longer fail an entire ticker by default.

SEC-7D then documented a controlled local diagnostic:

- requested tickers: 5;
- output rows: 1283;
- unique tickers in output: 5;
- rows with `TRANSFORMED` status: 1283;
- rows with `review_required=true`: 1267;
- rows with `review_required=false`: 16;
- parsed notes failures: 0;
- derived fields:
  - `DERIVED_FIELDS_MISSING_OR_REVIEW_REQUIRED`: 748;
  - `DERIVED_FIELDS_PARTIAL`: 461;
  - `DERIVED_FIELDS_PRESENT`: 74;
- fiscal period coverage:
  - `FY`: 454;
  - `Q1`: 236;
  - `Q2`: 285;
  - `Q3`: 308;
- fiscal-year range: 2009 through 2027;
- skipped-fact and conflict evidence remains noisy;
- pipeline integration readiness conclusion: no.

SEC-7E defined explicit review modes and annual-first policy, but SEC-7F must go deeper than applying filters and shortening notes.

The current review output can make every emitted row look affected by every skipped fact because skipped facts are collected globally at payload level and repeated into output row notes/evidence. That is safe from an audit perspective, but it is too noisy and not semantically precise enough for future governed ingestion.

## Root Cause

The root cause is missing review-model separation.

The existing output still blends:

- accepted source facts;
- rejected or skipped source facts;
- fact-level review evidence;
- field/component review evidence;
- period-level review evidence;
- row-level review evidence;
- ticker-level review evidence;
- run-level diagnostic summaries;
- selected review-mode output rows.

Without this separation, period filtering can reduce output volume but cannot solve the deeper ambiguity. SEC-7F must first create the conceptual and implementation structure that allows review evidence to be attached at the correct level.

## Anti-Patterns To Avoid

SEC-7F must explicitly avoid:

- filtering old periods before classifying facts;
- hiding excluded facts silently;
- repeating all skipped facts on every row;
- treating review-required as business weakness;
- treating missing values as zero;
- silently selecting winners for conflicting facts;
- mixing annual and quarterly periods implicitly;
- using filtering to make the output look clean;
- feeding SEC output into the pipeline before review semantics are stable;
- creating ranking, scoring, eligibility, tradeability, urgency, conviction, allocation, buy/sell, or final-action semantics.

## Required Structural Solution

SEC-7F should implement or prepare a structured local SEC review model that separates at least conceptually:

```text
AcceptedFact
RejectedFact
PeriodReviewResult
RunReviewSummary
```

Exact names may differ, but the implementation must preserve these responsibilities.

The model should make it possible to answer:

- which raw SEC facts were accepted;
- which raw SEC facts were rejected or skipped;
- why a fact was rejected or skipped;
- whether rejected/skipped evidence has usable period metadata;
- which period a fact belongs to when period assignment is possible;
- which period-level rows are emitted under the selected review mode;
- which evidence belongs in a row-level note;
- which evidence belongs at period, ticker, or run level only;
- what was excluded by period selection;
- whether pipeline integration remains blocked.

## Proposed Internal Review Model

The future implementation may choose dataclasses, dictionaries, typed records, or small helper functions, but it should conceptually distinguish these entities.

### AcceptedFact

Represents a parsed SEC fact that can support a direct field or derived component.

Expected responsibilities:

- preserve source tag;
- preserve unit;
- preserve source value;
- preserve fiscal year;
- preserve fiscal period;
- preserve period end date;
- preserve filed date, form, frame, accession when available;
- preserve field or component target;
- preserve deterministic-selection notes where needed.

### RejectedFact

Represents a parsed or partially parsed SEC fact that cannot support a field/component.

Expected responsibilities:

- preserve available source metadata;
- preserve rejection reason;
- preserve whether fiscal year, fiscal period, and period end date are usable;
- preserve whether the rejected fact can be attached to a specific period;
- avoid forcing unrelated rows to carry this evidence.

Rejected facts include skipped facts, invalid-unit facts, unsupported-period facts, missing-metadata facts, and conflicting facts that must remain review-required.

### PeriodReviewResult

Represents the review state for one ticker and one fiscal period where a period can be identified.

Expected responsibilities:

- group accepted facts by ticker/period;
- group rejected facts that can be tied to the same ticker/period;
- compute direct fields and approved derived fields only from accepted facts;
- mark conflicting or missing fields review-required;
- include row-level notes only when relevant to that period row;
- allow the row emitter to select periods according to `review_mode`.

### RunReviewSummary

Represents aggregate evidence that should not be repeated across unrelated rows.

Expected responsibilities:

- count skipped/rejected facts by reason;
- count rejected facts without usable period metadata;
- count rows excluded by `review_mode`;
- count older periods excluded by `min_fiscal_year`;
- count periods excluded by `max_annual_periods`;
- summarize ticker-level failures such as missing CIK or missing Company Facts file;
- remain local and uncommitted unless a later governance policy explicitly approves generated summaries.

## Fact Lifecycle

SEC-7F should document and implement this lifecycle:

```text
raw SEC fact
-> parsed candidate fact
-> accepted fact OR rejected/skipped fact
-> period assignment where possible
-> field/component selection
-> review-required classification where needed
-> row-level evidence OR period-level/ticker-level/run-level summary
```

Important constraints:

- missing values are not zero;
- rejected/skipped facts remain auditable;
- facts with usable period metadata should attach only to affected period evidence;
- facts without usable period metadata should be summarized at run level and optionally ticker level;
- conflicts remain review-required unless an approved deterministic policy applies;
- no hidden filtering is allowed.

## Evidence Lifecycle

Evidence must be attached at the narrowest correct level.

### Row-level evidence

Row-level notes should contain only evidence relevant to the emitted row and period.

Examples:

- selected direct field evidence;
- selected derived-component evidence;
- duplicate same-value deterministic-selection note for that row;
- conflict evidence for a field in that row;
- missing-field evidence for that row;
- rejected/skipped facts tied to that exact period.

### Period-level evidence

Period-level evidence may contain facts and review issues attached to a fiscal year/period even when not all of them are emitted directly in the row notes.

Examples:

- rejected facts with the same fiscal year/fiscal period/end date;
- period-specific conflicting tags;
- period-specific invalid units;
- period-level field/component review state.

### Ticker-level evidence

Ticker-level evidence should be used when the issue affects a ticker but cannot be assigned to a specific emitted row.

Examples:

- missing Company Facts file;
- malformed payload;
- ticker-level mapping issue;
- rejected facts with ticker context but no usable period metadata.

### Run-level evidence

Run-level evidence should summarize aggregate diagnostic information.

Examples:

- skipped-fact reason counts;
- facts without usable period metadata;
- rows excluded by `review_mode`;
- older periods excluded by `min_fiscal_year`;
- generated summary categories.

Run-level evidence must remain local by default and must not be committed unless a later governance policy approves sanitized artifacts.

## Review Modes

SEC-7F must support or prepare these explicit review modes.

### ALL_PERIODS_REVIEW

Purpose:

- broad local diagnostic mode;
- preserves broad behavior;
- useful for audit and source-data understanding.

Policy:

- not pipeline-ready;
- not a first fundamentals integration candidate;
- may include annual and quarterly rows;
- must still use structured evidence separation.

### RECENT_ANNUAL_REVIEW

Purpose:

- annual-only review mode;
- first future candidate for fundamentals history;
- bounded by explicit `min_fiscal_year` and/or `max_annual_periods`.

Policy:

- not automatically pipeline-ready;
- excludes quarterly rows from emitted output;
- excludes older FY rows according to explicit parameters;
- must summarize excluded evidence rather than hiding it silently;
- must not mix quarterly and annual data.

### RECENT_MIXED_REVIEW

Purpose:

- review-only annual + quarterly diagnostic mode;
- used only when quarterly behavior is intentionally being evaluated.

Policy:

- not pipeline-ready until quarterly policy is approved;
- must be explicitly requested;
- must not imply TTM, inferred annuals, or approved operational quarterly ingestion.

## Period Selection Semantics

Filtering must happen after fact classification, not before.

Required lifecycle:

```text
parse/classify facts first
-> build review model
-> apply explicit review_mode selection
-> emit selected review rows
-> emit or compute run-level diagnostic summary
```

Required semantics:

- `min_fiscal_year` excludes older rows after review-model construction;
- `max_annual_periods` selects a deterministic bounded number of FY periods;
- excluded facts and periods must be represented in summary evidence;
- annual and quarterly periods must not be mixed implicitly;
- `RECENT_ANNUAL_REVIEW` is a contract, not a data-cleaning workaround;
- `ALL_PERIODS_REVIEW` remains broad and review-only.

## Row-Level Versus Run-Level Note Policy

Row-level notes:

- include only evidence relevant to that emitted row/period;
- may include accepted fact evidence;
- may include period-specific rejected fact evidence;
- may include field/component conflicts for that row;
- must not repeat global skipped-fact lists.

Run-level summaries:

- include aggregate skipped/rejected categories;
- include facts without usable period metadata;
- include excluded-period categories;
- include review-mode selection summaries;
- remain local and uncommitted by default.

Ticker-level summaries may be used as an intermediate layer for issues that affect one ticker but no specific emitted period.

## Local Summary Requirements

SEC-7F should produce or make computable a local summary covering:

- requested tickers;
- unique tickers represented;
- output rows;
- emitted rows by `review_mode`;
- fiscal period distribution;
- fiscal-year range in emitted output;
- excluded row/period count by review-mode reason;
- skipped/rejected fact count by reason;
- facts with usable period metadata versus without usable period metadata;
- rows with review-required true/false;
- derived-field status distribution;
- conflict count by category;
- missing CIK row preservation;
- missing Company Facts row preservation.

Generated run-level outputs must remain local and uncommitted unless explicitly approved.

## Implementation Boundaries

SEC-7F is allowed to change only the isolated SEC local review and transformer surfaces required to implement the review model and period selection.

It must not introduce pipeline integration.

It must not change runtime behavior outside the SEC fundamentals review path.

It must not run live SEC calls in tests.

It must not commit real SEC data or generated outputs.

## Allowed Future Files

Allowed future implementation files:

- `scripts/fundamentals/sec_companyfacts_transform.py`
- `scripts/fundamentals/run_sec_transformation_review.py`
- optional isolated local summary helper under `scripts/fundamentals/`
- `tests/fundamentals/test_sec_companyfacts_transform.py`
- `tests/fundamentals/test_run_sec_transformation_review.py`
- optional focused summary-helper tests
- `docs/sprints/sec_7f_sec_review_model_and_period_selection_implementation.md`

## Forbidden Future Files

Forbidden future implementation files and areas:

- `scripts/core/decision_engine.py`
- `scripts/reporting/`
- `scripts/telegram/`
- `scripts/run_full_pipeline.py`
- `scripts/run_scan.py`
- `.github/workflows/`
- `data/`
- `reports/`
- generated files
- CSV files
- real SEC cache files
- real SEC review outputs
- portfolio files
- scanner runtime files
- validation runtime files
- context runtime files
- timing runtime files
- portfolio intelligence runtime files

## Required Future SEC-7F Tests

Required tests:

- accepted facts are represented in row-level evidence;
- rejected/skipped facts with period metadata are attached only to relevant period evidence;
- rejected/skipped facts without usable period metadata are not repeated on every row;
- run-level skipped fact summary is produced or computable;
- `ALL_PERIODS_REVIEW` preserves broad behavior;
- `RECENT_ANNUAL_REVIEW` includes only recent FY rows;
- `RECENT_MIXED_REVIEW` includes recent annual and quarterly rows only when explicitly requested;
- `min_fiscal_year` excludes older rows with summary evidence;
- `max_annual_periods` is deterministic;
- conflicts remain review-required;
- missing values are not treated as zero;
- missing CIK and missing Company Facts row preservation remains intact;
- no generated output writes unless explicit path is provided;
- no live SEC calls;
- no pipeline integration.

## Validation Commands For Future SEC-7F

Required validation commands:

```bash
git diff --check
.venv/bin/python -m pytest tests/fundamentals/test_sec_companyfacts_transform.py
.venv/bin/python -m pytest tests/fundamentals/test_run_sec_transformation_review.py
```

If an optional summary helper is introduced, add the focused summary-helper test command.

Optional broader validation, if runtime boundaries remain isolated:

```bash
.venv/bin/python -m pytest tests/fundamentals
```

## Expected Future SEC-7F PR Title

```text
feat: implement SEC review model and period selection
```

## Expected Future SEC-7F PR Body

Expected PR body for the future Codex implementation sprint:

```markdown
## Summary

Implements SEC-7F review-model and period-selection behavior for the isolated SEC Company Facts local review path.

- separates accepted facts from rejected/skipped facts
- separates row-level, period-level, ticker-level, and run-level evidence
- supports `ALL_PERIODS_REVIEW`, `RECENT_ANNUAL_REVIEW`, and `RECENT_MIXED_REVIEW`
- supports deterministic annual period selection through `min_fiscal_year` and/or `max_annual_periods`
- prevents global skipped facts from being repeated on unrelated rows
- preserves review-required behavior for conflicts and missing values
- keeps SEC output blocked from pipeline integration

## Boundaries

- no live SEC calls
- no real SEC data committed
- no generated data committed
- no pipeline integration
- no Decision Engine changes
- no Reporting changes
- no Telegram changes
- no scanner/validation/context/timing/portfolio intelligence changes
- tests use fixtures/temp paths
- generated outputs write only to explicit paths

## Validation

- `git diff --check`
- `.venv/bin/python -m pytest tests/fundamentals/test_sec_companyfacts_transform.py`
- `.venv/bin/python -m pytest tests/fundamentals/test_run_sec_transformation_review.py`
```

## Recommended Next Sprint After SEC-7F

Recommended next sprint after SEC-7F:

```text
SEC-7G — Controlled SEC Review Re-Run and Integration Readiness Assessment
```

SEC-7G should run a controlled local diagnostic using the new review model and explicit review modes. It should document aggregate-only results and decide whether SEC output is ready for a separate pipeline-integration specification sprint.

SEC-7G must remain diagnostic and documentation/governance oriented unless a later approved sprint explicitly authorizes integration work.

## Backlog Impact Assessment

Backlog impact assessment:

- No new backlog item is required by this pre-implementation note.

BL-0015 and BL-0017 still cover the current SEC source-data, quality classification, and governed ingestion strategy work. This note reframes SEC-7F under the existing SEC/fundamentals backlog drivers rather than opening a new backlog gap.

## Governance Boundary Confirmation

This note preserves:

- classification upstream;
- allocation downstream;
- Decision Engine as the only allocation authority;
- Reporting as communication only;
- no hidden filtering;
- no upstream tradeability;
- no eligibility, ranking, scoring, urgency, conviction, allocation, buy/sell, or final-action semantics;
- deterministic architecture;
- row preservation;
- auditability;
- separation of concerns;
- English-only repository content.

No runtime behavior changed. No pipeline integration was introduced.
