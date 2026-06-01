# SEC-7E Period Filtering and Fact Selection Policy

Status: IMPLEMENTED
Branch: codex/sec-7e-period-filtering-fact-selection-policy

## Purpose

SEC-7E defines explicit policy for SEC Company Facts period filtering, annual versus quarterly review modes, skipped-fact summarization, deterministic fact-selection behavior, and the deeper review-model semantics required before SEC output can be considered for pipeline integration.

This sprint is documentation-only. It does not implement period filtering, fact-selection changes, pipeline integration, generated output changes, or runtime review-model changes.

## Source Inputs

SEC-7E uses these source inputs:

- `AGENTS.md`
- `docs/audits/sec_interim_documentation_code_alignment_audit.md`
- `docs/sprints/sec_7c_real_sec_fact_tolerance_and_period_selection.md`
- `docs/sprints/sec_7d_controlled_local_review_diagnostics.md`
- `docs/active/source_data/sec_edgar_fundamentals_source_strategy.md`
- `docs/active/source_data/sec_edgar_source_architecture.md`
- `docs/active/source_data/sec_xbrl_mapping_investigation.md`
- `docs/active/source_data/sec_fundamental_analysis_rationalization.md`
- `docs/active/source_data/sec_derived_formula_policy.md`
- `docs/active/contracts/fundamentals_platform_contract.md`
- `docs/active/contracts/fundamental_calculations_technical_spec.md`
- `docs/sprints/project_backlog.md`
- `scripts/fundamentals/sec_companyfacts_transform.py`
- `scripts/fundamentals/run_sec_transformation_review.py`
- `tests/fundamentals/test_sec_companyfacts_transform.py`
- `tests/fundamentals/test_run_sec_transformation_review.py`

## SEC-7D Diagnostic Summary

SEC-7D ran a controlled local diagnostic from explicit local ignored inputs after SEC-7C fact-tolerance changes.

Aggregate findings:

| diagnostic | result |
|---|---:|
| Requested tickers | 5 |
| Output rows | 1283 |
| Unique tickers in output | 5 |
| Rows with `TRANSFORMED` status | 1283 |
| Rows with `review_required=true` | 1267 |
| Rows with `review_required=false` | 16 |
| Parsed notes failures | 0 |

Derived field status:

| derived_fields_status | rows |
|---|---:|
| DERIVED_FIELDS_MISSING_OR_REVIEW_REQUIRED | 748 |
| DERIVED_FIELDS_PARTIAL | 461 |
| DERIVED_FIELDS_PRESENT | 74 |

Fiscal period coverage:

| fiscal_period | rows |
|---|---:|
| FY | 454 |
| Q1 | 236 |
| Q2 | 285 |
| Q3 | 308 |

The fiscal-year range was 2009 through 2027. Skipped-fact and conflict evidence remained noisy. SEC-7D concluded that SEC output is not ready for pipeline integration.

## Scope

SEC-7E defines policy only.

This sprint covers:

- period filtering policy;
- annual versus quarterly policy;
- recent period selection policy;
- skipped-fact note summarization policy;
- deterministic fact-selection policy;
- review-required policy;
- structured review-model framing for SEC-7F;
- generated-output and safe-summary policy;
- pipeline integration readiness assessment;
- future implementation handoff.

## Explicit Non-Scope

SEC-7E does not include:

- code changes;
- test changes;
- SEC calls;
- SEC downloads;
- generated data;
- generated CSV commits;
- real SEC cache file commits;
- real SEC review output commits;
- pipeline integration;
- pipeline orchestration changes;
- Decision Engine changes;
- Reporting changes;
- Telegram changes;
- portfolio changes;
- scanner, validation, context, timing, or portfolio intelligence changes;
- fundamental quality runtime changes;
- fundamental analysis runtime changes.

## Structural Review-Model Finding For SEC-7F

SEC-7F must not be limited to simple period filtering or cosmetic note summarization.

The deeper architectural issue is that the current SEC review output does not yet expose a sufficiently explicit review model. SEC-7C made the transformer more tolerant of messy individual facts, and SEC-7D proved that broad local transformation now preserves many rows. However, the remaining output still combines several different evidence levels into row notes. That makes the diagnostic safe but too noisy for governed interpretation.

The structural problem is not only:

- too many periods;
- too many review-required rows;
- too much skipped-fact noise.

The root problem is missing review semantics. SEC output needs a model that separates:

- accepted facts that can support a field value;
- rejected or skipped facts that cannot support a field value;
- fact-level review evidence;
- period-level review evidence;
- row-level review evidence;
- ticker-level review evidence;
- run-level diagnostic summaries;
- explicit review-mode period selection.

Current known issue:

- skipped facts are collected globally at payload level;
- the global skipped-fact list is then repeated into every transformed output row's notes/evidence;
- this is safe because it does not hide evidence;
- it is noisy because every row can look affected by facts that may not belong to that row or period.

SEC-7F must therefore implement or prepare a structured local SEC review model rather than building around the limitation. Filtering and summaries are still needed, but they must be outputs of the review model, not substitutes for the model.

## Period Filtering Policy

SEC review and future transformation should support explicit review modes instead of implicit filtering.

Approved policy direction:

- parse and classify SEC facts before filtering;
- build a review model that knows accepted facts, rejected facts, period evidence, ticker evidence, and run evidence;
- apply explicit `review_mode` selection after fact classification and review-model construction;
- emit selected review rows according to the requested mode;
- emit or compute run-level diagnostic summaries for excluded, skipped, or rejected evidence;
- keep raw local review capable of broad transformation;
- do not hide facts silently;
- preserve evidence for excluded or skipped facts through local diagnostics or review summaries;
- make filtering an explicit operator-selected mode;
- keep filtering local to SEC review and transformation utilities until a separate pipeline-integration specification is approved.

Approved future review modes:

| review_mode | Purpose | Pipeline readiness |
|---|---|---|
| ALL_PERIODS_REVIEW | Broad local review of all transformable periods for audit, diagnostics, source-data understanding, and review-model validation. | Not pipeline-ready. |
| RECENT_ANNUAL_REVIEW | Annual-only review for a recent bounded fiscal-year window using explicit `min_fiscal_year` and/or `max_annual_periods`. | First future candidate for fundamentals history, but not automatically pipeline-ready. |
| RECENT_MIXED_REVIEW | Recent annual and quarterly review for diagnostics where quarterly behavior is explicitly being evaluated. | Review-only until a quarterly policy is separately approved. |

Default future local review behavior may remain broad, but any future operational candidate must use an explicit limited mode. The first pipeline-integration candidate should not consume every historical and quarterly period from broad review output.

`RECENT_ANNUAL_REVIEW` is not a workaround to hide bad rows. It is an explicit review contract for annual-only source-data evaluation. It narrows the period set only after facts have been parsed, classified, and summarized, and it must preserve evidence for excluded or rejected facts at the correct summary level.

## Annual Versus Quarterly Policy

FY periods and quarterly periods must be treated as different review contexts.

Approved policy direction:

- FY facts remain the preferred basis for the first SEC-to-fundamentals integration candidate;
- Q1, Q2, Q3, and Q4 facts may remain available for local review diagnostics;
- quarterly facts must not be mixed with annual facts silently;
- missing Q4 must not be inferred from FY;
- FY must not be inferred from Q1, Q2, Q3, or Q4 unless a future formula policy explicitly approves a derived annual or trailing calculation;
- trailing annual comparison must use comparable annual periods only unless a future sprint approves TTM or mixed-period behavior;
- quarterly facts that create noisy partial data should remain review-only until a separate quarterly integration policy exists.

First pipeline integration should use annual periods only unless a later sprint explicitly approves quarterly integration.

## Recent Period Selection Policy

Future implementation should support explicit period-window parameters.

Approved policy direction:

- support `min_fiscal_year` for review runs that need a lower fiscal-year boundary;
- support `max_annual_periods` for annual-focused review runs;
- optionally support `max_periods` only for diagnostics where annual and quarterly periods are intentionally mixed;
- keep the exact period window as a policy parameter, not a hardcoded rule in unrelated layers;
- never silently drop excluded periods without run-level summary evidence;
- never filter old periods before facts are classified;
- summarize excluded older periods at run level and, where useful, ticker level.

Recommended default for first implementation:

```text
review_mode=RECENT_ANNUAL_REVIEW
max_annual_periods=5
```

A three-year window may be useful for minimum metric coverage, but five fiscal years better supports the current fundamentals contract preference for three to five years of raw history. The exact value should remain configurable by the SEC review utility.

## Skipped-Fact Note Summarization Policy

Skipped-fact evidence must not be discarded, but it should not be repeated noisily in every review row when a run-level or ticker-level summary can preserve auditability more clearly.

Approved policy direction:

- local generated diagnostics may preserve full skipped-fact details;
- committed documentation should use aggregate reason categories only;
- future review output should separate row-level notes from period-level, ticker-level, and run-level diagnostic summaries;
- skipped facts with identifiable period metadata may be attached to affected periods;
- skipped facts without usable period metadata should be summarized at run level and, where useful, ticker level;
- row-level notes should contain only evidence relevant to that row or period;
- skipped facts must not be repeated across unrelated rows;
- future generated summaries must not be committed by default.

Recommended future summary categories:

- missing fiscal year;
- unsupported fiscal period;
- missing period end date;
- invalid unit;
- conflicting value;
- conflicting unit;
- derived component conflict;
- mixed component family;
- capex sign review;
- excluded by review mode;
- older than selected period window;
- other review-required fact issue.

## Deterministic Fact-Selection Policy

Deterministic selection is allowed only where the selected value is source-supported, units and periods are compatible, and no conflicting value is being silently resolved.

Allowed deterministic selection:

- duplicate same-value facts for the same tag, unit, period, and field may be selected deterministically with evidence;
- documented primary/alternate tag candidate order may be used when values do not conflict and units and periods are consistent;
- deterministic ordering may use tag candidate order, filing date, form, accession, and value only as a stability mechanism, not as a ranking or quality score.

Review-required fact cases:

- conflicting same-tag/unit/period values;
- conflicting units;
- invalid units;
- missing fiscal metadata;
- missing period end date;
- unresolved amended fact conflicts;
- same field with incompatible annual and quarterly contexts;
- same field with unresolved currency mismatch;
- derived component conflicts;
- mixed debt component families;
- capex sign ambiguity;
- company-specific extension tags not already mapped by policy.

Never allowed:

- silently choose a winner for conflicting values;
- treat missing values as zero;
- infer missing components;
- mix quarterly and annual data silently;
- convert currencies without approved policy;
- invent TTM;
- create ranking, scoring, eligibility, tradeability, urgency, conviction, allocation, buy/sell, final-action, or hidden filtering semantics.

## Review-Required Policy

Review-required output is a source-data readiness signal only.

Review-required must be used when source evidence is incomplete, ambiguous, conflicting, unsupported, outside approved period and unit policy, or excluded from the selected review mode while still needing audit visibility.

Review-required must not:

- imply poor business quality;
- imply investment weakness;
- imply allocation eligibility or ineligibility;
- rank or score tickers;
- change Decision Engine authority;
- hide rows from review output.

Future review output should distinguish:

- fact-level review-required states, such as invalid unit, unsupported period, or missing metadata;
- row-level review-required states, such as missing fields in the emitted period row;
- period-level review-required states, such as conflicting field evidence for a fiscal period;
- ticker-level review-required states, such as missing CIK or missing Company Facts file;
- run-level review-required summaries, such as skipped facts without usable period metadata or excluded older periods.

## Review Model Direction For SEC-7F

SEC-7F should implement a structured local SEC review model that separates at least conceptually:

```text
AcceptedFact
RejectedFact
PeriodReviewResult
RunReviewSummary
```

Exact implementation names may differ, but the design intent must remain clear.

Fact lifecycle:

```text
raw SEC fact
-> parsed candidate fact
-> accepted fact OR rejected/skipped fact
-> period assignment where possible
-> field/component selection
-> review-required classification where needed
-> row-level evidence OR period-level/ticker-level/run-level summary
```

Filtering lifecycle:

```text
parse/classify facts first
-> build review model
-> apply explicit review_mode selection
-> emit selected review rows
-> emit or compute run-level diagnostic summary
```

This preserves source-data auditability while allowing `RECENT_ANNUAL_REVIEW` to produce a narrower, more interpretable review output.

## Anti-Patterns Explicitly Rejected

SEC-7E rejects the following implementation directions for SEC-7F and later work:

- filtering old periods before classifying facts;
- hiding excluded facts silently;
- repeating all skipped facts on every row;
- treating review-required as business weakness;
- treating missing values as zero;
- silently selecting winners for conflicting facts;
- mixing annual and quarterly periods implicitly;
- using filtering to make the output look clean;
- feeding SEC output into the pipeline before review semantics are stable;
- adding allocation, tradeability, eligibility, ranking, urgency, conviction, buy/sell, or final-action semantics to SEC review output.

## Generated-Output and Safe-Summary Policy

Generated SEC review outputs remain local unless a later repository policy explicitly approves committing sanitized artifacts.

Allowed committed content:

- documentation-only policy decisions;
- aggregate diagnostic counts;
- generic reason categories;
- synthetic fixture examples if clearly marked and minimized.

Forbidden committed content:

- raw SEC JSON;
- real SEC payload excerpts;
- full generated CSV rows;
- raw financial values from real SEC facts;
- copied real company fact values;
- downloaded SEC files;
- local absolute user paths;
- generated operational artifacts.

Future local generated summaries should write only to explicit operator-provided paths or temporary test paths. No SEC review utility should write to operational pipeline paths by default.

## Pipeline Integration Readiness Assessment

Is SEC output ready for pipeline integration?

```text
No.
```

SEC output is not ready for pipeline integration because the project still needs an approved implementation of structured review semantics, explicit review-mode period selection, row-level versus run-level note separation, and controlled local summary behavior. A later controlled re-run must demonstrate stable transformable coverage and acceptable review-required patterns before any separate pipeline-integration specification can be considered.

SEC-7E does not approve pipeline integration. Any future pipeline integration must be handled by a separate specification sprint after local review-model behavior is implemented and reviewed.

## Future Implementation Handoff

Recommended implementation target:

```text
SEC-7F — SEC Review Model and Period Selection Implementation
```

Allowed future implementation scope:

- implement a structured local SEC review model;
- separate accepted facts from rejected or skipped facts;
- separate fact-level, period-level, row-level, ticker-level, and run-level evidence;
- add explicit local review-mode selection to the SEC transformation or review utility;
- support `ALL_PERIODS_REVIEW`, `RECENT_ANNUAL_REVIEW`, and `RECENT_MIXED_REVIEW`;
- support `min_fiscal_year` and `max_annual_periods`;
- add local review summary generation or computable summary structures for aggregate skipped-fact and review-required categories;
- preserve existing broad review behavior when explicitly requested;
- keep all outputs local or temp-path only unless explicitly provided by the operator.

Likely allowed files for SEC-7F:

- `scripts/fundamentals/sec_companyfacts_transform.py`
- `scripts/fundamentals/run_sec_transformation_review.py`
- optional isolated local summary helper under `scripts/fundamentals/`
- `tests/fundamentals/test_sec_companyfacts_transform.py`
- `tests/fundamentals/test_run_sec_transformation_review.py`
- optional focused test for any summary helper
- `docs/sprints/sec_7f_sec_review_model_and_period_selection_implementation.md`

Forbidden future implementation scope:

- live SEC calls in tests;
- SEC downloads;
- generated operational output commits;
- full pipeline integration;
- Decision Engine changes;
- Reporting changes;
- Telegram changes;
- portfolio changes;
- scanner, validation, context, timing, or portfolio intelligence changes;
- fundamental quality or analysis runtime changes;
- writing to `data/processed/`, `reports/`, or operational paths by default.

Required SEC-7F tests:

- accepted facts are represented in row-level evidence;
- rejected/skipped facts with period metadata are attached only to relevant period evidence;
- rejected/skipped facts without usable period metadata are not repeated on every row;
- run-level skipped fact summary is produced or computable;
- explicit `ALL_PERIODS_REVIEW` preserves broad behavior;
- `RECENT_ANNUAL_REVIEW` includes only recent FY rows;
- `RECENT_MIXED_REVIEW` includes recent annual and quarterly rows only when explicitly requested;
- `min_fiscal_year` excludes older rows with summary evidence;
- `max_annual_periods` is deterministic;
- conflicts remain review-required;
- missing values are not treated as zero;
- missing CIK and missing Company Facts row preservation remains intact;
- generated outputs write only to provided temp paths;
- no live SEC calls;
- no pipeline integration.

Validation for SEC-7F should include focused SEC fundamentals tests and `git diff --check`.

## Tests / Validation Performed

SEC-7E validation:

```bash
git diff --check
git status
```

No tests were run because SEC-7E is documentation-only and no code or tests were changed.

## Governance Boundary Confirmation

SEC-7E preserves:

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

## Backlog Impact Assessment

Backlog impact assessment:
- No new backlog items identified.

BL-0015 and BL-0017 still cover the current SEC source-data, quality classification, and governed ingestion strategy work. SEC-7E does not reveal a new backlog gap.

## Recommended Next Sprint

Recommended next sprint:

```text
SEC-7F — SEC Review Model and Period Selection Implementation
```

SEC-7F should implement a structured local SEC review model, explicit review modes, recent annual period selection, configurable period windows, and aggregate review-summary behavior. It must remain isolated from pipeline integration and generated operational outputs.
