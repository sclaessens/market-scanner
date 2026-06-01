# SEC-7E Period Filtering and Fact Selection Policy

Status: IMPLEMENTED
Branch: codex/sec-7e-period-filtering-fact-selection-policy

## Purpose

SEC-7E defines explicit policy for SEC Company Facts period filtering, annual versus quarterly review modes, skipped-fact summarization, and deterministic fact-selection behavior before SEC output can be considered for pipeline integration.

This sprint is documentation-only. It does not implement period filtering, fact-selection changes, pipeline integration, or generated output changes.

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

## Period Filtering Policy

SEC review and future transformation should support explicit review modes instead of implicit filtering.

Approved policy direction:

- keep raw local review capable of broad transformation;
- do not hide facts silently;
- preserve evidence for excluded or skipped facts through local diagnostics or review summaries;
- make filtering an explicit operator-selected mode;
- keep filtering local to SEC review and transformation utilities until a separate pipeline-integration specification is approved.

Approved future review modes:

| review_mode | Purpose | Pipeline readiness |
|---|---|---|
| ALL_PERIODS_REVIEW | Broad local review of all transformable periods for audit, diagnostics, and source-data understanding. | Not suitable for first pipeline integration. |
| RECENT_ANNUAL_REVIEW | Annual-only review for a recent bounded fiscal-year window. | Preferred candidate mode for first future integration specification. |
| RECENT_MIXED_REVIEW | Recent annual and quarterly review for diagnostics where quarterly behavior is explicitly being evaluated. | Review-only until a quarterly policy is separately approved. |

Default future local review behavior may remain broad, but any future operational candidate must use an explicit limited mode. The first pipeline-integration candidate should not consume every historical and quarterly period from broad review output.

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
- never silently drop excluded periods without run-level summary evidence.

Recommended default for first implementation:

```text
review_mode=RECENT_ANNUAL_REVIEW
max_annual_periods=5
```

A three-year window may be useful for minimum metric coverage, but five fiscal years better supports the current fundamentals contract preference for three to five years of raw history. The exact value should remain configurable by the SEC review utility.

## Skipped-Fact Note Summarization Policy

Skipped-fact evidence must not be discarded, but it should not be repeated noisily in every review row when a run-level summary can preserve auditability more clearly.

Approved policy direction:

- local generated diagnostics may preserve full skipped-fact details;
- committed documentation should use aggregate reason categories only;
- future review output should separate row-level notes from run-level diagnostic summaries when practical;
- skipped facts with identifiable period metadata may be attached to affected periods;
- skipped facts without enough period metadata should be summarized at run level and, where useful, ticker level;
- row-level notes should include skipped-fact details only when the skipped fact can be tied to that row or period;
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

Review-required must be used when source evidence is incomplete, ambiguous, conflicting, unsupported, or outside approved period and unit policy.

Review-required must not:

- imply poor business quality;
- imply investment weakness;
- imply allocation eligibility or ineligibility;
- rank or score tickers;
- change Decision Engine authority;
- hide rows from review output.

Future review output should distinguish:

- ticker-level review-required states, such as missing CIK or missing Company Facts file;
- period-level review-required states, such as missing or ambiguous period metadata;
- field-level review-required states, such as conflicting direct facts;
- derived-component review-required states, such as mixed debt component families or capex ambiguity;
- run-level review-required summaries, such as skipped facts without usable period metadata.

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

SEC output is not ready for pipeline integration because the project still needs approved implementation of explicit period filtering and review-summary behavior, followed by a controlled re-run that demonstrates stable transformable coverage and acceptable review-required patterns.

SEC-7E does not approve pipeline integration. Any future pipeline integration must be handled by a separate specification sprint after local review-mode filtering and summary behavior are implemented and reviewed.

## Future Implementation Handoff

Recommended implementation target:

```text
SEC-7F — Period Filtering and Review Summary Implementation
```

Allowed future implementation scope:

- add explicit local review-mode filtering to the SEC transformation or review utility;
- support `ALL_PERIODS_REVIEW`, `RECENT_ANNUAL_REVIEW`, and `RECENT_MIXED_REVIEW`;
- support `min_fiscal_year` and `max_annual_periods`;
- add local review summary generation for aggregate skipped-fact and review-required categories;
- preserve existing broad review behavior when explicitly requested;
- keep all outputs local or temp-path only unless explicitly provided by the operator.

Likely allowed files for SEC-7F:

- `scripts/fundamentals/sec_companyfacts_transform.py`
- `scripts/fundamentals/run_sec_transformation_review.py`
- optional isolated local summary helper under `scripts/fundamentals/`
- `tests/fundamentals/test_sec_companyfacts_transform.py`
- `tests/fundamentals/test_run_sec_transformation_review.py`
- optional focused test for any summary helper
- `docs/sprints/sec_7f_period_filtering_and_review_summary_implementation.md`

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

- explicit `ALL_PERIODS_REVIEW` preserves broad behavior;
- `RECENT_ANNUAL_REVIEW` includes only annual periods in the configured recent window;
- `RECENT_MIXED_REVIEW` includes recent annual and quarterly periods only when explicitly requested;
- `min_fiscal_year` excludes older periods with run-level summary evidence;
- `max_annual_periods` is deterministic;
- skipped facts are summarized without being repeated on unrelated rows;
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
SEC-7F — Period Filtering and Review Summary Implementation
```

SEC-7F should implement explicit local review modes, recent annual period filtering, configurable period windows, and aggregate review-summary behavior. It must remain isolated from pipeline integration and generated operational outputs.
