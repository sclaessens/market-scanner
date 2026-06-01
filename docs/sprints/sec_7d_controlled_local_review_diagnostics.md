# SEC-7D Controlled Local Review Diagnostics

Status: IMPLEMENTED
Branch: codex/sec-7d-controlled-local-review-diagnostics

## Purpose

SEC-7D documents a controlled local diagnostic review of the SEC transformation output after SEC-7C fact-tolerance changes.

The goal is to summarize whether the isolated local SEC review path now preserves transformable rows, where review-required patterns remain, and whether the project is closer to safe SEC-to-fundamentals pipeline integration.

## Scope

SEC-7D is documentation-only.

The sprint uses explicit local review inputs and aggregate-only summaries. It does not add a new diagnostics helper because the existing controlled review runner already produces the required review columns and summary output for this sprint.

## Explicit Non-Scope

SEC-7D does not include:

- live SEC calls;
- SEC downloads;
- generated output commits;
- real SEC payload commits;
- generated CSV commits;
- full pipeline integration;
- scheduled SEC refresh;
- operational fundamentals output changes;
- Decision Engine changes;
- Reporting changes;
- Telegram changes;
- portfolio changes;
- scanner, validation, context, timing, or portfolio intelligence changes;
- fundamental quality runtime changes;
- fundamental analysis runtime changes;
- ticker-category runtime logic.

## SEC-7C Baseline

SEC-7C improved the isolated SEC Company Facts transformer so messy individual facts no longer fail an entire ticker by default.

The SEC-7C transformer now:

- skips facts with missing or invalid fiscal metadata or invalid units with review evidence;
- isolates conflicting direct fields or derived components;
- preserves transformable clean periods;
- keeps duplicate same-value facts deterministic;
- keeps derived `total_debt` and `free_cash_flow` conservative;
- remains isolated from the operational pipeline.

## Controlled Local Diagnostic Procedure

The controlled local review runner must be invoked only with explicit local inputs and an explicit output path:

```bash
.venv/bin/python scripts/fundamentals/run_sec_transformation_review.py \
  --project-tickers <local_project_tickers_file> \
  --ticker-cik-source <local_ticker_cik_source_file> \
  --companyfacts-dir <local_companyfacts_dir> \
  --source-freshness-date <YYYY-MM-DD> \
  --extraction-date <YYYY-MM-DD> \
  --output <local_ignored_or_temp_output_path>
```

For this sprint, the existing ignored local SEC review input bundle was used and the generated diagnostic output was written to a temporary path outside the repository. The generated output was not committed.

## Safe Summary Policy

Committed diagnostics may include only aggregate summaries:

- number of requested tickers;
- number of output rows;
- transformation status distribution;
- review-required distribution;
- derived field status distribution;
- missing-fields distribution;
- generic review reason categories;
- skipped fact reason categories;
- whether output remains row-preserving;
- whether transformable periods are present;
- whether recurring blockers suggest period filtering or fact-selection policy.

Committed diagnostics must not include:

- raw SEC JSON;
- real SEC payload excerpts;
- full generated CSV rows;
- raw financial values;
- copied real company fact values;
- downloaded SEC files;
- local absolute user paths;
- generated operational artifacts.

## Diagnostic Findings

Aggregate-only local diagnostic findings:

| diagnostic | result |
|---|---:|
| Requested tickers | 5 |
| Output rows | 1283 |
| Unique tickers in output | 5 |
| Rows with `TRANSFORMED` status | 1283 |
| Rows with `review_required=true` | 1267 |
| Rows with `review_required=false` | 16 |
| Rows with parsed notes failures | 0 |

Transformation status distribution:

| transformation_status | rows |
|---|---:|
| TRANSFORMED | 1283 |

Derived field status distribution:

| derived_fields_status | rows |
|---|---:|
| DERIVED_FIELDS_MISSING_OR_REVIEW_REQUIRED | 748 |
| DERIVED_FIELDS_PARTIAL | 461 |
| DERIVED_FIELDS_PRESENT | 74 |

Most common missing-field combinations:

| missing_fields | rows |
|---|---:|
| revenue, gross_profit, operating_income, net_income, diluted_eps, total_debt, free_cash_flow | 439 |
| revenue, gross_profit, operating_income, net_income, diluted_eps, total_debt | 131 |
| operating_income, total_debt, total_equity, free_cash_flow | 116 |
| total_debt | 100 |
| total_debt, free_cash_flow | 62 |

Review reason categories:

| review_reason_category | rows |
|---|---:|
| missing fields | 1267 |
| all reviewed fields present | 16 |

Skipped fact and conflict evidence:

| diagnostic | result |
|---|---:|
| Rows with skipped-fact notes | 704 |
| Unique skipped facts represented in notes | 37 |
| Unique skipped facts with missing fiscal year reason | 37 |
| Rows with conflict or unit-conflict notes | 341 |
| Rows with review-required field or derived-component evidence | 387 |

Fiscal period coverage:

| fiscal_period | rows |
|---|---:|
| FY | 454 |
| Q1 | 236 |
| Q2 | 285 |
| Q3 | 308 |

The fiscal-year range in the local review output spans 2009 through 2027.

## Interpretation

SEC-7C materially improved the review path: the controlled local review now produces transformed rows instead of whole-ticker `TRANSFORM_REVIEW_REQUIRED` output for the local review set.

The diagnostic still shows that the output is not pipeline-ready. Nearly all rows remain review-required, missing-field combinations are frequent, derived field coverage is uneven, and skipped-fact notes are noisy enough to need a clearer diagnostic policy.

The broad fiscal-year range and mixed annual/quarterly output confirm that local review can preserve data, but the project still needs an explicit period-selection policy before any operational integration is considered.

## Decision On Period Filtering

Period filtering is needed before pipeline integration.

SEC-7D does not implement filtering. The diagnostic indicates that a future policy should decide whether controlled review output should support an explicit `min_fiscal_year`, `max_periods`, annual-only review mode, or another deterministic period-selection rule.

Any future filtering must remain explicit and reviewable. It must not hide excluded facts without review evidence and must not create ranking, scoring, eligibility, tradeability, urgency, conviction, allocation, buy/sell, or final-action semantics.

## Decision On Further Fact-Selection Policy

Further deterministic fact-selection policy is needed.

SEC-7C correctly avoided silent winners for conflicting facts. SEC-7D shows conflicts and review-required evidence still occur often enough that a future policy should decide:

- whether certain duplicate/amended fact patterns may be deterministically resolved;
- whether older periods should be separated from current review periods;
- whether skipped-fact notes should be summarized separately from every transformed row;
- whether review outputs should distinguish field-level, component-level, period-level, and ticker-level review-required states more explicitly.

## Pipeline Integration Readiness Assessment

Is SEC output ready for pipeline integration?

No. SEC output is not yet ready for pipeline integration until controlled local diagnostics show stable transformable coverage, acceptable review-required patterns, and an approved policy for any remaining period filtering or fact-selection ambiguity.

SEC-7D does not approve pipeline integration. Any future pipeline work should start with a separate pipeline-integration specification sprint after period and fact-selection policy is accepted.

## Tests / Validation Performed

Validation performed for SEC-7D:

- started from latest `main`;
- ran the controlled local review runner with explicit local inputs and a temporary output path;
- inspected only aggregate counts and generic categories from the generated temporary output;
- ran `git diff --check`;
- ran `git status`.

No test suite changes were made because SEC-7D is documentation-only and no helper was added.

## Governance Boundary Confirmation

SEC-7D preserves:

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

BL-0015 and BL-0017 still cover the current SEC source-data, quality classification, and governed ingestion strategy work. SEC-7D does not reveal a new backlog gap.

## Recommended Next Sprint

Recommended next sprint:

```text
SEC-7E — Period Filtering and Fact Selection Policy
```

SEC-7E should define explicit policy for recent-period selection, annual versus quarterly review modes, skipped-fact note summarization, and any deterministic fact-selection rules that may safely reduce review-required noise before SEC output is considered for a later pipeline-integration specification sprint.
