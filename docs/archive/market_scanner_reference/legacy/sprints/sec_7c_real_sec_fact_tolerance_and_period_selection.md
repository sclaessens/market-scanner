# SEC-7C Real SEC Fact Tolerance and Period Selection

Status: IMPLEMENTED
Branch: codex/sec-7c-real-sec-fact-tolerance-period-selection

## Problem Statement

The SEC-7B controlled local diagnostic showed that local ticker/CIK mapping and Company Facts JSON availability worked for the review tickers, but the transformation path was too strict for messy real SEC facts. Individual facts with missing fiscal-year metadata or conflicting values could cause the whole ticker to become `TRANSFORM_REVIEW_REQUIRED`.

SEC-7C addresses that tolerance gap without integrating SEC output into the operational pipeline.

## Implemented Scope

SEC-7C updates the isolated SEC Company Facts transformer so problematic fact records and conflicting field/component groups are isolated at fact, field, component, or period level where possible.

Implemented behavior:

- facts with missing fiscal year, unsupported fiscal period, missing period end date, or invalid units are skipped with review evidence;
- same-tag/unit/period value conflicts leave the affected field or component blank for review instead of raising through the whole ticker;
- unit conflicts leave the affected field or component blank for review instead of selecting a silent winner;
- clean fields and clean periods remain transformable;
- duplicate same-value facts remain deterministic;
- derived `total_debt` and `free_cash_flow` remain available only under the approved local/fixture conditions from SEC-6C;
- derived components with conflicts remain blank with review notes and evidence.

## Explicit Non-Scope

SEC-7C does not implement:

- live SEC calls;
- SEC downloads;
- full pipeline integration;
- scheduled SEC refresh;
- real operational data transformation;
- generated operational output commits;
- Decision Engine changes;
- Reporting changes;
- Telegram changes;
- portfolio changes;
- scanner, validation, context, timing, or portfolio intelligence changes;
- fundamental quality runtime changes;
- fundamental analysis runtime changes;
- ticker-category runtime logic.

## Real SEC-7B Diagnostic Interpretation

The SEC-7B diagnostic result was not treated as a complete failure of SEC data. It showed that the review path needed better tolerance around messy facts. Missing fiscal-year facts and conflicting SEC facts should not automatically invalidate every other field and period for the same ticker.

SEC-7C preserves the interpretation that local SEC review output is source-data review material only. It does not create ranking, scoring, tradeability, urgency, conviction, eligibility, buy/sell, allocation, or final-action semantics.

## Fact Tolerance Behavior

The transformer now records skipped fact evidence for invalid individual facts, including available source tag, unit, value, fiscal year, fiscal period, period end date, filed date, form, frame, accession, and reason.

Conflicting fact groups are isolated:

- direct-field conflicts leave the field blank for the affected period;
- derived-component conflicts block the affected derived field;
- duplicate same-value facts still use deterministic first filed/form/accession order;
- no conflicting value winner is selected silently;
- missing values are not treated as zero.

## Period Preservation / Period Selection Behavior

SEC-7C preserves transformable periods even when another period contains bad facts. If an older period has conflicting facts and a recent period is clean, the clean period still transforms.

No new recent-period CLI filter was added in SEC-7C. The bounded implementation keeps default behavior as broad local transformation of all transformable periods and relies on period isolation rather than hidden filtering. A future sprint may add explicit `max_periods` or `min_fiscal_year` support if operator review needs it.

## Review Evidence Behavior

Review evidence remains row-level through the existing `notes` payload. Skipped facts are recorded under `skipped_facts`, and conflicted direct fields or derived components are recorded under review-required evidence keys.

Evidence is descriptive only. It supports auditability and source-data review; it does not imply eligibility, ranking, scoring, tradeability, urgency, conviction, allocation, buy/sell action, or final action.

## Tests Added Or Updated

Updated fixture-based tests cover:

- missing fiscal-year facts without whole-ticker failure;
- invalid fiscal periods without whole-ticker failure;
- missing period end dates without whole-ticker failure;
- same-tag/unit/period conflicts without whole-ticker failure;
- conflicted field/period review evidence;
- clean-period preservation when older periods conflict;
- deterministic duplicate same-value handling;
- clean `total_debt` derivation under approved SEC-6C conditions;
- clean `free_cash_flow` derivation under approved SEC-6C conditions;
- conflicting derived components remaining blank with review notes;
- controlled review runner output from explicit local fixture paths;
- no live SEC/network dependency in tests;
- output writes only to explicit temporary paths.

## Governance Boundary Confirmation

SEC-7C preserves:

- classification upstream;
- allocation downstream;
- Decision Engine as the only allocation authority;
- Reporting as communication only;
- no hidden filtering;
- no upstream tradeability;
- deterministic architecture;
- row preservation;
- auditability;
- separation of concerns;
- English-only repository content.

No downstream runtime behavior was changed. The implementation remains isolated to SEC transformer and controlled local review test coverage.

## Backlog Impact Assessment

Backlog impact assessment:
- No new backlog items identified.

BL-0015 and BL-0017 still cover the current SEC source-data, quality classification, and governed ingestion strategy work. SEC-7C does not reveal a new backlog gap.

## Recommended Next Sprint

Recommended next sprint: SEC-7D controlled local review diagnostics.

SEC-7D should use explicit operator-selected local inputs to review transformed outputs, summarize recurring review-required reasons, and decide whether explicit period filtering or additional deterministic fact-selection policy is needed before any pipeline integration is considered.
