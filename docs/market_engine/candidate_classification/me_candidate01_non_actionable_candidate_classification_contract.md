# ME-CANDIDATE01 - Non-actionable candidate classification contract

Sprint: ME-CANDIDATE01 - Define non-actionable candidate classification contract

Status: Defined

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-CANDIDATE - Candidate Classification

## Purpose

ME-CANDIDATE01 defines the first formal Candidate Classification contract for Market Engine output that is downstream of readable operator reporting and upstream of any later implementation work.

The contract classifies existing Market Engine artifact evidence into review-only candidate buckets that help a human operator understand which tickers may deserve further manual review. It is explicitly non-actionable and must not create trading, allocation, ranking, scoring, urgency, conviction, target-price, watchlist-mutation, broker, delivery, or execution authority.

ME-CANDIDATE01 is documentation-only. It does not implement Python code, tests, CLI behavior, runtime classification, provider calls, source refresh, live market data, broker calls, portfolio writes, watchlist writes, Telegram/email delivery, scheduler behavior, UI behavior, production reporting, or Decision Engine behavior.

## Upstream Basis

ME-CANDIDATE01 builds on the completed supported-universe output chain:

* ME-RUN20 executed the supported active cached-source subset and produced local dry-run artifacts.
* ME-RUN21 inspected those artifacts and confirmed structural completeness for the supported subset.
* ME-RUN22 produced the first human-readable interpretation report from those artifacts.
* ME-OUT01 defined `market-engine-readable-operator-report-v1`.
* ME-OUT02 implemented deterministic local readable operator report generation.

Candidate Classification does not replace readable operator reporting. It defines a narrower downstream interpretation layer that may group artifact evidence into non-actionable candidate review categories after readable output exists.

## Contract Name

Approved future contract family:

```text
market-engine-candidate-classification-v1
```

Recommended future Markdown filename:

```text
candidate_classification_report.md
```

Recommended future machine-readable companion filename:

```text
candidate_classification_summary.json
```

Recommended future local output path category:

```text
artifacts/market_engine/<candidate_classification_run_id>/
```

Generated candidate-classification artifacts must remain local and non-production unless a later sprint explicitly authorizes a different persistence, delivery, publication, or UI boundary.

## Approved Inputs

Candidate Classification may consume only existing local artifacts and existing local summaries produced by approved Market Engine runs.

Approved input artifact families:

```text
market-engine-readable-operator-report-v1
market-engine-local-dry-run-artifact-v1
market-engine-local-dry-run-artifact-manifest-v1
market-engine-end-to-end-dry-run-v1
market-engine-interpretation-report-v1
```

Approved input file categories:

```text
artifacts/market_engine/<operator_report_run_id>/operator_report.md
artifacts/market_engine/<operator_report_run_id>/operator_report_summary.json
artifacts/market_engine/<dry_run_artifact_root>/<TICKER>/dry_run.json
artifacts/market_engine/<dry_run_artifact_root>/<TICKER>/manifest.json
artifacts/market_engine/<interpretation_report_run_id>/market_engine_interpretation_report.md
artifacts/market_engine/<interpretation_report_run_id>/market_engine_interpretation_report_summary.json
```

Candidate Classification must not call providers, refresh source snapshots, repair artifacts, infer missing upstream values, query broker APIs, read live market data, read portfolio state outside an already-approved local dry-run artifact, or mutate any upstream artifact.

## Classification Audience

The primary audience is the human operator who wants to know which completed supported-universe ticker artifacts deserve further manual review.

The classification is allowed to answer these questions:

* Which tickers have complete artifact evidence available for candidate review?
* Which tickers cannot be classified because artifact evidence is missing, stale, blocked, malformed, or unsupported?
* Which non-actionable review bucket best describes the current artifact evidence?
* Which evidence references justify the bucket?
* Which data limitations must be reviewed before any future downstream work?
* Which safe implementation sprint could build this contract later?

The classification is not allowed to answer whether the operator should buy, sell, hold, allocate capital, enter a trade, exit a trade, size a position, prioritize an execution order, create a broker order, mutate a watchlist, send an alert, or route anything to a delivery channel.

## Required Top-Level Output Sections

A compliant future Markdown candidate classification report must contain these sections in deterministic order:

1. Classification metadata
2. Source artifact boundary
3. Non-actionable boundary
4. Input coverage summary
5. Classification method
6. Candidate bucket summary
7. Per-ticker candidate classifications
8. Unclassified, blocked, skipped, stale, and malformed ticker notes
9. Missing-data and stale-data notes
10. Provenance summary
11. Human-review checklist
12. Safe next-step candidate
13. Appendix: machine-readable summary reference

The exact wording may evolve in implementation, but these sections must remain present unless a future contract version explicitly replaces them.

## Required Metadata

The classification metadata section must include:

* candidate classification format version;
* candidate classification run id;
* generated-at timestamp when supplied by the caller;
* input operator report root when supplied;
* input dry-run artifact root when supplied;
* input interpretation report root when supplied;
* included ticker count;
* classified ticker count;
* unclassified ticker count;
* blocked ticker count when available from upstream artifacts;
* skipped ticker count when available from upstream artifacts;
* stale-data marker count when available from upstream artifacts;
* local-only / non-production marker;
* non-actionable boundary marker.

The output must preserve original upstream artifact identifiers and must not create hidden run identifiers.

## Allowed Candidate Buckets

A compliant future implementation may emit only these candidate buckets:

```text
ready_for_manual_candidate_review
requires_missing_data_review
requires_stale_data_review
requires_blocked_state_review
requires_source_coverage_review
requires_portfolio_context_review
requires_human_interpretation_review
unclassified_due_to_malformed_artifact
unclassified_due_to_unsupported_input
unclassified_due_to_insufficient_evidence
```

Bucket semantics:

* `ready_for_manual_candidate_review` means the artifact evidence is structurally complete enough for a human to inspect manually. It does not mean attractive, buyable, tradable, urgent, high conviction, or portfolio-ready.
* `requires_missing_data_review` means upstream missing-data markers are present and must remain visible.
* `requires_stale_data_review` means upstream stale-data markers are present and must remain visible.
* `requires_blocked_state_review` means one or more upstream stages were blocked or did not complete.
* `requires_source_coverage_review` means source support, source identity, source snapshot, provider-error, or coverage evidence needs review.
* `requires_portfolio_context_review` means classification evidence depends on incomplete, stale, absent, or blocked portfolio-context evidence already present upstream.
* `requires_human_interpretation_review` means evidence is readable but not specific enough for a narrower non-actionable bucket.
* `unclassified_due_to_malformed_artifact` means the artifact could not be read or parsed safely.
* `unclassified_due_to_unsupported_input` means an input contract or version is not approved by this contract.
* `unclassified_due_to_insufficient_evidence` means the artifact is readable but lacks enough approved evidence to classify.

Candidate buckets must not be ordered as a ranking. The recommended display order is the fixed order above, followed by alphabetical ticker ordering inside each bucket.

## Explicitly Forbidden Buckets And Fields

Candidate Classification must not emit buckets, fields, labels, or text that create or imply:

* BUY / SELL / HOLD;
* entry price;
* exit price;
* target price;
* target weight;
* position size;
* allocation instruction;
* rebalance instruction;
* broker-ready order;
* execution instruction;
* ticker ranking;
* score;
* grade;
* urgency;
* conviction;
* tradeability;
* expected return;
* risk/reward rating;
* probability of success;
* watchlist mutation;
* alert delivery;
* Telegram/email delivery instruction;
* automatic Decision Engine request.

The term `candidate` means candidate for human review only. It must not mean candidate trade, candidate position, candidate order, candidate allocation, or candidate watchlist entry.

## Classification Evidence Requirements

Each per-ticker classification must preserve or reference:

* ticker symbol;
* upstream artifact file references;
* upstream format versions;
* run state;
* input mode;
* output families present;
* completed and non-completed stages;
* missing-data markers;
* stale-data markers;
* blocked reasons;
* Decision Engine handoff state when present, without creating new Decision Engine conclusions;
* Delivery / Reporting state when present, without creating delivery authority;
* portfolio-context state when present;
* source snapshot reference when present;
* manifest metadata when present;
* provenance references when present;
* numeric-zero evidence when present.

A future implementation may summarize evidence concisely, but it must not suppress the source marker that caused a ticker to be classified into a review-required or unclassified bucket.

## Numeric-Zero And Missing-Data Rules

Candidate Classification must preserve the distinction between numeric zero and missing data.

The classifier must not:

* convert missing numeric values to zero;
* hide missing values because a ticker has enough other evidence;
* infer a missing value from another field;
* repair stale metadata;
* suppress blocked states because downstream stages are absent;
* convert blocked, missing, or stale states into investment quality judgements.

Numeric zero may be listed as evidence only when it is explicitly present in upstream artifacts.

## Provenance Requirements

Candidate Classification must preserve references to upstream evidence when available.

Allowed provenance references include:

* local artifact paths;
* operator report summary paths;
* interpretation report summary paths;
* source snapshot identifiers;
* SEC CompanyFacts source references already present in upstream artifacts;
* stage output family names;
* upstream observation references;
* manifest metadata;
* blocked-reason references;
* missing-data references;
* stale-data references.

The classifier must not fabricate source references or cite external sources that were not part of the inspected local artifacts.

## Machine-Readable Companion Summary

A future implementation should emit a JSON companion summary with this minimum metadata:

```json
{
  "candidate_classification_format_version": "market-engine-candidate-classification-v1",
  "candidate_classification_run_id": "<candidate_classification_run_id>",
  "generated_at": "<timestamp-or-null>",
  "input_operator_report_root": "<path-or-null>",
  "input_artifact_root": "<path-or-null>",
  "input_interpretation_report_root": "<path-or-null>",
  "included_tickers": [],
  "classified_tickers": [],
  "unclassified_tickers": [],
  "bucket_counts": {},
  "per_ticker_classifications": [],
  "missing_data_notes_present": false,
  "stale_data_notes_present": false,
  "blocked_notes_present": false,
  "provenance_references_present": false,
  "numeric_zero_evidence_present": false,
  "non_actionable_boundary": true,
  "advisory_language_guardrail": {
    "forbidden_action_terms_checked": true,
    "candidate_classification_contains_trading_instruction": false
  }
}
```

The JSON companion summary is for validation, automation, and audit support only. It must not become a broker-ready payload, alert payload, ranking payload, scoring payload, watchlist payload, or production delivery payload.

## Human-Review Checklist

The report must include a human-review checklist that helps the operator decide whether more Market Engine work is needed.

Allowed checklist items:

* input operator report readable;
* input artifact root readable;
* all expected ticker directories present;
* all expected `dry_run.json` files present;
* all expected `manifest.json` files present;
* all JSON files parse successfully;
* all required output families present;
* missing-data notes reviewed;
* stale-data notes reviewed;
* blocked tickers reviewed;
* provenance references reviewed;
* numeric-zero evidence reviewed;
* candidate buckets reviewed as human-review buckets only;
* non-actionable boundary preserved.

The checklist must not include trade-entry, trade-exit, allocation, position-sizing, broker, order, alert, Telegram, email, watchlist, or external delivery steps.

## Safe Next-Step Candidate

The report may include exactly one safe next-step candidate.

Allowed next-step semantics:

* propose implementation of this Candidate Classification contract;
* propose QA contract regression around candidate-classification artifact inputs;
* propose human review of source coverage limitations;
* propose human review of missing or stale data;
* propose human review of malformed, blocked, or unsupported artifacts.

Forbidden next-step semantics:

* trade now;
* buy below a price;
* sell above a price;
* hold a current position;
* allocate a percentage;
* rebalance a portfolio;
* add to a watchlist;
* send a broker order;
* send Telegram or email delivery;
* treat the classification as production advice.

## Fail-Closed Requirements

A future implementation must fail closed or skip explicitly when:

* the input operator report root is missing when required by invocation;
* the input artifact root is missing when required by invocation;
* an input root is not a directory;
* a required summary file is missing;
* a ticker directory lacks `dry_run.json`;
* a ticker directory lacks `manifest.json`;
* JSON parsing fails;
* an artifact declares an unsupported format version;
* an operator report summary declares an unsupported format version;
* the candidate classification run id is unsafe;
* the output path would escape the approved output root;
* the output directory already exists.

Failure output must be operator-readable and must preserve the reason without suppressing the affected ticker.

## Determinism Requirements

A future implementation must be deterministic for identical inputs.

Required deterministic behavior:

* stable ticker ordering;
* stable bucket ordering;
* stable section ordering;
* stable skipped-reason ordering;
* stable output-family ordering;
* stable stage-state ordering;
* stable JSON key ordering when practical;
* caller-supplied generated-at timestamp for reproducible test output.

## Explicit Non-Scope

ME-CANDIDATE01 does not authorize:

* Python implementation;
* tests;
* CLI changes;
* runtime candidate classification;
* provider calls;
* SEC or EDGAR calls;
* yfinance calls;
* live market data;
* source refresh;
* broker integration;
* Telegram or email delivery;
* production reports;
* portfolio writes;
* watchlist writes;
* scheduler behavior;
* UI behavior;
* upstream dry-run artifact mutation;
* readable operator report mutation;
* Analysis Review changes;
* Recommendation Review changes;
* Portfolio Review changes;
* Decision Engine behavior changes;
* Delivery / Reporting behavior changes;
* BUY / SELL / HOLD action semantics;
* allocation advice;
* target prices;
* target weights;
* position sizing;
* order generation;
* execution advice;
* ranking;
* scoring;
* urgency;
* conviction;
* tradeability authority.

## Future Implementation Requirements

A future ME-CANDIDATE02 implementation sprint should:

* implement a deterministic builder for `market-engine-candidate-classification-v1`;
* consume only existing local operator report summaries and/or approved local dry-run artifact roots;
* write Markdown and JSON companion outputs under an explicit local output root;
* refuse overwrite;
* preserve all missing, stale, blocked, malformed, unsupported, numeric-zero, and provenance markers;
* emit only the allowed candidate buckets defined by this contract;
* include advisory-language guardrail coverage;
* include tests for complete artifacts, missing files, malformed JSON, unsupported versions, deterministic ordering, unsafe run ids, overwrite refusal, bucket assignment, and non-actionable wording;
* update documentation, audit, backlog, and roadmap.

## Acceptance Criteria

ME-CANDIDATE01 is complete when:

* the candidate classification contract is defined;
* approved input artifact families are listed;
* output format version is named;
* allowed candidate buckets are defined;
* forbidden action, ranking, scoring, urgency, conviction, tradeability, target-price, allocation, and execution semantics are explicit;
* required Markdown sections are defined;
* required JSON companion metadata is defined;
* missing, stale, blocked, malformed, unsupported, numeric-zero, and provenance preservation rules are defined;
* advisory-language guardrails are defined;
* fail-closed behavior is defined;
* future implementation requirements are documented;
* explicit non-scope preserves all provider, broker, delivery, portfolio, watchlist, runtime, reporting, and action-authority boundaries.
