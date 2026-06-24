# ME-OUT01 - Readable operator report contract from dry-run artifacts

Sprint: ME-OUT01 - Define readable operator report contract from dry-run artifacts

Status: Defined

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-OUT - Output / Operator Reporting

## Purpose

ME-OUT01 defines the first formal readable operator report contract for Market Engine dry-run artifacts.

The contract turns completed local dry-run artifacts into an operator-facing report shape that is readable, deterministic, source-grounded, provenance-preserving, and explicitly non-actionable.

ME-OUT01 is documentation-only. It does not implement Python code, tests, CLI behavior, data refresh, report generation, delivery, scheduler behavior, UI behavior, broker integration, portfolio mutation, watchlist mutation, or Decision Engine behavior.

## Upstream Basis

ME-OUT01 builds on the completed ME-RUN20 through ME-RUN22 chain:

* ME-RUN20 executed a supported-universe cached-source dry-run and produced local per-ticker artifacts.
* ME-RUN21 inspected those artifacts and confirmed they were complete and structurally usable for interpretation.
* ME-RUN22 implemented `market-engine-interpretation-report-v1` as the first deterministic local human-readable interpretation report.

ME-OUT01 does not replace ME-RUN22. It formalizes the next operator-reporting contract so future output work has a stable boundary before further implementation.

## Contract Name

Approved contract family:

```text
market-engine-readable-operator-report-v1
```

Recommended future Markdown filename:

```text
operator_report.md
```

Recommended future machine-readable companion filename:

```text
operator_report_summary.json
```

Recommended future local output path category:

```text
artifacts/market_engine/<operator_report_run_id>/
```

Generated report artifacts must remain local and non-production unless a later sprint explicitly authorizes a different persistence, delivery, or publication boundary.

## Approved Inputs

The readable operator report may consume only existing local dry-run artifacts and existing local report summaries produced by approved Market Engine runs.

Approved input artifact families:

```text
market-engine-local-dry-run-artifact-v1
market-engine-local-dry-run-artifact-manifest-v1
market-engine-end-to-end-dry-run-v1
market-engine-interpretation-report-v1
```

Approved input file categories:

```text
artifacts/market_engine/<dry_run_artifact_root>/<TICKER>/dry_run.json
artifacts/market_engine/<dry_run_artifact_root>/<TICKER>/manifest.json
artifacts/market_engine/<interpretation_report_run_id>/market_engine_interpretation_report.md
artifacts/market_engine/<interpretation_report_run_id>/market_engine_interpretation_report_summary.json
```

The operator report must not call providers, refresh source snapshots, repair artifacts, infer missing upstream values, query broker APIs, read live market data, read portfolio state outside an already-approved local dry-run artifact, or mutate any upstream artifact.

## Report Audience

The primary audience is the human operator reviewing whether the Market Engine run is understandable and operationally usable.

The report is allowed to answer these questions:

* What artifact root was inspected?
* Which tickers were included?
* Which tickers were skipped or blocked, and why?
* Which Market Engine stages completed?
* Which output families are present?
* Which missing-data, stale-data, blocked-state, and provenance markers are visible?
* Which evidence is available for human review?
* Which next operational sprint or review step is safe to consider?

The report is not allowed to answer whether the operator should buy, sell, hold, allocate capital, enter a trade, exit a trade, size a position, prioritize an execution order, or route an order to a broker.

## Required Top-Level Report Sections

A compliant Markdown operator report must contain these sections in deterministic order:

1. Report metadata
2. Source artifact boundary
3. Non-actionable boundary
4. Universe coverage
5. Artifact integrity summary
6. Stage completion summary
7. Per-ticker operator summaries
8. Missing-data and stale-data notes
9. Blocked and skipped ticker notes
10. Provenance summary
11. Human-review checklist
12. Safe next-step candidate
13. Appendix: machine-readable summary reference

The exact wording may evolve in implementation, but these sections must remain present unless a future contract version explicitly replaces them.

## Required Report Metadata

The report metadata section must include:

* report format version;
* report run id;
* generated-at timestamp when supplied by the caller;
* input artifact root;
* optional interpretation report root when supplied by the caller;
* included ticker count;
* skipped ticker count;
* blocked ticker count when available from dry-run artifacts;
* completed ticker count when available from dry-run artifacts;
* local-only / non-production marker.

The report must preserve the original upstream artifact identifiers and must not create hidden run identifiers.

## Universe Coverage Requirements

The universe coverage section must show:

* requested tickers when available;
* included tickers;
* skipped tickers;
* blocked tickers;
* artifact-present tickers;
* artifact-missing tickers;
* malformed-artifact tickers;
* manual-review-only or excluded tickers when that information is already present in approved upstream artifacts.

Ticker ordering must be deterministic. Recommended ordering is alphabetical by ticker symbol unless the upstream artifact already defines a canonical deterministic order.

The report may group tickers by status, but it must not rank them by investment attractiveness, conviction, urgency, expected return, or tradeability.

## Artifact Integrity Requirements

The artifact integrity section must make structural validity visible:

* `dry_run.json` present or missing;
* `manifest.json` present or missing;
* JSON parse status;
* artifact format version;
* dry-run format version;
* manifest format version;
* input mode;
* run state;
* completed stages;
* non-completed stages;
* output families present.

Malformed, missing, or unsupported artifacts must be reported explicitly and must not be silently excluded.

## Stage Completion Requirements

The stage completion section must summarize known stage states without changing semantics.

Allowed stage-summary language:

* completed;
* completed with limitations;
* blocked;
* skipped;
* missing;
* stale;
* unsupported;
* malformed;
* not present in artifact;
* unknown because the artifact could not be read.

The report must not translate these states into investment actions, ticker preference, capital guidance, or execution guidance.

## Per-Ticker Operator Summary Requirements

Each per-ticker summary must preserve:

* ticker symbol;
* artifact file references;
* run state;
* input mode;
* source snapshot reference when present;
* output families present;
* completed and non-completed stages;
* missing-data notes;
* stale-data notes;
* blocked reasons;
* provenance references;
* delivery/reporting summary state when present;
* Decision Engine handoff state when present, without creating new Decision Engine conclusions.

Per-ticker summaries may use concise operator wording. They must remain factual summaries of the artifact contents.

## Missing, Stale, and Blocked Data Requirements

The report must preserve upstream missing-data, stale-data, and blocked-state markers.

The report must not:

* convert missing numeric values to zero;
* hide missing data because the rest of the ticker completed;
* infer a missing value from another source;
* repair stale metadata;
* suppress a blocked stage because downstream stages are absent;
* convert blocked state into negative investment judgement.

Numeric zero must remain distinguishable from missing data.

## Provenance Requirements

The report must preserve references to upstream evidence when available.

Allowed provenance references include:

* source snapshot identifiers;
* local artifact paths;
* SEC CompanyFacts source references already present in upstream artifacts;
* stage output family names;
* upstream observation references;
* manifest metadata;
* generated report summary references.

The report must not fabricate source references or cite external sources that were not part of the inspected local artifacts.

## Human-Review Checklist

The report must include a human-review checklist that helps the operator decide whether more Market Engine work is needed.

Allowed checklist items:

* artifact root readable;
* all expected ticker directories present;
* all expected `dry_run.json` files present;
* all expected `manifest.json` files present;
* all JSON files parse successfully;
* all required output families present;
* missing-data notes reviewed;
* stale-data notes reviewed;
* blocked tickers reviewed;
* provenance references reviewed;
* non-actionable boundary preserved.

The checklist must not include trade-entry, trade-exit, allocation, position-sizing, broker, order, alert, Telegram, or external delivery steps.

## Safe Next-Step Candidate

The report may include exactly one safe next-step candidate.

Allowed next-step semantics:

* propose a follow-up sprint;
* propose human review of source coverage;
* propose implementation of this operator report contract;
* propose QA contract regression around dry-run artifacts;
* propose non-actionable candidate classification only if a separate approved sprint defines that job-family boundary.

Forbidden next-step semantics:

* trade now;
* buy below a price;
* sell above a price;
* hold a current position;
* allocate a percentage;
* rebalance a portfolio;
* send a broker order;
* send Telegram or email delivery;
* create a watchlist mutation;
* treat the report as production advice.

## Machine-Readable Companion Summary

A future implementation should emit a JSON companion summary with this minimum metadata:

```json
{
  "report_format_version": "market-engine-readable-operator-report-v1",
  "report_run_id": "<operator_report_run_id>",
  "generated_at": "<timestamp-or-null>",
  "input_artifact_root": "<path>",
  "interpretation_report_root": "<path-or-null>",
  "included_tickers": [],
  "skipped_tickers": [],
  "blocked_tickers": [],
  "completed_tickers": [],
  "output_families_present": [],
  "missing_data_notes_present": false,
  "stale_data_notes_present": false,
  "blocked_notes_present": false,
  "provenance_references_present": false,
  "non_actionable_boundary": true,
  "advisory_language_guardrail": {
    "forbidden_action_terms_checked": true,
    "operator_report_contains_trading_instruction": false
  }
}
```

The JSON companion summary is for validation, automation, and audit support. It must not become a broker-ready payload, alert payload, ranking payload, or production delivery payload.

## Advisory-Language Guardrails

Normal report text must avoid creating market-participation instructions.

Forbidden generated meanings include:

* buy, sell, or hold as instructions;
* entry price or exit price;
* target price;
* target weight;
* position size;
* allocation instruction;
* conviction label;
* urgency label;
* tradeability label;
* ticker ranking;
* score;
* broker-ready order;
* execution instruction;
* delivery instruction.

The report may mention upstream contract names that contain `Decision Engine handoff` or `Delivery / Reporting` when those names are already present in dry-run artifacts, but it must not create new authority beyond the source artifact.

## Fail-Closed Requirements

A future implementation must fail closed or skip explicitly when:

* the input artifact root is missing;
* the input artifact root is not a directory;
* a ticker directory lacks `dry_run.json`;
* a ticker directory lacks `manifest.json`;
* JSON parsing fails;
* an artifact declares an unsupported format version;
* the report run id is unsafe;
* the output path would escape the approved output root;
* the output directory already exists.

Failure output must be operator-readable and must preserve the reason without suppressing the affected ticker.

## Determinism Requirements

A future implementation must be deterministic for identical inputs.

Required deterministic behavior:

* stable ticker ordering;
* stable section ordering;
* stable skipped-reason ordering;
* stable output-family ordering;
* stable stage-state ordering;
* stable JSON key ordering when practical;
* caller-supplied generated-at timestamp for reproducible test output.

## Explicit Non-Scope

ME-OUT01 does not authorize:

* Python implementation;
* tests;
* CLI changes;
* report generation at runtime;
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
* Analysis Review changes;
* Recommendation Review changes;
* Portfolio Review changes;
* Decision Engine behavior changes;
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

A future ME-OUT02 implementation sprint should:

* implement a deterministic builder for `market-engine-readable-operator-report-v1`;
* consume only existing local artifact roots and optional existing interpretation report summaries;
* write Markdown and JSON companion outputs under an explicit local output root;
* refuse overwrite;
* preserve all missing, stale, blocked, and provenance markers;
* include advisory-language guardrail coverage;
* include tests for complete artifacts, missing files, malformed JSON, unsupported versions, deterministic ordering, unsafe report ids, overwrite refusal, and non-actionable wording;
* update documentation, audit, backlog, and roadmap.

## Acceptance Criteria

ME-OUT01 is complete when:

* the readable operator report contract is defined;
* approved input artifact families are listed;
* output format version is named;
* required Markdown sections are defined;
* required JSON companion metadata is defined;
* missing, stale, blocked, numeric-zero, and provenance preservation rules are defined;
* advisory-language guardrails are defined;
* fail-closed behavior is defined;
* future implementation requirements are documented;
* explicit non-scope preserves all provider, broker, delivery, portfolio, watchlist, runtime, and action-authority boundaries.
