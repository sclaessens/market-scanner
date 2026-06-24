# ME-CANDIDATE02 - Non-actionable candidate classification implementation

Sprint: ME-CANDIDATE02 - Implement non-actionable candidate classification from readable operator output

Status: Implemented

Job family: ME-CANDIDATE - Candidate Classification

## Purpose

ME-CANDIDATE02 implements the `market-engine-candidate-classification-v1` contract defined by ME-CANDIDATE01.

The implementation classifies existing readable operator output into fixed review-only candidate buckets for human triage. It does not create action authority, trade guidance, output delivery, portfolio mutation, watchlist mutation, or provider access.

## Implemented Runtime

Implemented package:

```text
src/market_engine/candidate_classification/
```

Public API:

```text
classify_non_actionable_candidate_from_readable_output(...)
build_candidate_classification_report(...)
CandidateClassificationInput
CandidateTickerClassification
CandidateEvidenceReference
CandidateSafetyFlags
CandidateClassificationReportResult
CandidateClassificationError
```

CLI module:

```text
market_engine.candidate_classification.non_actionable_candidate_classification_command
```

Example command:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m market_engine.candidate_classification.non_actionable_candidate_classification_command \
  --input-operator-report-root artifacts/market_engine/me-out02-readable-operator-report-me-run20-supported-universe-20260623T120000Z \
  --output-root artifacts/market_engine \
  --candidate-classification-run-id me-candidate02-non-actionable-classification-me-run20-supported-universe-20260623T120000Z \
  --generated-at 2026-06-24T13:00:00Z
```

## Output Contract

Implemented format version:

```text
market-engine-candidate-classification-v1
```

Generated filenames:

```text
candidate_classification_report.md
candidate_classification_summary.json
```

Local output path pattern:

```text
artifacts/market_engine/<candidate_classification_run_id>/
```

Generated artifacts are local and non-production. They are not committed by default.

## Classification Buckets

The implementation uses the exact ME-CANDIDATE01 bucket set:

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

These buckets are not ordered as ranking, preference, attractiveness, urgency, conviction, tradeability, or execution priority.

## Fail-Closed Behavior

The implementation fails closed for:

* missing input operator report root;
* input root that is not a directory;
* missing `operator_report_summary.json`;
* malformed summary JSON;
* unsafe candidate classification run id;
* path traversal;
* output directory overwrite.

Per-ticker classification fails closed into unclassified or review-required buckets when:

* input contract version is unsupported;
* readable output is missing;
* an upstream run did not complete;
* stale-data notes are present;
* missing-data notes are present;
* blocked notes are present;
* required output-family evidence is incomplete;
* forbidden action-oriented wording is detected in the input text.

## Preserved Evidence

The classifier preserves:

* ticker symbol;
* output-family references;
* evidence references;
* blocking reasons;
* safety flags;
* missing-data markers;
* stale-data markers;
* blocked-state markers;
* provenance presence;
* numeric-zero evidence presence;
* source input roots where supplied.

## Non-Scope

ME-CANDIDATE02 does not add provider calls, SEC or EDGAR calls, yfinance calls, live market data, source refresh, broker integration, Telegram or email delivery, production reports, portfolio writes, watchlist writes, scheduler behavior, UI behavior, readable operator report mutation, Analysis Review changes, Recommendation Review changes, Portfolio Review changes, Decision Engine behavior changes, Delivery / Reporting behavior changes, allocation advice, target prices, target weights, position sizing, order generation, execution advice, ranking, scoring, urgency, conviction, or tradeability authority.

## Tests

Added focused tests under:

```text
tests/market_engine/candidate_classification/test_non_actionable_candidate_classification.py
```

Coverage includes:

* complete readable operator context;
* missing readable output;
* incomplete dry-run;
* stale-data notes;
* forbidden action-oriented wording;
* format version;
* evidence references;
* generated output safety;
* unsupported operator summary version;
* missing operator summary;
* unsafe run id;
* overwrite refusal;
* CLI generation.

## Validation

Validation commands:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/candidate_classification -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q -k "candidate or readable or operator or dry_run"
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q
git diff --check
```

## Next Planning Note

No additional downstream sprint is inserted by ME-CANDIDATE02. Future candidate-classification output review should be defined in backlog and roadmap only when a concrete operator-review or governance need is identified.
