# ME-CANDIDATE01 Backlog Entry - Non-actionable candidate classification contract

Sprint: ME-CANDIDATE01 - Define non-actionable candidate classification contract

Owner roles: Product Owner / Operator / Technical Architect / Development Lead / QA Lead / Governance Auditor

Job family: ME-CANDIDATE - Candidate Classification

Status: COMPLETED BY ME-CANDIDATE01

## Goal

Define a non-actionable candidate classification contract from readable operator output and dry-run artifacts without introducing action authority, ranking, scoring, allocation, urgency, conviction, tradeability, target prices, watchlist mutation, broker behavior, delivery behavior, or execution behavior.

## Rationale

ME-OUT02 implemented deterministic readable operator reporting from local dry-run artifacts.

A separate Candidate Classification contract sprint is required before any implementation can group ticker artifacts into candidate-review buckets. The contract must make it explicit that `candidate` means candidate for human review only, not candidate trade, candidate position, candidate order, candidate allocation, candidate watchlist entry, or candidate delivery item.

## Scope

ME-CANDIDATE01 defines:

* candidate classification contract family;
* approved input artifact families;
* recommended future local output path category;
* required Markdown report sections;
* required JSON companion summary metadata;
* classification metadata requirements;
* allowed candidate buckets;
* forbidden buckets, fields, labels, and meanings;
* classification evidence requirements;
* missing-data, stale-data, blocked-state, malformed-artifact, unsupported-input, numeric-zero, and provenance preservation requirements;
* human-review checklist requirements;
* safe next-step candidate semantics;
* advisory-language guardrails;
* fail-closed behavior;
* deterministic output requirements;
* ME-CANDIDATE02 future implementation requirements.

## Outcome

ME-CANDIDATE01 defined:

```text
market-engine-candidate-classification-v1
```

Implemented documentation:

```text
docs/market_engine/candidate_classification/me_candidate01_non_actionable_candidate_classification_contract.md
docs/market_engine/audits/me_candidate01_non_actionable_candidate_classification_contract_audit.md
docs/market_engine/backlog/me_candidate01_non_actionable_candidate_classification_contract_backlog_entry.md
docs/market_engine/roadmap/me_candidate01_non_actionable_candidate_classification_contract_roadmap_entry.md
```

## Allowed Future Candidate Buckets

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

These buckets are review-only. They must not be sorted or interpreted as investment attractiveness, conviction, urgency, tradeability, ranking, score, allocation priority, or execution priority.

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

## Acceptance Criteria

Completed:

* candidate classification contract defined;
* approved input artifact families listed;
* output format version named;
* allowed candidate buckets defined;
* forbidden action, ranking, scoring, urgency, conviction, tradeability, target-price, allocation, and execution semantics explicit;
* required Markdown sections defined;
* required JSON companion metadata defined;
* missing, stale, blocked, malformed, unsupported, numeric-zero, and provenance preservation rules defined;
* advisory-language guardrails defined;
* fail-closed behavior defined;
* future implementation requirements documented;
* explicit non-scope preserves all provider, broker, delivery, portfolio, watchlist, runtime, reporting, and action-authority boundaries.

## Next Sprint

```text
ME-CANDIDATE02 - Implement non-actionable candidate classification from readable operator output
```
