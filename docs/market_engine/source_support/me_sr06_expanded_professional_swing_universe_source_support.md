# ME-SR06 — Expanded Professional Swing Universe source-support classification

Status: COMPLETED BY ME-SR06

## Purpose

ME-SR06 adds a controlled source-support layer for the expanded/proposed Professional Swing Universe produced by ME-UNI09.

The sprint keeps the scale-first path intact:

```text
ME-SR06 → ME-RUN23
```

ME-SR06 does not refine candidate classification, operator reporting, Decision Engine behavior, portfolio behavior, delivery behavior, or trade semantics.

## Runtime contract

ME-SR06 introduces:

```text
market-engine-expanded-professional-swing-source-support-v1
```

Runtime entry point:

```text
market_engine.source_support.classify_expanded_professional_swing_universe_source_support
```

The classifier consumes a validated `market-engine-professional-swing-universe-expansion-v1` result from ME-UNI09.

## Implementation approach

ME-SR06 deliberately reuses the existing ME-SR05 source-support classifier.

The ME-SR06 layer:

1. validates that the input is a ME-UNI09 expansion result;
2. validates the expanded `final_universe_entries` structure;
3. rejects duplicate ticker/market keys before classification;
4. builds a temporary in-memory filesystem CSV shaped like the editable Professional Swing Universe contract;
5. runs the existing ME-SR05 Professional Swing source-support classifier against that temporary CSV;
6. adds expanded-universe provenance on top of each classified ticker.

This avoids duplicating SEC CompanyFacts artifact discovery, provider-error handling, required-field mapping, malformed-artifact handling, manual-review handling, unsupported-source handling, and summary-state classification.

## Output

Each expanded-universe source-support entry preserves:

* ticker;
* name;
* market;
* asset type;
* whether the row originated from the existing universe or from an ME-UNI09 expansion candidate;
* universe/source-row provenance;
* candidate source id/reference when available;
* source-support status;
* source-support reason;
* the underlying ME-SR05 source-support entry, including source artifact references, provider error references, missing-field evidence, numeric-zero evidence, and universe row reference.

Summary counts include:

* `total_expanded_universe_entries`;
* `supported_cached`;
* `missing_snapshot`;
* `unsupported_sec_companyfacts`;
* `missing_required_source_field`;
* `malformed_or_unreadable_source_artifact`;
* `ambiguous_identity`;
* `manual_review_only`;
* `excluded`;
* `blocked_unsupported_or_manual_review_total`.

## Fail-closed behavior

ME-SR06 fails closed when:

* the input is not `market-engine-professional-swing-universe-expansion-v1`;
* `final_universe_entries` is malformed;
* final rows are not mapping objects;
* ticker/market keys are duplicated;
* paths contain parent traversal;
* the referenced input Professional Swing Universe is invalid;
* ME-SR05 rejects the temporary expanded universe CSV;
* source-row provenance cannot be mapped back to either the existing universe or an included expansion candidate.

Unreadable or malformed cached source artifacts remain classified by the reused ME-SR05 source-support rules.

## Boundaries

ME-SR06 is source-support-only.

It does not add live provider calls, SEC/EDGAR fetches, yfinance, broker calls, Telegram, portfolio writes, watchlist writes, Decision Engine changes, BUY/SELL/HOLD advice, target prices, ranking, scoring, urgency, conviction, tradeability, allocation, order, or execution behavior.

## Tests

Implemented tests:

```text
tests/market_engine/source_support/test_expanded_professional_swing_source_support.py
```

Coverage includes:

* expanded candidate with valid cached source → `supported_cached`;
* expanded candidate without snapshot → `missing_snapshot`;
* manual-review-only existing universe row → `manual_review_only`;
* unsupported SEC CompanyFacts identity/provider state → `unsupported_sec_companyfacts`;
* malformed source artifact → `malformed_or_unreadable_source_artifact`;
* existing universe entries remain classifiable;
* newly included ME-UNI09 candidate entries remain classifiable;
* deterministic ordering;
* summary counts;
* unsupported input contract fail-closed behavior;
* duplicate ticker/market fail-closed behavior;
* provider/network import guardrails;
* normal output action-authority wording guardrails.

## Local validation

This GitHub sprint was prepared without access to Steven's local `.venv` and local artifact tree. Steven must run the local validation commands before merge.
