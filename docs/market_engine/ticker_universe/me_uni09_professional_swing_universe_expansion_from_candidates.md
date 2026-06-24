# ME-UNI09 - Professional Swing Universe Expansion From Candidates

## Sprint Status

Completed by ME-UNI09.

## Purpose

ME-UNI09 implements a controlled, deterministic universe-maintenance layer that can consume non-actionable candidate-classification output and propose additions to the editable Professional Swing Universe.

This sprint is limited to universe maintenance. It does not add provider access, source refresh, live data, report delivery, portfolio mutation, watchlist mutation, Decision Engine behavior, allocation authority, order authority, or action-oriented investment output.

## Implemented Contract

ME-UNI09 introduces:

```text
market-engine-professional-swing-universe-expansion-v1
```

The implementation consumes:

* an existing editable Professional Swing Universe CSV using `market-engine-editable-professional-swing-universe-v1`;
* a candidate-classification summary using `market-engine-candidate-classification-v1`.

The implementation returns a structured in-memory expansion result. It does not write files, mutate the canonical universe CSV, or update production data.

## Runtime Surface

Implemented module:

```text
src/market_engine/ticker_universe/professional_swing_expansion.py
```

Public API:

```text
build_professional_swing_universe_expansion(...)
ProfessionalSwingUniverseExpansionResult
ProfessionalSwingUniverseExpansionDecision
ProfessionalSwingUniverseExpansionError
```

The builder:

* loads the existing Professional Swing Universe with `include_inactive=True`;
* reads the candidate-classification summary JSON;
* validates candidate-classification format identity;
* accepts only known candidate buckets;
* includes only `ready_for_manual_candidate_review` candidates when the proposed universe entry is valid;
* preserves existing universe entries;
* avoids duplicate ticker/market entries;
* preserves deterministic ordering by operator priority, ticker, and market;
* records inclusion, exclusion, duplicate, blocked, and manual-review reasons.

## Inclusion Rules

A candidate may be included only when:

* the candidate-classification bucket is `ready_for_manual_candidate_review`;
* the candidate provides a `proposed_universe_entry`;
* the proposed entry satisfies the editable Professional Swing Universe schema;
* ticker identity matches between the classification record and proposed entry;
* the ticker/market key is not already present;
* the ticker/market key is not duplicated in the candidate input;
* the candidate is not marked manual-review-only, blocked, rejected, unsupported, ambiguous, malformed, or missing required source evidence.

Optional `operator_approved_tickers` may further restrict inclusion. When supplied, candidates not present in that explicit approval set are excluded with `not_operator_approved`.

## Fail-Closed Behavior

The builder fails closed on:

* missing input paths;
* path traversal in input paths;
* non-file input paths;
* malformed candidate-classification JSON;
* unsupported candidate-classification format versions;
* missing `per_ticker_classifications`;
* non-object candidate records;
* missing required candidate fields;
* unknown candidate buckets;
* invalid ticker formats;
* conflicting ticker identity;
* invalid proposed Professional Swing Universe rows.

The builder excludes, rather than includes, candidates that are:

* ineligible by candidate bucket;
* already present;
* duplicated within the candidate input;
* manual-review-only;
* ambiguous;
* unsupported;
* missing source coverage;
* non-equity;
* malformed or unsupported by safety flags.

## Output Shape

The result includes:

* format version;
* input universe path;
* input candidate-classification path;
* existing universe count;
* candidate count;
* included count;
* excluded count;
* duplicate count;
* blocked/manual-review count;
* resulting universe count;
* summary counts;
* included candidate decisions;
* excluded candidate decisions;
* duplicate candidate decisions;
* blocked/manual-review decisions;
* proposed final universe entries;
* non-actionable boundary marker.

## Non-Goals Preserved

ME-UNI09 does not:

* fetch providers;
* call SEC, EDGAR, yfinance, broker, Telegram, email, or external services;
* refresh source data;
* mutate portfolio or watchlist state;
* mutate the canonical Professional Swing Universe file;
* add scheduler, UI, reporting delivery, or production writes;
* change Candidate Classification, Source Support, Run, Output, Portfolio Review, Decision Engine, or Delivery behavior;
* introduce action-oriented instructions, ranking, scoring, target prices, urgency, conviction, allocation, order, or execution semantics.

## Tests

Added focused tests:

```text
tests/market_engine/ticker_universe/test_professional_swing_universe_expansion.py
```

Coverage includes eligible inclusion, existing-entry preservation, duplicate handling, already-present handling, manual-review exclusion, ambiguous exclusion, unsupported/source-coverage exclusion, malformed input fail-closed behavior, unknown bucket fail-closed behavior, deterministic ordering, summary counts, optional operator approval, dependency boundary checks, and non-actionable summary wording.

## Next Sprint

The next active direction is:

```text
ME-SR06 - Classify source support for expanded Professional Swing Universe
```

ME-RUN23 should follow ME-SR06 to execute the expanded supported-universe cached-source run and produce readable/candidate outputs.
