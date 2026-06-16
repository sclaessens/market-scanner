# ME-AR04 — Analysis Review Setup Detection implementation

## Status

COMPLETED BY ME-AR04

## Sprint

ME-AR04 — Implement Analysis Review consumption of Setup Detection

## Job family

ME-AR — Analysis Review jobs

## Purpose

ME-AR04 implements Analysis Review consumption of `sec-companyfacts-setup-detection-v1` while preserving the existing `sec-companyfacts-analysis-review-v1` output contract.

The implementation keeps Analysis Review descriptive, provenance-preserving, missing-data-aware, numeric-zero-safe, non-recommendation, and non-actionable.

## Governing Contract

Implemented contract:

* `docs/market_engine/analysis_review/me_ar03_setup_detection_input_contract.md`

Approved Setup Detection input contract:

* `sec-companyfacts-setup-detection-v1`

Preserved Analysis Review output contract:

* `sec-companyfacts-analysis-review-v1`

## Implementation Strategy

ME-AR04 extends the existing `build_sec_companyfacts_analysis_review(...)` builder with an optional Setup Detection input.

When no Setup Detection input is provided, the builder preserves existing ME-AR02 behavior.

When Setup Detection input is provided, the builder:

* validates Setup Detection contract version;
* validates input alignment across Fundamental Observations, Derived Observations, and Setup Detection;
* appends Setup Detection-aware Analysis Review items;
* preserves Setup Detection references, categories, states, evidence, limitations, missing observations, source references, and derived references;
* preserves top-level Setup Detection format version, run ID, and non-actionable boundary metadata.

## Input Alignment

ME-AR04 validates alignment where available on:

* ticker;
* CIK;
* provider name;
* source context format version;
* source context state;
* source refresh snapshot ID;
* source refresh fetched timestamp;
* source refresh payload format version;
* fundamental observation format version;
* derived observation format version;
* setup detection format version.

Mismatched or unsupported Setup Detection input fails closed with `ValueError`.

## Setup-Aware Review Categories

ME-AR04 adds these Analysis Review categories:

* `SETUP_DETECTION_REVIEW`;
* `SETUP_EVIDENCE_COMPLETENESS_REVIEW`;
* `SETUP_LIMITATION_REVIEW`;
* `SETUP_HUMAN_REVIEW_REQUIREMENT`.

Existing Analysis Review categories remain valid and unchanged.

## Setup-Aware Review States

ME-AR04 adds these Analysis Review states:

* `SETUP_DETECTED`;
* `SETUP_PARTIALLY_DETECTED`;
* `SETUP_NOT_DETECTED`;
* `SETUP_CONFLICTED`;
* `SETUP_BLOCKED_BY_MISSING_DATA`;
* `SETUP_NOT_ASSESSED`;
* `SETUP_REQUIRES_HUMAN_REVIEW`.

These states are review states only. They do not imply action authority.

## Mapping Behavior

Setup Detection states map deterministically into Analysis Review states:

| Setup Detection state           | Analysis Review state           |
| ------------------------------- | ------------------------------- |
| `setup_detected`                | `SETUP_DETECTED`                |
| `setup_partially_detected`      | `SETUP_PARTIALLY_DETECTED`      |
| `setup_not_detected`            | `SETUP_NOT_DETECTED`            |
| `setup_conflicted`              | `SETUP_CONFLICTED`              |
| `setup_blocked_by_missing_data` | `SETUP_BLOCKED_BY_MISSING_DATA` |
| `setup_not_assessed`            | `SETUP_NOT_ASSESSED`            |

Setup Detection categories map deterministically into Analysis Review categories:

| Setup Detection category         | Analysis Review category             |
| -------------------------------- | ------------------------------------ |
| `cash_generation_setup`          | `SETUP_DETECTION_REVIEW`             |
| `fundamental_availability_setup` | `SETUP_EVIDENCE_COMPLETENESS_REVIEW` |
| `profitability_evidence_setup`   | `SETUP_DETECTION_REVIEW`             |
| `revenue_evidence_setup`         | `SETUP_DETECTION_REVIEW`             |
| `balance_sheet_evidence_setup`   | `SETUP_DETECTION_REVIEW`             |
| `data_limitation_setup`          | `SETUP_LIMITATION_REVIEW`            |
| `not_assessed_setup`             | `SETUP_HUMAN_REVIEW_REQUIREMENT`     |

Partial, conflicted, blocked, and not-assessed setup evidence also routes to a Setup Detection human-review item.

## Provenance Behavior

Setup Detection-aware review items preserve:

* setup category;
* setup state;
* setup message;
* setup evidence;
* setup limitations;
* missing observations;
* source observation references;
* derived observation references;
* non-actionable boundary marker.

Fundamental Observation and Derived Observation review behavior remains unchanged.

## Missing-Data Behavior

Missing setup observations remain explicit.

Analysis Review does not:

* convert missing values to zero;
* omit missing-observation lists;
* treat missing setup evidence as negative evidence;
* treat missing setup evidence as recommendation evidence;
* proceed as if incomplete setup evidence were complete.

## Numeric-Zero Behavior

Numeric zero remains a valid observed or derived value.

If Setup Detection preserves numeric zero in setup evidence, Analysis Review preserves the value or reference.

Zero evidence is not treated as missing and is not converted into action text.

## Persistence Behavior

The existing Analysis Review persistence path remains:

```text
data/market_engine/analysis_reviews/<analysis_review_run_id>/<ticker>/analysis_review.json
```

Persistence now preserves Setup Detection references when the Analysis Review object includes Setup Detection-aware review items.

Tests use temporary directories only.

## Non-Scope

ME-AR04 does not introduce:

* Recommendation Review behavior;
* Portfolio Review behavior;
* Decision Engine behavior;
* Delivery / Reporting behavior;
* Telegram behavior;
* provider behavior;
* source refresh behavior;
* source context behavior;
* production data writes;
* live provider calls;
* action-authority semantics.

## Next Sprint

Recommended next sprint:

```text
ME-RR03 — Extend Recommendation Review contract for Setup Detection-aware Analysis Review
```
