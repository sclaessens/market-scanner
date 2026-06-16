# ME-AR04 — Analysis Review Setup Detection implementation audit

## Status

COMPLETED BY ME-AR04

## Sprint

ME-AR04 — Implement Analysis Review consumption of Setup Detection

## Branch

me-ar04-implement-analysis-review-setup-detection-consumption

## Sprint Goal

Implement Analysis Review consumption of Setup Detection output according to the ME-AR03 contract.

The implementation must preserve the existing `sec-companyfacts-analysis-review-v1` output contract and must not introduce recommendation authority, portfolio behavior, delivery behavior, Telegram behavior, reporting behavior, provider behavior, production data writes, or Decision Engine behavior.

## Files Added

Documentation:

* `docs/market_engine/analysis_review/me_ar04_analysis_review_setup_detection_implementation.md`
* `docs/market_engine/audits/me_ar04_analysis_review_setup_detection_implementation_audit.md`

## Files Changed

Runtime:

* `src/market_engine/analysis_review/sec_companyfacts_analysis_review.py`

Tests:

* `tests/market_engine/analysis_review/test_sec_companyfacts_analysis_review.py`

Documentation:

* `docs/market_engine/backlog/market_engine_backlog.md`
* `docs/market_engine/roadmap/market_engine_roadmap.md`

## Contract Implemented

Implemented governing contract:

* `docs/market_engine/analysis_review/me_ar03_setup_detection_input_contract.md`

## Input Contracts

Approved input contracts:

* `sec-companyfacts-fundamental-observations-v1`
* `sec-companyfacts-derived-cash-generation-observations-v1`
* `sec-companyfacts-setup-detection-v1`

Unsupported Setup Detection input contracts fail closed with `ValueError`.

## Output Contract Preserved

Preserved output contract:

* `sec-companyfacts-analysis-review-v1`

ME-AR04 does not introduce `sec-companyfacts-analysis-review-v2`.

## Implementation Summary

ME-AR04 extends `build_sec_companyfacts_analysis_review(...)` with an optional Setup Detection input.

When Setup Detection input is omitted, existing ME-AR02 behavior is preserved.

When Setup Detection input is provided, the builder:

* validates Setup Detection format version;
* validates alignment across Fundamental Observations, Derived Observations, and Setup Detection;
* appends Setup Detection-aware Analysis Review items;
* preserves setup categories and setup states;
* preserves setup evidence;
* preserves setup limitations;
* preserves missing observations;
* preserves source observation references;
* preserves derived observation references;
* preserves top-level Setup Detection format version, run ID, and non-actionable boundary metadata.

## Categories Implemented

Additional Analysis Review categories:

* `SETUP_DETECTION_REVIEW`
* `SETUP_EVIDENCE_COMPLETENESS_REVIEW`
* `SETUP_LIMITATION_REVIEW`
* `SETUP_HUMAN_REVIEW_REQUIREMENT`

Existing Analysis Review categories remain valid.

## States Implemented

Additional Analysis Review states:

* `SETUP_DETECTED`
* `SETUP_PARTIALLY_DETECTED`
* `SETUP_NOT_DETECTED`
* `SETUP_CONFLICTED`
* `SETUP_BLOCKED_BY_MISSING_DATA`
* `SETUP_NOT_ASSESSED`
* `SETUP_REQUIRES_HUMAN_REVIEW`

These states are descriptive Analysis Review states only.

## Setup Detection Reference Behavior

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

## Missing-Data Behavior

Missing setup observations remain explicit.

Missing setup evidence is not converted to zero and is not converted into negative evidence or recommendation evidence.

## Numeric-Zero Behavior

Numeric zero remains present when it is preserved by upstream Setup Detection evidence.

Analysis Review does not treat numeric zero as missing.

## Persistence Behavior

The existing Analysis Review persistence helper remains in place.

Persistence now serializes Setup Detection metadata and Setup Detection-aware review item references when present.

Tests use temporary directories only.

## Tests Added

ME-AR04 tests cover:

* complete Setup Detection input creates Setup Detection-aware Analysis Review;
* partial setup input creates partial Setup Detection-aware review;
* missing setup evidence creates blocked Setup Detection-aware review;
* conflicted setup input creates conflicted review and human-review routing;
* not-assessed setup input remains not assessed;
* unsupported Setup Detection input contract fails closed;
* Setup Detection input alignment mismatch fails closed;
* numeric zero remains present and is not treated as missing;
* Setup Detection references are preserved;
* Fundamental Observation and Derived Observation references remain preserved;
* existing Analysis Review behavior remains valid;
* persistence preserves Setup Detection references;
* forbidden action-authority terms are not emitted;
* no legacy `scripts` or old `market_scanner` imports are introduced.

## Validation Commands And Results

Targeted Analysis Review tests:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/analysis_review -q
```

Result:

```text
18 passed
```

Full Market Engine tests:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine -q
```

Result:

```text
156 passed
```

Repository validation:

```bash
git diff --check
git status --short
git diff --stat
git diff --name-only
grep -R "from scripts\|import scripts\|from market_scanner\|import market_scanner" src/market_engine tests/market_engine || true
```

Results:

* `git diff --check` passed.
* `git status --short` showed only the planned Analysis Review runtime, Analysis Review tests, implementation documentation, audit, backlog, and roadmap changes.
* `git diff --stat` and `git diff --name-only` showed only the expected ME-AR04 areas.
* The legacy dependency grep found only pre-existing negative assertion strings in Recommendation Review tests and no active imports from legacy `scripts` or old `market_scanner`.
* A focused forbidden-authority scan of Analysis Review runtime/tests found no runtime emission of action-authority terms; matches were limited to synthetic SEC tag text and negative assertion test data.

## Boundaries Preserved

Confirmed:

* no live provider calls were introduced;
* no SEC calls were introduced;
* no EDGAR calls were introduced;
* no yfinance calls were introduced;
* no production data writes were introduced;
* no portfolio mutation was introduced;
* no watchlist mutation was introduced;
* no Telegram behavior was introduced;
* no reporting output was introduced;
* no Decision Engine behavior was introduced;
* no Recommendation Review behavior was changed;
* no Portfolio Review behavior was changed;
* no Delivery / Reporting behavior was changed;
* no provider behavior was changed;
* no source refresh behavior was changed;
* no source context behavior was changed;
* no legacy `scripts` imports were introduced;
* no old `market_scanner` imports were introduced.

## Next Recommended Sprint

Recommended next sprint:

```text
ME-RR03 — Extend Recommendation Review contract for Setup Detection-aware Analysis Review
```
