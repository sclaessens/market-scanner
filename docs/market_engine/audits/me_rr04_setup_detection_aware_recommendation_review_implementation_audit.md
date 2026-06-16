# ME-RR04 — Setup Detection-aware Recommendation Review implementation audit

## Status

COMPLETED BY ME-RR04

## Sprint

ME-RR04 — Implement Recommendation Review consumption of Setup Detection-aware Analysis Review output

## Branch

me-rr04-implement-recommendation-review-setup-detection-consumption

## Sprint Goal

Implement the ME-RR03 contract so Recommendation Review can consume Setup Detection-aware `sec-companyfacts-analysis-review-v1` output while preserving Recommendation Review as non-actionable and downstream of Analysis Review.

## Files Added

Documentation:

* `docs/market_engine/recommendation_review/me_rr04_setup_detection_aware_recommendation_review_implementation.md`
* `docs/market_engine/audits/me_rr04_setup_detection_aware_recommendation_review_implementation_audit.md`

## Files Changed

Runtime:

* `src/market_engine/recommendation_review/sec_companyfacts_recommendation_review.py`

Tests:

* `tests/market_engine/recommendation_review/test_sec_companyfacts_recommendation_review.py`

Documentation:

* `docs/market_engine/backlog/market_engine_backlog.md`
* `docs/market_engine/roadmap/market_engine_roadmap.md`

## Contract Implemented

Implemented governing contract:

* `docs/market_engine/recommendation_review/me_rr03_setup_detection_aware_analysis_review_contract.md`

## Input Contract

Preserved approved input contract:

* `sec-companyfacts-analysis-review-v1`

Unsupported Analysis Review input contracts continue to fail closed with `ValueError`.

## Output Contract Preserved

Preserved output contract:

* `sec-companyfacts-recommendation-review-v1`

ME-RR04 does not introduce `sec-companyfacts-recommendation-review-v2`.

## Implementation Summary

ME-RR04 extends Recommendation Review to detect Setup Detection-aware Analysis Review items when present.

The implementation preserves:

* setup-aware Analysis Review references;
* setup categories;
* setup states;
* setup evidence;
* setup limitations;
* missing setup observations;
* source observation references;
* derived observation references;
* setup detection format version;
* setup detection run ID;
* setup non-actionable boundary metadata.

Existing ME-RR02 behavior remains valid for Analysis Review inputs without setup-aware fields.

## Routing Behavior

Setup-aware Analysis Review states route only to non-actionable Recommendation Review states:

* detected setup evidence routes to `human_review_required`;
* partial setup evidence routes to `human_review_required` with uncertainty preserved;
* conflicted setup evidence routes to `human_review_required` with conflict preserved;
* blocked setup evidence routes to `blocked_by_missing_data`;
* not-assessed setup evidence routes to `insufficient_evidence`;
* not-detected setup evidence routes to `insufficient_evidence`.

Detected setup evidence is not converted into action authority.

Not-detected setup evidence is not converted into a negative recommendation.

## Missing-Data Behavior

Missing setup observations remain explicit.

Missing setup evidence is not converted into zero, negative evidence, ranking, scoring, or recommendation evidence.

## Numeric-Zero Behavior

Numeric zero remains present when preserved by setup-aware Analysis Review evidence.

Recommendation Review does not treat numeric zero as missing.

## Persistence Behavior

The existing Recommendation Review persistence helper remains in place.

Persistence now serializes setup-aware provenance fields when present.

Tests use temporary directories only.

No production data writes were introduced.

## Tests Added

ME-RR04 tests cover:

* detected setup-aware Analysis Review creates non-actionable human-review Recommendation Review;
* partial setup-aware Analysis Review preserves uncertainty and missing setup evidence;
* blocked setup-aware Analysis Review preserves missing data and blocks Recommendation Review;
* conflicted setup-aware Analysis Review preserves conflict and routes to human review;
* not-assessed and not-detected setup-aware Analysis Review remain insufficient evidence and do not become negative recommendations;
* missing or empty setup-aware fields are handled safely;
* legacy Analysis Review outputs remain valid;
* setup-aware persistence preserves provenance;
* forbidden action-authority terms are not emitted;
* no legacy `scripts` or old `market_scanner` imports are introduced.

## Validation Commands And Results

Targeted Recommendation Review tests:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/recommendation_review -q
```

Result:

```text
14 passed
```

Full Market Engine tests:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine -q
```

Result:

```text
163 passed
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
* `git status --short` showed only the planned Recommendation Review runtime, Recommendation Review tests, implementation documentation, audit, backlog, and roadmap changes.
* `git diff --stat` and `git diff --name-only` showed only the expected ME-RR04 areas.
* The legacy dependency grep found only negative assertion strings in Recommendation Review tests and no active imports from legacy `scripts` or old `market_scanner`.
* A focused authority-term scan found only structured forbidden-action constants and negative assertion test data, not emitted Recommendation Review guidance.

## Boundaries Preserved

Confirmed:

* no Setup Detection logic was modified;
* no Analysis Review production logic was modified;
* no provider calls were introduced;
* no network calls were introduced;
* no market data writes were introduced;
* no portfolio behavior was modified;
* no watchlist behavior was modified;
* no Telegram behavior was introduced;
* no reporting or delivery behavior was introduced;
* no Decision Engine behavior was modified;
* no BUY / SELL / HOLD authority was introduced;
* no allocation, position sizing, execution advice, ranking, scoring, conviction, urgency, or tradeability behavior was introduced;
* no legacy `scripts` imports were introduced;
* no old `market_scanner` imports were introduced.

## Next Recommended Sprint

Recommended next sprint:

```text
ME-PR01 — Define Portfolio Review contract from Recommendation Review
```
