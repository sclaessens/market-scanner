# ME-RR04 — Setup Detection-aware Recommendation Review implementation

## Status

COMPLETED BY ME-RR04

## Sprint

ME-RR04 — Implement Recommendation Review consumption of Setup Detection-aware Analysis Review output

## Job Family

ME-RR — Recommendation Review jobs

## Purpose

ME-RR04 implements Recommendation Review consumption of Setup Detection-aware `sec-companyfacts-analysis-review-v1` output.

Recommendation Review reads, preserves, and exposes setup-aware evidence from Analysis Review output when available.

Recommendation Review remains non-actionable and does not become an execution, portfolio, delivery, or Decision Engine authority.

## Files Changed

Runtime:

* `src/market_engine/recommendation_review/sec_companyfacts_recommendation_review.py`

Tests:

* `tests/market_engine/recommendation_review/test_sec_companyfacts_recommendation_review.py`

Documentation:

* `docs/market_engine/recommendation_review/me_rr04_setup_detection_aware_recommendation_review_implementation.md`
* `docs/market_engine/audits/me_rr04_setup_detection_aware_recommendation_review_implementation_audit.md`
* `docs/market_engine/backlog/market_engine_backlog.md`
* `docs/market_engine/roadmap/market_engine_roadmap.md`

## Contract Behavior Implemented

Preserved input contract:

* `sec-companyfacts-analysis-review-v1`

Preserved output contract:

* `sec-companyfacts-recommendation-review-v1`

ME-RR04 does not introduce a new Recommendation Review contract version.

The implementation detects setup-aware Analysis Review items when present and preserves:

* setup categories;
* setup states;
* setup evidence;
* setup limitations;
* missing setup observations;
* setup-aware Analysis Review references;
* source observation references;
* derived observation references;
* setup detection format version;
* setup detection run ID;
* Setup Detection non-actionable boundary metadata when available.

## Routing Behavior

Setup-aware Analysis Review states route only to non-actionable Recommendation Review outcomes:

* detected setup evidence routes to `human_review_required`;
* partial setup evidence routes to `human_review_required` with uncertainty preserved;
* conflicted setup evidence routes to `human_review_required` with conflict preserved;
* blocked setup evidence routes to `blocked_by_missing_data`;
* not-assessed or not-detected setup evidence routes to `insufficient_evidence`.

These are review-routing outcomes only.

## Backward Compatibility

Recommendation Review remains tolerant when setup-aware fields are absent, incomplete, null, empty, or legacy-shaped.

Existing ME-RR02 behavior remains valid for Analysis Review outputs that do not contain setup-aware fields.

Unsupported Analysis Review contract versions still fail closed.

## Authority Boundaries

ME-RR04 does not introduce:

* BUY / SELL / HOLD authority;
* recommendation authority beyond existing non-actionable human-review routing;
* allocation;
* position sizing;
* execution advice;
* ranking;
* scoring;
* conviction;
* urgency;
* tradeability decisions;
* portfolio mutation;
* watchlist mutation;
* delivery behavior;
* Telegram behavior;
* reporting behavior;
* Decision Engine behavior.

## Test Coverage Summary

Tests cover:

* detected setup-aware Analysis Review input;
* partial setup-aware Analysis Review input;
* blocked setup-aware Analysis Review input;
* conflicted setup-aware Analysis Review input;
* not-assessed setup-aware Analysis Review input;
* not-detected setup-aware Analysis Review input;
* missing or empty setup-aware fields;
* legacy Analysis Review input without setup-aware fields;
* setup-aware persistence provenance;
* forbidden action-authority output boundaries.

Tests use local synthetic objects only.

No provider or network calls are used.

## Known Non-Goals

ME-RR04 does not:

* modify Setup Detection logic;
* modify Analysis Review production logic;
* create Portfolio Review behavior;
* create Delivery / Reporting behavior;
* call providers;
* write market data;
* mutate portfolio or watchlist state;
* call or modify the Decision Engine.

## Next Sprint

Recommended next sprint:

```text
ME-PR01 — Define Portfolio Review contract from Recommendation Review
```
