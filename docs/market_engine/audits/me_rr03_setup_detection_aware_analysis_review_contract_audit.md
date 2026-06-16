# ME-RR03 — Setup Detection-aware Recommendation Review contract audit

## Status

COMPLETED BY ME-RR03

## Sprint

ME-RR03 — Extend Recommendation Review contract for Setup Detection-aware Analysis Review

## Branch

me-rr03-extend-recommendation-review-contract-for-setup-detection

## Sprint type

Documentation-only contract sprint.

## Objective

Define how Recommendation Review may consume Setup Detection-aware Analysis Review output while preserving the existing Recommendation Review authority boundary.

ME-RR03 keeps the approved Recommendation Review input contract as:

```text
sec-companyfacts-analysis-review-v1
```

The sprint does not introduce runtime implementation, tests, provider calls, data writes, Portfolio Review behavior, Delivery behavior, Telegram behavior, reporting behavior, or Decision Engine behavior.

## Files added

Documentation:

* `docs/market_engine/recommendation_review/me_rr03_setup_detection_aware_analysis_review_contract.md`
* `docs/market_engine/audits/me_rr03_setup_detection_aware_analysis_review_contract_audit.md`

## Files changed

Documentation:

* `docs/market_engine/backlog/market_engine_backlog.md`
* `docs/market_engine/roadmap/market_engine_roadmap.md`

## Contract inputs inspected

ME-RR03 is based on the existing Recommendation Review and Analysis Review contract chain:

* `docs/market_engine/recommendation_review/me_rr01_recommendation_review_contract.md`
* `docs/market_engine/audits/me_rr02_recommendation_review_implementation_audit.md`
* `docs/market_engine/analysis_review/me_ar03_setup_detection_input_contract.md`
* `docs/market_engine/analysis_review/me_ar04_analysis_review_setup_detection_implementation.md`
* `docs/market_engine/audits/me_ar04_analysis_review_setup_detection_implementation_audit.md`

## Contract decisions

ME-RR03 confirms:

* Recommendation Review remains downstream of Analysis Review.
* Recommendation Review may not consume Setup Detection output directly.
* Recommendation Review may consume Setup Detection-aware Analysis Review only through validated `sec-companyfacts-analysis-review-v1`.
* `sec-companyfacts-analysis-review-v1` remains the approved input contract.
* `sec-companyfacts-recommendation-review-v1` remains the future output contract family.
* Setup-aware evidence must remain provenance-preserving.
* Missing setup evidence must remain explicit.
* Numeric zero must not be treated as missing.
* Detected setup evidence may route only to non-actionable human review.
* Partial setup evidence must preserve uncertainty.
* Conflicted setup evidence must preserve conflict.
* Blocked setup evidence must preserve missing-data blocking.
* Not-assessed setup evidence must remain not assessed or insufficient evidence.
* Not-detected setup evidence must not become a negative recommendation.

## Non-actionable routing defined

ME-RR03 defines future non-actionable routing for these Analysis Review setup-aware states:

* `SETUP_DETECTED`
* `SETUP_PARTIALLY_DETECTED`
* `SETUP_NOT_DETECTED`
* `SETUP_CONFLICTED`
* `SETUP_BLOCKED_BY_MISSING_DATA`
* `SETUP_NOT_ASSESSED`
* `SETUP_REQUIRES_HUMAN_REVIEW`

All routing remains human-review-only or blocked/insufficient-evidence routing.

No action, portfolio, ranking, scoring, conviction, urgency, tradeability, allocation, position-sizing, execution, order, Telegram, reporting, or delivery semantics are introduced.

## ME-RR04 requirements defined

ME-RR03 defines that ME-RR04 must:

* consume only validated `sec-companyfacts-analysis-review-v1`;
* preserve existing ME-RR02 behavior when Setup Detection-aware Analysis Review items are absent;
* detect Setup Detection-aware Analysis Review items where present;
* preserve setup-aware provenance;
* preserve setup categories and states;
* preserve setup evidence and limitations;
* preserve missing setup observations;
* preserve source and derived references;
* preserve numeric-zero semantics;
* preserve non-actionable boundary markers;
* route detected setup evidence to human review only;
* route partial setup evidence to human review with explicit uncertainty;
* route conflicted setup evidence to human review with explicit conflict;
* route blocked setup evidence to blocked-by-missing-data;
* route not-assessed setup evidence to insufficient-evidence or blocked routing;
* fail closed for unsupported Analysis Review input contracts;
* add local synthetic tests only;
* avoid live provider calls;
* avoid production data writes;
* avoid legacy `scripts` or old `market_scanner` imports.

## Boundaries preserved

ME-RR03 does not introduce:

* Python implementation;
* tests;
* runtime behavior;
* provider calls;
* SEC calls;
* EDGAR calls;
* yfinance calls;
* production data writes;
* generated artifacts;
* Recommendation Review runtime changes;
* Analysis Review runtime changes;
* Setup Detection runtime changes;
* Portfolio Review behavior;
* portfolio mutation;
* watchlist mutation;
* Delivery behavior;
* Telegram behavior;
* reporting behavior;
* Decision Engine behavior;
* BUY / SELL / HOLD action semantics;
* target price;
* rating;
* score;
* ranking;
* conviction;
* urgency;
* tradeability;
* allocation;
* position sizing;
* execution advice;
* order generation.

## Backlog and roadmap requirements

ME-RR03 updates must ensure:

* backlog marks ME-RR03 as completed;
* backlog marks ME-RR04 as the only recommended next sprint;
* roadmap status becomes `ACTIVE ROADMAP AFTER ME-RR03`;
* roadmap completed chain includes ME-RR03;
* roadmap marks ME-RR04 as the only recommended next sprint;
* future sequence remains:

  1. ME-RR04
  2. ME-PR01
  3. ME-PR02
  4. ME-DE01
  5. ME-DE02
  6. ME-DL01
  7. ME-DL02

## Audit conclusion

ME-RR03 is complete when the contract document, this audit, backlog, and roadmap are updated together.

The sprint extends Recommendation Review on paper only, keeps `sec-companyfacts-analysis-review-v1` as the approved input contract, preserves setup-aware evidence and provenance, defines non-actionable setup-state routing, and prepares ME-RR04 without creating runtime behavior or authority changes.