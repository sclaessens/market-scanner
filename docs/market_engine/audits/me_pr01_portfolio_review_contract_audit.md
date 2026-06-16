# ME-PR01 — Portfolio Review contract audit

## Status

COMPLETED BY ME-PR01

## Sprint

ME-PR01 — Define Portfolio Review contract from Recommendation Review

## Branch

me-pr01-define-portfolio-review-contract

## Sprint goal

Define the Portfolio Review contract after Setup Detection-aware Recommendation Review exists.

## Files added

Documentation:

* `docs/market_engine/portfolio_review/me_pr01_portfolio_review_contract.md`
* `docs/market_engine/audits/me_pr01_portfolio_review_contract_audit.md`

## Files changed

Documentation:

* `docs/market_engine/roadmap/market_engine_roadmap.md`

## Contract defined

Defined Portfolio Review as a non-actionable review layer downstream of Recommendation Review and upstream of Decision Engine handoff.

Approved Portfolio Review input contract:

* `sec-companyfacts-recommendation-review-v1`

Approved portfolio-context input family:

* `market-engine-portfolio-context-v1`

Recommended Portfolio Review output contract:

* `sec-companyfacts-portfolio-review-v1`

Recommended future persistence path:

* `data/market_engine/portfolio_reviews/<portfolio_review_run_id>/<ticker>/portfolio_review.json`

ME-PR01 does not create or write this path.

## Portfolio-context boundary

ME-PR01 defined that Portfolio Review requires explicitly supplied portfolio context.

Portfolio Review must not silently infer holdings, exposure, concentration, or policy constraints from broker exports, reports, old runtime files, generated output folders, watchlists, or source/provider data unless a later sprint explicitly approves that input path.

## Approved review categories

ME-PR01 approved initial Portfolio Review categories:

* `position_context_review`
* `exposure_context_review`
* `concentration_context_review`
* `portfolio_fit_context_review`
* `portfolio_data_limitation_review`
* `downstream_handoff_readiness_review`
* `input_contract_invalid`

These categories are review categories only.

## Approved review states

ME-PR01 approved initial Portfolio Review states:

* `portfolio_review_required`
* `portfolio_context_supported`
* `portfolio_context_partial`
* `portfolio_context_missing`
* `portfolio_context_stale`
* `portfolio_context_invalid`
* `position_already_held`
* `position_not_held`
* `position_unknown`
* `exposure_known`
* `exposure_missing`
* `concentration_within_context`
* `concentration_requires_review`
* `blocked_by_missing_portfolio_context`
* `blocked_by_invalid_input`
* `ready_for_decision_engine_handoff_review`
* `not_applicable`

These states are non-actionable portfolio-review states only.

## Missing-data behavior

Missing Recommendation Review input must fail closed or produce invalid-input output in a future implementation.

Missing portfolio context must block Portfolio Review or produce explicit missing-context output.

Missing portfolio fields must remain explicit and must not be converted into zero.

Unknown position state remains unknown.

Partial portfolio data remains partial.

## Stale-data behavior

Portfolio context must include a timestamp or equivalent freshness marker.

If freshness cannot be assessed, a future implementation must mark context as partial, stale, or blocked according to the implementation contract.

Stale portfolio context must not be treated as current and must not generate action guidance.

## Numeric-zero behavior

Numeric zero remains present when explicitly supplied and source-grounded.

Examples include explicit zero quantity or explicit zero exposure for a non-held ticker.

Missing numeric values must not be coerced to zero.

## Provenance preservation

ME-PR01 requires a future Portfolio Review implementation to preserve:

* Recommendation Review provenance;
* Analysis Review references preserved by Recommendation Review;
* Setup Detection-aware provenance when present upstream;
* source observation references when preserved upstream;
* derived observation references when preserved upstream;
* portfolio-context provenance;
* missing and stale portfolio-context markers.

## Authority boundaries preserved

Confirmed ME-PR01 does not introduce:

* Python code;
* tests;
* runtime behavior;
* provider calls;
* broker calls;
* data writes;
* generated artifacts;
* portfolio mutation;
* watchlist mutation;
* Decision Engine behavior;
* Delivery / Reporting behavior;
* Telegram behavior;
* BUY / SELL / HOLD action semantics;
* allocation execution;
* order generation;
* target weights;
* position sizing instructions;
* ranking;
* scoring;
* conviction;
* urgency;
* tradeability authority.

## Decision Engine boundary

Portfolio Review remains upstream of Decision Engine handoff.

Portfolio Review may prepare structurally valid non-actionable portfolio-context review output.

Only a later Decision Engine handoff contract may define how Portfolio Review output becomes eligible for Decision Engine evaluation.

Only the Decision Engine may later own action/allocation authority if explicitly implemented by an approved future sprint.

## Delivery boundary

Portfolio Review does not create delivery output, reports, Telegram messages, or user-facing delivery eligibility.

Delivery / Reporting remains a later job family.

## ME-PR02 readiness

ME-PR01 defines ME-PR02 implementation requirements.

ME-PR02 must consume only validated Recommendation Review input and explicitly supplied approved portfolio context.

ME-PR02 must remain non-actionable and must not mutate portfolio state, call providers, call brokers, call the Decision Engine, send Telegram, report, deliver, rank, score, allocate, size positions, or emit BUY / SELL / HOLD action semantics.

## Validation performed

Connector-based repository inspection confirmed the governing context from:

* `docs/market_engine/roadmap/market_engine_roadmap.md`
* `docs/market_engine/backlog/market_engine_backlog.md`
* `docs/market_engine/recommendation_review/me_rr03_setup_detection_aware_analysis_review_contract.md`
* `docs/market_engine/recommendation_review/me_rr04_setup_detection_aware_recommendation_review_implementation.md`

No local checkout was available in this runtime, so no local pytest or `git diff --check` command was executed here.

Because ME-PR01 is documentation-only, no Python test suite was required or modified.

## Next recommended sprint

Recommended next sprint:

```text
ME-PR02 — Implement Portfolio Review
```
