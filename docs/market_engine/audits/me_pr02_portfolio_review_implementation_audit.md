# ME-PR02 — Portfolio Review implementation audit

## Status

COMPLETED BY ME-PR02

## Sprint

ME-PR02 — Implement Portfolio Review

## Branch

me-pr02-implement-portfolio-review

## Sprint Goal

Implement Portfolio Review as the downstream consumer of approved Recommendation Review output and explicitly supplied portfolio context.

The implementation must remain non-actionable and must not introduce provider calls, broker calls, portfolio writes, watchlist writes, reporting delivery, Telegram behavior, Decision Engine behavior, BUY / SELL / HOLD action semantics, allocation, position sizing, execution advice, ranking, scoring, conviction, urgency, or tradeability behavior.

## Files Added

Runtime:

* `src/market_engine/portfolio_review/__init__.py`
* `src/market_engine/portfolio_review/sec_companyfacts_portfolio_review.py`

Tests:

* `tests/market_engine/portfolio_review/test_sec_companyfacts_portfolio_review.py`

Documentation:

* `docs/market_engine/portfolio_review/me_pr02_portfolio_review_implementation.md`
* `docs/market_engine/audits/me_pr02_portfolio_review_implementation_audit.md`

## Files Changed

Documentation:

* `docs/market_engine/backlog/market_engine_backlog.md`
* `docs/market_engine/roadmap/market_engine_roadmap.md`

## Contract Implemented

Implemented governing contract:

* `docs/market_engine/portfolio_review/me_pr01_portfolio_review_contract.md`

## Input Contracts

Approved Recommendation Review input contract:

* `sec-companyfacts-recommendation-review-v1`

Approved portfolio context input family:

* `market-engine-portfolio-context-v1`

Unsupported Recommendation Review input contracts fail closed with `ValueError`.

Unsupported portfolio context contracts produce controlled invalid-input Portfolio Review output.

## Output Contract

Implemented output contract:

* `sec-companyfacts-portfolio-review-v1`

## Implementation Summary

ME-PR02 implements:

* `MarketEnginePortfolioContext`;
* `MarketEnginePortfolioPositionState`;
* `SecCompanyFactsPortfolioReviewCategory`;
* `SecCompanyFactsPortfolioReviewState`;
* `SecCompanyFactsPortfolioReviewItem`;
* `SecCompanyFactsPortfolioReview`;
* `build_sec_companyfacts_portfolio_review(...)`;
* `persist_sec_companyfacts_portfolio_review(...)`.

The builder consumes validated Recommendation Review input and explicitly supplied portfolio context.

It emits non-actionable position, exposure, concentration, portfolio-fit, data-limitation, and downstream-handoff-readiness review items.

## Categories Implemented

Implemented Portfolio Review categories:

* `position_context_review`
* `exposure_context_review`
* `concentration_context_review`
* `portfolio_fit_context_review`
* `portfolio_data_limitation_review`
* `downstream_handoff_readiness_review`
* `input_contract_invalid`

## States Implemented

Implemented Portfolio Review states:

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

## Provenance Behavior

Portfolio Review preserves:

* Recommendation Review provenance;
* Recommendation Review item references;
* Setup Detection-aware provenance when present upstream;
* portfolio-context provenance;
* missing portfolio-context fields;
* stale portfolio-context fields.

## Missing And Stale Data Behavior

Missing Recommendation Review input produces controlled invalid-input output.

Missing portfolio context produces `blocked_by_missing_portfolio_context`.

Stale portfolio context produces `portfolio_context_stale`.

Partial portfolio context produces `portfolio_context_partial`.

Invalid portfolio context produces `portfolio_context_invalid` or controlled invalid-input output.

Missing portfolio fields remain explicit and are not converted to zero.

## Numeric-Zero Behavior

Numeric zero remains valid when explicitly supplied by portfolio context.

Tests cover zero quantity, zero market value, zero total value, and zero ticker exposure.

## Persistence Behavior

Persistence writes deterministic JSON to:

```text
data/market_engine/portfolio_reviews/<portfolio_review_run_id>/<ticker>/portfolio_review.json
```

Tests use temporary directories only.

Persistence refuses overwrite with `FileExistsError`.

## Tests Added

ME-PR02 tests cover:

* valid Recommendation Review and valid portfolio context;
* missing Recommendation Review input;
* unsupported Recommendation Review contract;
* non-reviewable Recommendation Review input;
* missing portfolio context;
* stale portfolio context;
* partial portfolio context;
* held ticker state;
* not-held ticker state;
* unknown position state;
* numeric-zero preservation;
* Recommendation Review provenance preservation;
* Setup Detection-aware provenance preservation;
* unsupported portfolio context contract;
* persistence and overwrite refusal;
* forbidden action-authority output boundaries;
* no legacy `scripts` or old `market_scanner` imports.

## Validation Commands And Results

Targeted Portfolio Review tests:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/portfolio_review -q
```

Result:

```text
16 passed
```

Relevant upstream review tests:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/recommendation_review -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine/analysis_review -q
```

Result:

```text
14 passed
18 passed
```

Full Market Engine tests:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/market_engine -q
```

Result:

```text
179 passed
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
* `git status --short` showed only planned Portfolio Review runtime, Portfolio Review tests, implementation documentation, audit, backlog, and roadmap changes.
* `git diff --stat` and `git diff --name-only` showed only the expected tracked ME-PR02 documentation updates before staging.
* The legacy dependency grep found only negative assertion strings in Recommendation Review and Portfolio Review tests and no active imports from legacy `scripts` or old `market_scanner`.
* Backlog and roadmap each have one `Status: RECOMMENDED NEXT` marker, now assigned to `ME-DE01 — Define Decision Engine handoff contract`.
* A focused authority-term scan found only structured forbidden-action constants, boundary text, and negative assertion test data, not Portfolio Review guidance.

## Boundaries Preserved

Confirmed:

* no provider calls were introduced;
* no network calls were introduced;
* no broker calls were introduced;
* no portfolio files were written;
* no real portfolio data was modified;
* no watchlist data was modified;
* no Telegram behavior was introduced;
* no reporting delivery behavior was introduced;
* no Decision Engine behavior was introduced;
* no BUY / SELL / HOLD action authority was introduced;
* no allocation advice was introduced;
* no target weights were introduced;
* no position sizing advice was introduced;
* no execution advice was introduced;
* no ranking, scoring, conviction, urgency, or tradeability behavior was introduced;
* no legacy `scripts` imports were introduced;
* no old `market_scanner` imports were introduced.

## Follow-Up Candidate

Possible future Portfolio Review follow-up candidate:

```text
ME-PR03 — Define approved portfolio context source and persistence contract
```

This candidate is not inserted ahead of ME-DE01 because ME-PR02 did not uncover a blocker.

## Next Recommended Sprint

Recommended next sprint:

```text
ME-DE01 — Define Decision Engine handoff contract
```
