# ME-SA08 - Company Profile-Only Recommendation Review Boundary Audit

Sprint ID: ME-SA08
Status: COMPLETED BY ME-SA08
Job family: ME-SA / Recommendation Review
Date: 2026-06-29
Branch: `me-sa08-company-profile-only-recommendation-review-boundary`

## Purpose

ME-SA08 closes the risk that descriptive company-profile context could be
mistaken for sufficient evidence for a Recommendation Review outcome.

Consumed company-profile context may identify and describe a company, but it
does not provide fundamental financial, financial-market, valuation, or setup
evidence. Recommendation Review must therefore remain deterministically
non-actionable and fail closed.

## Behavior Added

The existing Recommendation Review schema now represents profile-only analysis
context with:

```text
review_state: blocked_by_missing_data
review_category: company_profile_only_context_non_actionable
blocked_reasons:
  - company_profile_only_context_non_actionable
```

The review item records that fundamental financial, financial-market, and setup
evidence are missing. Descriptive profile context remains available through
input provenance only.

No recommendation, action, entry price, stop loss, target price, conviction,
position sizing, trade instruction, or Decision Engine-ready result is created.

## Combined Context Behavior

When company-profile context accompanies an existing SEC CompanyFacts Analysis
Review, the profile context is additive provenance only. Existing financial and
setup evidence continues to determine the Recommendation Review state.

A limited financial Analysis Review remains blocked when profile context is
present. Company metadata cannot upgrade missing or insufficient evidence.

## Dry-Run Integration

Valid profile-only cached-source dry-runs now build the explicit Recommendation
Review boundary rather than a generic not-started placeholder.

The dry-run still stops at Recommendation Review. Portfolio Review, Decision
Engine handoff, and Delivery / Reporting remain not started with the same
profile-only non-actionable reason.

Written local dry-run artifacts preserve the Recommendation Review blocker and
company-profile input provenance.

## Files Changed

Runtime:

```text
src/market_engine/recommendation_review/sec_companyfacts_recommendation_review.py
src/market_engine/run/cached_source_execution.py
```

Tests:

```text
tests/market_engine/recommendation_review/test_sec_companyfacts_recommendation_review.py
tests/market_engine/run/test_me_run10_cached_source_local_execution.py
tests/market_engine/run/test_me_run27_company_profile_cross_ticker_dry_run.py
```

Documentation:

```text
docs/market_engine/audits/me_sa08_company_profile_only_recommendation_review_boundary.md
docs/market_engine/backlog/me_sa08_company_profile_only_recommendation_review_boundary_backlog_entry.md
docs/market_engine/backlog/market_engine_backlog.md
docs/market_engine/roadmap/me_sa08_company_profile_only_recommendation_review_boundary_roadmap_entry.md
docs/market_engine/roadmap/market_engine_roadmap.md
```

## Explicit Non-Goals

ME-SA08 does not add:

* provider or network access;
* yfinance, SEC, EDGAR, or external API calls;
* financial derivation from company-profile fields;
* valuation or setup inference;
* recommendation, allocation, conviction, ranking, or urgency logic;
* portfolio or watchlist mutation;
* Telegram, reporting, delivery, broker, or production behavior;
* Decision Engine authority;
* cached-source validation changes;
* legacy scanner runtime changes.

## Validation

```text
16 passed - Recommendation Review tests
21 passed - cached-source local execution tests
2 passed - ME-RUN27 cross-ticker tests
520 passed - tests/market_engine
1187 passed - full pytest
```

## Known Limitation and Follow-Up

The cached-source command still accepts one snapshot path. ME-SA08 does not add
multi-source discovery or orchestration.

Profile-only dry-runs intentionally stop at Recommendation Review. A future
explicitly governed sprint may define non-actionable downstream communication
of this blocked review state without creating recommendation or allocation
authority.

## Final Status

```text
PASS
```
