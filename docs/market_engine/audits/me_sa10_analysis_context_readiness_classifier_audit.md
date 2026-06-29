# ME-SA10 - Analysis-Context Readiness Classifier Audit

Sprint ID: ME-SA10
Status: COMPLETED BY ME-SA10
Job family: ME-SA / Analysis Review
Date: 2026-06-29
Branch: `me-sa10-analysis-context-readiness-classifier`

## Purpose

ME-SA10 implements the typed, deterministic, fail-closed analysis-context
readiness classifier defined by:

```text
docs/market_engine/analysis_review/me_sa09_multi_source_analysis_context_readiness_contract.md
```

The classifier describes evidence completeness only. It does not produce a
recommendation, determine action or allocation, or authorize execution,
delivery, portfolio mutation, or Decision Engine behavior.

## Implementation

The implementation adds:

```text
market-engine-analysis-context-readiness-v1
```

Typed readiness levels:

```text
descriptive_only
partial_analysis
recommendation_eligible
actionable_review
decision_ready
```

Typed evidence families:

```text
company_profile
fundamentals
valuation
setup_price_market
portfolio_context
provenance_manifest_staleness
delivery_reporting_handoff
```

The frozen result object preserves:

* readiness level;
* ordered present and missing evidence families;
* deterministic blocked reasons;
* Recommendation Review eligibility;
* fixed-false actionable-review and Decision Engine readiness gates;
* provenance and staleness state;
* unknown-input notes;
* prohibited inference and safety notes;
* a non-authority boundary;
* straightforward JSON-compatible serialization through `to_payload()`.

## Classification Behavior

| Input | Result | Required blocker |
| --- | --- | --- |
| company profile only | `descriptive_only` | `company_profile_only_context_non_actionable` |
| empty or unknown input | `descriptive_only` | `insufficient_analysis_context` |
| fundamentals without setup/price/market | `partial_analysis` | `missing_setup_or_price_context` |
| setup/price/market without fundamentals | `partial_analysis` | `missing_fundamental_evidence` |
| fundamentals plus setup/price/market plus valid provenance/freshness | `recommendation_eligible` | none |
| stale or unprovenanced analytical context | `partial_analysis` at most | `stale_or_unprovenanced_analysis_context` |
| missing required valuation | `partial_analysis` | `insufficient_analysis_context` |
| malformed or unknown family input | `descriptive_only` | `insufficient_analysis_context` |

Company-profile context never upgrades fundamentals-only or setup-only input.

## Fail-Closed Rules

The classifier fails closed when:

* no recognized evidence exists;
* an unknown evidence-family value is supplied;
* evidence input is not iterable;
* provenance, staleness, or valuation-required gates are not booleans;
* the provenance family is absent;
* provenance is invalid;
* context is stale;
* required fundamental evidence is absent;
* required setup, price, or market context is absent;
* strategy-required valuation evidence is absent.

Invalid or stale provenance is represented by an explicit blocker and gate
state. An invalid family is not simultaneously reported as missing when it is
present.

## Reserved Levels

`actionable_review` and `decision_ready` exist in the enum so the runtime type
matches the ME-SA09 contract.

They are unreachable in ME-SA10:

```text
actionable_review_allowed = false
decision_engine_ready = false
```

No evidence-family combination can activate either level. Separate future
governance and contract approval are required before reachability may change.

## Integration Boundary

ME-SA10 exposes the classifier through the `market_engine.analysis_review`
package and tests it directly.

It is not injected into existing Analysis Review, Recommendation Review,
cached-source dry-run, Portfolio Review, handoff, delivery, or persisted
artifact schemas. This avoids an unapproved output-contract change.

Runtime orchestration and artifact persistence remain follow-up work.

## Files Changed

Runtime:

```text
src/market_engine/analysis_review/analysis_context_readiness.py
src/market_engine/analysis_review/__init__.py
```

Tests:

```text
tests/market_engine/analysis_review/test_analysis_context_readiness.py
```

Documentation:

```text
docs/market_engine/audits/me_sa10_analysis_context_readiness_classifier_audit.md
docs/market_engine/backlog/me_sa10_analysis_context_readiness_classifier_backlog_entry.md
docs/market_engine/backlog/market_engine_backlog.md
docs/market_engine/roadmap/me_sa10_analysis_context_readiness_classifier_roadmap_entry.md
docs/market_engine/roadmap/market_engine_roadmap.md
```

## Non-Goals

ME-SA10 does not:

* create recommendations or recommendation outcomes;
* activate `actionable_review` or `decision_ready`;
* determine BUY, SELL, HOLD, conviction, urgency, tradeability, rank, or score;
* allocate capital or size positions;
* generate orders or call brokers;
* add provider, network, yfinance, SEC, EDGAR, or market-data access;
* change cached-source validation;
* change portfolio, watchlist, Telegram, delivery, reporting, production-write,
  or Decision Engine behavior;
* change existing persisted artifact formats;
* modify legacy scanner runtime.

## Validation

```text
15 passed - new readiness classifier tests
40 passed - tests/market_engine/analysis_review
16 passed - tests/market_engine/recommendation_review
535 passed - tests/market_engine
1202 passed - full pytest
PASS - git diff --check
```

## Known Limitations and Follow-Up

ME-SA10 accepts explicit typed evidence families and control gates. It does not
discover or infer evidence families from existing stage payloads.

A future sprint may define a versioned adapter from approved Analysis Review
metadata into the classifier and, separately, an additive artifact persistence
contract. That work must preserve the standalone classifier semantics, require
explicit source-family applicability, and keep reserved levels unreachable
unless governance changes first.

## Final Status

```text
PASS
```
