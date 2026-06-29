# ME-SA11 - Analysis-Context Readiness Adapter and Artifact Metadata Audit

Sprint ID: ME-SA11
Status: COMPLETED BY ME-SA11
Job family: ME-SA / Analysis Review and local dry-run metadata
Date: 2026-06-29
Branch: `me-sa11-analysis-context-readiness-adapter-artifact-metadata`

## Purpose

ME-SA11 connects the standalone ME-SA10 readiness classifier to existing
Market Engine dry-run stage payloads and makes the resulting non-authoritative
metadata visible in local dry-run artifacts.

Contract references:

```text
docs/market_engine/analysis_review/me_sa09_multi_source_analysis_context_readiness_contract.md
src/market_engine/analysis_review/analysis_context_readiness.py
```

## Chosen Adapter Contract

The adapter entry point is:

```text
classify_analysis_context_readiness_from_stage_payloads(stage_payloads)
```

It accepts the existing dry-run stage-payload mapping and returns the typed
ME-SA10 `AnalysisContextReadinessResult`.

Mapping is explicit and conservative:

* `COMPANY_PROFILE` requires either:
  * consumed Company Profile Source Context plus
    `market-engine-company-profile-analysis-context-v1`; or
  * the existing ME-SA07 `company_profile_context` extension on a valid SEC
    Analysis Review;
* `FUNDAMENTALS` requires a usable
  `sec-companyfacts-fundamental-observations-v1` payload;
* `SETUP_PRICE_MARKET` requires a usable
  `sec-companyfacts-setup-detection-v1` payload;
* `VALUATION` is never inferred because no approved valuation stage exists;
* `PORTFOLIO_CONTEXT` requires a usable Portfolio Review plus the approved
  portfolio-context version;
* `PROVENANCE_MANIFEST_STALENESS` requires a usable source contract, source
  snapshot identity, local cached-source or deterministic fixture root,
  evidence lineage, ticker alignment, and no stale markers;
* `DELIVERY_REPORTING_HANDOFF` requires both usable handoff and delivery
  contracts.

Unsupported versions, blocked stages, malformed values, missing lineage,
ticker mismatch, missing provenance roots, and stale markers cannot satisfy a
higher readiness gate.

When real observation or setup-item collections are present, empty, malformed,
missing-data, not-assessed, source-limited, partial, conflicted, or blocked
collections do not satisfy their evidence-family gate. Minimal approved
contract fixtures may omit item collections.

## Artifact Location and Shape

Readiness is added to the top-level end-to-end dry-run payload:

```text
analysis_context_readiness
```

Local dry-run persistence already stores the full dry-run payload, so the
persisted location is:

```text
artifact["payload"]["analysis_context_readiness"]
```

The section contains:

```text
readiness_format_version
readiness_level
evidence_families_present
evidence_families_missing
blocked_reasons
recommendation_review_eligible
actionable_review_allowed
decision_engine_ready
provenance_valid
context_stale
unknown_evidence_families
input_notes
prohibited_inferences
safety_notes
non_authority_boundary
```

This is an additive metadata extension. These versions remain unchanged:

```text
market-engine-end-to-end-dry-run-v1
market-engine-local-dry-run-artifact-v1
```

No required stage was added and no existing field was removed or redefined.

## Semantic Results

Company-profile-only cached-source dry-runs persist:

```text
readiness_level: descriptive_only
evidence_families_present:
  - company_profile
  - provenance_manifest_staleness
blocked_reasons:
  - company_profile_only_context_non_actionable
recommendation_review_eligible: false
actionable_review_allowed: false
decision_engine_ready: false
```

Other adapter results:

| Context | Readiness |
| --- | --- |
| fundamentals only | `partial_analysis` with `missing_setup_or_price_context` |
| setup/price/market only | `partial_analysis` with `missing_fundamental_evidence` |
| fundamentals plus setup/price/market plus valid provenance | `recommendation_eligible` at most |
| stale or unprovenanced analytical context | `partial_analysis` at most with `stale_or_unprovenanced_analysis_context` |
| unknown or missing stage output | fail-closed and non-authoritative |

Additional company-profile context cannot upgrade partial non-profile evidence.

## Reserved Levels

`actionable_review` and `decision_ready` remain reserved and unreachable.

The adapter calls the ME-SA10 classifier and does not add an override path.
Every persisted result therefore retains:

```text
actionable_review_allowed: false
decision_engine_ready: false
```

Recommendation eligibility remains permission to evaluate only. It is not a
recommendation or Decision Engine handoff authorization.

## Files Changed

Runtime:

```text
src/market_engine/analysis_review/analysis_context_readiness_adapter.py
src/market_engine/analysis_review/__init__.py
src/market_engine/run/end_to_end_dry_run.py
```

Tests:

```text
tests/market_engine/analysis_review/test_analysis_context_readiness_adapter.py
tests/market_engine/run/test_end_to_end_dry_run.py
tests/market_engine/run/test_local_dry_run_artifacts.py
tests/market_engine/run/test_me_run10_cached_source_local_execution.py
```

Documentation:

```text
docs/market_engine/audits/me_sa11_analysis_context_readiness_adapter_artifact_metadata_audit.md
docs/market_engine/backlog/me_sa11_analysis_context_readiness_adapter_artifact_metadata_backlog_entry.md
docs/market_engine/backlog/market_engine_backlog.md
docs/market_engine/roadmap/me_sa11_analysis_context_readiness_adapter_artifact_metadata_roadmap_entry.md
docs/market_engine/roadmap/market_engine_roadmap.md
```

## Safety and Non-Goals

ME-SA11 adds no:

* recommendation output or action semantics;
* reachable `actionable_review` or `decision_ready` state;
* BUY, SELL, HOLD, conviction, urgency, tradeability, ranking, or scoring;
* allocation, position sizing, order, execution, or broker behavior;
* provider, network, yfinance, SEC, EDGAR, external API, or live market-data
  access;
* cached-source validation change;
* portfolio or watchlist mutation;
* Telegram sending;
* production write;
* Decision Engine behavior;
* legacy scanner runtime change.

The existing local artifact write remains explicitly opt-in.

## Validation

```text
11 passed - new readiness adapter tests
51 passed - tests/market_engine/analysis_review
16 passed - tests/market_engine/recommendation_review
114 passed - tests/market_engine/run
546 passed - tests/market_engine
1213 passed - full pytest
PASS - git diff --check
```

## Known Limitations and Follow-Up

The adapter consumes existing stage payloads; it does not discover new sources
or create valuation evidence.

The first recommended follow-up is:

```text
ME-RUN28A - Run NVDA/AMD/ASML through persisted readiness and Recommendation Review boundary
```

Later output-oriented work may include:

```text
ME-DL03 - Create a Telegram preview artifact without sending
ME-RUN28 - Run expanded supported-universe acquisition and dry-run classification
```

Any future adapter expansion must use explicit approved contracts and remain
fail closed.

## Final Status

```text
PASS
```
