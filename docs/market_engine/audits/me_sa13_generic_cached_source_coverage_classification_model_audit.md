# ME-SA13 - Generic Cached-Source Coverage Classification Model Audit

Sprint ID: ME-SA13
Status: COMPLETED BY ME-SA13
Job family: ME-SA / Source Acquisition and Source Coverage
Date: 2026-07-02
Branch: `me-sa13-implement-generic-cached-source-coverage-classification-model`

## 1. Purpose and Scope

ME-SA13 implements the pure, deterministic generic coverage classifier defined
by ME-SA12:

```text
market-engine-supported-universe-cached-source-coverage-v1
```

Core rule:

```text
tickers are data, not logic
```

The implementation classifies configured capabilities, source-family
requirements, evidence gates, blockers, and readiness. It does not acquire
data, read files, call providers, execute Recommendation Review, prepare a
Decision Engine handoff, mutate state, or persist artifacts.

## 2. Source Basis

Source main commit:

```text
cb65de4 Merge pull request #418 from sclaessens/me-sa12-generic-supported-universe-cached-source-coverage-contract
```

Contract:

```text
docs/market_engine/source_support/me_sa12_generic_supported_universe_cached_source_coverage_contract.md
```

## 3. Implementation Overview

New runtime module:

```text
src/market_engine/source_support/cached_source_coverage.py
```

Public package exports:

```text
src/market_engine/source_support/__init__.py
```

Tests:

```text
tests/market_engine/source_support/test_cached_source_coverage.py
```

The module uses frozen dataclasses and `StrEnum` values. Classification is
pure Python and has no filesystem, clock, provider, network, portfolio,
watchlist, Telegram, Recommendation Review, or Decision Engine dependency.

## 4. API and Model

Primary entry points:

```python
classify_cached_source_coverage(coverage_input)
classify_cached_source_coverage_batch(coverage_inputs)
```

Serialization helper:

```python
to_plain_dict(classification)
```

### Input model

`CachedSourceCoverageInput` contains:

* ticker as opaque data;
* supported-universe status;
* target capability;
* explicit generic source-family requirements;
* source-family evidence;
* optional explicit blockers.

Target capabilities add generic minimum requirements. They do not inspect the
ticker value.

### Source-family evidence gates

Each `SourceFamilyEvidence` independently represents:

* source support;
* availability;
* manifest status;
* provenance status;
* freshness status;
* consumability status;
* completeness status;
* optional evidence reference.

### Output model

`CachedSourceCoverageClassification` contains:

* contract version;
* ticker;
* target capability;
* aggregate coverage status;
* readiness status;
* per-family results;
* deterministic blockers;
* Recommendation Review allowance;
* actionable flag;
* DE-ready flag;
* Decision Engine handoff allowance;
* audit notes;
* non-authority boundary.

`CachedSourceCoverageBatchClassification` preserves input order and emits
coverage/readiness counts plus actionable and DE-ready counts.

## 5. Domain Enums

Source families:

```text
company_profile
fundamental_facts
price_history
setup_detection_input
portfolio_context
recommendation_review_input
decision_engine_handoff_input
```

Coverage statuses:

```text
supported
unsupported
available
unavailable
partial
accepted
rejected
stale
unprovenanced
invalid_manifest
missing_snapshot
descriptive_only
blocked
not_consumable
not_required
```

Readiness statuses:

```text
unavailable
partial
descriptive_only
analysis_ready
recommendation_review_ready
actionable
de_ready
blocked
```

`actionable` and `de_ready` are schema values only. They are deliberately
unreachable under the current ME-SA12, ME-RR, and ME-DE governance.

## 6. Classification Rules

Rule precedence is generic and fail-closed:

1. Unsupported universe blocks the requested capability.
2. Unsupported required source family blocks.
3. Missing required evidence becomes `missing_snapshot`.
4. Invalid or rejected manifest blocks.
5. Required missing provenance blocks.
6. Required stale evidence blocks.
7. Required non-consumable evidence blocks.
8. Incomplete required evidence becomes partial with a family blocker.
9. Valid company-profile-only evidence becomes `descriptive_only`.
10. Accepted fundamental coverage may become `analysis_ready`.
11. Missing setup or price input prevents Recommendation Review readiness.
12. Complete generic Recommendation Review requirements may become
    `recommendation_review_ready`.
13. Missing required portfolio context blocks the handoff capability.
14. Current actionable-review and Decision Engine readiness authority remains
    explicitly blocked even when every evidence-family gate passes.

Per-family failures preserve both the gate blocker and the relevant capability
blocker, for example:

```text
missing_cached_source_snapshot
missing_fundamental_evidence
missing_setup_or_price_context
blocked_missing_portfolio_context
recommendation_review_blocked
decision_engine_handoff_blocked
```

## 7. Authority Boundary

ME-SA13 does not determine investment action or allocation.

The output fields:

```text
actionable
de_ready
decision_engine_handoff_allowed
```

remain false under this contract version.

A complete handoff evidence set is classified as structurally complete
coverage and at most `recommendation_review_ready`, with:

```text
actionable_contract_not_approved
decision_engine_handoff_blocked
```

This preserves the ME-SA12 rule that reserved authority states are modeled but
not activated. A future authority-contract change must not be smuggled into a
source-support classifier.

## 8. Test Matrix

| Test family | Expected result |
|---|---|
| Unsupported universe | `unsupported` / `blocked` |
| Unsupported source family | `unsupported` / `blocked` |
| Missing snapshot | `missing_snapshot` / `unavailable` |
| Invalid manifest | `invalid_manifest` / `blocked` |
| Missing provenance | `unprovenanced` / `blocked` |
| Stale required source | `stale` / `blocked` |
| Non-consumable source | `not_consumable` / `blocked` |
| Company-profile-only | `descriptive_only`; non-actionable |
| Partial family coverage | `partial` with explicit blockers |
| Complete fundamental analysis input | `analysis_ready`; non-actionable |
| Missing setup input | Recommendation Review blocked |
| Missing portfolio context | handoff blocked |
| Complete Recommendation Review input | `recommendation_review_ready`; non-actionable |
| Complete handoff evidence | reserved authority blockers; not DE-ready |
| Not-required source family | does not block |
| Invalid/duplicate input | fail-closed exception |
| Batch classification | deterministic order and counts |
| Plain-dict conversion | contract/audit fields preserved |
| Same evidence with different ticker strings | identical classification except ticker |

Targeted new tests:

```text
39 passed
```

Full source-support tests:

```text
63 passed
```

## 9. ME-RUN28 Regression Cases

ME-RUN28 names appear only in parameterized tests:

| Regression class | Fixture examples | Expected generic outcome |
|---|---|---|
| Valid company profile; analytical requirements absent | NVDA, AMD, ASML | `descriptive_only`; not actionable; not DE-ready |
| Missing cached snapshot | AAPL, GOOGL, AMZN, MU | `missing_snapshot` / `unavailable` |
| Fundamental evidence present; complementary requirements absent | AVGO, CLS, VRT, COST, META, MSFT, TSM, CRDO, IREN | `partial` with explicit blockers |

No production rule refers to any of those ticker values.

## 10. Anti-Ticker-Specific Governance

The explicit anti-regression test classifies:

```text
AAA
NVDA
FUTURE_XYZ
```

with identical coverage input and asserts equal output after removing only the
ticker field.

Required governance commands:

```text
rg -n 'ticker\s*==|ticker\s+in\s+\[|symbol\s*==|symbol\s+in\s+\[' src tests scripts docs/market_engine
rg -n 'NVDA|AMD|ASML|AVGO|CLS|VRT|COST|META|MSFT|AAPL|GOOGL|AMZN|TSM|MU|CRDO|IREN' src/market_engine scripts tests/market_engine docs/market_engine
```

Final results and interpretation are recorded before commit.

Results:

```text
ticker/symbol control-flow grep:
  src/market_engine: no hits
  new ME-SA13 runtime module: no hits
  src/market_scanner: three existing empty-string identity checks
  tests: fixture assertions, fixture lookup, and test-only data branches
  docs: ME-SA12 prohibited examples and governance text

ME-RUN28 ticker-value grep:
  new ME-SA13 runtime module: no hits
  existing source_acquisition runtime: bounded ticker allowlist and profile
    fixture mapping already identified by ME-RUN28
  existing end_to_end_dry_run_command: embedded command fixture examples
  tests/docs: fixture and regression evidence
  scripts: existing legacy command/help examples
```

The broad value grep also produces substring matches, including `MU` inside
unrelated uppercase identifiers. Such matches are not ticker logic.

Mandatory Decision Engine authority greps:

```text
BUY: existing scripts/portfolio legacy text and ignored bytecode only
SELL: existing scripts/portfolio legacy text and ignored bytecode only
tradeable: ignored bytecode only
```

ME-SA13 adds no runtime or script hit outside its generic module, and that
module contains no hard-coded ticker value.

## 11. Validation

Completed:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/source_support/test_cached_source_coverage.py -q
39 passed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine/source_support -q
63 passed

import smoke
PASS
```

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/market_engine -q
585 passed in 2.54s

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
1252 passed in 3.77s

git diff --check
PASS

governance greps
PASS - no new ticker-specific runtime logic

staged safety grep
PASS - matches are contract/audit text, generic enum values, boundary strings,
and tests; no provider/network, Telegram, broker/order, portfolio/watchlist,
or production-write behavior was added
```

## 12. Files Changed

Runtime:

```text
src/market_engine/source_support/cached_source_coverage.py
src/market_engine/source_support/__init__.py
```

Tests:

```text
tests/market_engine/source_support/test_cached_source_coverage.py
```

Documentation:

```text
docs/market_engine/audits/me_sa13_generic_cached_source_coverage_classification_model_audit.md
docs/market_engine/backlog/me_sa13_generic_cached_source_coverage_classification_model_backlog_entry.md
docs/market_engine/backlog/market_engine_backlog.md
docs/market_engine/roadmap/me_sa13_generic_cached_source_coverage_classification_model_roadmap_entry.md
docs/market_engine/roadmap/market_engine_roadmap.md
```

## 13. Known Limitations

* The classifier is not yet wired to staging/import validation artifacts.
* Callers must provide already interpreted evidence-gate statuses.
* The classifier does not read manifests or calculate freshness.
* No persistence or dry-run reporting adapter is included.
* Current `actionable` and `de_ready` states remain unreachable.
* Batch input uses one unique ticker identifier per classification.
* The classifier does not broaden provider or ticker coverage.

## 14. Recommended Next Sprint

```text
ME-SA14 - Adapt cached-source staging validation into generic coverage input
```

ME-SA14 should map existing staging-validation and manifest evidence into
`CachedSourceCoverageInput` without changing validation semantics. Dry-run
reporting integration should follow only after this adapter remains
deterministic and fail-closed.

## 15. Final Status

```text
PASS
```
