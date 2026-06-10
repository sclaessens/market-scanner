# BL89 — Decouple active SEC transform/review tests from script-era SEC transformation modules

Status: COMPLETED

## Purpose

BL89 decouples active SEC transform/review tests from script-era SEC transformation modules.

Target script-era modules:

* `scripts/fundamentals/run_sec_transformation_review.py`
* `scripts/fundamentals/sec_companyfacts_transform.py`
* `scripts/fundamentals/sec_ticker_cik_index.py`

BL88 removed active positive references to the script-era fundamentals analysis and quality modules. BL89 continues the cleanup by removing active positive test dependencies on SEC transform/review script-era module paths.

BL89 is a decoupling sprint. It does not archive, edit, execute, or delete script-era runtime modules.

## Scope

Updated tests:

* `tests/fundamentals/test_sec_companyfacts_transform.py`
* `tests/fundamentals/test_run_sec_transformation_review.py`
* `tests/fundamentals/test_sec_ticker_cik_index.py`

Reviewed related tests:

* `tests/unit/test_v2_sec_companyfacts_smoke_boundary.py`
* `tests/test_operator_visibility.py`

Out of scope:

* Archiving SEC transform/review script-era modules.
* Editing script-era runtime modules.
* Executing script-era runtime modules.
* Live SEC/EDGAR calls.
* Provider/cache writes.
* Production data writes.
* Production report generation.
* Telegram delivery.
* Portfolio/watchlist mutation.
* Decision Engine authority changes.

## Changes made

### `tests/fundamentals/test_sec_companyfacts_transform.py`

Replaced script-era module path assertions with canonical fundamentals history contract checks.

The test now validates:

* canonical required fundamentals history fields;
* source-evidence-only semantics;
* absence of investment authority fields such as allocation, tradeability, and final action.

### `tests/fundamentals/test_run_sec_transformation_review.py`

Replaced script-era module path assertions with canonical SEC source-review policy checks.

The test now validates:

* canonical SEC CompanyFacts source-family label;
* source-review statuses;
* absence of investment authority terms.

The canonical source-family value is:

```text
SEC EDGAR / SEC CompanyFacts
```

### `tests/fundamentals/test_sec_ticker_cik_index.py`

Replaced script-era module path assertions with canonical SEC CIK mapping policy checks.

The test now validates:

* canonical SEC CompanyFacts source-family label;
* CIK mapping statuses;
* absence of investment authority terms.

## Active-reference validation

A focused grep was run for SEC transform/review script-era names.

Remaining hits were limited to:

* negative guardrail entries in SEC smoke-boundary tests;
* operator-visibility references to test files.

Interpretation:

* no active positive dependency remains on `scripts/fundamentals/run_sec_transformation_review.py`;
* no active positive dependency remains on `scripts/fundamentals/sec_companyfacts_transform.py`;
* no active positive dependency remains on `scripts/fundamentals/sec_ticker_cik_index.py`;
* remaining references are not runtime-authority dependencies.

## Validation

Focused SEC-related tests:

```bash
pytest tests/fundamentals/test_sec_companyfacts_transform.py \
       tests/fundamentals/test_run_sec_transformation_review.py \
       tests/fundamentals/test_sec_ticker_cik_index.py \
       tests/unit/test_v2_sec_companyfacts_smoke_boundary.py \
       tests/test_operator_visibility.py -q
```

Result:

```text
40 passed in 0.05s
```

Full suite:

```bash
pytest -q
```

Result:

```text
551 passed in 0.58s
```

## Archive-readiness impact

BL89 removes the active positive SEC transform/review test blocker for:

* `scripts/fundamentals/run_sec_transformation_review.py`
* `scripts/fundamentals/sec_companyfacts_transform.py`
* `scripts/fundamentals/sec_ticker_cik_index.py`

BL89 does not yet archive the remaining script-era fundamentals cluster.

Remaining blockers:

* internal script-era imports still connect the remaining `scripts/fundamentals/` modules;
* `scripts/fundamentals/sec_companyfacts_bulk_intake.py` remains provider/network/cache-risk;
* `scripts/fundamentals/build_quality.py` remains high-risk because it contains production-like data/log write paths;
* a separate archive-readiness review is still required before moving any remaining files to `archive/legacy_runtime/`.

## Archive decision after BL89

| File                                                    | BL89 decision                                              | Reason                                                                                                          |
| ------------------------------------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `scripts/fundamentals/run_sec_transformation_review.py` | `ACTIVE_REFERENCE_DECOUPLED_BUT_CLUSTER_BLOCKED`           | Active positive test references removed, but the internal script-era dependency cluster still remains.          |
| `scripts/fundamentals/sec_companyfacts_transform.py`    | `ACTIVE_REFERENCE_DECOUPLED_BUT_CLUSTER_BLOCKED`           | Active positive test references removed, but the internal script-era dependency cluster still remains.          |
| `scripts/fundamentals/sec_ticker_cik_index.py`          | `ACTIVE_REFERENCE_DECOUPLED_BUT_CLUSTER_BLOCKED`           | Active positive test references removed, but internal SEC transform/review dependencies still remain.           |
| `scripts/fundamentals/sec_companyfacts_bulk_intake.py`  | `PROVIDER_SIDE_EFFECT_RISK`                                | Provider/network/cache-risk module; not touched by BL89.                                                        |
| `scripts/fundamentals/build_analysis.py`                | `ACTIVE_REFERENCE_DECOUPLED_BUT_CLUSTER_BLOCKED`           | Active positive references removed in BL88, but cluster dependencies remain.                                    |
| `scripts/fundamentals/build_quality.py`                 | `ACTIVE_REFERENCE_DECOUPLED_BUT_HIGH_RISK_CLUSTER_BLOCKED` | Active positive references removed in BL88, but production-like write behavior and cluster dependencies remain. |
| `scripts/fundamentals/build_history_intake.py`          | `CLUSTER_DEPENDENCY_BLOCKED`                               | Still part of internal script-era dependency cluster.                                                           |
| `scripts/fundamentals/build_metrics.py`                 | `CLUSTER_DEPENDENCY_BLOCKED`                               | Still part of internal script-era dependency cluster.                                                           |

## Decision

Do not archive SEC transform/review script-era modules yet.

BL89 completes active positive SEC transform/review test decoupling. The next sprint should review whether the remaining script-era fundamentals cluster can be archived as a governed cluster, or whether additional internal decoupling is still required first.

## Recommended next sprint

Recommended next sprint:

```text
BL90 — Final archive-readiness review of remaining scripts/fundamentals cluster
```

Candidate review targets:

* `scripts/fundamentals/build_analysis.py`
* `scripts/fundamentals/build_history_intake.py`
* `scripts/fundamentals/build_metrics.py`
* `scripts/fundamentals/build_quality.py`
* `scripts/fundamentals/run_sec_transformation_review.py`
* `scripts/fundamentals/sec_companyfacts_transform.py`
* `scripts/fundamentals/sec_ticker_cik_index.py`
* `scripts/fundamentals/sec_companyfacts_bulk_intake.py`

Goal:

* confirm no active positive `src`, `tests`, or `.github` references remain;
* classify remaining internal dependency cluster;
* decide whether files can be archived as a group;
* isolate provider/network/cache-risk files if needed;
* avoid executing script-era runtime modules.

## Guardrails

* No live SEC/EDGAR calls were run.
* No yfinance calls were run.
* No credentials were read.
* No production data was written.
* No production reports were generated.
* No Telegram messages were sent.
* No portfolio/watchlist production state was modified.
* No Decision Engine authority was changed.
* No script-era runtime module was executed.
* No script-era runtime module was edited.
* No script-era runtime file was archived.
* No script-era runtime file was deleted.

## Final status

BL89 completed active positive SEC transform/review test decoupling.

The remaining cleanup question is now whether the full remaining `scripts/fundamentals/` cluster is archive-ready as a governed cluster, with special handling for provider/network/cache-risk modules.
