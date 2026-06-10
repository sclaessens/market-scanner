# BL88 — Decouple active tests from remaining script-era fundamentals analysis and quality modules

Status: COMPLETED

## Purpose

BL88 decouples active tests and analysis metadata from the remaining script-era fundamentals analysis and quality modules:

* `scripts/fundamentals/build_analysis.py`
* `scripts/fundamentals/build_quality.py`

BL87 confirmed that `build_history_intake.py` and `build_metrics.py` should not be archived in isolation because they remain part of a broader internal script-era dependency cluster.

BL88 addresses the next blocker: active positive references to the analysis and quality script-era modules.

BL88 is a decoupling sprint. It does not archive, edit, execute, or delete script-era runtime modules.

## Scope

Updated source metadata:

* `src/market_scanner/analysis/analysis_boundary.py`

Updated tests:

* `tests/unit/test_v2_canonical_analysis.py`
* `tests/core/test_fundamentals_runtime_organization.py`
* `tests/core/test_fundamentals_operational_validation.py`
* `tests/core/test_build_fundamental_analysis.py`
* `tests/core/test_build_fundamental_layer.py`

Out of scope:

* Archiving `scripts/fundamentals/build_analysis.py`.
* Archiving `scripts/fundamentals/build_quality.py`.
* Editing script-era runtime modules.
* Executing script-era runtime modules.
* Decoupling SEC transform/review tests.
* Provider calls.
* Production data writes.
* Production report generation.
* Telegram delivery.
* Portfolio/watchlist mutation.
* Decision Engine authority changes.

## Changes made

### `src/market_scanner/analysis/analysis_boundary.py`

BL88 removed the remaining legacy-analysis authority entries.

Before BL88, canonical analysis metadata still listed:

```text
scripts/fundamentals/build_analysis.py
scripts/fundamentals/build_quality.py
```

After BL88:

```text
LEGACY_ANALYSIS_AUTHORITIES = ()
```

The boundary now tracks migrated analysis contract authorities through canonical modules:

```text
src/market_scanner/analysis/analysis_boundary.py
src/market_scanner/analysis/analysis_contracts.py
```

It also continues to track migrated fundamentals contract authorities:

```text
src/market_scanner/fundamentals/fundamental_contracts.py
src/market_scanner/fundamentals/fundamentals_metrics_contracts.py
```

### `tests/unit/test_v2_canonical_analysis.py`

Updated canonical analysis tests so they no longer read or depend on:

* `scripts/fundamentals/build_analysis.py`
* `scripts/fundamentals/build_quality.py`

The test now validates:

* canonical analysis owner;
* migrated fundamentals contract authorities;
* migrated analysis contract authorities;
* no remaining legacy script authorities;
* no script-era imports;
* no side effects;
* no final investment behavior.

### `tests/core/test_fundamentals_runtime_organization.py`

Updated runtime organization checks so they validate canonical ownership instead of script-era entrypoint requirements.

Canonical modules now checked include:

* `market_scanner.fundamentals.fundamental_contracts`
* `market_scanner.fundamentals.fundamentals_metrics_contracts`
* `market_scanner.analysis.analysis_boundary`
* `market_scanner.analysis.analysis_contracts`

### `tests/core/test_fundamentals_operational_validation.py`

Updated operational validation checks so they refer only to canonical contract paths under `src/market_scanner/`.

The test no longer lists script-era analysis or quality modules as operational requirements.

### `tests/core/test_build_fundamental_analysis.py`

Replaced script-era `build_analysis.py` path checks with canonical analysis boundary checks.

The test now validates:

* canonical analysis ownership;
* evidence review stage names;
* absence of final decision authority.

### `tests/core/test_build_fundamental_layer.py`

Replaced script-era `build_quality.py` path checks with canonical fundamentals contract checks.

The test now validates:

* canonical fundamentals dataset roles;
* source-data readiness states;
* absence of final action/allocation/position-size fields.

## Active-reference validation

A focused grep was run:

```bash
grep -RIn \
  "scripts/fundamentals/build_analysis.py\|scripts/fundamentals/build_quality.py\|build_analysis\|build_quality" \
  src tests .github \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

Observed remaining hits:

* canonical function names such as `build_analysis_plan`;
* canonical tests importing `build_analysis_plan`;
* one negative guardrail entry: `scripts.fundamentals.build_quality` inside `FORBIDDEN_SCRIPT_IMPORTS`.

Interpretation:

* no active positive dependency on `scripts/fundamentals/build_analysis.py` remains;
* no active positive dependency on `scripts/fundamentals/build_quality.py` remains;
* remaining hits are canonical names or negative guardrails.

## Validation

Focused tests:

```bash
pytest tests/unit/test_v2_canonical_analysis.py \
       tests/core/test_fundamentals_runtime_organization.py \
       tests/core/test_fundamentals_operational_validation.py \
       tests/core/test_build_fundamental_analysis.py \
       tests/core/test_build_fundamental_layer.py -q
```

Result:

```text
20 passed in 0.03s
```

Full suite:

```bash
pytest -q
```

Result:

```text
551 passed in 0.55s
```

## Archive-readiness impact

BL88 removes the active source/test metadata blocker for:

* `scripts/fundamentals/build_analysis.py`
* `scripts/fundamentals/build_quality.py`

However, BL88 does not yet make the full `scripts/fundamentals/` cluster archive-ready.

Remaining blockers:

* SEC transform/review script-era tests still reference `run_sec_transformation_review.py` and `sec_companyfacts_transform.py`;
* `build_quality.py` still has high-risk production-like write behavior;
* internal script-era imports still exist inside the remaining `scripts/fundamentals/` cluster.

## Archive decision after BL88

| File                                           | BL88 decision                                              | Reason                                                                                                                   |
| ---------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `scripts/fundamentals/build_analysis.py`       | `ACTIVE_REFERENCE_DECOUPLED_BUT_CLUSTER_BLOCKED`           | Active positive test/source references removed, but internal cluster dependencies and archive sequencing remain.         |
| `scripts/fundamentals/build_quality.py`        | `ACTIVE_REFERENCE_DECOUPLED_BUT_HIGH_RISK_CLUSTER_BLOCKED` | Active positive test/source references removed, but production-like write risk and internal cluster dependencies remain. |
| `scripts/fundamentals/build_history_intake.py` | `CLUSTER_DEPENDENCY_BLOCKED`                               | Still part of internal script-era dependency cluster.                                                                    |
| `scripts/fundamentals/build_metrics.py`        | `CLUSTER_DEPENDENCY_BLOCKED`                               | Still part of internal script-era dependency cluster.                                                                    |

## Decision

Do not archive analysis/quality script-era modules yet.

BL88 completes active positive reference decoupling for analysis and quality modules. The next cleanup step should target SEC transform/review active tests and remaining SEC-boundary references.

## Recommended next sprint

Recommended next sprint:

```text
BL89 — Decouple active SEC transform/review tests from script-era SEC transformation modules
```

Candidate files/tests:

* `tests/fundamentals/test_sec_companyfacts_transform.py`
* `tests/fundamentals/test_run_sec_transformation_review.py`
* `tests/unit/test_v2_sec_companyfacts_smoke_boundary.py`
* `tests/test_operator_visibility.py`

Candidate script-era modules to decouple from active positive references:

* `scripts/fundamentals/run_sec_transformation_review.py`
* `scripts/fundamentals/sec_companyfacts_transform.py`
* `scripts/fundamentals/sec_ticker_cik_index.py`

BL89 should avoid editing or executing script-era runtime modules.

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

BL88 completed active positive reference decoupling for script-era fundamentals analysis and quality modules.

The remaining cleanup blocker is now concentrated around SEC transform/review tests and the remaining internal script-era dependency cluster.
