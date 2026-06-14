# BL92 — Archive remaining scripts/fundamentals cluster after final no-active-reference check

Status: COMPLETED

## Purpose

BL92 archives the remaining `scripts/fundamentals/` script-era runtime cluster after the final no-active-reference check confirmed that active `src`, `tests`, and `.github` code no longer depends on these files.

This sprint follows the decoupling sequence:

* BL86 — decoupled active references to `build_history_intake.py` and `build_metrics.py`
* BL87 — reviewed internal script-era dependency clustering
* BL88 — decoupled active references to `build_analysis.py` and `build_quality.py`
* BL89 — decoupled active SEC transform/review tests
* BL90 — final archive-readiness review found the remaining bulk-intake blocker
* BL91 — decoupled the final active bulk SEC CompanyFacts intake test reference

BL92 is an archive sprint. It archives files by moving them to `archive/legacy_runtime/`. It does not execute or modify script-era runtime behavior.

## Final pre-archive checks

### Remaining files before archive

BL92 confirmed the remaining active script-era fundamentals files before archive:

```text
scripts/fundamentals/build_analysis.py
scripts/fundamentals/build_history_intake.py
scripts/fundamentals/build_metrics.py
scripts/fundamentals/build_quality.py
scripts/fundamentals/run_sec_transformation_review.py
scripts/fundamentals/sec_companyfacts_bulk_intake.py
scripts/fundamentals/sec_companyfacts_transform.py
scripts/fundamentals/sec_ticker_cik_index.py
```

### Active path reference check

BL92 checked active references from `src`, `tests`, and `.github` to the remaining script-era file paths.

Result:

```text
No output
```

Interpretation:

* no active positive references remain to the remaining `scripts/fundamentals/*.py` file paths.

### Active script import check

BL92 checked active `scripts.fundamentals` imports from `src`, `tests`, and `.github`.

Result:

```text
No output
```

Interpretation:

* no active positive `scripts.fundamentals` imports remain in active source, tests, or workflows.

## Archived files

BL92 moved the remaining script-era fundamentals cluster to `archive/legacy_runtime/scripts/fundamentals/`:

```text
scripts/fundamentals/build_analysis.py -> archive/legacy_runtime/scripts/fundamentals/build_analysis.py
scripts/fundamentals/build_history_intake.py -> archive/legacy_runtime/scripts/fundamentals/build_history_intake.py
scripts/fundamentals/build_metrics.py -> archive/legacy_runtime/scripts/fundamentals/build_metrics.py
scripts/fundamentals/build_quality.py -> archive/legacy_runtime/scripts/fundamentals/build_quality.py
scripts/fundamentals/run_sec_transformation_review.py -> archive/legacy_runtime/scripts/fundamentals/run_sec_transformation_review.py
scripts/fundamentals/sec_companyfacts_bulk_intake.py -> archive/legacy_runtime/scripts/fundamentals/sec_companyfacts_bulk_intake.py
scripts/fundamentals/sec_companyfacts_transform.py -> archive/legacy_runtime/scripts/fundamentals/sec_companyfacts_transform.py
scripts/fundamentals/sec_ticker_cik_index.py -> archive/legacy_runtime/scripts/fundamentals/sec_ticker_cik_index.py
```

The archived folder now contains:

```text
archive/legacy_runtime/scripts/fundamentals/__init__.py
archive/legacy_runtime/scripts/fundamentals/build_analysis.py
archive/legacy_runtime/scripts/fundamentals/build_history_intake.py
archive/legacy_runtime/scripts/fundamentals/build_metrics.py
archive/legacy_runtime/scripts/fundamentals/build_quality.py
archive/legacy_runtime/scripts/fundamentals/run_sec_transformation_review.py
archive/legacy_runtime/scripts/fundamentals/sec_companyfacts_bulk_intake.py
archive/legacy_runtime/scripts/fundamentals/sec_companyfacts_transform.py
archive/legacy_runtime/scripts/fundamentals/sec_ticker_cik_index.py
```

The `__init__.py` file was already archived in an earlier sprint.

## Post-archive active scripts/fundamentals check

BL92 confirmed that the active `scripts/fundamentals/` location no longer contains Python files.

Result:

```text
No active scripts/fundamentals/*.py files remain.
```

## Validation

Focused regression suite:

```bash
pytest tests/fundamentals/test_sec_companyfacts_bulk_intake.py \
       tests/fundamentals/test_sec_companyfacts_transform.py \
       tests/fundamentals/test_run_sec_transformation_review.py \
       tests/fundamentals/test_sec_ticker_cik_index.py \
       tests/unit/test_v2_sec_companyfacts_smoke_boundary.py \
       tests/test_operator_visibility.py \
       tests/unit/test_v2_canonical_analysis.py \
       tests/core/test_fundamentals_runtime_organization.py \
       tests/core/test_fundamentals_operational_validation.py \
       tests/core/test_build_fundamental_analysis.py \
       tests/core/test_build_fundamental_layer.py \
       tests/contract/test_v2_fundamentals_metrics_contracts.py \
       tests/contract/test_v2_fundamental_history_validation_contracts.py -q
```

Result:

```text
89 passed in 0.07s
```

Full suite:

```bash
pytest -q
```

Result:

```text
553 passed in 0.58s
```

## Archive decision

BL92 decision:

```text
ARCHIVED
```

The remaining `scripts/fundamentals/` cluster is now archived under:

```text
archive/legacy_runtime/scripts/fundamentals/
```

## Impact

After BL92:

* active `src`, `tests`, and `.github` code no longer depends on `scripts/fundamentals/*.py`;
* active `scripts/fundamentals/` no longer contains Python runtime files;
* canonical fundamentals and analysis contracts live under `src/market_scanner/`;
* historical script-era implementation remains preserved in `archive/legacy_runtime/`;
* provider-risk SEC CompanyFacts bulk-intake code is preserved as historical legacy evidence only, not active runtime.

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
* No script-era runtime behavior was modified.
* Files were archived, not deleted.

## Recommended next sprint

Recommended next sprint:

```text
BL93 — Review remaining active scripts/ tree after fundamentals archive
```

Goal:

* inspect remaining `scripts/**/*.py` files;
* confirm which script-era files still exist outside `scripts/fundamentals/`;
* classify remaining files as active, legacy, archive-ready, or blocked;
* avoid runtime execution;
* continue reducing `scripts/` to non-runtime governance or archived historical material.
