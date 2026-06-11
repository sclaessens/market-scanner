# BL91 — Decouple active bulk SEC CompanyFacts intake test from provider-risk script-era module

Status: COMPLETED

## Purpose

BL91 decouples the final active positive test reference to the provider-risk script-era bulk SEC CompanyFacts intake module:

* `scripts/fundamentals/sec_companyfacts_bulk_intake.py`

BL90 determined that the remaining `scripts/fundamentals/` cluster was not archive-ready because `tests/fundamentals/test_sec_companyfacts_bulk_intake.py` still contained a positive script-era module path reference.

BL91 removes that active blocker while preserving provider-risk governance.

BL91 is a decoupling sprint. It does not archive, edit, execute, or delete script-era runtime modules.

## Scope

Updated test:

* `tests/fundamentals/test_sec_companyfacts_bulk_intake.py`

Out of scope:

* Archiving `scripts/fundamentals/sec_companyfacts_bulk_intake.py`
* Editing script-era runtime modules
* Executing script-era runtime modules
* Live SEC/EDGAR calls
* Provider/cache writes
* Production data writes
* Production report generation
* Telegram delivery
* Portfolio/watchlist mutation
* Decision Engine authority changes

## Changes made

### `tests/fundamentals/test_sec_companyfacts_bulk_intake.py`

Before BL91, the test contained a positive script-era module path reference:

```text id="cjgjum"
LEGACY_MODULE_PATH = Path("scripts/fundamentals/sec_companyfacts_bulk_intake.py")
```

After BL91, the test no longer references the script-era file path.

The test now validates canonical provider-governance policy:

* SEC CompanyFacts source family remains explicit;
* SEC bulk source URL remains known as provider evidence;
* network access requires explicit operator action;
* cache writes require explicit operator action;
* manifest writes require explicit operator action;
* tests may not download SEC bulk data;
* tests may not write SEC bulk cache;
* no investment authority is introduced.

The canonical source-family value remains:

```text id="sp164c"
SEC EDGAR / SEC CompanyFacts
```

## Active-reference validation

A focused grep was run:

```bash id="fcof3l"
grep -RIn \
  "scripts/fundamentals/sec_companyfacts_bulk_intake.py\|sec_companyfacts_bulk_intake" \
  src tests .github \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=__pycache__ \
  --exclude="*.pyc"
```

Observed remaining hits:

```text id="br71vy"
tests/fundamentals/test_sec_companyfacts_bulk_intake.py
tests/unit/test_v2_sec_companyfacts_smoke_boundary.py
tests/test_operator_visibility.py
```

Interpretation:

* remaining hits in `test_sec_companyfacts_bulk_intake.py` are canonical policy test names;
* remaining hit in `test_v2_sec_companyfacts_smoke_boundary.py` is a negative `FORBIDDEN_SCRIPT_IMPORTS` guardrail;
* remaining hit in `test_operator_visibility.py` is an operator-visibility reference to the test file;
* no positive `scripts/fundamentals/sec_companyfacts_bulk_intake.py` path reference remains.

## Validation

Focused bulk-intake test:

```bash id="q2ey3r"
pytest tests/fundamentals/test_sec_companyfacts_bulk_intake.py -q
```

Result:

```text id="i0y6x4"
4 passed in 0.02s
```

BL90/BL91 focused suite:

```bash id="z6ho4r"
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

```text id="n02qac"
89 passed in 0.07s
```

Full suite:

```bash id="scsnzv"
pytest -q
```

Result:

```text id="c8bx4s"
553 passed in 0.57s
```

## Archive-readiness impact

BL91 removes the final active positive test reference to:

* `scripts/fundamentals/sec_companyfacts_bulk_intake.py`

After BL91, the remaining script-era fundamentals files appear to have no active positive `src`, `tests`, or `.github` references by module path.

However, BL91 does not archive files. A final no-active-reference check is still required before archive.

## File-level decisions after BL91

| File                                                    | BL91 decision                                           | Reason                                                                                    |
| ------------------------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `scripts/fundamentals/sec_companyfacts_bulk_intake.py`  | `ACTIVE_REFERENCE_DECOUPLED_PROVIDER_RISK_REMAINS`      | Positive path reference removed, but provider/network/cache-risk remains.                 |
| `scripts/fundamentals/build_analysis.py`                | `CANDIDATE_FOR_CLUSTER_ARCHIVE_AFTER_FINAL_CHECK`       | Active positive references decoupled in BL88.                                             |
| `scripts/fundamentals/build_history_intake.py`          | `CANDIDATE_FOR_CLUSTER_ARCHIVE_AFTER_FINAL_CHECK`       | Active positive references decoupled in BL86.                                             |
| `scripts/fundamentals/build_metrics.py`                 | `CANDIDATE_FOR_CLUSTER_ARCHIVE_AFTER_FINAL_CHECK`       | Active positive references decoupled in BL86.                                             |
| `scripts/fundamentals/build_quality.py`                 | `HIGH_RISK_CLUSTER_ARCHIVE_CANDIDATE_AFTER_FINAL_CHECK` | Active positive references decoupled in BL88, but production-like write behavior remains. |
| `scripts/fundamentals/run_sec_transformation_review.py` | `CANDIDATE_FOR_CLUSTER_ARCHIVE_AFTER_FINAL_CHECK`       | Active positive references decoupled in BL89.                                             |
| `scripts/fundamentals/sec_companyfacts_transform.py`    | `CANDIDATE_FOR_CLUSTER_ARCHIVE_AFTER_FINAL_CHECK`       | Active positive references decoupled in BL89.                                             |
| `scripts/fundamentals/sec_ticker_cik_index.py`          | `CANDIDATE_FOR_CLUSTER_ARCHIVE_AFTER_FINAL_CHECK`       | Active positive references decoupled in BL89.                                             |

## Decision

BL91 is complete.

Do not archive in BL91.

The next sprint should run a final no-active-reference check and then archive the remaining `scripts/fundamentals/` cluster if the check confirms no active positive references remain.

## Recommended next sprint

Recommended next sprint:

```text id="o1rds8"
BL92 — Archive remaining scripts/fundamentals cluster after final no-active-reference check
```

Candidate archive targets:

* `scripts/fundamentals/build_analysis.py`
* `scripts/fundamentals/build_history_intake.py`
* `scripts/fundamentals/build_metrics.py`
* `scripts/fundamentals/build_quality.py`
* `scripts/fundamentals/run_sec_transformation_review.py`
* `scripts/fundamentals/sec_companyfacts_bulk_intake.py`
* `scripts/fundamentals/sec_companyfacts_transform.py`
* `scripts/fundamentals/sec_ticker_cik_index.py`

Required BL92 guardrail:

* run final no-active-positive-reference check before moving files;
* do not execute script-era runtime modules;
* archive, do not delete;
* keep audit trail in `docs/audits/legacy_runtime/`;
* update backlog.

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

BL91 removed the final active positive test reference to the provider-risk bulk SEC CompanyFacts intake script-era module.

The remaining `scripts/fundamentals/` cluster is now a candidate for archive after one final no-active-reference check.
