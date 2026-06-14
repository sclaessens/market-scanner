# BL90 — Final archive-readiness review of remaining scripts/fundamentals cluster

Status: COMPLETED

## Purpose

BL90 performs a final archive-readiness review of the remaining `scripts/fundamentals/` cluster after BL86, BL87, BL88, and BL89 decoupled active positive references from most script-era fundamentals modules.

BL90 is a review sprint. It does not archive, edit, execute, or delete script-era runtime modules.

## Scope

Reviewed remaining script-era fundamentals files:

* `scripts/fundamentals/build_analysis.py`
* `scripts/fundamentals/build_history_intake.py`
* `scripts/fundamentals/build_metrics.py`
* `scripts/fundamentals/build_quality.py`
* `scripts/fundamentals/run_sec_transformation_review.py`
* `scripts/fundamentals/sec_companyfacts_bulk_intake.py`
* `scripts/fundamentals/sec_companyfacts_transform.py`
* `scripts/fundamentals/sec_ticker_cik_index.py`

Out of scope:

* Archiving any script-era runtime file.
* Editing script-era runtime modules.
* Executing script-era runtime modules.
* Live SEC/EDGAR calls.
* Provider/cache writes.
* Production data writes.
* Production report generation.
* Telegram delivery.
* Portfolio/watchlist mutation.
* Decision Engine authority changes.

## Remaining files

BL90 confirmed the remaining `scripts/fundamentals/*.py` files:

```text id="cy6o4i"
scripts/fundamentals/build_analysis.py
scripts/fundamentals/build_history_intake.py
scripts/fundamentals/build_metrics.py
scripts/fundamentals/build_quality.py
scripts/fundamentals/run_sec_transformation_review.py
scripts/fundamentals/sec_companyfacts_bulk_intake.py
scripts/fundamentals/sec_companyfacts_transform.py
scripts/fundamentals/sec_ticker_cik_index.py
```

## Active source/test/.github reference review

BL90 checked active references from `src`, `tests`, and `.github` to the remaining script-era fundamentals files.

Finding:

```text id="ht9lia"
tests/fundamentals/test_sec_companyfacts_bulk_intake.py:6:LEGACY_MODULE_PATH = Path("scripts/fundamentals/sec_companyfacts_bulk_intake.py")
```

Interpretation:

* active positive references to `build_analysis.py` were already decoupled in BL88;
* active positive references to `build_quality.py` were already decoupled in BL88;
* active positive references to SEC transform/review/index files were decoupled in BL89;
* one active positive test reference remains for `sec_companyfacts_bulk_intake.py`.

Therefore the full remaining `scripts/fundamentals/` cluster is not archive-ready yet.

## Internal dependency review

BL90 confirmed the internal script-era dependency cluster remains:

```text id="a8dwwn"
scripts/fundamentals/build_analysis.py imports from scripts.fundamentals.build_metrics
scripts/fundamentals/run_sec_transformation_review.py imports from scripts.fundamentals.build_history_intake
scripts/fundamentals/run_sec_transformation_review.py imports from scripts.fundamentals.sec_companyfacts_transform
scripts/fundamentals/run_sec_transformation_review.py imports from scripts.fundamentals.sec_ticker_cik_index
scripts/fundamentals/build_metrics.py imports from scripts.fundamentals.build_history_intake
scripts/fundamentals/build_quality.py imports from scripts.fundamentals.build_metrics
scripts/fundamentals/build_quality.py imports from scripts.fundamentals.build_history_intake
scripts/fundamentals/sec_companyfacts_transform.py imports from scripts.fundamentals.build_history_intake
scripts/fundamentals/sec_companyfacts_transform.py imports from scripts.fundamentals.sec_ticker_cik_index
```

Interpretation:

* internal dependencies remain contained inside `scripts/fundamentals/`;
* these internal dependencies do not by themselves block archive if the entire governed cluster is archived together;
* however, the remaining active positive bulk-intake test reference still blocks full archive.

## Side-effect and provider-risk review

BL90 reviewed side-effect and provider-risk patterns across the remaining cluster.

### `build_analysis.py`

Observed:

* `argparse`
* `pd.read_csv(...)`
* `Path(output_path).parent.mkdir(...)`
* `analysis_df.to_csv(...)`
* `main()`
* `if __name__ == "__main__"`

Classification:

```text id="j43qoy"
SCRIPT_ENTRYPOINT
CSV_READ
DIRECTORY_CREATE
OPTIONAL_CSV_WRITE
```

### `build_history_intake.py`

Observed:

* `argparse`
* `Path(input_path)`
* `pd.read_csv(...)`
* `Path(args.report_path).write_text(...)`
* `main()`
* `if __name__ == "__main__"`

Classification:

```text id="2ifxmm"
SCRIPT_ENTRYPOINT
CSV_READ
OPTIONAL_REPORT_WRITE
```

### `build_metrics.py`

Observed:

* `argparse`
* `pd.read_csv(...)`
* `Path(output_path).parent.mkdir(...)`
* `metrics_df.to_csv(...)`
* `main()`
* `if __name__ == "__main__"`

Classification:

```text id="7qruj3"
SCRIPT_ENTRYPOINT
CSV_READ
DIRECTORY_CREATE
OPTIONAL_CSV_WRITE
```

### `build_quality.py`

Observed:

* `argparse`
* default production-like paths:

  * `data/processed/context_strength.csv`
  * `data/raw/fundamentals.csv`
  * `data/processed/fundamental_quality.csv`
  * `data/logs/fundamental_layer_log.csv`
* `pd.read_csv(...)`
* `LOG_PATH.parent.mkdir(...)`
* `to_csv(LOG_PATH)`
* `OUTPUT_PATH.parent.mkdir(...)`
* `output_df.to_csv(OUTPUT_PATH)`
* `main()`
* `if __name__ == "__main__"`

Classification:

```text id="jzavmf"
SCRIPT_ENTRYPOINT
CSV_READ
DEFAULT_PRODUCTION_DATA_PATHS
DIRECTORY_CREATE
PRODUCTION_CSV_WRITE
LOG_CSV_WRITE
HIGH_RISK_LEGACY_ARTIFACT
```

### `run_sec_transformation_review.py`

Observed:

* `argparse`
* `Path(companyfacts_dir)`
* `Path(output_path)`
* `output.parent.mkdir(...)`
* `review_df.to_csv(...)`
* `main(...)`
* `if __name__ == "__main__"`

Classification:

```text id="qv7tfi"
SCRIPT_ENTRYPOINT
LOCAL_SEC_REVIEW_INPUT
DIRECTORY_CREATE
OPTIONAL_REVIEW_CSV_WRITE
```

### `sec_companyfacts_transform.py`

Observed:

* `argparse`
* `Path(path)`
* `Path(output_path)`
* `output.parent.mkdir(...)`
* `df.to_csv(...)`
* `main(...)`
* `if __name__ == "__main__"`

Classification:

```text id="s8h8h8"
SCRIPT_ENTRYPOINT
LOCAL_SEC_JSON_READ
DIRECTORY_CREATE
OPTIONAL_CSV_WRITE
```

### `sec_ticker_cik_index.py`

Observed:

* `argparse`
* `Path(path)`
* `pd.read_csv(...)`
* `Path(output_path)`
* `output.parent.mkdir(...)`
* `coverage_df.to_csv(...)`
* `main()`
* `if __name__ == "__main__"`

Classification:

```text id="7cw542"
SCRIPT_ENTRYPOINT
CSV_READ
DIRECTORY_CREATE
OPTIONAL_COVERAGE_CSV_WRITE
```

### `sec_companyfacts_bulk_intake.py`

Observed:

* `argparse`
* `urllib.error`
* `urllib.parse`
* `urllib.request.Request`
* `urllib.request.urlopen`
* default cache path: `data/local/sec_edgar/companyfacts`
* cache directory creation
* `Path(...).open("rb")`
* manifest `write_text(...)`
* target directory creation
* `urlopen(...)`
* binary file write
* `main()`
* `if __name__ == "__main__"`

Classification:

```text id="iwj9rg"
SCRIPT_ENTRYPOINT
PROVIDER_NETWORK_RISK
SEC_EDGAR_BULK_DOWNLOAD_RISK
LOCAL_CACHE_WRITE_RISK
MANIFEST_WRITE_RISK
BINARY_FILE_WRITE_RISK
REQUIRES_EXPLICIT_OPERATOR_ACTION
```

## Bulk-intake test review

BL90 inspected `tests/fundamentals/test_sec_companyfacts_bulk_intake.py`.

Current test content still includes:

```text id="13zeur"
LEGACY_MODULE_PATH = Path("scripts/fundamentals/sec_companyfacts_bulk_intake.py")
SEC_COMPANYFACTS_BULK_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
```

The test still validates the script-era path:

```text id="82b5lf"
assert LEGACY_MODULE_PATH.parts == ("scripts", "fundamentals", "sec_companyfacts_bulk_intake.py")
```

Interpretation:

* this is an active positive test reference;
* it blocks archive readiness for `sec_companyfacts_bulk_intake.py`;
* because `sec_companyfacts_bulk_intake.py` is also a provider/network/cache-risk module, it should be decoupled in a separate sprint before cluster archive is attempted.

## Validation

Focused review suite:

```bash id="l8aqzq"
pytest tests/fundamentals/test_sec_companyfacts_transform.py \
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

```text id="ln68lv"
85 passed in 0.08s
```

Full suite:

```bash id="aacps5"
pytest -q
```

Result:

```text id="c3oi3k"
551 passed in 0.67s
```

## Archive-readiness decision

BL90 decision:

```text id="run1ns"
NOT_ARCHIVE_READY_YET
```

Reason:

* one active positive test reference remains for `scripts/fundamentals/sec_companyfacts_bulk_intake.py`;
* `sec_companyfacts_bulk_intake.py` is provider/network/cache-risk;
* cluster archive should not proceed until bulk-intake provider-risk test is decoupled.

## File-level decisions after BL90

| File                                                    | BL90 decision                                                      | Reason                                                                             |
| ------------------------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| `scripts/fundamentals/build_analysis.py`                | `CANDIDATE_FOR_CLUSTER_ARCHIVE_AFTER_BULK_INTAKE_DECOUPLING`       | Active positive references decoupled; internal cluster dependency remains.         |
| `scripts/fundamentals/build_history_intake.py`          | `CANDIDATE_FOR_CLUSTER_ARCHIVE_AFTER_BULK_INTAKE_DECOUPLING`       | Active positive references decoupled; internal cluster dependency remains.         |
| `scripts/fundamentals/build_metrics.py`                 | `CANDIDATE_FOR_CLUSTER_ARCHIVE_AFTER_BULK_INTAKE_DECOUPLING`       | Active positive references decoupled; internal cluster dependency remains.         |
| `scripts/fundamentals/build_quality.py`                 | `HIGH_RISK_CLUSTER_ARCHIVE_CANDIDATE_AFTER_BULK_INTAKE_DECOUPLING` | Active positive references decoupled, but has production-like write behavior.      |
| `scripts/fundamentals/run_sec_transformation_review.py` | `CANDIDATE_FOR_CLUSTER_ARCHIVE_AFTER_BULK_INTAKE_DECOUPLING`       | Active positive references decoupled; internal cluster dependency remains.         |
| `scripts/fundamentals/sec_companyfacts_transform.py`    | `CANDIDATE_FOR_CLUSTER_ARCHIVE_AFTER_BULK_INTAKE_DECOUPLING`       | Active positive references decoupled; internal cluster dependency remains.         |
| `scripts/fundamentals/sec_ticker_cik_index.py`          | `CANDIDATE_FOR_CLUSTER_ARCHIVE_AFTER_BULK_INTAKE_DECOUPLING`       | Active positive references decoupled; internal cluster dependency remains.         |
| `scripts/fundamentals/sec_companyfacts_bulk_intake.py`  | `NOT_ARCHIVE_READY_PROVIDER_RISK_ACTIVE_TEST_REFERENCE`            | Active positive test reference remains and provider/network/cache-risk is present. |

## Recommended next sprint

Recommended next sprint:

```text id="zcm81z"
BL91 — Decouple active bulk SEC CompanyFacts intake test from provider-risk script-era module
```

Candidate test:

* `tests/fundamentals/test_sec_companyfacts_bulk_intake.py`

Candidate canonical policy target:

* SEC CompanyFacts provider/source-readiness policy;
* explicit operator-action requirement;
* no network execution during tests;
* no cache writes during tests;
* no script-era module-path dependency.

Goal:

* remove active positive reference to `scripts/fundamentals/sec_companyfacts_bulk_intake.py`;
* preserve provider-risk governance;
* avoid executing or editing the script-era provider module.

Likely follow-up after BL91:

```text id="wsagc7"
BL92 — Archive remaining scripts/fundamentals cluster after final no-active-reference check
```

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

BL90 confirms that the remaining `scripts/fundamentals/` cluster is close to archive-ready but still blocked.

The only active positive test reference remaining is for the provider-risk script-era bulk SEC CompanyFacts intake module.

The next sprint should decouple the bulk-intake test from the script-era module path before any cluster archive sprint is attempted.
