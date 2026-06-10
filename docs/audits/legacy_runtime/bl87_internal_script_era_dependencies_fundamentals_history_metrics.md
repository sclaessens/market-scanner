# BL87 — Review internal script-era dependencies on fundamentals history and metrics modules

Status: COMPLETED

## Purpose

BL87 reviews the remaining internal `scripts/fundamentals/` dependencies that still block archiving:

* `scripts/fundamentals/build_history_intake.py`
* `scripts/fundamentals/build_metrics.py`

BL86 removed active source/test metadata dependencies from these files. BL87 verifies what still blocks archive inside the remaining script-era fundamentals tree.

BL87 is a review sprint. It does not archive, edit, execute, or delete script-era runtime modules.

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

Reviewed canonical coverage:

* `src/market_scanner/fundamentals/fundamental_contracts.py`
* `src/market_scanner/fundamentals/fundamentals_metrics_contracts.py`
* `docs/audits/legacy_runtime/bl83_canonical_fundamentals_metrics_contract_tests.md`
* `docs/audits/legacy_runtime/bl84_canonical_fundamentals_history_validation_contract_tests.md`
* `docs/audits/legacy_runtime/bl85_archive_readiness_review_fundamentals_history_and_metrics.md`
* `docs/audits/legacy_runtime/bl86_decouple_active_tests_and_metadata_from_script_era_fundamentals_history_metrics.md`

Out of scope:

* Archiving script-era files.
* Editing script-era runtime modules.
* Executing script-era runtime modules.
* Provider calls.
* Production data writes.
* Production report generation.
* Telegram delivery.
* Portfolio/watchlist mutation.
* Decision Engine authority changes.

## Remaining script-era fundamentals files

BL87 confirmed the remaining script-era fundamentals files:

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

## Internal dependency findings

### Direct imports into `build_history_intake.py` and `build_metrics.py`

BL87 confirmed the following internal script-era imports:

```text
scripts/fundamentals/build_analysis.py: imports from scripts.fundamentals.build_metrics
scripts/fundamentals/run_sec_transformation_review.py: imports REQUIRED_COLUMNS from scripts.fundamentals.build_history_intake
scripts/fundamentals/build_metrics.py: imports validate_fundamentals_history from scripts.fundamentals.build_history_intake
scripts/fundamentals/build_quality.py: imports from scripts.fundamentals.build_metrics
scripts/fundamentals/build_quality.py: imports validate_fundamentals_history from scripts.fundamentals.build_history_intake
scripts/fundamentals/sec_companyfacts_transform.py: imports REQUIRED_COLUMNS from scripts.fundamentals.build_history_intake
```

This confirms that both target files still act as internal dependency anchors inside `scripts/fundamentals/`.

### Full internal `scripts.fundamentals` import graph

BL87 also confirmed additional internal script-era imports:

```text
scripts/fundamentals/run_sec_transformation_review.py imports sec_companyfacts_transform
scripts/fundamentals/run_sec_transformation_review.py imports sec_ticker_cik_index
scripts/fundamentals/sec_companyfacts_transform.py imports sec_ticker_cik_index
```

This means the remaining script-era fundamentals files form a connected internal legacy cluster rather than isolated files.

## Side-effect review

BL87 reviewed side-effect risk patterns in the dependent script-era files.

### `scripts/fundamentals/build_analysis.py`

Observed:

```text
argparse
pd.read_csv(...)
Path(output_path).parent.mkdir(...)
analysis_df.to_csv(...)
main()
if __name__ == "__main__"
```

Classification:

```text
SCRIPT_ENTRYPOINT
CSV_READ
DIRECTORY_CREATE
OPTIONAL_CSV_WRITE
```

### `scripts/fundamentals/build_quality.py`

Observed:

```text
argparse
Path("data/processed/context_strength.csv")
Path("data/raw/fundamentals.csv")
Path("data/processed/fundamental_quality.csv")
Path("data/logs/fundamental_layer_log.csv")
pd.read_csv(...)
LOG_PATH.parent.mkdir(...)
to_csv(LOG_PATH)
OUTPUT_PATH.parent.mkdir(...)
output_df.to_csv(OUTPUT_PATH)
main()
if __name__ == "__main__"
```

Classification:

```text
SCRIPT_ENTRYPOINT
CSV_READ
DEFAULT_PRODUCTION_DATA_PATHS
DIRECTORY_CREATE
PRODUCTION_CSV_WRITE
LOG_CSV_WRITE
```

This is higher risk than `build_history_intake.py` and `build_metrics.py`.

### `scripts/fundamentals/run_sec_transformation_review.py`

Observed:

```text
argparse
Path(companyfacts_dir)
Path(output_path)
output.parent.mkdir(...)
review_df.to_csv(...)
main(...)
if __name__ == "__main__"
```

Classification:

```text
SCRIPT_ENTRYPOINT
LOCAL_SEC_REVIEW_INPUT
DIRECTORY_CREATE
OPTIONAL_REVIEW_CSV_WRITE
```

### `scripts/fundamentals/sec_companyfacts_transform.py`

Observed:

```text
argparse
Path(path)
Path(output_path)
output.parent.mkdir(...)
df.to_csv(...)
main(...)
if __name__ == "__main__"
```

Classification:

```text
SCRIPT_ENTRYPOINT
LOCAL_SEC_JSON_READ
DIRECTORY_CREATE
OPTIONAL_CSV_WRITE
```

### `scripts/fundamentals/build_metrics.py`

Observed:

```text
argparse
pd.read_csv(...)
Path(output_path).parent.mkdir(...)
metrics_df.to_csv(...)
main()
if __name__ == "__main__"
```

Classification:

```text
SCRIPT_ENTRYPOINT
CSV_READ
DIRECTORY_CREATE
OPTIONAL_CSV_WRITE
```

### `scripts/fundamentals/build_history_intake.py`

Observed:

```text
argparse
Path(input_path)
pd.read_csv(...)
main()
Path(args.report_path).write_text(...)
if __name__ == "__main__"
```

Classification:

```text
SCRIPT_ENTRYPOINT
CSV_READ
OPTIONAL_REPORT_WRITE
```

## Active source/test reference findings

BL87 confirmed that active references still exist for the broader remaining script-era cluster:

```text
src/market_scanner/analysis/analysis_boundary.py
tests/fundamentals/test_sec_companyfacts_transform.py
tests/fundamentals/test_run_sec_transformation_review.py
tests/unit/test_v2_sec_companyfacts_smoke_boundary.py
tests/core/test_build_fundamental_analysis.py
tests/core/test_build_fundamental_layer.py
tests/core/test_fundamentals_operational_validation.py
tests/test_operator_visibility.py
```

Important distinction:

* BL86 removed active positive references to `build_history_intake.py` and `build_metrics.py`.
* BL87 confirms that other script-era fundamentals files still have active references.
* Therefore the remaining blocker is now the broader script-era fundamentals cluster, not only the original two target files.

## Canonical coverage review

BL87 confirmed that the relevant canonical coverage exists:

```text
docs/audits/legacy_runtime/bl83_canonical_fundamentals_metrics_contract_tests.md
docs/audits/legacy_runtime/bl84_canonical_fundamentals_history_validation_contract_tests.md
docs/audits/legacy_runtime/bl85_archive_readiness_review_fundamentals_history_and_metrics.md
docs/audits/legacy_runtime/bl86_decouple_active_tests_and_metadata_from_script_era_fundamentals_history_metrics.md
src/market_scanner/fundamentals/fundamental_contracts.py
src/market_scanner/fundamentals/fundamentals_metrics_contracts.py
```

This confirms the original BL81/BL82 history and metrics contract gaps are now covered canonically.

## Validation

Focused related tests:

```bash
pytest tests/contract/test_v2_fundamentals_metrics_contracts.py \
       tests/contract/test_v2_fundamental_history_validation_contracts.py \
       tests/unit/test_v2_canonical_analysis.py \
       tests/core/test_fundamentals_runtime_organization.py \
       tests/core/test_fundamentals_operational_validation.py \
       tests/core/test_build_fundamental_metrics.py \
       tests/core/test_build_fundamentals_history_intake.py -q
```

Result:

```text
44 passed in 0.04s
```

Full suite:

```bash
pytest -q
```

Result:

```text
550 passed in 0.55s
```

## Archive decision after BL87

BL87 decision:

```text
DO_NOT_ARCHIVE_HISTORY_OR_METRICS_IN_ISOLATION
```

| File                                                    | BL87 decision                              | Reason                                                                                                                             |
| ------------------------------------------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/fundamentals/build_history_intake.py`          | `CLUSTER_DEPENDENCY_BLOCKED`               | Still imported by `build_metrics.py`, `build_quality.py`, `run_sec_transformation_review.py`, and `sec_companyfacts_transform.py`. |
| `scripts/fundamentals/build_metrics.py`                 | `CLUSTER_DEPENDENCY_BLOCKED`               | Still imported by `build_analysis.py` and `build_quality.py`.                                                                      |
| `scripts/fundamentals/build_analysis.py`                | `ACTIVE_REFERENCE_AND_OPTIONAL_WRITE_RISK` | Still active in metadata/tests and has CSV read/write behavior.                                                                    |
| `scripts/fundamentals/build_quality.py`                 | `HIGH_RISK_PRODUCTION_WRITE_BLOCKER`       | Still active in metadata/tests and writes production-like outputs and logs.                                                        |
| `scripts/fundamentals/run_sec_transformation_review.py` | `SEC_REVIEW_RUNNER_BLOCKER`                | Still active in tests and depends on SEC transform/index/history constants.                                                        |
| `scripts/fundamentals/sec_companyfacts_transform.py`    | `SEC_TRANSFORM_BLOCKER`                    | Still active in tests and depends on history constants and CIK/ticker index helpers.                                               |
| `scripts/fundamentals/sec_ticker_cik_index.py`          | `SEC_MAPPING_DEPENDENCY`                   | Still used by SEC transform/review scripts.                                                                                        |
| `scripts/fundamentals/sec_companyfacts_bulk_intake.py`  | `PROVIDER_SIDE_EFFECT_RISK`                | Still a provider/network/cache-risk script and should remain separately governed.                                                  |

## Decision

Do not archive `build_history_intake.py` or `build_metrics.py` as isolated files.

The remaining script-era fundamentals files should be handled as a dependency cluster.

The safer path is to first decouple or retire active references to:

* `build_analysis.py`
* `build_quality.py`
* `run_sec_transformation_review.py`
* `sec_companyfacts_transform.py`

before attempting archival of `build_history_intake.py` and `build_metrics.py`.

## Recommended next sprint

Recommended next sprint:

```text
BL88 — Decouple active tests from remaining script-era fundamentals analysis and quality modules
```

Candidate files/tests:

* `tests/core/test_build_fundamental_analysis.py`
* `tests/core/test_build_fundamental_layer.py`
* `tests/core/test_fundamentals_operational_validation.py`
* `src/market_scanner/analysis/analysis_boundary.py`
* `tests/unit/test_v2_canonical_analysis.py`

Goal:

* remove active positive references to `scripts/fundamentals/build_analysis.py`;
* remove active positive references to `scripts/fundamentals/build_quality.py`;
* keep only canonical/source metadata or negative guardrails;
* avoid editing or executing script-era runtime modules.

Recommended follow-up after that:

```text
BL89 — Decouple active SEC transform/review tests from script-era SEC transformation modules
```

Candidate files/tests:

* `tests/fundamentals/test_sec_companyfacts_transform.py`
* `tests/fundamentals/test_run_sec_transformation_review.py`
* `tests/unit/test_v2_sec_companyfacts_smoke_boundary.py`
* `tests/test_operator_visibility.py`

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

BL87 confirms that `build_history_intake.py` and `build_metrics.py` are no longer blocked by missing canonical contract coverage or active positive references from `src/tests/.github`.

They remain blocked by internal script-era dependency clustering.

The next cleanup step should target active references to the broader analysis/quality script-era modules before attempting archive.
