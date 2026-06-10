# BL85 — Review archive-readiness of script-era fundamentals history and metrics modules

Status: COMPLETED

## Purpose

BL85 reviews whether the remaining script-era fundamentals history and metrics modules can be archived after BL83 and BL84 added canonical contract coverage.

Target files:

* `scripts/fundamentals/build_history_intake.py`
* `scripts/fundamentals/build_metrics.py`

BL85 is primarily an archive-readiness and active-reference review. It does not archive the target files.

## Scope

Reviewed script-era files:

* `scripts/fundamentals/build_history_intake.py`
* `scripts/fundamentals/build_metrics.py`

Reviewed canonical coverage:

* `src/market_scanner/fundamentals/fundamental_contracts.py`
* `src/market_scanner/fundamentals/fundamentals_metrics_contracts.py`
* `tests/contract/test_v2_fundamental_history_validation_contracts.py`
* `tests/contract/test_v2_fundamentals_metrics_contracts.py`
* `docs/audits/legacy_runtime/bl83_canonical_fundamentals_metrics_contract_tests.md`
* `docs/audits/legacy_runtime/bl84_canonical_fundamentals_history_validation_contract_tests.md`

Updated during BL85:

* `tests/contract/test_v2_fundamental_history_validation_contracts.py`

Out of scope:

* Archiving script-era files.
* Editing script-era runtime modules.
* Provider calls.
* Production data writes.
* Production reports.
* Portfolio/watchlist state changes.
* Telegram delivery.
* Decision Engine authority changes.

## Existing files confirmed

Both target files still exist:

```text
scripts/fundamentals/build_history_intake.py
scripts/fundamentals/build_metrics.py
```

Observed file sizes:

```text
scripts/fundamentals/build_history_intake.py — 7658 bytes
scripts/fundamentals/build_metrics.py — 8098 bytes
```

## Active-reference review

A static search over active areas was run against:

```text
src
tests
.github
docs
```

Search target:

```text
build_history_intake|build_metrics
```

Findings:

### Active source reference

`src/market_scanner/analysis/analysis_boundary.py` still statically lists:

```text
scripts/fundamentals/build_metrics.py
```

This appears to be metadata/boundary evidence rather than an executable import, but it is still an active source reference and should be handled before archival.

### Active test references

The following tests still reference one or both script-era paths:

```text
tests/unit/test_v2_canonical_analysis.py
tests/core/test_fundamentals_runtime_organization.py
tests/core/test_fundamentals_operational_validation.py
tests/core/test_build_fundamental_metrics.py
tests/core/test_build_fundamentals_history_intake.py
```

The key finding is that script-era references still exist in active test files. These may be static legacy-policy checks rather than direct runtime imports, but they must be reviewed or updated before archiving the target files.

### Documentation references

Many documentation references remain. These are expected and do not block archival by themselves, provided they are historical/audit references.

## Internal script-era dependency review

A static search inside `scripts/` found active internal script-era imports:

```text
scripts/fundamentals/build_analysis.py: imports from scripts.fundamentals.build_metrics
scripts/fundamentals/run_sec_transformation_review.py: imports REQUIRED_COLUMNS from scripts.fundamentals.build_history_intake
scripts/fundamentals/build_metrics.py: imports validate_fundamentals_history from scripts.fundamentals.build_history_intake
scripts/fundamentals/build_quality.py: imports from scripts.fundamentals.build_metrics
scripts/fundamentals/build_quality.py: imports validate_fundamentals_history from scripts.fundamentals.build_history_intake
scripts/fundamentals/sec_companyfacts_transform.py: imports REQUIRED_COLUMNS from scripts.fundamentals.build_history_intake
```

This means the target files are still internal dependency anchors for other remaining script-era fundamentals files.

This does not prove active production runtime usage, but it does mean the files should not be archived before the dependent script-era modules are reviewed or decoupled.

## Side-effect review

Static side-effect search found:

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

Risk classification:

```text
OPTIONAL_REPORT_WRITE
SCRIPT_ENTRYPOINT
CSV_READ
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

Risk classification:

```text
OPTIONAL_CSV_WRITE
SCRIPT_ENTRYPOINT
CSV_READ
DIRECTORY_CREATE
```

Neither target file should be executed as part of cleanup. Any future archival must remain file-move-only and test-driven.

## Canonical coverage review

BL85 confirmed canonical coverage exists from BL83 and BL84.

Canonical files present:

```text
src/market_scanner/fundamentals/fundamentals_metrics_contracts.py
src/market_scanner/fundamentals/fundamental_contracts.py
tests/contract/test_v2_fundamentals_metrics_contracts.py
tests/contract/test_v2_fundamental_history_validation_contracts.py
docs/audits/legacy_runtime/bl83_canonical_fundamentals_metrics_contract_tests.md
docs/audits/legacy_runtime/bl84_canonical_fundamentals_history_validation_contract_tests.md
```

This confirms that the main BL81 contracts for metrics and history validation now have canonical coverage.

However, canonical coverage alone is not enough to archive the target files because active source/test references and script-internal dependencies still exist.

## BL85 corrective test-stability fix

During the focused BL85 validation run, the combined related tests initially failed:

```text
6 failed, 33 passed
```

Failure location:

```text
tests/contract/test_v2_fundamental_contracts.py
```

Cause:

`tests/contract/test_v2_fundamental_history_validation_contracts.py` imported and called `reload(fundamental_contracts)` in a no-file-side-effect test.

That reload recreated class objects for:

```text
FundamentalContractIssue
FundamentalContractIssueCode
```

The values appeared identical, but tuple equality failed because the compared dataclass/enum instances came from different module object identities.

BL85 corrected this by removing the unnecessary module reload from the no-file-side-effect test.

Corrected validation result:

```text
pytest tests/contract/test_v2_fundamentals_metrics_contracts.py \
       tests/contract/test_v2_fundamental_history_validation_contracts.py \
       tests/contract/test_v2_fundamental_contracts.py -q
```

Result:

```text
39 passed in 0.03s
```

Full suite:

```text
pytest -q
```

Result:

```text
547 passed in 0.52s
```

## Archive-readiness decision

BL85 result:

```text
NOT_READY_FOR_ARCHIVE_YET
```

For both target files:

| File                                           | BL85 decision               | Reason                                                                                                                                             |
| ---------------------------------------------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/fundamentals/build_history_intake.py` | `NOT_READY_FOR_ARCHIVE_YET` | Still imported by other script-era fundamentals files and still referenced by active tests.                                                        |
| `scripts/fundamentals/build_metrics.py`        | `NOT_READY_FOR_ARCHIVE_YET` | Still imported by other script-era fundamentals files, still listed in canonical analysis boundary metadata, and still referenced by active tests. |

## What blocks archive now

The remaining blockers are no longer missing canonical contract coverage. BL83 and BL84 addressed that.

The remaining blockers are:

1. Active test references to script-era paths.
2. Active metadata reference in `src/market_scanner/analysis/analysis_boundary.py`.
3. Internal script-era imports from:

   * `build_analysis.py`
   * `build_quality.py`
   * `run_sec_transformation_review.py`
   * `sec_companyfacts_transform.py`
4. Existing optional file-write behavior inside the target scripts.

## Decision

Do not archive `build_history_intake.py` or `build_metrics.py` yet.

The correct next step is to update or retire the remaining active tests/source references so they no longer require the files to remain in their current active `scripts/fundamentals/` location.

## Recommended next sprint

Recommended next sprint:

```text
BL86 — Decouple active tests and metadata references from script-era fundamentals history and metrics files
```

BL86 should be focused on active references only.

Candidate actions:

* Review `tests/core/test_build_fundamental_metrics.py`.
* Review `tests/core/test_build_fundamentals_history_intake.py`.
* Review `tests/core/test_fundamentals_runtime_organization.py`.
* Review `tests/core/test_fundamentals_operational_validation.py`.
* Review `tests/unit/test_v2_canonical_analysis.py`.
* Review `src/market_scanner/analysis/analysis_boundary.py`.

BL86 should not archive the files yet unless the reference removal is proven and separately validated.

Alternative split:

```text
BL86 — Decouple active tests from build_history_intake.py and build_metrics.py
BL87 — Update canonical analysis boundary metadata for archived script-era metrics ownership
BL88 — Archive reviewed script-era fundamentals history and metrics modules
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
* No script-era Python runtime files were executed.
* No script-era runtime module was edited.
* No script-era runtime file was archived.
* No script-era runtime file was deleted.

## Final status

BL85 confirms that canonical coverage from BL83 and BL84 is present and stable.

BL85 also corrected a BL84 focused-test-order instability.

The target script-era files are closer to archival, but still not archive-ready because active tests/source metadata and internal script-era dependencies remain.
