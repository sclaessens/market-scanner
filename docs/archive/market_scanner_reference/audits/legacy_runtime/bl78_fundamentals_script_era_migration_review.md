# BL78 — Fundamentals script-era migration and archive-readiness review

Status: COMPLETED

## Purpose

BL78 reviews the remaining script-era fundamentals Python files after BL74 removed active runtime imports from `scripts.fundamentals`.

This sprint exists to prevent blind deletion of the remaining fundamentals files. The project direction remains clear: the canonical codebase is `src/market_scanner/`, and the old `scripts/` tree is legacy. However, several remaining fundamentals scripts may still contain useful source-data, transformation, metric, validation, quality, or review logic that must either be migrated into canonical modules or explicitly retired.

BL78 is a documentation-only classification sprint.

It does not archive, delete, move, refactor, or execute runtime Python files.

## Registry basis

Primary registry and audit basis:

* `docs/audits/legacy_runtime/bl70_python_cleanup_registry_lock.md`
* `docs/audits/legacy_runtime/v2_script_era_python_cleanup_inventory.md`
* `docs/audits/legacy_runtime/v2_high_risk_script_era_side_effect_cleanup_review.md`
* `docs/audits/legacy_runtime/v2_high_risk_script_era_test_execution_cleanup.md`
* `docs/audits/legacy_runtime/v2_fundamentals_script_era_side_effect_migration_review.md`
* `docs/audits/legacy_runtime/bl74_decouple_active_tests_from_script_era_fundamentals.md`
* `docs/audits/legacy_runtime/bl76_remaining_script_era_python_dependency_classification.md`

Recent cleanup context:

* BL74 removed active `scripts.fundamentals` imports from `tests/`, `src/`, and `.github/`.
* BL75 archived the previously blocked `scripts/fundamentals/__init__.py`.
* BL76 classified the remaining `scripts/` tree and marked the remaining fundamentals files as migration-required or side-effect-risk.
* BL77 archived the first low-risk non-fundamentals script-era Python files.

## Scope

Reviewed fundamentals files:

* `scripts/fundamentals/build_analysis.py`
* `scripts/fundamentals/build_history_intake.py`
* `scripts/fundamentals/build_metrics.py`
* `scripts/fundamentals/build_quality.py`
* `scripts/fundamentals/run_sec_transformation_review.py`
* `scripts/fundamentals/sec_companyfacts_bulk_intake.py`
* `scripts/fundamentals/sec_companyfacts_transform.py`
* `scripts/fundamentals/sec_ticker_cik_index.py`

Related script-era compatibility wrappers observed during the reference check:

* `scripts/core/build_fundamental_analysis.py`
* `scripts/core/build_fundamental_layer.py`
* `scripts/core/build_fundamental_metrics.py`
* `scripts/core/build_fundamentals_history_intake.py`

Canonical areas inspected for possible replacement ownership:

* `src/market_scanner/analysis/`
* `src/market_scanner/fundamentals/`

## Active reference check

Command:

```bash
grep -R "scripts/fundamentals\|scripts.fundamentals" -n . \
  --include="*.py" \
  --include="*.md" \
  --include="*.yml" \
  --include="*.yaml" \
  --exclude-dir=".git" \
  --exclude-dir=".venv" \
  --exclude-dir="archive"
```

Result summary:

* Active tests no longer import `scripts.fundamentals` as a runtime package.
* Active tests still contain static path/evidence references to legacy fundamentals files.
* `scripts/fundamentals/` still contains internal script-era imports between fundamentals modules.
* `scripts/core/build_fundamental_*` wrappers still import from `scripts.fundamentals`.
* `src/market_scanner/analysis/analysis_boundary.py` still statically lists selected legacy fundamentals files as boundary evidence.
* Multiple audit and provider-smoke documents still reference legacy fundamentals files as historical evidence.
* Legacy documentation references remain expected and are not active runtime blockers.

Observed active static test references include:

* `tests/fundamentals/test_sec_companyfacts_transform.py`
* `tests/fundamentals/test_sec_ticker_cik_index.py`
* `tests/fundamentals/test_sec_companyfacts_bulk_intake.py`
* `tests/fundamentals/test_run_sec_transformation_review.py`
* `tests/core/test_build_fundamental_analysis.py`
* `tests/core/test_build_fundamental_layer.py`
* `tests/core/test_fundamentals_operational_validation.py`
* `tests/core/test_build_fundamental_metrics.py`
* `tests/core/test_build_fundamentals_history_intake.py`
* `tests/unit/test_v2_canonical_analysis.py`
* `tests/unit/test_v2_sec_companyfacts_smoke_boundary.py`

Observed script-era internal imports include:

* `scripts/fundamentals/build_analysis.py`

  * imports from `scripts.fundamentals.build_metrics`
* `scripts/fundamentals/build_metrics.py`

  * imports from `scripts.fundamentals.build_history_intake`
* `scripts/fundamentals/build_quality.py`

  * imports from `scripts.fundamentals.build_metrics`
  * imports from `scripts.fundamentals.build_history_intake`
* `scripts/fundamentals/sec_companyfacts_transform.py`

  * imports from `scripts.fundamentals.build_history_intake`
  * imports from `scripts.fundamentals.sec_ticker_cik_index`
* `scripts/fundamentals/run_sec_transformation_review.py`

  * imports from `scripts.fundamentals.build_history_intake`
  * imports from `scripts.fundamentals.sec_companyfacts_transform`
  * imports from `scripts.fundamentals.sec_ticker_cik_index`

Observed compatibility wrappers include:

* `scripts/core/build_fundamental_analysis.py`

  * imports from `scripts.fundamentals.build_analysis`
* `scripts/core/build_fundamental_layer.py`

  * imports from `scripts.fundamentals.build_quality`
* `scripts/core/build_fundamental_metrics.py`

  * imports from `scripts.fundamentals.build_metrics`
* `scripts/core/build_fundamentals_history_intake.py`

  * imports from `scripts.fundamentals.build_history_intake`

## Remaining fundamentals files

Command:

```bash
find scripts/fundamentals -type f -name "*.py" | sort
```

Result:

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

## Canonical replacement check

Command:

```bash
find src/market_scanner -type f -name "*.py" | sort | grep -Ei "fundamental|sec|companyfacts|metrics|quality|cik|intake|history|analysis"
```

Result:

```text
src/market_scanner/analysis/__init__.py
src/market_scanner/analysis/analysis_boundary.py
src/market_scanner/analysis/analysis_contracts.py
src/market_scanner/fundamentals/__init__.py
src/market_scanner/fundamentals/fundamental_contracts.py
src/market_scanner/fundamentals/fundamentals_normalization_adapter.py
src/market_scanner/fundamentals/fundamentals_normalization_contracts.py
src/market_scanner/fundamentals/fundamentals_persistence.py
src/market_scanner/fundamentals/fundamentals_provider_adapter.py
src/market_scanner/fundamentals/fundamentals_provider_contracts.py
src/market_scanner/fundamentals/fundamentals_real_source_smoke.py
src/market_scanner/fundamentals/sec_companyfacts_live_smoke.py
src/market_scanner/fundamentals/sec_companyfacts_smoke_boundary.py
src/market_scanner/fundamentals/source_data_readiness.py
src/market_scanner/fundamentals/source_data_records.py
```

Interpretation:

* Canonical fundamentals and analysis modules exist under `src/market_scanner/`.
* Their presence does not prove complete behavioral parity with the script-era fundamentals files.
* Several script-era files may still contain useful validation, metric, quality, analysis, SEC transformation, or ticker/CIK mapping knowledge.
* Provider/network/data-write files remain blocked until explicitly governed or retired.

## Classification

| File                                                    | Classification                    | Rationale                                                                                                                                                             | Next action                                                                                                    |
| ------------------------------------------------------- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `scripts/fundamentals/build_analysis.py`                | MIGRATION_REQUIRED                | Contains script-era fundamental analysis behavior and imports script-era metrics constants. It is referenced by historical audits and by canonical boundary evidence. | Compare against `src/market_scanner/analysis/` and migrate only approved review-safe contracts before archive. |
| `scripts/fundamentals/build_history_intake.py`          | MIGRATION_REQUIRED                | Contains history schema/intake validation knowledge used by metrics and SEC transform scripts.                                                                        | Extract required schema and validation policy into canonical fundamentals contracts if still needed.           |
| `scripts/fundamentals/build_metrics.py`                 | MIGRATION_REQUIRED                | Contains script-era metric derivation logic and depends on history validation.                                                                                        | Compare formulas and missing-value policy against canonical fundamentals/analysis contracts.                   |
| `scripts/fundamentals/build_quality.py`                 | MIGRATION_REQUIRED                | Contains script-era quality/readiness behavior and depends on metrics/history logic. Prior audits indicate data-write/log behavior.                                   | Separate source-data readiness from legacy output generation before migration or archive.                      |
| `scripts/fundamentals/sec_companyfacts_transform.py`    | MIGRATION_REQUIRED                | Contains SEC CompanyFacts transformation and depends on history schema plus ticker/CIK normalization. May contain useful pure transform rules.                        | Compare against canonical SEC CompanyFacts smoke/source boundaries before archive.                             |
| `scripts/fundamentals/sec_ticker_cik_index.py`          | MIGRATION_REQUIRED_SOURCE_MAPPING | Contains ticker/CIK normalization, mapping, and coverage behavior. Identifier mapping policy may not yet have a complete canonical owner.                             | Define or confirm canonical ticker/CIK source-metadata ownership before archive.                               |
| `scripts/fundamentals/run_sec_transformation_review.py` | BLOCKED_REVIEW_RUNNER_RISK        | Script-era review runner around SEC transformation. May read local files and write review evidence.                                                                   | Do not execute. Retire or archive only after canonical SEC transform/review parity decision.                   |
| `scripts/fundamentals/sec_companyfacts_bulk_intake.py`  | BLOCKED_PROVIDER_SIDE_EFFECT_RISK | SEC bulk intake is provider/network/data-write capable and remains high-risk. Prior audits classify this as side-effect sensitive.                                    | Do not execute. Do not migrate live behavior without explicit provider governance approval.                    |

## Compatibility wrapper finding

The following files are compatibility wrappers over `scripts.fundamentals`:

* `scripts/core/build_fundamental_analysis.py`
* `scripts/core/build_fundamental_layer.py`
* `scripts/core/build_fundamental_metrics.py`
* `scripts/core/build_fundamentals_history_intake.py`

These wrappers should not be archived in the same sprint as the fundamentals modules. They should be handled in a follow-up cleanup after BL78 determines which underlying fundamentals logic has canonical ownership or can be retired.

Recommended follow-up:

* BL79A: classify and retire/archive fundamentals compatibility wrappers only if no active runtime references remain.
* BL79B: migrate or retire pure fundamentals logic after canonical parity check.
* BL79C: keep SEC bulk/provider behavior blocked until explicit provider-source governance approval.

## Findings

1. BL74 successfully removed active runtime imports from `scripts.fundamentals` in tests, source, and workflows.

2. The remaining active test references are static evidence/path references, not direct runtime imports.

3. The fundamentals scripts still depend on each other internally. Archiving one fundamentals module without a planned group migration may break remaining legacy-only script behavior.

4. The `scripts/core/build_fundamental_*` files remain compatibility wrappers over the script-era fundamentals modules.

5. Canonical modules exist under `src/market_scanner/analysis/` and `src/market_scanner/fundamentals/`, but complete parity with legacy metrics, analysis, history validation, SEC transform, and ticker/CIK mapping is not proven by BL78.

6. Provider and SEC-related files remain high-risk because they may involve network, cache, local SEC files, generated reports, or evidence writes.

7. BL78 does not identify any fundamentals file as immediately archive-ready without additional migration/parity work.

## Decision

Do not archive the remaining `scripts/fundamentals/*.py` files in a single batch.

The correct next step is to split the fundamentals cleanup into smaller governed follow-up sprints:

1. **Fundamentals compatibility wrappers**

   * Review and archive `scripts/core/build_fundamental_*` wrappers if they have no active runtime references and only forward to legacy fundamentals modules.

2. **Pure reusable fundamentals logic**

   * Review `build_history_intake.py`, `build_metrics.py`, `build_analysis.py`, `build_quality.py`, `sec_companyfacts_transform.py`, and `sec_ticker_cik_index.py` for canonical-equivalence or migration.

3. **SEC/provider runner and intake logic**

   * Keep `sec_companyfacts_bulk_intake.py` and `run_sec_transformation_review.py` blocked until explicit provider/data-write/review-runner governance is approved.

## Recommended BL79 scope

Recommended next sprint:

```text
BL79 — Archive fundamentals compatibility wrappers after active-reference check
```

Proposed BL79 candidates:

* `scripts/core/build_fundamental_analysis.py`
* `scripts/core/build_fundamental_layer.py`
* `scripts/core/build_fundamental_metrics.py`
* `scripts/core/build_fundamentals_history_intake.py`

BL79 should first run a focused active-reference check for those wrapper files. If they are referenced only as legacy evidence and not as active runtime entrypoints, they can be archived under `archive/legacy_runtime/scripts/core/`.

BL79 must not archive the underlying `scripts/fundamentals/*.py` files unless a separate canonical parity decision has been made.

## Validation

Full test suite was run for repository safety.

Command:

```bash
pytest -q
```

Result:

```text
522 passed in 0.59s
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

## Final status

BL78 is complete as a documentation-only migration and archive-readiness review.

No remaining `scripts/fundamentals/*.py` file is declared archive-ready by BL78.

The next cleanup should address the `scripts/core/build_fundamental_*` compatibility wrappers first, or separately perform canonical parity work for pure fundamentals logic.
