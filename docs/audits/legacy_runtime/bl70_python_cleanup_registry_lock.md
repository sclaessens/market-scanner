# BL70 — Python cleanup registry lock

## Status

Proposed for merge.

## Purpose

Freeze the existing Python cleanup and legacy-runtime inventories as the canonical cleanup execution registry.

This document exists to stop repeated broad Python re-inventory loops. Future cleanup work must use the existing audit records as the starting point and perform only narrow delta checks for the files included in a specific cleanup batch.

## Scope

This is a documentation-only governance task.

BL70 does not modify, move, delete, archive, rename, refactor, or execute Python files.

BL70 does not change tests, workflows, data files, reports, Telegram behavior, portfolio/watchlist behavior, provider behavior, or Decision Engine behavior.

## Registry sources

The following existing records are promoted as the canonical Python cleanup registry basis:

* `docs/audits/reset_cleanup/v2_python_architecture_cleanup_and_legacy_decoupling_review.md`
* `docs/audits/legacy_runtime/v2_script_era_python_cleanup_inventory.md`
* `docs/audits/legacy_runtime/v2_high_risk_script_era_side_effect_cleanup_review.md`
* `docs/audits/legacy_runtime/v2_high_risk_script_era_test_execution_cleanup.md`
* `docs/audits/legacy_runtime/v2_fundamentals_script_era_side_effect_migration_review.md`
* `docs/audits/legacy_runtime/v2_legacy_runtime_archive_validation_and_entrypoint_certification.md`
* `docs/active/architecture/v2_canonical_runtime_architecture.md`

These records are not perfect current-state source code manifests. They are the approved governance basis for cleanup sequencing, risk classification, canonical ownership, and migration/archive/delete decisions.

## Registry interpretation

The existing registry establishes the following current cleanup doctrine:

* `src/market_scanner/app.py` is the certified active runtime entrypoint.
* `scripts/run_scan.py` and `scripts/run_full_pipeline.py` are no longer active runtime-path entrypoints and are retained only under `archive/legacy_runtime/scripts/` as historical references.
* Remaining `scripts/` Python files are script-era surfaces unless later migrated, wrapped, retired, archived, or explicitly reapproved.
* Script-era files must not be treated as canonical runtime authorities merely because they still exist.
* Static references in governance records, canonical metadata, or legacy evidence documents do not automatically make a script-era file active runtime code.
* Cleanup must remain staged and reversible where possible.

## No repeated broad inventory rule

Future cleanup sprints must not repeat a full repository-wide Python inventory unless the registry itself is missing, corrupted, or explicitly invalidated by a governance decision.

Instead, each cleanup sprint must:

1. reference the registry records above;
2. select a small file batch from the existing classifications;
3. run a narrow delta check for only that batch;
4. confirm active imports, workflow references, test references, and manual-entrypoint risk for the selected files;
5. perform only the approved cleanup action for those files;
6. update the registry/backlog with the result.

A broad `find . -name "*.py"` inventory may be used only as a verification aid. It must not become a new competing cleanup registry.

## Required pre-checks for future cleanup batches

Before any future archive, delete, move, rename, or migration cleanup PR, run targeted checks for the selected files:

```bash
git checkout main
git pull origin main
git status

grep -R "<selected-path>" -n src tests .github docs --include="*.py" --include="*.yml" --include="*.yaml" --include="*.md"

grep -R "from scripts\|import scripts" -n src tests --include="*.py"
```

If a selected cleanup batch touches code or tests, run the relevant focused tests and the full active suite:

```bash
pytest -q
```

Documentation-only cleanup registry updates may skip tests only when no code, tests, data, workflows, reports, scripts, or runtime files changed.

## Cleanup execution rules

Future cleanup PRs must be small.

A cleanup PR may do one of the following, not all at once:

* archive confirmed inactive files;
* delete confirmed obsolete files;
* migrate a narrowly defined behavior into a canonical owner;
* replace a legacy behavior test with canonical boundary coverage;
* mark a file as temporarily retained because risk remains;
* update registry/backlog status after a completed cleanup action.

A cleanup PR must not combine unrelated domains such as fundamentals, Decision Engine, Telegram delivery, portfolio/watchlist command handling, SEC provider governance, reporting, and scanner execution unless explicitly approved.

## Protected areas

The following areas remain protected until separately scoped:

* Decision Engine final/allocation semantics;
* Telegram delivery and command polling;
* portfolio/watchlist mutation;
* provider/live-source network access;
* SEC bulk intake/cache/download behavior;
* production data writes;
* reporting and message delivery side effects;
* generated CSV/report artifacts;
* workflow execution behavior.

## Preferred next cleanup sequence

The next cleanup work should follow this order:

1. select only the safest low-risk P3 archive/delete candidates already identified in the registry;
2. avoid fundamentals, SEC, Decision Engine, Telegram, portfolio, watchlist, provider, report, and workflow files in the first cleanup batch;
3. prove the selected files have no active imports, workflow references, or active pytest dependency;
4. archive or delete only that small approved batch;
5. update the registry/backlog with the completed result.

After that, a separate sprint may handle duplicate fundamentals wrappers or migration-after-parity candidates.

## BL70 result

BL70 creates the cleanup-registry lock.

The project should stop redoing broad Python inventories before each cleanup discussion. Future cleanup work must use the existing registry records plus narrow delta checks and must proceed through small, explicit cleanup batches.
