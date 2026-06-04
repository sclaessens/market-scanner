# v2 Legacy Runtime Entrypoint Migration

## Status

Completed by RESET-10L-BL29.

## Reset stage

RESET-10L-BL29 — Migrate Legacy Runtime Entrypoint Logic.

## Purpose

This sprint starts migration of runtime entrypoint authority away from legacy runners and toward the canonical v2 runtime architecture defined in BL28.

The sprint establishes the canonical v2 application boundary under:

```text
src/market_scanner/app.py
```

The new boundary is intentionally minimal, deterministic, and dry-run only. It defines the approved runtime sequence without executing scanner logic, provider calls, production data writes, report generation, Telegram delivery, portfolio/watchlist updates, or Decision Engine investment behavior.

## Policies applied

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_python_architecture_cleanup_and_legacy_decoupling_review.md`
- `docs/active/v2_canonical_runtime_architecture.md`

Policy result:

- A new canonical Python file was created only because BL28 explicitly approved `src/market_scanner/app.py` as the canonical v2 application entrypoint.
- A new unit test file was created because no existing test file owned canonical app-level orchestration behavior.
- No legacy runner authority was expanded.
- No runtime implementation was wired to production scanner, reporting, Telegram, provider, portfolio/watchlist, or Decision Engine behavior.

## Files inspected

Governance and backlog:

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_python_architecture_cleanup_and_legacy_decoupling_review.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/backlog.md`

Legacy runtime files:

- `scripts/run_scan.py`
- `scripts/run_full_pipeline.py`

Current package structure and related tests:

- `src/market_scanner/`
- `src/market_scanner/orchestration/pipeline_core.py`
- `src/market_scanner/shared/records.py`
- `tests/integration/test_v2_minimal_pipeline_core.py`
- `tests/unit/test_v2_package_bootstrap.py`
- `tests/core/test_run_full_pipeline.py`
- `tests/test_operator_visibility.py`
- reporting, Telegram, runtime, and fundamentals test locations discovered by static search.

## Files changed

- `src/market_scanner/app.py`
- `tests/unit/test_v2_canonical_app.py`
- `docs/active/v2_legacy_runtime_entrypoint_migration.md`
- `docs/active/backlog.md`

No legacy runtime files were changed.

## Canonical app boundary created or certified

The canonical app boundary is now created at `src/market_scanner/app.py`.

Public functions:

- `build_canonical_runtime_plan`
- `run_canonical_app`

Core records:

- `RuntimeStage`
- `SideEffectGuarantees`
- `CanonicalRuntimePlan`
- `CanonicalAppResult`

The canonical runtime plan exposes the approved BL28 sequence:

```text
application_entrypoint
-> scanner_universe_selection
-> provider_source_access
-> fundamentals_normalization_evidence
-> analysis
-> decision_review_boundary
-> message_composition
-> report_generation_where_approved
-> delivery_telegram_where_approved
```

The initial app boundary is a planning/orchestration boundary only. It does not implement production scanner execution, provider execution, analysis execution, report artifact generation, Telegram delivery, portfolio/watchlist mutation, or Decision Engine investment behavior.

## Runtime authority result

Runtime entrypoint authority migration has started.

The repository now has a canonical v2 app boundary that defines the approved runtime sequence without relying on the legacy full-pipeline scripts. This does not yet replace legacy runtime behavior, but it prevents the legacy scripts from being the only place where application runtime authority exists.

## Legacy runner status

`scripts/run_scan.py` and `scripts/run_full_pipeline.py` are not approved as permanent canonical runtime authorities.

They remain present as legacy dependencies and migration references. They were not edited, expanded, moved, deleted, or wrapped in this sprint.

Remaining legacy status:

- `scripts/run_scan.py` still owns the existing broad legacy runtime flow, including scanner execution, production data writes, reporting artifact creation, and Telegram delivery.
- `scripts/run_full_pipeline.py` still shells into `scripts/run_scan.py`.
- Both remain migration/archive candidates under the legacy decoupling policy.

## Tests added

Added:

```text
tests/unit/test_v2_canonical_app.py
```

The tests prove:

- the canonical runtime plan exposes the approved stages;
- all stages are side-effect-free by default;
- dry-run result metadata is deterministic;
- provider calls are not made;
- production data writes are not made;
- reports are not generated;
- Telegram artifacts are not created;
- portfolio/watchlist files are not modified;
- legacy scripts are not imported or invoked;
- no final investment behavior is produced;
- non-dry-run execution fails closed.

## Side-effect guarantees

The canonical app dry-run records these guarantees:

```text
provider_calls_made = False
production_data_writes = False
reports_generated = False
telegram_artifacts_created = False
portfolio_or_watchlist_updates = False
legacy_runners_invoked = False
```

The module has no import-time side effects and does not create files during dry-run execution.

## Python file creation justification

`src/market_scanner/app.py` was created because BL28 approved it as the canonical v2 application entrypoint. It replaces legacy runtime authority over time rather than adding a competing one-off runner. The existing relevant modules were inspected first, including `src/market_scanner/orchestration/pipeline_core.py`, `scripts/run_scan.py`, and `scripts/run_full_pipeline.py`. `pipeline_core.py` owns the minimal synthetic pipeline scaffold, while BL28 identified `app.py` as the application entrypoint owner. The legacy scripts are migration targets and should not receive new v2 runtime authority.

`tests/unit/test_v2_canonical_app.py` was created because no existing test file owned canonical app-level orchestration behavior. Existing runtime tests focus on legacy scripts or the minimal synthetic pipeline scaffold, not the BL28 application entrypoint boundary.

No one-off runtime helper files, temporary migration files, ticker-specific files, or parallel shortcut runners were created.

## Remaining migration work

Recommended next migration work:

1. Define or create the canonical scanner runtime boundary under the BL28-approved scanner ownership.
2. Migrate scanner/universe planning logic away from `scripts/run_scan.py` without provider calls or production writes.
3. Keep production data writes, reporting, Telegram delivery, portfolio/watchlist updates, and Decision Engine behavior disconnected until separately approved.
4. Add import and side-effect tests for each migrated canonical boundary.
5. Only after canonical callers exist and tests pass, review whether legacy runner files can become certified bridges, archive candidates, or delete-after-confirmation candidates.

## Guardrails confirmation

- No credentials committed.
- No raw live payloads committed.
- No production data writes.
- No reports generated.
- No Telegram artifacts generated.
- No unsafe production pipeline execution.
- No portfolio/watchlist updates.
- No final BUY/SELL/HOLD recommendation.
- No missing values converted to zero.
- No one-off temporary helper files created.
- Legacy runners were not expanded.

## Next recommended step

RESET-10L-BL30 — Migrate Scanner Runtime Logic to Canonical V2 Boundary.
