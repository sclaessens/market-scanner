# v2 Scanner Runtime Boundary Migration

## Status

Completed by RESET-10L-BL30.

## Reset stage

RESET-10L-BL30 — Migrate Scanner Runtime Logic to Canonical V2 Boundary.

## Purpose

This sprint starts scanner and universe-selection runtime migration away from legacy scripts and toward the canonical v2 runtime architecture defined in BL28.

The sprint establishes a side-effect-free canonical scanner boundary under:

```text
src/market_scanner/scanner/
```

The new boundary is deterministic and dry-run/planning-only. It does not execute a real scan, fetch market data, fetch fundamentals, read or write production CSVs, modify portfolio/watchlist files, generate reports, send Telegram messages, trigger Decision Engine behavior, or produce investment recommendations.

## Policies applied

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_python_architecture_cleanup_and_legacy_decoupling_review.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_legacy_runtime_entrypoint_migration.md`

Policy result:

- New scanner Python files were created only because BL28 approved `src/market_scanner/scanner/` as the canonical scanner/universe ownership boundary.
- A new scanner unit test file was created because no existing test file owned canonical scanner-boundary behavior.
- Existing legacy runner scripts were inspected but not edited.
- No one-off scanner helper, quick scanner, migration bridge, or parallel runtime shortcut was created.

## Files inspected

Governance and backlog:

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_python_architecture_cleanup_and_legacy_decoupling_review.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_legacy_runtime_entrypoint_migration.md`
- `docs/active/backlog.md`

Canonical app and tests:

- `src/market_scanner/app.py`
- `tests/unit/test_v2_canonical_app.py`

Legacy runtime and scanner files:

- `scripts/run_scan.py`
- `scripts/run_full_pipeline.py`
- `scripts/core/scanner.py`
- `scripts/core/data_fetcher.py`

Repository structure and references were inspected with static file inventory and grep searches for scanner, scan, universe, ticker, watchlist, portfolio, `run_scan`, and `run_full_pipeline`.

## Files changed

- `src/market_scanner/app.py`
- `src/market_scanner/scanner/__init__.py`
- `src/market_scanner/scanner/scanner_contracts.py`
- `src/market_scanner/scanner/scanner_boundary.py`
- `tests/unit/test_v2_canonical_app.py`
- `tests/unit/test_v2_canonical_scanner.py`
- `docs/active/v2_scanner_runtime_boundary_migration.md`
- `docs/active/backlog.md`

No legacy runtime files were changed.

## Canonical scanner boundary result

The canonical scanner boundary is now established under:

```text
src/market_scanner/scanner/
```

Public functions:

- `build_scanner_plan`
- `build_universe_selection_plan`

Core records:

- `ScannerPlan`
- `ScannerStage`

The scanner plan describes:

- scanner stage name;
- input source category;
- candidate universe source;
- whether provider calls are allowed;
- whether data writes are allowed;
- whether portfolio/watchlist mutation is allowed;
- whether reports are allowed;
- whether Telegram delivery is allowed;
- legacy migration status.

The initial scanner boundary has two planning stages:

```text
universe_selection
candidate_construction
```

Both stages are side-effect-free by default.

## App integration result

`src/market_scanner/app.py` now imports the canonical scanner boundary and includes the deterministic scanner plan in the canonical app dry-run runtime plan.

The canonical app scanner stage now references:

```text
src/market_scanner/scanner/
```

with status:

```text
canonical_boundary_established
```

The app still does not import or call `scripts/run_scan.py` or `scripts/run_full_pipeline.py`.

## Legacy runner status

`scripts/run_scan.py` and `scripts/run_full_pipeline.py` remain legacy migration/archive candidates and were not expanded as canonical runtime authorities.

They remain present as legacy dependencies and migration references:

- `scripts/run_scan.py` still owns the existing broad legacy runtime flow, including scanner execution, production data writes, reporting artifact creation, and Telegram delivery.
- `scripts/run_full_pipeline.py` still shells into `scripts/run_scan.py`.
- Neither file was edited, moved, deleted, wrapped, or given new canonical authority.

## Tests added

Added:

```text
tests/unit/test_v2_canonical_scanner.py
```

Updated:

```text
tests/unit/test_v2_canonical_app.py
```

The tests prove:

- scanner plan is deterministic;
- scanner plan is side-effect-free;
- scanner plan forbids provider calls by default;
- scanner plan forbids production data writes by default;
- scanner plan forbids reports by default;
- scanner plan forbids Telegram delivery by default;
- scanner plan forbids portfolio/watchlist mutation by default;
- scanner plan produces no investment recommendation behavior;
- canonical app uses the canonical scanner boundary;
- canonical app does not import or invoke legacy runner scripts;
- legacy runner scripts were not expanded to import canonical scanner/app modules.

## Side-effect guarantees

The canonical scanner plan records these guarantees for every scanner stage:

```text
provider_calls_allowed = False
data_writes_allowed = False
portfolio_watchlist_mutation_allowed = False
reports_allowed = False
telegram_delivery_allowed = False
```

The scanner boundary has no import-time side effects and does not create files during plan construction.

## Python file creation justification

The canonical scanner files were created because BL28 approved a dedicated scanner/universe ownership boundary. They replace legacy scanner responsibility over time rather than adding competing temporary scanner helpers. Existing relevant modules were inspected first, including `src/market_scanner/app.py`, `scripts/run_scan.py`, `scripts/run_full_pipeline.py`, `scripts/core/scanner.py`, and `scripts/core/data_fetcher.py`. The script-era scanner modules contain live data access and runtime behavior, so the new canonical scanner boundary is intentionally metadata-only until migration is separately approved.

`tests/unit/test_v2_canonical_scanner.py` was created because no existing test file owned canonical scanner-boundary behavior. Existing tests cover legacy runtime scripts, the canonical app boundary, or unrelated v2 scaffolds.

No one-off runtime helper files, temporary migration files, quick scanner files, or parallel shortcut runners were created.

## Remaining migration work

Recommended next migration work:

1. Define the canonical analysis boundary and migrate analysis runtime ownership away from legacy script-era analysis files.
2. Later, migrate scanner/universe logic incrementally from `scripts/core/scanner.py` and `scripts/core/data_fetcher.py` into canonical scanner modules with tests.
3. Keep live provider/data access disconnected until an explicit source-access sprint approves it.
4. Keep production data writes, reporting, Telegram delivery, portfolio/watchlist updates, and Decision Engine behavior disconnected until separately approved.
5. Only after canonical callers exist and tests pass, review whether legacy scanner files can become certified bridges, archive candidates, or delete-after-confirmation candidates.

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

RESET-10L-BL31 — Migrate Analysis Runtime Logic to Canonical V2 Boundary.
