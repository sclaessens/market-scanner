# v2 Analysis Runtime Boundary Migration

## Status

Completed by RESET-10L-BL31.

## Reset stage

RESET-10L-BL31 — Migrate Analysis Runtime Logic to Canonical V2 Boundary.

## Purpose

This sprint starts analysis runtime migration away from legacy script-era analysis files and toward the canonical v2 runtime architecture defined in BL28.

The sprint establishes a side-effect-free canonical analysis boundary under:

```text
src/market_scanner/analysis/
```

The new boundary is deterministic and dry-run/planning-only. It does not execute real analysis, fetch providers, read or write production CSVs, modify portfolio/watchlist files, generate reports, send Telegram messages, trigger delivery behavior, or produce final investment recommendations.

## Policies applied

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_python_architecture_cleanup_and_legacy_decoupling_review.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_legacy_runtime_entrypoint_migration.md`
- `docs/active/v2_scanner_runtime_boundary_migration.md`

Policy result:

- New analysis Python files were created only because BL28 approved `src/market_scanner/analysis/` as the canonical analysis ownership boundary.
- A new analysis unit test file was created because no existing test file owned canonical analysis-boundary behavior.
- Existing legacy runner and legacy analysis files were inspected but not edited.
- No one-off analysis helper, quick analysis runner, migration bridge, ticker-specific file, or parallel runtime shortcut was created.

## Files inspected

Governance and backlog:

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_python_architecture_cleanup_and_legacy_decoupling_review.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_legacy_runtime_entrypoint_migration.md`
- `docs/active/v2_scanner_runtime_boundary_migration.md`
- `docs/active/backlog.md`

Canonical app, scanner, and tests:

- `src/market_scanner/app.py`
- `src/market_scanner/scanner/__init__.py`
- `src/market_scanner/scanner/scanner_contracts.py`
- `src/market_scanner/scanner/scanner_boundary.py`
- `tests/unit/test_v2_canonical_app.py`
- `tests/unit/test_v2_canonical_scanner.py`

Legacy runtime and analysis files:

- `scripts/run_scan.py`
- `scripts/run_full_pipeline.py`
- `scripts/fundamentals/build_analysis.py`
- `scripts/fundamentals/build_metrics.py`
- `scripts/fundamentals/build_quality.py`
- `scripts/core/build_fundamental_analysis.py`
- `scripts/core/decision_engine.py`

Canonical v2 fundamentals and review-adjacent files:

- `src/market_scanner/fundamentals/fundamentals_provider_adapter.py`
- `src/market_scanner/decisions/decision_engine.py`
- `src/market_scanner/decisions/decision_records.py`

Repository structure and references were inspected with static file inventory and grep searches for analysis, fundamental analysis, portfolio review, portfolio intelligence, Decision Engine, review, legacy runner references, and forbidden investment semantics.

## Files changed

- `src/market_scanner/app.py`
- `src/market_scanner/analysis/__init__.py`
- `src/market_scanner/analysis/analysis_contracts.py`
- `src/market_scanner/analysis/analysis_boundary.py`
- `tests/unit/test_v2_canonical_app.py`
- `tests/unit/test_v2_canonical_analysis.py`
- `docs/active/v2_analysis_runtime_boundary_migration.md`
- `docs/active/backlog.md`

No legacy runtime or legacy analysis files were changed.

## Canonical analysis boundary result

The canonical analysis boundary is now established under:

```text
src/market_scanner/analysis/
```

Public functions:

- `build_analysis_plan`
- `build_analysis_input_policy`
- `build_fundamental_analysis_plan`

Core records:

- `AnalysisPlan`
- `AnalysisStage`
- `AnalysisInputPolicy`

The analysis plan describes:

- analysis stage name;
- input evidence category;
- required upstream ownership;
- whether provider calls are allowed;
- whether data writes are allowed;
- whether reports are allowed;
- whether Telegram delivery is allowed;
- whether portfolio/watchlist mutation is allowed;
- whether final output behavior is allowed;
- whether capital-action, priority-label, numeric-rank, price-projection, or execution-quality outputs are allowed;
- legacy migration status.

The initial analysis boundary has three planning stages:

```text
fundamental_evidence_review
profile_evidence_review
limitation_review
```

All stages are side-effect-free by default.

The boundary can reference already-governed evidence concepts, including source-derived FreeCashFlow, governed growth evidence, cash-flow profile evidence, growth profile evidence, and review limitations. It does not compute new financial features or execute legacy analysis.

## App integration result

`src/market_scanner/app.py` now imports the canonical analysis boundary and includes the deterministic analysis plan in the canonical app dry-run runtime plan.

The canonical app analysis stage now references:

```text
src/market_scanner/analysis/
```

with status:

```text
canonical_boundary_established
```

The app still does not import or call `scripts/run_scan.py` or `scripts/run_full_pipeline.py`.

## Legacy analysis status

Legacy analysis responsibilities remain migration targets and were not expanded.

Current legacy analysis authority remains concentrated in:

- `scripts/fundamentals/build_analysis.py`
- `scripts/fundamentals/build_metrics.py`
- `scripts/fundamentals/build_quality.py`
- `scripts/core/build_fundamental_analysis.py`

These files still own script-era CSV analysis behavior, metrics construction, quality construction, and compatibility re-export behavior. They remain present as legacy dependencies and migration references. They were not edited, moved, deleted, wrapped, or given new canonical authority.

## Legacy runner status

`scripts/run_scan.py` and `scripts/run_full_pipeline.py` remain legacy migration/archive candidates and were not expanded as canonical runtime authorities.

They remain present as legacy dependencies and migration references:

- `scripts/run_scan.py` still owns the existing broad legacy runtime flow, including scanner execution, production data writes, reporting artifact creation, and Telegram delivery.
- `scripts/run_full_pipeline.py` still shells into `scripts/run_scan.py`.
- Neither file was edited, moved, deleted, wrapped, or given new canonical authority.

## Tests added

Added:

```text
tests/unit/test_v2_canonical_analysis.py
```

Updated:

```text
tests/unit/test_v2_canonical_app.py
```

The tests prove:

- analysis plan is deterministic;
- analysis plan is side-effect-free;
- analysis plan forbids provider calls by default;
- analysis plan forbids production data writes by default;
- analysis plan forbids reports by default;
- analysis plan forbids Telegram delivery by default;
- analysis plan forbids portfolio/watchlist mutation by default;
- analysis plan forbids final BUY/SELL/HOLD by default;
- analysis plan forbids allocation/conviction/urgency/scoring/target-price/tradeability/recommendation behavior;
- canonical app uses the canonical analysis boundary;
- canonical app does not import or invoke legacy runner scripts;
- legacy runner and legacy analysis files were not expanded to import canonical analysis/app modules.

## Side-effect guarantees

The canonical analysis plan records these guarantees for every analysis stage:

```text
provider_calls_allowed = False
data_writes_allowed = False
reports_allowed = False
telegram_delivery_allowed = False
portfolio_watchlist_mutation_allowed = False
final_outcomes_allowed = False
capital_action_outputs_allowed = False
priority_label_outputs_allowed = False
numeric_rank_outputs_allowed = False
price_projection_outputs_allowed = False
execution_quality_outputs_allowed = False
```

The analysis boundary has no import-time side effects and does not create files during plan construction.

## Python file creation justification

The canonical analysis files were created because BL28 approved a dedicated analysis ownership boundary. They replace scattered or legacy analysis responsibility over time rather than adding competing temporary analysis helpers. Existing relevant modules were inspected first, including `src/market_scanner/app.py`, `src/market_scanner/scanner/`, `scripts/run_scan.py`, `scripts/run_full_pipeline.py`, `scripts/fundamentals/build_analysis.py`, `scripts/fundamentals/build_metrics.py`, `scripts/fundamentals/build_quality.py`, and `scripts/core/build_fundamental_analysis.py`. The script-era analysis files contain CSV runtime behavior and optional production writes, so the new canonical analysis boundary is intentionally metadata-only until migration is separately approved.

`tests/unit/test_v2_canonical_analysis.py` was created because no existing test file owned canonical analysis-boundary behavior. Existing tests cover legacy analysis behavior, canonical app behavior, canonical scanner behavior, or unrelated v2 scaffolds.

No one-off runtime helper files, temporary migration files, ticker-specific files, quick analysis files, or parallel shortcut runners were created.

## Remaining migration work

Recommended next migration work:

1. Define the canonical Decision Review runtime boundary and keep Decision Engine allocation authority isolated.
2. Later, migrate review-only analysis behavior from `scripts/fundamentals/build_analysis.py` into canonical analysis modules with tests.
3. Keep legacy metrics and quality construction decoupling separate from analysis runtime ownership.
4. Keep live provider/data access disconnected until an explicit source-access sprint approves it.
5. Keep production data writes, reporting, Telegram delivery, portfolio/watchlist updates, and Decision Engine behavior disconnected until separately approved.
6. Only after canonical callers exist and tests pass, review whether legacy analysis files can become certified bridges, archive candidates, or delete-after-confirmation candidates.

## Guardrails confirmation

- No credentials committed.
- No raw live payloads committed.
- No production data writes.
- No reports generated.
- No Telegram artifacts generated.
- No unsafe production pipeline execution.
- No portfolio/watchlist updates.
- No final BUY/SELL/HOLD recommendation.
- No final BUY/SELL/HOLD, allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior was added.
- No missing values converted to zero.
- No one-off temporary helper files created.
- Legacy runners were not expanded.
- Legacy analysis files were not expanded.

## Next recommended step

RESET-10L-BL32 — Migrate Decision Review Runtime Logic to Canonical V2 Boundary.

