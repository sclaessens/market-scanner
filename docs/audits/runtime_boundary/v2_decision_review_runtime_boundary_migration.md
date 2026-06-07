# v2 Decision Review Runtime Boundary Migration

## Status

Completed by RESET-10L-BL32.

## Reset stage

RESET-10L-BL32 — Migrate Decision Review Runtime Logic to Canonical V2 Boundary.

## Purpose

This sprint starts decision/review runtime migration away from legacy Decision Engine execution paths and toward the canonical v2 runtime architecture defined in BL28.

The sprint establishes a side-effect-free canonical decision/review boundary under:

```text
src/market_scanner/decision/
```

The new boundary is deterministic and dry-run/planning-only. It is not a recommendation engine. It does not execute the legacy Decision Engine, produce investment decisions, fetch providers, read or write production CSVs, modify portfolio/watchlist files, generate reports, send Telegram messages, or trigger delivery behavior.

## Policies applied

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_python_architecture_cleanup_and_legacy_decoupling_review.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_legacy_runtime_entrypoint_migration.md`
- `docs/active/v2_scanner_runtime_boundary_migration.md`
- `docs/active/v2_analysis_runtime_boundary_migration.md`

Policy result:

- New decision/review Python files were created only because BL28 approved `src/market_scanner/decision/` as the canonical decision/review ownership boundary.
- A new decision unit test file was created because no existing test file owned canonical decision/review-boundary behavior.
- Existing legacy runner, legacy Decision Engine, reporting, Telegram, portfolio, and watchlist files were inspected but not edited.
- No one-off decision helper, recommendation engine, quick decision runner, migration bridge, or parallel runtime shortcut was created.

## Files inspected

Governance and backlog:

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_python_architecture_cleanup_and_legacy_decoupling_review.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_legacy_runtime_entrypoint_migration.md`
- `docs/active/v2_scanner_runtime_boundary_migration.md`
- `docs/active/v2_analysis_runtime_boundary_migration.md`
- `docs/active/backlog.md`

Canonical app, scanner, analysis, and tests:

- `src/market_scanner/app.py`
- `src/market_scanner/scanner/`
- `src/market_scanner/analysis/`
- `tests/unit/test_v2_canonical_app.py`
- `tests/unit/test_v2_canonical_scanner.py`
- `tests/unit/test_v2_canonical_analysis.py`

Legacy runtime and decision/review files:

- `scripts/run_scan.py`
- `scripts/run_full_pipeline.py`
- `scripts/core/decision_engine.py`
- `src/market_scanner/decisions/decision_engine.py`
- `src/market_scanner/decisions/decision_records.py`
- `scripts/core/build_portfolio_intelligence.py`
- `scripts/portfolio/evaluate_positions.py`
- `scripts/reporting/build_reporting_layer.py`
- `scripts/reporting/build_telegram_summary.py`
- `scripts/reporting/send_telegram.py`
- `src/market_scanner/reporting/`
- `src/market_scanner/portfolio/`

Repository structure and references were inspected with static file inventory and grep searches for decision, review, final decision, portfolio review, portfolio intelligence, reporting, Telegram, and forbidden investment semantics.

## Files changed

- `src/market_scanner/app.py`
- `src/market_scanner/decision/__init__.py`
- `src/market_scanner/decision/decision_contracts.py`
- `src/market_scanner/decision/decision_boundary.py`
- `tests/unit/test_v2_canonical_app.py`
- `tests/unit/test_v2_canonical_decision.py`
- `docs/active/v2_decision_review_runtime_boundary_migration.md`
- `docs/active/backlog.md`

No legacy runtime, legacy Decision Engine, reporting, Telegram, portfolio, or watchlist files were changed.

## Canonical decision/review boundary result

The canonical decision/review boundary is now established under:

```text
src/market_scanner/decision/
```

Public functions:

- `build_decision_review_plan`
- `build_decision_review_stage`
- `build_review_policy`

Core records:

- `DecisionReviewPlan`
- `DecisionReviewStage`
- `ReviewPolicy`

The decision/review plan describes:

- decision/review stage name;
- upstream analysis evidence category;
- allowed review states;
- blocked final state codes;
- blocked behavior codes;
- whether provider calls are allowed;
- whether data writes are allowed;
- whether reports are allowed;
- whether Telegram delivery is allowed;
- whether portfolio/watchlist mutation is allowed;
- whether final output behavior is allowed;
- whether capital-action, priority-label, numeric-rank, price-projection, or execution-quality outputs are allowed;
- legacy migration status.

The initial decision/review boundary has two planning stages:

```text
review_state_boundary
policy_block_review
```

All stages are side-effect-free by default.

Allowed review-oriented states:

```text
review_required
limited_analysis
insufficient_evidence
evidence_available
blocked_by_policy
```

## App integration result

`src/market_scanner/app.py` now imports the canonical decision/review boundary and includes the deterministic decision/review plan in the canonical app dry-run runtime plan.

The canonical app decision/review stage now references:

```text
src/market_scanner/decision/
```

with status:

```text
canonical_boundary_established
```

The app still does not import or call `scripts/run_scan.py`, `scripts/run_full_pipeline.py`, legacy Decision Engine execution, legacy report files, or legacy Telegram files.

## Legacy Decision Engine status

Legacy Decision Engine responsibilities remain migration targets and were not expanded.

Current legacy and review-adjacent decision authority remains concentrated in:

- `scripts/core/decision_engine.py`
- `src/market_scanner/decisions/decision_engine.py`
- `src/market_scanner/decisions/decision_records.py`

`scripts/core/decision_engine.py` still owns script-era final decision CSV behavior and production output paths. `src/market_scanner/decisions/` still contains the older v2 review-only scaffold. These files remain present as legacy dependencies and migration references. They were not edited, moved, deleted, wrapped, or given new canonical authority.

## Legacy runner status

`scripts/run_scan.py` and `scripts/run_full_pipeline.py` remain legacy migration/archive candidates and were not expanded as canonical runtime authorities.

They remain present as legacy dependencies and migration references:

- `scripts/run_scan.py` still owns the existing broad legacy runtime flow, including scanner execution, production data writes, reporting artifact creation, Telegram delivery, and legacy Decision Engine execution.
- `scripts/run_full_pipeline.py` still shells into `scripts/run_scan.py`.
- Neither file was edited, moved, deleted, wrapped, or given new canonical authority.

## Tests added

Added:

```text
tests/unit/test_v2_canonical_decision.py
```

Updated:

```text
tests/unit/test_v2_canonical_app.py
```

The tests prove:

- decision/review plan is deterministic;
- decision/review plan is side-effect-free;
- decision/review plan forbids provider calls by default;
- decision/review plan forbids production data writes by default;
- decision/review plan forbids reports by default;
- decision/review plan forbids Telegram delivery by default;
- decision/review plan forbids portfolio/watchlist mutation by default;
- decision/review plan forbids final BUY/SELL/HOLD by default;
- decision/review plan forbids allocation/conviction/urgency/scoring/target-price/tradeability/recommendation behavior;
- decision/review plan exposes review-oriented states only;
- decision/review plan explicitly lists blocked final decision semantics;
- canonical app uses the canonical decision/review boundary;
- canonical app does not import or invoke legacy runner scripts;
- canonical decision/review boundary does not import or invoke legacy Decision Engine execution;
- legacy runner and legacy Decision Engine files were not expanded to import canonical decision/app modules.

## Side-effect guarantees

The canonical decision/review plan records these guarantees for every decision/review stage:

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

The decision/review boundary has no import-time side effects and does not create files during plan construction.

## Python file creation justification

The canonical decision/review files were created because BL28 approved a dedicated decision/review ownership boundary. They replace legacy Decision Engine responsibility over time rather than adding competing temporary decision helpers or recommendation logic. Existing relevant modules were inspected first, including `src/market_scanner/app.py`, `src/market_scanner/scanner/`, `src/market_scanner/analysis/`, `scripts/run_scan.py`, `scripts/run_full_pipeline.py`, `scripts/core/decision_engine.py`, `src/market_scanner/decisions/decision_engine.py`, `src/market_scanner/decisions/decision_records.py`, reporting modules, Telegram modules, and portfolio/review modules. The legacy Decision Engine files contain runtime behavior and production writes, so the new canonical decision/review boundary is intentionally metadata-only until migration is separately approved.

`tests/unit/test_v2_canonical_decision.py` was created because no existing test file owned canonical decision/review-boundary behavior. Existing tests cover legacy Decision Engine behavior, canonical app behavior, canonical scanner behavior, canonical analysis behavior, or unrelated v2 scaffolds.

No one-off runtime helper files, temporary migration files, recommendation engines, quick decision files, or parallel shortcut runners were created.

## Blocked investment semantics

The canonical decision/review boundary explicitly blocks these final state codes:

```text
buy
sell
hold
allocate
increase_position
reduce_position
target_price
tradeable
not_tradeable
```

The boundary also blocks these behavior codes:

```text
allocation
conviction
urgency
scoring
target-price
tradeability
recommendation
```

These values are listed only as blocked policy metadata. They are not emitted as review outcomes, not executed, and not connected to legacy Decision Engine behavior.

## Remaining migration work

Recommended next migration work:

1. Define the canonical message composition runtime boundary while keeping message composition separate from report generation and delivery.
2. Later, decide whether the older `src/market_scanner/decisions/` scaffold should be migrated into `src/market_scanner/decision/`, retained temporarily, or archived after confirmation.
3. Keep legacy final decision CSV generation disconnected until an explicit Decision Engine migration sprint approves it.
4. Keep production data writes, reporting, Telegram delivery, portfolio/watchlist updates, and investment behavior disconnected until separately approved.
5. Only after canonical callers exist and tests pass, review whether legacy Decision Engine files can become certified bridges, archive candidates, or delete-after-confirmation candidates.

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
- Legacy Decision Engine files were not expanded.

## Next recommended step

RESET-10L-BL33 — Migrate Message Composition Runtime Logic to Canonical V2 Boundary.

