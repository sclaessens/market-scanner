# v2 Legacy Runtime Script Archive Readiness Review

## Status

Completed by RESET-10L-BL36.

## Reset stage

RESET-10L-BL36 — Legacy Runtime Script Archive Readiness Review.

## Purpose

This review determines whether the primary legacy runtime scripts are ready to archive now that canonical v2 runtime boundaries exist for the app, scanner, analysis, decision/review, messaging, reporting, and delivery layers.

Reviewed legacy runtime scripts:

```text
scripts/run_scan.py
scripts/run_full_pipeline.py
```

Conclusion: neither script is archive-ready. Both remain active legacy dependencies, and `scripts/run_scan.py` still owns broad executable runtime behavior and side effects that are not yet migrated into executable canonical v2 boundaries.

This sprint is review-only. No Python files, tests, workflows, data files, report files, portfolio/watchlist files, or legacy scripts were changed.

## Policies applied

- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_python_architecture_cleanup_and_legacy_decoupling_review.md`
- `docs/active/v2_legacy_runtime_entrypoint_migration.md`
- `docs/active/v2_scanner_runtime_boundary_migration.md`
- `docs/active/v2_analysis_runtime_boundary_migration.md`
- `docs/active/v2_decision_review_runtime_boundary_migration.md`
- `docs/active/v2_message_composition_runtime_boundary_migration.md`
- `docs/active/v2_report_artifact_runtime_boundary_migration.md`
- `docs/active/v2_delivery_runtime_boundary_migration.md`
- `docs/active/backlog.md`

Policy application:

- A legacy Python file that is still used is not automatically approved for long-term retention.
- The canonical v2 packages now define ownership boundaries, but most are planning-only and do not yet replace legacy executable behavior.
- Archive or deletion requires removing active source/test/workflow/operator dependencies and migrating or retiring unique logic first.

## Files reviewed

Primary review targets:

- `scripts/run_scan.py`
- `scripts/run_full_pipeline.py`

Canonical runtime boundary files and packages:

- `src/market_scanner/app.py`
- `src/market_scanner/scanner/`
- `src/market_scanner/analysis/`
- `src/market_scanner/decision/`
- `src/market_scanner/messaging/`
- `src/market_scanner/reporting/`
- `src/market_scanner/delivery/`

Active workflow/test reference locations inspected:

- `.github/workflows/daily-market-scan.yml`
- `tests/core/test_fundamentals_runtime_organization.py`
- `tests/core/test_run_full_pipeline.py`
- `tests/test_operator_visibility.py`
- canonical v2 boundary tests that read legacy scripts as migration-target evidence

## Inspection method

Static inspection only. The legacy scripts were not executed.

Commands and inspection patterns used:

```bash
grep -R "scripts/run_scan.py\|scripts/run_full_pipeline.py\|run_scan.py\|run_full_pipeline.py\|run_scan\|run_full_pipeline" -n . \
  --exclude-dir=.venv \
  --exclude-dir=venv \
  --exclude-dir=__pycache__ \
  --exclude-dir=.git

grep -R "import .*run_scan\|from .*run_scan\|import .*run_full_pipeline\|from .*run_full_pipeline" -n . --include="*.py" \
  --exclude-dir=.venv \
  --exclude-dir=venv \
  --exclude-dir=__pycache__ \
  --exclude-dir=.git

grep -R "if __name__ == .__main__." -n . --include="*.py" \
  --exclude-dir=.venv \
  --exclude-dir=venv \
  --exclude-dir=__pycache__ \
  --exclude-dir=.git

grep -R "run_scan\|run_full_pipeline\|scripts/" -n .github docs scripts tests src \
  --exclude-dir=.venv \
  --exclude-dir=venv \
  --exclude-dir=__pycache__ \
  --exclude-dir=.git || true

grep -n "telegram\|Telegram\|send\|report\|write\|csv\|portfolio\|watchlist\|decision\|BUY\|SELL\|HOLD\|recommendation\|tradeability" scripts/run_scan.py scripts/run_full_pipeline.py || true
```

Notes:

- The broad legacy-reference grep also surfaced `.pytest_cache` node IDs from prior local test runs. Those cache entries are not treated as active dependencies.
- Historical `docs/archive/` references are treated as evidence only. Active `docs/active/`, `.github`, `tests`, `scripts`, and `src` references are treated as active or migration-relevant.

## Static search summary

Reference search found:

- active workflow invocation of `python scripts/run_scan.py`;
- active tests importing `scripts.run_scan` and `scripts.run_full_pipeline`;
- active canonical app and boundary metadata listing the scripts as legacy authorities or migration targets;
- active documentation repeatedly identifying the scripts as legacy migration/archive candidates;
- historical archive documentation with older operator instructions for both scripts;
- `scripts/run_full_pipeline.py` constructing and running `[sys.executable, "scripts/run_scan.py"]`.

Import search found direct test imports:

```text
tests/core/test_fundamentals_runtime_organization.py
tests/core/test_run_full_pipeline.py
tests/test_operator_visibility.py
```

Entrypoint search confirmed both reviewed scripts remain runnable:

```text
scripts/run_scan.py
scripts/run_full_pipeline.py
```

Workflow search found active runtime use:

```text
.github/workflows/daily-market-scan.yml
```

The workflow also invokes:

```text
scripts/telegram/process_telegram_commands.py
```

That Telegram command processing file is not part of this archive-readiness decision, but it confirms remaining legacy runtime coupling around delivery and command handling.

## Reference summary

| Reference area | `scripts/run_scan.py` | `scripts/run_full_pipeline.py` | Archive-readiness impact |
|---|---|---|---|
| Source code | Referenced as legacy authority metadata by canonical app/scanner/delivery boundaries; imports many script-era modules. | Constructs a subprocess command to run `scripts/run_scan.py`. | Metadata references alone do not block archive, but wrapper dependency does. |
| Tests | Imported directly and heavily monkeypatched in `tests/core/test_run_full_pipeline.py` and `tests/test_operator_visibility.py`; namespace expectations in `tests/core/test_fundamentals_runtime_organization.py`. | Imported directly in `tests/core/test_run_full_pipeline.py` and `tests/test_operator_visibility.py`. | Active test dependencies block immediate archive. |
| Docs | Active docs classify it as a legacy migration/archive candidate and note it still owns broad runtime behavior. | Active docs classify it as a legacy migration/archive candidate and wrapper around `run_scan.py`. | Docs support review/migration, not immediate archive. |
| Workflows | `.github/workflows/daily-market-scan.yml` invokes `python scripts/run_scan.py`. | No active workflow invocation found. | Active workflow dependency blocks immediate archive for `run_scan.py`. |
| Operator procedure | Historical docs and tests preserve script-era operator behavior. | Historical docs and tests preserve wrapper behavior. | Operator replacement needs an approved canonical path first. |

## Entrypoint summary

The repository still contains many runnable Python files. For this review, the important finding is that both primary legacy runtime scripts remain explicit entrypoints:

```text
scripts/run_scan.py: if __name__ == "__main__"
scripts/run_full_pipeline.py: if __name__ == "__main__"
```

`src/market_scanner/app.py` is the canonical v2 application boundary, but it currently returns a dry-run plan and fails closed for non-dry-run execution. It does not yet replace executable scanner, analysis, report, delivery, portfolio, or Decision Engine behavior.

## Side-effect summary

`scripts/run_scan.py` still contains or triggers:

- provider/data-source calls through `fetch_ohlcv_data`;
- scanner execution through `load_tickers`, `add_indicators`, `scan_ticker`, and `rank_setups`;
- production directory creation through `ensure_dirs`;
- CSV writes to `data/logs`, `data/processed`, and `data/portfolio`;
- optional fundamental metrics and analysis writes;
- validation, context, fundamentals, timing, portfolio, portfolio intelligence, and final decision layer execution;
- Decision Engine execution through `build_final_decisions`;
- report generation and report/dashboard/log writes through `build_reporting_layer` and `write_reporting_outputs`;
- Telegram message artifact creation through the reporting layer;
- Telegram sending through `send_daily_summary`;
- final decision/final action behavior by invoking the legacy Decision Engine.

`scripts/run_full_pipeline.py` still contains or triggers:

- subprocess execution of `scripts/run_scan.py`;
- full legacy pipeline execution through that subprocess;
- the same side-effect risk as `scripts/run_scan.py` whenever the subprocess is allowed to run.

## Per-script archive-readiness table

| File path | Primary status | References found | Unique logic remaining | Side-effect risk | Canonical owner | Archive recommendation | Required next action |
|---|---|---|---|---|---|---|---|
| `scripts/run_scan.py` | NOT_ARCHIVE_READY_DEPENDENCY_REMAINS | Workflow invocation, active test imports, active docs, canonical metadata references. | Broad runtime orchestration, scanner loop, output sequencing, optional fundamentals bridge, layer ordering, reporting/delivery ordering, operator progress messages. | High: provider calls, data writes, report writes, Telegram artifact creation, Telegram send attempt, portfolio/watchlist-adjacent behavior, Decision Engine execution. | `src/market_scanner/app.py` for entrypoint; `src/market_scanner/scanner/`, `analysis/`, `decision/`, `messaging/`, `reporting/`, `delivery/`, `fundamentals/`, `portfolio/`, and future config/shared owners for responsibilities. | Do not archive now. | Decouple workflow/tests from legacy runner, migrate or retire remaining executable responsibilities, then rerun archive-readiness review. |
| `scripts/run_full_pipeline.py` | NOT_ARCHIVE_READY_DEPENDENCY_REMAINS | Active test imports, historical/active docs, direct subprocess call to `scripts/run_scan.py`. | Wrapper command construction, operator progress/failure output, optional fundamentals argument forwarding. | High by delegation: running it executes `scripts/run_scan.py` and its side effects. | `src/market_scanner/app.py` for canonical app boundary; future canonical operator/runtime command policy for executable entrypoint. | Do not archive now. | Replace test/operator dependency with canonical dry-run or approved executable entrypoint, then retire wrapper after `run_scan.py` dependency is removed. |

## Per-script detailed findings

### `scripts/run_scan.py`

What it currently does:

- Owns the broad end-to-end script-era market scan runtime.
- Loads ticker universe and market regime data.
- Fetches OHLCV data for QQQ, SPY, and each ticker.
- Builds scanner output, validation, context, fundamentals, timing state, portfolio state, portfolio review, portfolio intelligence, final decisions, reporting outputs, and Telegram delivery.
- Prints operator progress and artifact messages.

Is it still referenced by source code?

- Yes, as legacy authority metadata in canonical v2 app/scanner/delivery boundaries.
- It is not imported by canonical v2 execution code.

Is it still referenced by tests?

- Yes. Active tests import and monkeypatch it directly.
- `tests/core/test_run_full_pipeline.py` validates its fundamentals pipeline and end-to-end ordering under monkeypatches.
- `tests/test_operator_visibility.py` validates operator output and runtime ordering.
- `tests/core/test_fundamentals_runtime_organization.py` validates namespace wiring through `scripts.run_scan`.

Is it still referenced by docs?

- Yes. Active docs classify it as a legacy migration/archive candidate.
- Historical docs preserve older operator instructions and audit references.

Is it still referenced by workflows?

- Yes. `.github/workflows/daily-market-scan.yml` invokes `python scripts/run_scan.py`.

Is it still the only owner of any runtime responsibility?

- Yes. It is still the only executable owner of the current full market scan sequence from live scanner through report and Telegram delivery.
- Canonical v2 boundaries now own planning metadata and responsibility maps, but they intentionally do not execute production runtime behavior.

Which canonical v2 boundary now owns or should own that responsibility?

- Application entrypoint: `src/market_scanner/app.py`
- Scanner/universe planning: `src/market_scanner/scanner/`
- Provider/fundamentals evidence: `src/market_scanner/fundamentals/`
- Analysis planning: `src/market_scanner/analysis/`
- Decision/review planning: `src/market_scanner/decision/`
- Message composition planning: `src/market_scanner/messaging/`
- Report artifact planning: `src/market_scanner/reporting/`
- Delivery planning: `src/market_scanner/delivery/`
- Portfolio/config executable ownership still needs a safe canonical migration decision before archive.

Does the script contain logic not yet represented in canonical boundaries?

- Yes. The canonical boundaries do not yet execute the scanner loop, live market data calls, output sequencing, layer calls, portfolio state construction, final Decision Engine execution, reporting writes, or Telegram delivery.

Does it produce side effects?

- Yes. It writes CSVs, builds report artifacts, creates report output paths, attempts Telegram delivery, and invokes modules that read/write production paths.

Does it write data or reports?

- Yes. It writes scanner and failed ticker CSVs directly and calls downstream builders that write processed, portfolio, log, report, and Telegram message artifacts.

Does it send Telegram messages?

- Yes, by calling `send_daily_summary`.

Does it use credentials or network calls?

- It imports `send_daily_summary`, which reads Telegram credentials when executed. It also calls scanner data fetch functions that can perform provider/network access.

Does it trigger Decision Engine or final recommendation behavior?

- Yes. It calls `build_final_decisions` from `scripts/core/decision_engine.py`.

Can the file be archived immediately?

- No.

What must be decoupled first?

- Active workflow invocation.
- Active tests that import and monkeypatch `scripts.run_scan`.
- Unique runtime sequencing and operator output behavior.
- Live scanner/data-fetch behavior.
- Production data/report write behavior.
- Decision Engine invocation.
- Telegram artifact/delivery behavior.
- Portfolio and portfolio-intelligence runtime calls.

What tests or controlled checks are needed before archive/delete?

- Tests proving `.github` no longer invokes `scripts/run_scan.py`.
- Tests proving canonical app or an approved operator entrypoint owns the runtime sequence.
- Tests proving canonical boundaries can execute or explicitly fail closed for scanner, analysis, decision/review, reporting, and delivery responsibilities.
- Controlled side-effect checks proving no production data/report/Telegram side effects move into the wrong boundary.
- Search checks proving no active source/test/workflow dependency remains.

### `scripts/run_full_pipeline.py`

What it currently does:

- Owns a script-era full-pipeline wrapper.
- Builds a subprocess command targeting `scripts/run_scan.py`.
- Forwards optional fundamentals paths.
- Prints wrapper-level start/completion/failure output.

Is it still referenced by source code?

- It is not imported by canonical v2 source code.
- It references `scripts/run_scan.py` directly as the command it runs.

Is it still referenced by tests?

- Yes. Active tests import `scripts.run_full_pipeline` directly and validate command construction, optional argument forwarding, and operator output.

Is it still referenced by docs?

- Yes. Active docs classify it as a legacy migration/archive candidate. Historical docs preserve older full-pipeline operator instructions.

Is it still referenced by workflows?

- No active workflow invocation was found for `scripts/run_full_pipeline.py`.

Is it still the only owner of any runtime responsibility?

- It is the only current wrapper preserving the old operator-facing full-pipeline command shape.
- It delegates actual runtime behavior to `scripts/run_scan.py`.

Which canonical v2 boundary now owns or should own that responsibility?

- `src/market_scanner/app.py` owns the canonical app boundary.
- A future approved operator/runtime command policy should define whether an executable wrapper remains necessary and where it lives.

Does the script contain logic not yet represented in canonical boundaries?

- Yes. It owns wrapper command construction and operator progress/failure output for the script-era full pipeline. Canonical app dry-run does not yet offer equivalent executable wrapper behavior.

Does it produce side effects?

- Yes by delegation. `run_step` executes the constructed subprocess command, which runs `scripts/run_scan.py`.

Does it write data or reports?

- Not directly, but executing it triggers `scripts/run_scan.py`, which writes data and reports.

Does it send Telegram messages?

- Not directly, but executing it triggers `scripts/run_scan.py`, which attempts Telegram delivery.

Does it use credentials or network calls?

- Not directly, but executing it triggers the network/credential risks of `scripts/run_scan.py`.

Does it trigger Decision Engine or final recommendation behavior?

- Not directly, but executing it triggers `scripts/run_scan.py`, which invokes the legacy Decision Engine.

Can the file be archived immediately?

- No.

What must be decoupled first?

- Active tests that import `scripts.run_full_pipeline`.
- Wrapper dependency on `scripts/run_scan.py`.
- Operator-facing command behavior, if still needed, must be represented by a canonical approved path or retired by governance.

What tests or controlled checks are needed before archive/delete?

- Search checks proving no active source/test/workflow dependency remains.
- Tests proving replacement operator path exists or wrapper behavior is intentionally retired.
- Tests proving no canonical app behavior depends on the wrapper.
- Controlled check that archiving it does not remove required CLI/operator behavior.

## Canonical boundary coverage assessment

Canonical boundaries now cover responsibility ownership, but mostly as side-effect-free planning boundaries:

| Responsibility | Canonical owner | Coverage today | Archive-readiness note |
|---|---|---|---|
| Application entrypoint | `src/market_scanner/app.py` | Dry-run plan and fail-closed non-dry-run behavior. | Does not yet replace executable `run_scan.py` behavior. |
| Scanner/universe | `src/market_scanner/scanner/` | Planning stages only. | Live scanner loop and data fetch remain in legacy scripts. |
| Provider/fundamentals | `src/market_scanner/fundamentals/` | Strongest canonical implementation coverage for governed evidence/persistence. | Existing script-era fundamentals metrics/quality/analysis still used by runtime tests. |
| Analysis | `src/market_scanner/analysis/` | Planning stages only. | Legacy CSV analysis behavior remains outside canonical execution. |
| Decision/review | `src/market_scanner/decision/` | Review planning only; explicitly blocks investment semantics. | Legacy Decision Engine remains current allocation authority when `run_scan.py` executes. |
| Messaging | `src/market_scanner/messaging/` | Composition planning only. | Legacy reporting layer still builds Telegram text. |
| Reporting | `src/market_scanner/reporting/` | Report artifact planning plus existing reporting scaffolds. | Legacy reporting scripts still write artifacts. |
| Delivery | `src/market_scanner/delivery/` | Delivery planning only; blocks network/credentials/Telegram execution. | Legacy Telegram send/polling remains outside canonical execution. |
| Portfolio/config/operator CLI | No fully approved executable owner yet. | Contracts and legacy scripts exist. | Runtime migration still needed before archiving broad runner scripts. |

## Remaining dependency map

Active dependencies blocking archive:

- `.github/workflows/daily-market-scan.yml` invokes `scripts/run_scan.py`.
- `tests/core/test_fundamentals_runtime_organization.py` imports `scripts.run_scan`.
- `tests/core/test_run_full_pipeline.py` imports `scripts.run_full_pipeline` and `scripts.run_scan`.
- `tests/test_operator_visibility.py` imports `scripts.run_full_pipeline` and `scripts.run_scan`.
- `scripts/run_full_pipeline.py` executes `scripts/run_scan.py` by subprocess.
- Canonical tests read legacy scripts as migration evidence; these references should be updated only after archive readiness is achieved.
- Active docs intentionally list the scripts as legacy migration/archive candidates.

## Logic still requiring migration

Before either script can be archived, the project must migrate, replace, or retire:

- workflow entrypoint behavior;
- operator progress and artifact messaging;
- scanner universe loading and live scan loop;
- market regime classification orchestration;
- scanner output preparation, sorting, duplicate handling, and failed ticker recording;
- validation/context/fundamentals/timing/portfolio/final-decision layer sequencing;
- optional raw fundamentals history path forwarding and build order;
- production path policy and directory creation;
- reporting output write sequence;
- Telegram delivery invocation;
- final Decision Engine invocation and its location in the canonical architecture;
- portfolio/portfolio-intelligence runtime ownership;
- tests that protect legacy command sequencing instead of canonical behavior.

## Archive readiness conclusion

Neither `scripts/run_scan.py` nor `scripts/run_full_pipeline.py` is archive-ready.

Primary conclusion:

```text
NOT_ARCHIVE_READY_DEPENDENCY_REMAINS
```

Reasons:

- active workflow dependency remains for `scripts/run_scan.py`;
- active test imports remain for both scripts;
- `scripts/run_full_pipeline.py` directly depends on `scripts/run_scan.py`;
- unique executable runtime logic remains in `scripts/run_scan.py`;
- canonical v2 boundaries are side-effect-free planning owners and do not yet replace broad executable runtime behavior;
- side-effect risk remains high if these scripts are executed.

Recommended next cleanup step:

```text
RESET-10L-BL37 — Decouple Remaining Legacy Runtime Dependencies
```

This is safer than archive/delete because the first blockers are active dependencies and runtime ownership gaps.

## Recommended next cleanup step

RESET-10L-BL37 should decouple the remaining legacy runtime dependencies before any archive sprint. Suggested scope:

- remove or replace `.github/workflows/daily-market-scan.yml` dependency on `scripts/run_scan.py` with an approved canonical dry-run or explicitly paused runtime path;
- migrate tests that import `scripts.run_scan` and `scripts.run_full_pipeline` toward canonical boundary tests or mark them as legacy-retirement candidates;
- define whether an executable canonical app command is needed beyond dry-run planning;
- keep provider calls, production writes, report generation, Telegram delivery, portfolio/watchlist updates, and Decision Engine behavior blocked unless separately approved.

## Guardrails confirmation

- No Python files changed.
- No tests changed.
- No workflows changed.
- No files moved.
- No files deleted.
- No legacy scripts modified.
- No provider calls made.
- No production pipeline executed.
- No production data writes.
- No reports generated.
- No Telegram artifacts generated.
- No Telegram delivery.
- No network calls.
- No credentials read.
- No portfolio/watchlist updates.
- No Decision Engine behavior changed.
- No BUY/SELL/HOLD/allocation/conviction/urgency/scoring/target-price/tradeability/recommendation behavior added.

## Known limitations

- This review did not execute tests because the sprint is documentation-only and tests were not required.
- This review did not execute legacy scripts or canonical app runtime behavior.
- Static grep output includes historical archive references and local pytest cache entries; those were separated from active source/test/workflow references in the conclusion.
- The review did not decide the future of `scripts/telegram/process_telegram_commands.py`, legacy portfolio command files, or other script-era entrypoints except where they block or inform the two primary reviewed scripts.
