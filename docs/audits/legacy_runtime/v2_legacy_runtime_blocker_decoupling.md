# Legacy Runtime Blocker Decoupling

## Status

Completed by RESET-10L-BL38.

## Reset stage

RESET-10L-BL38 — Decouple Remaining Legacy Runtime Blockers.

## Purpose

Remove the remaining active workflow, test, and wrapper blockers that kept the primary legacy runtime scripts from moving toward archive-readiness recheck.

Primary legacy runtime scripts:

- `scripts/run_scan.py`
- `scripts/run_full_pipeline.py`

This sprint did not delete, move, rename, or archive either script.

## Policies applied

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_python_architecture_cleanup_and_legacy_decoupling_review.md`
- `docs/active/v2_legacy_runtime_script_archive_readiness_review.md`
- `docs/active/v2_legacy_runtime_dependency_decoupling.md`
- `docs/active/backlog.md`
- canonical boundary migration records from RESET-10L-BL29 through RESET-10L-BL35

## Files inspected

- `scripts/run_scan.py`
- `scripts/run_full_pipeline.py`
- `src/market_scanner/app.py`
- `src/market_scanner/scanner/`
- `src/market_scanner/analysis/`
- `src/market_scanner/decision/`
- `src/market_scanner/messaging/`
- `src/market_scanner/reporting/`
- `src/market_scanner/delivery/`
- `.github/workflows/daily-market-scan.yml`
- `tests/core/test_run_full_pipeline.py`
- `tests/test_operator_visibility.py`
- canonical boundary tests under `tests/unit/`
- active governance and migration documents listed above

## Files changed

- `scripts/run_full_pipeline.py`
- `tests/core/test_run_full_pipeline.py`
- `tests/test_operator_visibility.py`
- `docs/active/v2_legacy_runtime_blocker_decoupling.md`
- `docs/active/backlog.md`

No new Python files were created.

## Blockers before BL38

After BL37:

- `scripts/run_scan.py` referenced by workflow: no
- `scripts/run_scan.py` imported or monkeypatched by tests: yes
- `scripts/run_full_pipeline.py` imported or monkeypatched by tests: yes
- `scripts/run_full_pipeline.py` shells into `scripts/run_scan.py`: yes

The remaining active test blockers were:

- `tests/core/test_run_full_pipeline.py`
- `tests/test_operator_visibility.py`

The remaining wrapper blocker was:

- `scripts/run_full_pipeline.py` built and executed a subprocess command for the legacy scan runtime.

## Blockers decoupled in BL38

- `tests/core/test_run_full_pipeline.py` no longer imports or monkeypatches `scripts.run_full_pipeline` or `scripts.run_scan`.
- `tests/test_operator_visibility.py` no longer imports or monkeypatches `scripts.run_full_pipeline` or `scripts.run_scan`.
- `scripts/run_full_pipeline.py` no longer imports `subprocess`.
- `scripts/run_full_pipeline.py` no longer builds a command for the legacy scan script.
- `scripts/run_full_pipeline.py` now fails closed and points operators to the canonical app dry-run boundary configured in the active workflow.

The active workflow command remains:

```bash
PYTHONPATH=src python -m market_scanner.app --dry-run
```

## Test dependency decoupling result

The test dependency blocker was decoupled.

The old runtime-import tests were rewritten as:

- canonical operator dry-run output tests;
- canonical fail-closed output tests;
- workflow static checks proving the active workflow remains canonical dry-run only;
- legacy wrapper fail-closed subprocess tests;
- static source checks proving the wrapper does not expose production behavior or invoke the legacy scan runtime.

The rewritten tests preserve coverage of the safe behavior now approved for these legacy surfaces: they must not execute production runtime, call providers, write files, create reports, send Telegram messages, read credentials, perform network calls, or produce investment semantics.

## Wrapper dependency result

The wrapper dependency was decoupled.

`scripts/run_full_pipeline.py` no longer shells into the legacy scan runtime. It remains present as a legacy script, but its executable behavior is now fail-closed. It accepts the old optional fundamentals arguments for compatibility parsing, but ignores them and exits with status code `2`.

The wrapper is not a new canonical runtime authority. It does not call the canonical app as a production replacement and does not import canonical app modules.

## Workflow dependency result

The workflow dependency remains decoupled after BL38.

`.github/workflows/daily-market-scan.yml` still runs only:

```bash
PYTHONPATH=src python -m market_scanner.app --dry-run
```

The workflow does not reference `scripts/run_scan.py` or `scripts/run_full_pipeline.py`.

## Legacy script status

- `scripts/run_scan.py` was not modified.
- `scripts/run_full_pipeline.py` was modified only to fail closed and remove its wrapper dependency.
- Neither script was deleted, moved, renamed, or archived.
- Legacy runner authority was not expanded.
- `scripts/run_scan.py` still exists and still contains broad legacy runtime logic, but active workflow/test/wrapper dependencies were decoupled.

## Remaining dependencies

After BL38:

- `scripts/run_scan.py` referenced by workflow: no
- `scripts/run_scan.py` imported or monkeypatched by tests: no
- `scripts/run_full_pipeline.py` imported or monkeypatched by tests: no
- `scripts/run_full_pipeline.py` shells into `scripts/run_scan.py`: no

Remaining references are static governance, canonical metadata, migration-target tests, historical documentation, and static assertions that classify the legacy scripts as non-canonical or migration/archive candidates.

## Remaining blockers

The direct workflow, test-import, test-monkeypatch, and wrapper blockers are removed.

Remaining archive-readiness questions:

- `scripts/run_scan.py` still contains broad executable legacy runtime logic and side effects if invoked manually.
- `scripts/run_scan.py` still has historical and active governance references as a legacy migration/archive candidate.
- A separate archive-readiness recheck is needed to confirm whether the remaining references are evidence-only, migration metadata, or still operator-active.

## Archive-readiness impact

Archive readiness improved materially.

The active workflow no longer depends on legacy scripts, tests no longer import or monkeypatch legacy runner modules, and the full-pipeline wrapper no longer invokes the scan script. The scripts remain present, so the next step should be a controlled archive-readiness recheck before any archive/delete/move decision.

## Python file creation justification if applicable

No new Python files were created.

Existing test files and the existing legacy wrapper were updated because those were the active dependency owners identified in BL36 and BL37.

## Side-effect guarantees

BL38 did not run a real scan, call providers, read credentials, perform network calls, write production data, generate reports, create Telegram artifacts, send Telegram messages, run the production pipeline, or mutate portfolio/watchlist files.

The only subprocess execution added in tests runs `scripts/run_full_pipeline.py` and verifies it fails closed without writing files or invoking the legacy scan runtime.

## Guardrails confirmation

- No credentials committed.
- No credentials read.
- No raw live payloads committed.
- No network calls performed.
- No production data writes.
- No reports generated.
- No report files written.
- No `reports/daily/telegram_message.txt` created or modified.
- No Telegram artifacts generated.
- No Telegram delivery added.
- No Telegram API calls made.
- No unsafe production pipeline execution.
- No portfolio/watchlist updates.
- No final BUY/SELL/HOLD recommendation.
- No allocation/conviction/urgency/scoring/target-price/tradeability/recommendation behavior.
- No missing values converted to zero.
- No one-off temporary helper files created.
- Legacy runtime scripts were not deleted, moved, renamed, or archived.
- Legacy runner authority was not expanded.

## Next recommended step

RESET-10L-BL39 — Legacy Runtime Script Archive Readiness Recheck.

The next sprint should re-run the archive-readiness review against the new dependency state and decide whether `scripts/run_scan.py` and `scripts/run_full_pipeline.py` are ready to archive, or whether remaining manual/operator or source-level references still require decoupling.
