# Archived Script Execution Test Cleanup

## Status

Completed by RESET-10L-BL43.

## Reset stage

RESET-10L-BL43 - Remove Archived Script Execution from Tests.

## Purpose

Remove the remaining active test behavior that executed an archived legacy runtime script while preserving useful coverage through static archive-status checks and canonical app validation.

BL41 certified `src/market_scanner/app.py` as the active runtime entrypoint, and BL42 confirmed that `tests/core/test_run_full_pipeline.py` still executed `archive/legacy_runtime/scripts/run_full_pipeline.py` to validate fail-closed behavior. Archived files must remain historical references only, not executable test targets.

Archived legacy runtime scripts are no longer executed by active tests.

Archived scripts remain historical references only and are not active runtime entrypoints.

The active runtime validation path remains `src/market_scanner/app.py` and the canonical dry-run CLI.

## Policies applied

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_legacy_runtime_archive_validation_and_entrypoint_certification.md`
- `docs/active/v2_script_era_python_cleanup_inventory.md`
- `docs/active/v2_legacy_runtime_script_archive_execution.md`
- `docs/active/v2_legacy_runtime_script_archive_readiness_recheck.md`
- `docs/active/v2_legacy_runtime_blocker_decoupling.md`
- `docs/active/v2_legacy_runtime_dependency_decoupling.md`
- `docs/active/backlog.md`
- Repository doctrine: classification upstream, allocation downstream, Decision Engine as the only allocation authority.
- English-only repository content governance.

## Files inspected

- `tests/core/test_run_full_pipeline.py`
- `tests/test_operator_visibility.py`
- `tests/core/test_fundamentals_runtime_organization.py`
- `tests/unit/test_v2_canonical_app.py`
- `archive/legacy_runtime/scripts/run_scan.py`
- `archive/legacy_runtime/scripts/run_full_pipeline.py`
- `src/market_scanner/app.py`
- `.github/workflows/daily-market-scan.yml`
- active governance and migration records listed above

## Files changed

- `tests/core/test_run_full_pipeline.py`
- `docs/active/v2_archived_script_execution_test_cleanup.md`
- `docs/active/backlog.md`

No production Python code, archived scripts, workflows, data files, report files, portfolio/watchlist files, or canonical boundary source files were changed.

## Archived script execution before BL43

Before BL43, `tests/core/test_run_full_pipeline.py` executed:

```text
archive/legacy_runtime/scripts/run_full_pipeline.py
```

The test used subprocess execution to verify the archived wrapper failed closed with exit code `2` and accepted legacy optional argument names without writing files.

That behavior conflicted with the archive rule that archived legacy runtime scripts must be retained only as historical references and must not be active executable test targets.

## Archived script execution after BL43

After BL43:

- active tests do not execute archived legacy runtime scripts;
- active tests do not import archived legacy runtime scripts;
- active tests do not monkeypatch archived legacy runtime scripts;
- `tests/core/test_run_full_pipeline.py` reads the archived wrapper only as static text;
- the fail-closed behavior is preserved as static archive evidence through marker checks, not runtime execution.

## Test changes

`tests/core/test_run_full_pipeline.py` was rewritten to remove subprocess execution of the archived wrapper.

The updated tests now assert:

- `scripts/run_scan.py` and `scripts/run_full_pipeline.py` are absent from the active `scripts/` path;
- `archive/legacy_runtime/scripts/run_scan.py` and `archive/legacy_runtime/scripts/run_full_pipeline.py` exist;
- the archived full-pipeline wrapper contains the fail-closed marker text;
- the archived full-pipeline wrapper points to the canonical app dry-run boundary;
- the archived full-pipeline wrapper preserves legacy optional argument names only as static historical evidence;
- the archived wrapper no longer contains the old scan command builder or `run_scan.py` wrapper dependency;
- the archived wrapper does not expose production behavior markers.

## Coverage preservation

Coverage was preserved without executing archived code.

The execution-based checks were replaced with static archive-status and source-marker checks. The canonical runtime behavior remains covered by active canonical app tests, including dry-run success, non-dry-run fail-closed behavior, no legacy script imports, no provider calls, no production data writes, no report generation, no Telegram artifacts, and no portfolio/watchlist updates.

## Canonical runtime validation

The active runtime validation path remains:

```text
src/market_scanner/app.py
```

and:

```bash
PYTHONPATH=src python -m market_scanner.app --dry-run
```

Canonical app validation remains in `tests/unit/test_v2_canonical_app.py`.

## Workflow status

`.github/workflows/daily-market-scan.yml` was not changed.

The workflow remains canonical dry-run only:

```bash
PYTHONPATH=src python -m market_scanner.app --dry-run
```

The workflow does not call active or archived legacy runtime scripts.

## Archived script status

Archived scripts remain historical references only:

- `archive/legacy_runtime/scripts/run_scan.py`
- `archive/legacy_runtime/scripts/run_full_pipeline.py`

They are not active runtime entrypoints and were not modified by BL43.

## Side-effect guarantees

BL43 did not execute archived scripts, run live providers, execute the production pipeline, read credentials, perform network calls, write production data, generate reports, create Telegram artifacts, send Telegram messages, mutate portfolio/watchlist files, or change Decision Engine behavior.

No final BUY/SELL/HOLD, allocation, conviction, urgency, scoring, target-price, tradeability, or recommendation behavior was added.

## Guardrails confirmation

- No archived scripts modified.
- No archived scripts executed.
- No archived scripts imported.
- No archived scripts monkeypatched.
- No Python production code changed.
- No workflow changes.
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
- No replacement runtime scripts created.
- Archived scripts remain historical references only.

## Known limitations

The archived `run_scan.py` file still contains historical side-effectful legacy runtime code if someone manually runs it from the archive. BL43 only removes active test execution of archived scripts; it does not modify archive contents.

Many script-era Python files remain under `scripts/` and require the follow-up cleanup sequence identified by BL42.

## Next recommended step

RESET-10L-BL44 - High-Risk Script-Era Side-Effect Cleanup Review.

Rationale: with archived-script execution removed from active tests, the project can safely proceed to the next script-era cleanup review without treating archive files as executable runtime targets.
