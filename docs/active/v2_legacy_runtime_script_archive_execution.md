# Legacy Runtime Script Archive Execution

## Status

Completed by RESET-10L-BL40.

## Reset stage

RESET-10L-BL40 — Archive Confirmed Legacy Runtime Scripts.

## Purpose

Archive the confirmed legacy runtime scripts that were no longer active workflow, source import, test import, test monkeypatch, or wrapper dependencies after RESET-10L-BL37 through RESET-10L-BL39.

Confirmed legacy runtime scripts:

- `scripts/run_scan.py`
- `scripts/run_full_pipeline.py`

scripts/run_scan.py and scripts/run_full_pipeline.py were removed from the active scripts runtime path and archived for historical reference.

The canonical v2 runtime authority remains src/market_scanner/app.py.

## Policies applied

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_python_architecture_cleanup_and_legacy_decoupling_review.md`
- `docs/active/v2_legacy_runtime_script_archive_readiness_review.md`
- `docs/active/v2_legacy_runtime_dependency_decoupling.md`
- `docs/active/v2_legacy_runtime_blocker_decoupling.md`
- `docs/active/v2_legacy_runtime_script_archive_readiness_recheck.md`
- canonical boundary migration records from RESET-10L-BL29 through RESET-10L-BL35
- `docs/active/backlog.md`

## Files inspected

- `scripts/run_scan.py`
- `scripts/run_full_pipeline.py`
- `.github/workflows/daily-market-scan.yml`
- `src/market_scanner/app.py`
- `tests/core/test_run_full_pipeline.py`
- `tests/test_operator_visibility.py`
- `tests/core/test_fundamentals_runtime_organization.py`
- canonical scanner, analysis, decision, messaging, reporting, and delivery boundary packages and tests
- active governance, migration, and backlog documents listed above

## Files moved

- `scripts/run_scan.py` -> `archive/legacy_runtime/scripts/run_scan.py`
- `scripts/run_full_pipeline.py` -> `archive/legacy_runtime/scripts/run_full_pipeline.py`

## Files changed

- `archive/legacy_runtime/scripts/run_scan.py`
- `archive/legacy_runtime/scripts/run_full_pipeline.py`
- `src/market_scanner/app.py`
- `src/market_scanner/scanner/scanner_boundary.py`
- `src/market_scanner/delivery/delivery_boundary.py`
- `tests/core/test_run_full_pipeline.py`
- `tests/test_operator_visibility.py`
- `tests/unit/test_v2_canonical_scanner.py`
- `tests/unit/test_v2_canonical_analysis.py`
- `tests/unit/test_v2_canonical_decision.py`
- `tests/unit/test_v2_canonical_messaging.py`
- `tests/unit/test_v2_canonical_reporting.py`
- `tests/unit/test_v2_canonical_delivery.py`
- `docs/active/v2_legacy_runtime_script_archive_execution.md`
- `docs/active/backlog.md`

The canonical metadata and static tests were updated only to reference archived legacy script paths and to assert the active runtime paths no longer exist. No replacement runtime scripts were created.

## Archive location

Approved archive location used:

```text
archive/legacy_runtime/scripts/
```

Archived files:

- `archive/legacy_runtime/scripts/run_scan.py`
- `archive/legacy_runtime/scripts/run_full_pipeline.py`

## Pre-archive dependency check

The pre-archive static checks showed:

- no active workflow dependency on either legacy script;
- no active Python import of `scripts.run_scan` or `scripts.run_full_pipeline`;
- no active test monkeypatch dependency on either legacy runner;
- no `scripts/run_full_pipeline.py` invocation of `scripts/run_scan.py`;
- remaining references were static governance records, canonical metadata, historical documents, and static tests.

Pre-archive active path references still existed in canonical metadata and static tests because the files had not yet been moved. Those references were updated during BL40 to archive paths or path-component assertions.

## Post-archive dependency check

Post-archive path checks confirmed:

```bash
test ! -f scripts/run_scan.py
test ! -f scripts/run_full_pipeline.py
test -f archive/legacy_runtime/scripts/run_scan.py
test -f archive/legacy_runtime/scripts/run_full_pipeline.py
```

Post-archive static checks confirmed:

- no `scripts/run_scan.py` or `scripts/run_full_pipeline.py` references remain in `.github`, `src`, or `tests`;
- no active imports of legacy runtime scripts remain;
- the active workflow does not reference active or archived legacy scripts;
- canonical app metadata points to archived legacy authority paths only;
- canonical app does not call archived scripts.

## Test updates

Tests were updated to assert the archive state without importing, executing, or monkeypatching archived scripts.

Updated test behavior includes:

- active `scripts/run_scan.py` and `scripts/run_full_pipeline.py` paths must not exist;
- archived legacy script paths must exist;
- archived `run_full_pipeline.py` remains fail-closed;
- canonical dry-run remains available;
- canonical non-dry-run still fails closed;
- workflow remains canonical dry-run only;
- canonical boundary tests read archived files only as static historical evidence.

Validation run:

- `tests/core/test_run_full_pipeline.py`: 5 passed;
- `tests/test_operator_visibility.py`: 4 passed;
- `tests/core/test_fundamentals_runtime_organization.py`: 2 passed;
- `tests/unit/test_v2_canonical_app.py`: 22 passed;
- canonical boundary suite: 79 passed;
- full test suite: 833 passed.

## Workflow status

`.github/workflows/daily-market-scan.yml` was not changed.

It remains canonical dry-run only:

```bash
PYTHONPATH=src python -m market_scanner.app --dry-run
```

The workflow does not call active or archived legacy scripts.

## Canonical runtime authority after archive

The canonical v2 runtime authority remains:

```text
src/market_scanner/app.py
```

The canonical app remains dry-run only by default and fails closed for non-dry-run execution.

## Archived script status

The archived scripts are historical reference files only.

- `archive/legacy_runtime/scripts/run_scan.py` preserves the old broad legacy runtime script for historical review.
- `archive/legacy_runtime/scripts/run_full_pipeline.py` preserves the fail-closed wrapper state from BL38.

Neither archived script is imported, invoked, monkeypatched, or referenced by the active workflow.

## Manual invocation risk resolution

BL39 identified `scripts/run_scan.py` as archive-ready with manual invocation risk because it still contained side-effectful legacy runtime behavior while present in the active scripts path.

BL40 resolves that active-path manual invocation risk by removing `scripts/run_scan.py` from the active `scripts/` runtime path.

The archived copy still contains historical side-effectful code, so operators must not treat the archive as an active runtime location.

## Side-effect guarantees

BL40 did not execute the archived scripts and did not run the legacy production pipeline.

No live providers were called, no credentials were read, no network calls were performed, no production data was written, no reports were generated, no Telegram artifacts were created, no Telegram messages were sent, and no portfolio/watchlist files were modified.

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
- No replacement runtime scripts created.
- Canonical app does not call archived scripts.
- Workflow does not call archived scripts.
- Archived scripts were not executed.

## Known limitations

- Historical and legacy documents still preserve old references to the scripts as historical evidence.
- The archived `run_scan.py` file still contains old side-effectful runtime code for historical reference; it must not be used as an active runtime entrypoint.
- BL40 archives only the two confirmed legacy runtime scripts. Other legacy script-era modules remain for separate migration or cleanup sprints.

## Next recommended step

RESET-10L-BL41 — Legacy Runtime Archive Validation and Active Entrypoint Certification.
