# Legacy Runtime Script Archive Readiness Recheck

## Status

Completed by RESET-10L-BL39.

## Reset stage

RESET-10L-BL39 — Legacy Runtime Script Archive Readiness Recheck.

## Purpose

Recheck whether the primary legacy runtime scripts are ready for a future archive sprint after BL37 and BL38 removed active workflow, test, and wrapper dependencies.

Primary legacy runtime scripts reviewed:

- `scripts/run_scan.py`
- `scripts/run_full_pipeline.py`

This review did not delete, move, rename, archive, or modify either script.

## Policies applied

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_python_architecture_cleanup_and_legacy_decoupling_review.md`
- `docs/active/v2_legacy_runtime_script_archive_readiness_review.md`
- `docs/active/v2_legacy_runtime_dependency_decoupling.md`
- `docs/active/v2_legacy_runtime_blocker_decoupling.md`
- canonical boundary migration records from RESET-10L-BL29 through RESET-10L-BL35
- `docs/active/backlog.md`

## Files reviewed

- `scripts/run_scan.py`
- `scripts/run_full_pipeline.py`
- `src/market_scanner/app.py`
- `.github/workflows/daily-market-scan.yml`
- `tests/core/test_run_full_pipeline.py`
- `tests/test_operator_visibility.py`
- `tests/core/test_fundamentals_runtime_organization.py`
- `src/market_scanner/scanner/`
- `src/market_scanner/analysis/`
- `src/market_scanner/decision/`
- `src/market_scanner/messaging/`
- `src/market_scanner/reporting/`
- `src/market_scanner/delivery/`
- active governance, migration, and backlog documents listed above

## Inspection method

The review used static inspection only.

Commands and inspection methods included:

- full file reads for the two legacy runtime scripts, canonical app, workflow, and named tests;
- static reads of canonical scanner, analysis, decision, messaging, reporting, and delivery boundaries;
- broad legacy reference search;
- active legacy import search;
- active legacy monkeypatch search;
- workflow legacy reference search;
- wrapper dependency search;
- side-effect keyword search in the legacy scripts.

No legacy runtime script was executed.

## Static search summary

Broad reference search still finds references to the legacy scripts, but they are no longer active workflow, source import, test import, or test monkeypatch dependencies.

Reference categories observed:

- static governance and migration documents;
- historical archive and legacy documents;
- canonical metadata that lists legacy authorities as migration/archive candidates;
- static test assertions that read legacy files without importing them;
- stale `.pytest_cache` node names from earlier test history;
- no active workflow invocation;
- no active Python import of `scripts.run_scan` or `scripts.run_full_pipeline`;
- no active test monkeypatch dependency on either legacy runner;
- no `scripts/run_full_pipeline.py` invocation of `scripts/run_scan.py`.

## Reference summary

`scripts/run_scan.py` and `scripts/run_full_pipeline.py` remain referenced by active documentation and canonical metadata as legacy migration/archive candidates. These references are evidence and governance references, not active runtime dependencies.

The canonical app still lists both files in `LEGACY_RUNTIME_AUTHORITIES`, and scanner/delivery boundaries list relevant legacy files as legacy authorities. This is static metadata documenting decoupling status, not execution.

## Workflow dependency summary

`.github/workflows/daily-market-scan.yml` does not reference:

- `scripts/run_scan.py`
- `scripts/run_full_pipeline.py`
- `run_scan.py`
- `run_full_pipeline.py`

The workflow uses only the canonical dry-run command:

```bash
PYTHONPATH=src python -m market_scanner.app --dry-run
```

Workflow dependency status: removed.

## Test dependency summary

Tests no longer import or monkeypatch:

- `scripts.run_scan`
- `scripts.run_full_pipeline`

Remaining test references are static file reads and assertions. They verify that:

- canonical boundaries do not invoke legacy scripts;
- the active workflow remains canonical dry-run only;
- `scripts/run_full_pipeline.py` remains fail-closed;
- `scripts/run_full_pipeline.py` no longer references `scripts/run_scan.py`.

Test dependency status: active import and monkeypatch dependencies removed.

## Wrapper dependency summary

`scripts/run_full_pipeline.py` no longer imports `subprocess`, no longer builds a command for `scripts/run_scan.py`, and no longer shells into or otherwise invokes the legacy scan runtime.

Wrapper status: fail-closed.

## Side-effect summary

`scripts/run_scan.py` still contains broad legacy runtime behavior if manually invoked:

- provider/live market data access through ticker loading and OHLCV fetch calls;
- scanner execution;
- production data directory creation;
- CSV writes under processed, log, portfolio, and report-related paths;
- validation, context, fundamentals, timing, portfolio, portfolio intelligence, and final Decision Engine sequencing;
- reporting layer execution and report/message artifact writes;
- Telegram summary delivery call;
- portfolio state and portfolio review builders;
- final decisions through the legacy Decision Engine.

`scripts/run_full_pipeline.py` is fail-closed:

- no provider calls;
- no network calls;
- no subprocess invocation;
- no production data writes;
- no report generation;
- no Telegram artifacts;
- no Telegram delivery;
- no portfolio/watchlist mutation;
- no Decision Engine execution.

## Per-script archive-readiness table

| file path | primary status | workflow dependency | test dependency | source import dependency | static references only | manual invocation risk | side-effect risk if invoked | canonical owner | archive recommendation | required next action |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| `scripts/run_scan.py` | ARCHIVE_READY_WITH_MANUAL_INVOCATION_RISK | no | no active import or monkeypatch | no active source import found | yes | yes | high | `src/market_scanner/app.py` plus scanner, fundamentals, analysis, decision, messaging, reporting, and delivery boundaries | Archive in the next approved sprint if the archive sprint removes it from active script paths and documents manual invocation risk. | Proceed to controlled archive sprint with pre/post static checks and no runtime execution. |
| `scripts/run_full_pipeline.py` | ARCHIVE_READY | no | no active import or monkeypatch | no active source import found | yes | low | low; fail-closed with exit code `2` | `src/market_scanner/app.py` | Archive in the next approved sprint. | Proceed to controlled archive sprint with pre/post static checks. |

## Per-script detailed findings

### `scripts/run_scan.py`

Current role:

- legacy broad runtime orchestrator;
- old scanner, layer, Decision Engine, reporting, and Telegram sequencing surface;
- still executable if manually invoked.

Active dependency findings:

- still referenced by workflow: no;
- still imported by source code: no active import found;
- still imported or monkeypatched by tests: no;
- still referenced only by governance/static documentation and static assertions: yes.

Runtime and side-effect findings:

- calls provider/data-fetch surfaces if manually invoked;
- creates directories;
- writes scanner and failed ticker CSVs;
- invokes validation, context, fundamentals, timing, portfolio, portfolio intelligence, final decisions, reporting, and Telegram delivery;
- writes reporting outputs through the legacy reporting layer;
- sends Telegram summary through legacy delivery code if manually invoked and delivery succeeds;
- executes legacy Decision Engine final decision behavior if manually invoked.

Archive readiness answer:

- It can be archived in the next sprint because active workflow, test, source import, and wrapper dependencies have been removed.
- The archive sprint must explicitly account for manual invocation risk because the script still contains side-effectful legacy runtime behavior while present.
- The archive sprint should not preserve it as a permanent fallback runtime authority.

### `scripts/run_full_pipeline.py`

Current role:

- fail-closed legacy wrapper;
- parses old optional fundamentals arguments for compatibility;
- prints a fail-closed message;
- exits with status code `2`.

Active dependency findings:

- still referenced by workflow: no;
- still imported by source code: no active import found;
- still imported or monkeypatched by tests: no;
- still referenced only by governance/static documentation and static assertions: yes.

Runtime and side-effect findings:

- no longer imports `subprocess`;
- no longer invokes `scripts/run_scan.py`;
- does not call providers;
- does not write data or reports;
- does not create Telegram artifacts;
- does not send Telegram messages;
- does not read credentials;
- does not perform network calls;
- does not trigger Decision Engine behavior.

Archive readiness answer:

- It is archive-ready.
- The archive sprint should remove it only through a separately approved archive/delete/move process with static dependency checks before and after.

## Canonical boundary coverage assessment

The canonical v2 architecture now has established boundaries for:

- application entrypoint: `src/market_scanner/app.py`;
- scanner/universe planning: `src/market_scanner/scanner/`;
- fundamentals/provider/evidence: `src/market_scanner/fundamentals/`;
- analysis planning: `src/market_scanner/analysis/`;
- decision/review planning: `src/market_scanner/decision/`;
- message composition planning: `src/market_scanner/messaging/`;
- report artifact planning: `src/market_scanner/reporting/`;
- delivery planning: `src/market_scanner/delivery/`.

These canonical boundaries are currently side-effect-free and dry-run/planning-oriented. They are sufficient to replace legacy runtime authority for the currently approved workflow state, which is canonical dry-run only.

They do not yet provide a production replacement for the old side-effectful full scan. That is acceptable for archive readiness because the old full scan is no longer an approved active workflow path; future production execution must be implemented through separately approved canonical migration sprints.

## Remaining dependency map

Active dependencies remaining:

- workflow dependency: none;
- source import dependency: none found;
- test import dependency: none;
- test monkeypatch dependency: none;
- wrapper dependency: none.

Static references remaining:

- active governance and migration documents;
- canonical metadata identifying legacy authorities;
- static test assertions;
- historical archive and legacy documents;
- stale `.pytest_cache` entries from earlier test history.

Static references do not block archive if the archive sprint updates or validates the relevant active governance metadata and ignores historical archive evidence appropriately.

## Archive readiness conclusion

Both primary legacy runtime scripts are ready for a controlled archive sprint.

Classification:

- `scripts/run_scan.py`: ARCHIVE_READY_WITH_MANUAL_INVOCATION_RISK.
- `scripts/run_full_pipeline.py`: ARCHIVE_READY.

The safest next step is not immediate deletion in this sprint. The next step should be a dedicated archive sprint that removes or archives the confirmed legacy runtime scripts, reruns static dependency checks, verifies no Python/test/workflow/runtime behavior was unintentionally changed beyond the approved archive action, and documents the manual invocation risk that is eliminated by removing `scripts/run_scan.py` from active script paths.

## Recommended next cleanup step

RESET-10L-BL40 — Archive Confirmed Legacy Runtime Scripts.

## Guardrails confirmation

- No Python files changed.
- No tests changed.
- No workflows changed.
- No files moved.
- No files deleted.
- No files archived.
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

- This was a static review only; no legacy runtime execution was performed.
- `scripts/run_scan.py` still contains side-effectful legacy runtime behavior until a future archive sprint removes or archives it.
- Historical archive and legacy documents still preserve old instructions where the scripts were active. Those records should remain historical evidence unless a later documentation cleanup sprint explicitly reconciles them.
- `.pytest_cache` may contain stale test node names from older runs. Those cache references are not repository source dependencies.
