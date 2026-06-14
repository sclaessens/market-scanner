# Legacy Runtime Dependency Decoupling

## Status

Completed by RESET-10L-BL37.

## Reset stage

RESET-10L-BL37 — Decouple Remaining Legacy Runtime Dependencies.

## Purpose

Reduce active workflow and test dependencies on the legacy runtime scripts so `scripts/run_scan.py` and `scripts/run_full_pipeline.py` can move closer to archive readiness without deleting, moving, archiving, or expanding those scripts.

Primary legacy runtime scripts:

- `scripts/run_scan.py`
- `scripts/run_full_pipeline.py`

## Policies applied

- `docs/active/v2_legacy_runtime_script_archive_readiness_review.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/backlog.md`

## Files inspected

- `docs/active/v2_legacy_runtime_script_archive_readiness_review.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/backlog.md`
- `scripts/run_scan.py`
- `scripts/run_full_pipeline.py`
- `.github/workflows/daily-market-scan.yml`
- `src/market_scanner/app.py`
- `src/market_scanner/scanner/`
- `src/market_scanner/analysis/`
- `src/market_scanner/decision/`
- `src/market_scanner/messaging/`
- `src/market_scanner/reporting/`
- `src/market_scanner/delivery/`
- `tests/unit/test_v2_canonical_app.py`
- `tests/unit/test_v2_canonical_scanner.py`
- `tests/unit/test_v2_canonical_analysis.py`
- `tests/unit/test_v2_canonical_decision.py`
- `tests/unit/test_v2_canonical_messaging.py`
- `tests/unit/test_v2_canonical_reporting.py`
- `tests/unit/test_v2_canonical_delivery.py`
- `tests/core/test_fundamentals_runtime_organization.py`
- `tests/core/test_run_full_pipeline.py`
- `tests/test_operator_visibility.py`

## Files changed

- `.github/workflows/daily-market-scan.yml`
- `src/market_scanner/app.py`
- `tests/unit/test_v2_canonical_app.py`
- `tests/core/test_fundamentals_runtime_organization.py`
- `docs/active/v2_legacy_runtime_dependency_decoupling.md`
- `docs/active/backlog.md`

## Dependencies before BL37

- `.github/workflows/daily-market-scan.yml` invoked `python scripts/run_scan.py`.
- `.github/workflows/daily-market-scan.yml` also created `.env`, called Telegram webhook deletion, processed Telegram commands, listed generated data/report files, and committed data/report artifacts.
- `tests/core/test_fundamentals_runtime_organization.py` imported `scripts.run_scan`.
- `tests/core/test_run_full_pipeline.py` imported and monkeypatched `scripts.run_full_pipeline` and `scripts.run_scan`.
- `tests/test_operator_visibility.py` imported and monkeypatched `scripts.run_full_pipeline` and `scripts.run_scan`.
- `scripts/run_full_pipeline.py` still builds a subprocess command for `scripts/run_scan.py`.

## Dependencies decoupled in BL37

- The active workflow no longer invokes `scripts/run_scan.py`.
- The active workflow no longer writes `.env`, reads Telegram secrets, calls Telegram webhook endpoints, processes Telegram commands, lists generated production artifacts, commits production data, or commits report files.
- The active workflow now runs the canonical app dry-run command:

```bash
PYTHONPATH=src python -m market_scanner.app --dry-run
```

- `src/market_scanner/app.py` now exposes a guarded CLI. The default and `--dry-run` path return the deterministic canonical runtime dry-run plan. The `--execute` path fails closed.
- `tests/unit/test_v2_canonical_app.py` now covers the canonical app CLI dry-run, module execution, fail-closed non-dry-run behavior, no legacy script imports, no network module imports, and no credential module imports.
- `tests/core/test_fundamentals_runtime_organization.py` no longer imports `scripts.run_scan`; it now verifies only the canonical fundamentals namespace and legacy core compatibility wrappers.

## Workflow decoupling result

Workflow dependency on `scripts/run_scan.py` was decoupled.

The workflow is now a canonical dry-run only. It does not perform a production scan, does not call providers, does not write production data, does not generate reports, does not create `reports/daily/telegram_message.txt`, does not read Telegram credentials, does not call Telegram APIs, and does not commit data or report artifacts.

## Test decoupling result

Test dependency was partially decoupled.

Removed:

- `tests/core/test_fundamentals_runtime_organization.py` no longer imports `scripts.run_scan`.

Remaining by design:

- `tests/core/test_run_full_pipeline.py` still imports and monkeypatches `scripts.run_full_pipeline` and `scripts.run_scan`.
- `tests/test_operator_visibility.py` still imports and monkeypatches `scripts.run_full_pipeline` and `scripts.run_scan`.

Those tests were left intact because they protect currently unmigrated legacy runtime behavior: subprocess command construction, operator output, broad sequencing, artifact message formatting, report-write ordering, and Telegram-delivery ordering. Removing or rewriting them before the corresponding canonical runtime behavior is migrated would reduce coverage and create false archive readiness.

## Canonical CLI result

Canonical CLI support was added to the existing approved application entrypoint:

```bash
python -m market_scanner.app --dry-run
```

The CLI behavior is intentionally narrow:

- default invocation runs the same dry-run plan as `--dry-run`;
- `--execute` fails closed with `Only dry-run canonical app planning is approved.`;
- no legacy scripts are imported or invoked;
- no provider calls are made;
- no production data writes are performed;
- no reports are generated;
- no Telegram artifacts are created;
- no credentials are read;
- no network calls are made;
- no portfolio or watchlist files are mutated;
- no final investment recommendations are produced.

No `src/market_scanner/__main__.py` file was created because `python -m market_scanner.app --dry-run` is sufficient for the approved workflow replacement.

## Legacy script status

- `scripts/run_scan.py` was not modified.
- `scripts/run_full_pipeline.py` was not modified.
- Neither legacy script was deleted, moved, renamed, archived, or expanded.
- Both scripts remain legacy migration/archive candidates.

## Remaining dependencies

After BL37:

- `scripts/run_scan.py` referenced by workflow: no
- `scripts/run_scan.py` imported or monkeypatched by tests: yes
- `scripts/run_full_pipeline.py` imported or monkeypatched by tests: yes
- `scripts/run_full_pipeline.py` shells into `scripts/run_scan.py`: yes

Remaining active test dependencies:

- `tests/core/test_run_full_pipeline.py`
- `tests/test_operator_visibility.py`

Remaining wrapper dependency:

- `scripts/run_full_pipeline.py` still shells into `scripts/run_scan.py`.

Remaining canonical metadata and governance references are expected and are not executable dependencies.

## Remaining blockers

- Canonical boundaries are still planning-oriented and do not yet own the broad executable sequencing currently covered by legacy runner tests.
- `scripts/run_scan.py` still owns broad runtime behavior: scanner execution, data writes, layer sequencing, report generation, Telegram delivery, portfolio/intelligence processing, and legacy Decision Engine execution.
- `scripts/run_full_pipeline.py` still wraps `scripts/run_scan.py`.
- Operator visibility behavior has not yet been migrated to canonical runtime owners.

## Archive-readiness impact

Archive readiness improved but is not complete.

The most significant active workflow dependency was removed. However, both legacy scripts remain not archive-ready because active tests still import or monkeypatch them and because unique executable runtime behavior has not yet been migrated, retired, or replaced by certified canonical owners.

## Python file creation justification if applicable

No new Python files were created.

`src/market_scanner/app.py` was updated because it is the BL28-approved canonical v2 application entrypoint. The update adds a guarded CLI dry-run surface to the existing canonical owner rather than creating a one-off runtime helper.

## Side-effect guarantees

BL37 did not execute a real scan, call providers, read credentials, perform network calls, write production data, generate reports, create Telegram artifacts, send Telegram messages, run the production pipeline, or mutate portfolio/watchlist files.

The workflow replacement is a dry-run plan only. The canonical CLI fails closed for non-dry-run execution.

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
- Legacy runtime scripts were not deleted, moved, archived, or expanded.

## Next recommended step

RESET-10L-BL38 — Decouple Remaining Legacy Runtime Blockers.

The next cleanup sprint should migrate or retire the remaining test-covered legacy runner responsibilities before archive-readiness is rechecked. Prefer preserving coverage by moving only validated behavior into canonical owners, then replacing the legacy runner tests with canonical boundary or canonical execution tests.
