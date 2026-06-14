# Legacy Runtime Archive Validation and Active Entrypoint Certification

## Status

Completed by RESET-10L-BL41.

## Reset stage

RESET-10L-BL41 — Legacy Runtime Archive Validation and Active Entrypoint Certification.

## Purpose

Validate that RESET-10L-BL40 correctly removed the confirmed legacy runtime scripts from the active `scripts/` runtime path, retained them only in the legacy archive, and left the canonical v2 application boundary as the certified active runtime entrypoint.

This sprint is review-only. No Python files, tests, workflows, archived scripts, data files, report files, portfolio/watchlist files, or runtime behavior were changed.

## Policies applied

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_python_architecture_cleanup_and_legacy_decoupling_review.md`
- `docs/active/v2_legacy_runtime_script_archive_readiness_review.md`
- `docs/active/v2_legacy_runtime_dependency_decoupling.md`
- `docs/active/v2_legacy_runtime_blocker_decoupling.md`
- `docs/active/v2_legacy_runtime_script_archive_readiness_recheck.md`
- `docs/active/v2_legacy_runtime_script_archive_execution.md`
- `docs/active/backlog.md`
- Repository doctrine: classification upstream, allocation downstream, Decision Engine as the only allocation authority.
- English-only repository content governance.

## Files inspected

- `docs/active/v2_python_file_creation_policy.md`
- `docs/active/v2_legacy_python_decoupling_policy.md`
- `docs/active/v2_canonical_runtime_architecture.md`
- `docs/active/v2_python_architecture_cleanup_and_legacy_decoupling_review.md`
- `docs/active/v2_legacy_runtime_script_archive_readiness_review.md`
- `docs/active/v2_legacy_runtime_dependency_decoupling.md`
- `docs/active/v2_legacy_runtime_blocker_decoupling.md`
- `docs/active/v2_legacy_runtime_script_archive_readiness_recheck.md`
- `docs/active/v2_legacy_runtime_script_archive_execution.md`
- `docs/active/backlog.md`
- `src/market_scanner/app.py`
- `.github/workflows/daily-market-scan.yml`
- `archive/legacy_runtime/scripts/run_scan.py`
- `archive/legacy_runtime/scripts/run_full_pipeline.py`
- `tests/core/test_run_full_pipeline.py`
- `tests/test_operator_visibility.py`
- `tests/core/test_fundamentals_runtime_organization.py`
- `src/market_scanner/scanner/`
- `src/market_scanner/analysis/`
- `src/market_scanner/decision/`
- `src/market_scanner/messaging/`
- `src/market_scanner/reporting/`
- `src/market_scanner/delivery/`

## Validation method

The validation used static repository inspection, file-presence checks, active workflow/source/test reference checks, canonical app inspection, and safe tests that do not execute archived scripts.

The archived scripts were inspected as historical text only. They were not executed.

## Active legacy script absence check

Commands:

```bash
test ! -f scripts/run_scan.py
test ! -f scripts/run_full_pipeline.py
```

Result: passed.

`scripts/run_scan.py` and `scripts/run_full_pipeline.py` are no longer present in the active `scripts/` runtime path.

## Archive presence check

Commands:

```bash
test -f archive/legacy_runtime/scripts/run_scan.py
test -f archive/legacy_runtime/scripts/run_full_pipeline.py
```

Result: passed.

`archive/legacy_runtime/scripts/run_scan.py` and `archive/legacy_runtime/scripts/run_full_pipeline.py` are retained only as historical legacy references and are not active runtime entrypoints.

## Workflow validation

Workflow reference check:

```bash
grep -R "scripts/run_scan.py\|scripts/run_full_pipeline.py\|archive/legacy_runtime/scripts/run_scan.py\|archive/legacy_runtime/scripts/run_full_pipeline.py" -n .github \
  --exclude-dir=.venv \
  --exclude-dir=venv \
  --exclude-dir=__pycache__ \
  --exclude-dir=.git || true
```

Result: no output.

Canonical dry-run check:

```bash
grep -R "python -m market_scanner.app --dry-run\|PYTHONPATH=src" -n .github/workflows/daily-market-scan.yml || true
```

Result:

```text
.github/workflows/daily-market-scan.yml:32:        run: PYTHONPATH=src python -m market_scanner.app --dry-run
```

The active workflow does not reference active or archived legacy runtime scripts. It remains canonical dry-run only.

## Active source/test import validation

Import check:

```bash
grep -R "import .*run_scan\|from .*run_scan\|import .*run_full_pipeline\|from .*run_full_pipeline" -n src tests .github \
  --include="*.py" \
  --exclude-dir=.venv \
  --exclude-dir=venv \
  --exclude-dir=__pycache__ \
  --exclude-dir=.git || true
```

Result: no output.

Old active path reference check:

```bash
grep -R "scripts/run_scan.py\|scripts/run_full_pipeline.py" -n src tests .github \
  --exclude-dir=.venv \
  --exclude-dir=venv \
  --exclude-dir=__pycache__ \
  --exclude-dir=.git || true
```

Result: no output.

There are no active source/test/workflow imports of the legacy runtime scripts and no active source/test/workflow references to the old active script paths.

## Archive invocation validation

Archive path check:

```bash
grep -R "archive/legacy_runtime/scripts/run_scan.py\|archive/legacy_runtime/scripts/run_full_pipeline.py" -n src tests .github \
  --exclude-dir=.venv \
  --exclude-dir=venv \
  --exclude-dir=__pycache__ \
  --exclude-dir=.git || true
```

Result: no output.

`src/market_scanner/app.py` contains `ARCHIVED_LEGACY_RUNTIME_ROOT = "archive/legacy_runtime/scripts"` as static metadata only. The canonical app does not import, invoke, monkeypatch, or execute archived scripts.

## Canonical app certification

The certified active runtime entrypoint is `src/market_scanner/app.py`.

Inspection confirms:

- canonical app dry-run exists through `run_canonical_app(dry_run=True)` and CLI `--dry-run`;
- canonical app non-dry-run fails closed through `run_canonical_app(dry_run=False)` and CLI `--execute`;
- canonical app does not call archived scripts;
- canonical app does not import archived scripts;
- canonical app does not run providers by default;
- canonical app does not write production data;
- canonical app does not generate reports;
- canonical app does not create Telegram artifacts;
- canonical app does not send Telegram messages;
- canonical app does not read credentials;
- canonical app does not perform network calls;
- canonical app does not mutate portfolio/watchlist files;
- canonical app does not produce investment recommendations.

## Canonical boundary status

Established canonical v2 boundaries remain present and side-effect-free by default:

- `src/market_scanner/app.py`
- `src/market_scanner/scanner/`
- `src/market_scanner/analysis/`
- `src/market_scanner/decision/`
- `src/market_scanner/messaging/`
- `src/market_scanner/reporting/`
- `src/market_scanner/delivery/`

The boundaries continue to represent scanner, analysis, decision/review, message composition, report artifact planning, and delivery planning as deterministic dry-run/planning surfaces.

## Test results

Executed:

```bash
.venv/bin/python -m pytest tests/unit/test_v2_canonical_app.py
```

Result: 22 passed.

Executed:

```bash
.venv/bin/python -m pytest \
  tests/unit/test_v2_canonical_app.py \
  tests/unit/test_v2_canonical_scanner.py \
  tests/unit/test_v2_canonical_analysis.py \
  tests/unit/test_v2_canonical_decision.py \
  tests/unit/test_v2_canonical_messaging.py \
  tests/unit/test_v2_canonical_reporting.py \
  tests/unit/test_v2_canonical_delivery.py
```

Result: 79 passed.

Executed safe archive-related static tests:

```bash
.venv/bin/python -m pytest tests/test_operator_visibility.py tests/core/test_fundamentals_runtime_organization.py
```

Result: 6 passed.

Not executed:

- `tests/core/test_run_full_pipeline.py`
- full suite

Reason: `tests/core/test_run_full_pipeline.py` executes `archive/legacy_runtime/scripts/run_full_pipeline.py` in a subprocess. BL41 explicitly forbids executing archived scripts and requires the guardrail statement "No archived scripts executed." The archive-related test command and the full suite would violate that sprint guardrail, so this validation preserved the stricter safety rule and recorded the test-scope limitation.

## Archive validation conclusion

Archive validation passed.

The confirmed legacy runtime scripts are absent from active `scripts/` paths, present in `archive/legacy_runtime/scripts/`, and not referenced by active workflow/source/test runtime paths.

## Active entrypoint certification conclusion

The certified active runtime entrypoint is `src/market_scanner/app.py`.

The workflow uses the canonical app dry-run command:

```bash
PYTHONPATH=src python -m market_scanner.app --dry-run
```

No active workflow, source, or test runtime path invokes active or archived legacy runtime scripts.

## Remaining risks

- The archived `run_scan.py` still contains historical side-effectful legacy runtime code, but it is no longer in the active `scripts/` path and was not executed in this sprint.
- Static governance and historical documents intentionally preserve legacy script references as evidence.
- Other script-era Python files remain outside this two-script archive validation and should be inventoried separately.
- A test-scope conflict remains: `tests/core/test_run_full_pipeline.py` verifies fail-closed behavior by executing an archived script, while BL41 forbids archived script execution. A future cleanup sprint should decide whether that test should become fully static.

## Guardrails confirmation

- No Python files changed.
- No tests changed.
- No workflows changed.
- No files moved.
- No files deleted.
- No files archived.
- No archived scripts modified.
- No archived scripts executed.
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

## Next recommended step

RESET-10L-BL42 — Script-Era Python Cleanup Inventory.

The next sprint should inventory remaining script-era Python files now that the primary legacy runtime scripts are archived and the active runtime entrypoint is certified.
