# RESET-9C2 Closeout: Legacy Runtime Test Retirement Map

## Files Created

- `docs/resets/reset_9c2_legacy_runtime_test_retirement_map.md`
- `docs/resets/reset_9c2_legacy_runtime_test_retirement_closeout.md`

## Scope Confirmation

RESET-9C2 was completed as documentation-only static inventory and governance mapping.

No test files were changed.

No runtime, code, data, report, generated output, or workflow files were changed.

No files were moved, deleted, archived, migrated, renamed, or refactored.

No files were changed under:

- `tests/`
- `scripts/`
- `src/`
- `data/`
- `reports/`
- `.github/workflows/`

## Calls Not Run

No production pipeline was run.

No `scripts/run_scan.py` execution was run.

No Telegram script was run.

No SEC diagnostics, SEC provider calls, market provider calls, network calls, Telegram API calls, or live data calls were run.

## Key Decision

Decision: `DO_NOT_DELETE_OR_ARCHIVE_LEGACY_TESTS_YET`

Rationale:

- legacy runtime remains the only complete manual fallback;
- many tests encode domain and edge-case knowledge required for v2 translation;
- v2 replacements are incomplete or scaffold-only for most runtime layers;
- SEC, provider, Telegram, portfolio, reporting, and generated-data tests require reapproval or policy decisions;
- tests are needed to support safe knowledge extraction.

## Validation Commands and Results

Commands run:

```bash
git diff --check
git status --short
git diff --stat
git diff --name-only
git diff -- tests scripts src data reports .github/workflows || true
```

Results:

- only the two RESET-9C2 documentation files are changed;
- no test/runtime/source/data/report/workflow diff exists;
- `git diff --check` passed;
- `git diff -- tests scripts src data reports .github/workflows || true` produced no diff.

No pytest is required because no code or tests were changed.

## Recommended Next Action

RESET-9C2A - Translate Legacy Validation Tests to V2 Contracts.
