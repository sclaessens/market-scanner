# RESET-9F2 Closeout: Pause Legacy Daily Scan Workflow

## Purpose

RESET-9F2 paused the scheduled automatic execution of the legacy daily scan workflow during the v2 cutover.

The goal was to prevent generated legacy CSV and report outputs from continuing to change on `main` while preserving an explicit manual fallback through GitHub Actions `workflow_dispatch`.

## Workflow Changed

- `.github/workflows/daily-market-scan.yml`

## Trigger Change

The active weekday scheduled trigger was removed:

```yaml
schedule:
  - cron: "17 20 * * 1-5"
```

Manual dispatch remains available:

```yaml
workflow_dispatch:
```

The workflow now includes governance comments explaining that scheduled legacy daily scan execution is paused during v2 cutover and should be re-enabled only after explicit governance approval.

## Scope Confirmation

Only the daily market scan workflow trigger and this closeout document were changed.

No changes were made to:

- `src/`
- `scripts/`
- `tests/`
- `data/`
- CSV files
- `reports/`
- generated files
- runtime behavior outside workflow scheduling

No production pipeline, Telegram script, SEC diagnostics, provider calls, network calls, or live data calls were run.

## Validation Commands and Results

Validated the workflow trigger state:

```bash
grep -n "schedule:" .github/workflows/daily-market-scan.yml || true
grep -n "cron:" .github/workflows/daily-market-scan.yml || true
grep -n "workflow_dispatch" .github/workflows/daily-market-scan.yml
```

Result:

- No active `schedule:` trigger remains.
- No active `cron:` line remains.
- `workflow_dispatch` remains present.

Documentation-safe validation:

```bash
git diff --check
git status --short
git diff --stat
git diff -- .github/workflows/daily-market-scan.yml
```

Result:

- `git diff --check` passed.
- `git status --short` showed only `.github/workflows/daily-market-scan.yml` and this closeout document changed.
- `git diff --stat` showed only the workflow trigger edit before the closeout document was staged.
- The workflow diff showed the scheduled trigger removed and manual dispatch preserved.

Optional YAML parsing with `.venv/bin/python` was skipped because PyYAML is not installed in the local virtual environment. No dependency was installed.

No pytest run is required for this workflow-only governance change.

## Recommended Next Action

RESET-9C - Legacy Runtime Inventory and Retirement Decision.
