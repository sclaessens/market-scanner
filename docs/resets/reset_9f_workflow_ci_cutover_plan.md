# RESET-9F — Workflow and CI Cutover Plan

## 1. Purpose

RESET-9F documents the current GitHub Actions workflow surface and defines a cutover plan for legacy automation during the v2 reset.

This is a documentation-only governance task. It does not modify workflows, code, tests, data, reports, or runtime behavior.

## 2. Triggering Finding

A recent pull from `main` included changes to generated CSV files:

- `data/portfolio/portfolio_review.csv`
- `data/processed/scanner_ranked.csv`

The corresponding commit was:

- `d69de548a1d0c9ae76f1be88f0f17c0e92f9a5d4`
- message: `Automated daily scan update`
- author: `github-actions[bot]`

This means a legacy GitHub Actions workflow is still writing generated legacy data to `main` while the repository is being reset toward v2.

## 3. Workflow Inventory

The workflow identified as responsible is:

```text
.github/workflows/daily-market-scan.yml
```

Workflow name:

```text
Daily Market Scan
```

Triggers:

```yaml
on:
  workflow_dispatch:
  schedule:
    - cron: "17 20 * * 1-5"
```

Permissions:

```yaml
permissions:
  contents: write
```

The workflow has permission to commit generated files back to the repository.

## 4. Legacy Runtime Actions Detected

The workflow executes legacy runtime scripts:

```yaml
- name: Process Telegram commands
  run: python scripts/telegram/process_telegram_commands.py

- name: Run scanner
  run: python scripts/run_scan.py
```

This confirms the workflow still operates on the legacy `scripts/` runtime surface.

## 5. Generated Files Committed by Workflow

The workflow stages these generated outputs:

```bash
git add data/watchlist/*.csv || true
git add data/portfolio/*.csv || true
git add data/processed/market_regime.csv || true
git add data/processed/scanner_ranked.csv || true
git add reports/daily/telegram_message.txt || true
```

It then commits and pushes directly to `main` when changes exist:

```bash
git commit -m "Automated daily scan update"
git push
```

## 6. Governance Assessment

This workflow conflicts with the current reset direction because:

- it runs legacy scripts while v2 is being introduced;
- it writes generated data to tracked repository paths;
- it pushes directly to `main`;
- it can alter CSV state independently of review PRs;
- it uses Telegram behavior during a phase where v2 reporting remains in-memory only;
- it keeps legacy generated outputs active while cleanup planning is underway.

This does not mean the workflow was wrong historically. It means it is no longer safe as an uncontrolled default during v2 reset and legacy cutover.

## 7. Recommended Decision

Decision: PAUSE_LEGACY_DAILY_SCAN_AUTOMATION_DURING_V2_CUTOVER

The legacy daily scan workflow should be disabled or restricted before continuing with legacy runtime/data cleanup.

The safest next step is a narrow Codex/local PR:

```text
RESET-9F2 — Pause Legacy Daily Scan Workflow
```

## 8. RESET-9F2 Recommended Scope

RESET-9F2 should modify only:

```text
.github/workflows/daily-market-scan.yml
```

Recommended change:

- remove or comment out the scheduled trigger;
- keep `workflow_dispatch` only if manual fallback is still desired;
- add a clear note that this is paused during v2 cutover;
- prevent automatic commits to `main` during reset.

Preferred temporary shape:

```yaml
on:
  workflow_dispatch:
```

Optional safer variant:

- keep the workflow file but disable all automatic scheduling;
- leave manual dispatch available only for explicitly intentional legacy fallback runs.

## 9. Future V2 CI Direction

A future v2 CI workflow should be introduced only after v2 commands stabilize.

Future CI should focus on:

- package import checks;
- contract tests;
- fixture tests;
- unit tests;
- integration tests for v2 scaffolds;
- no generated output commits;
- no live provider calls;
- no Telegram behavior;
- no direct writes to generated data paths.

A v2 CI workflow should not run the old production pipeline.

## 10. Stop Conditions Before Runtime Cleanup

Do not proceed with legacy runtime archive/delete execution while the legacy daily workflow still auto-runs and pushes generated outputs to `main`.

Before RESET-9C or later cleanup execution proceeds, confirm:

- whether the daily schedule is disabled;
- whether direct generated-data commits to `main` are paused;
- whether legacy runtime fallback remains manual and explicit;
- whether generated data cleanup has a separate approved plan.

## 11. Validation

No local commands were run because RESET-9F was executed through the GitHub connector as a documentation-only task.

Repository inspection confirmed:

- `.github/workflows/daily-market-scan.yml` exists;
- it has a weekday scheduled trigger;
- it has `contents: write` permission;
- it runs `scripts/telegram/process_telegram_commands.py`;
- it runs `scripts/run_scan.py`;
- it commits `data/watchlist/*.csv`, `data/portfolio/*.csv`, `data/processed/market_regime.csv`, `data/processed/scanner_ranked.csv`, and `reports/daily/telegram_message.txt`;
- it pushes using commit message `Automated daily scan update`.

## 12. Scope Confirmation

RESET-9F changed documentation only.

No workflow files were modified.
No code was changed.
No tests were changed.
No data or CSV files were changed.
No reports were changed.
No generated files were created outside this documentation file.
No production pipeline was run.
No SEC diagnostics were run.
No provider, Telegram, network, or live data calls were made.

## 13. Recommended Next Action

Recommended next action:

```text
RESET-9F2 — Pause Legacy Daily Scan Workflow
```

Executor: Codex/local implementation.

After RESET-9F2 is merged, continue with:

```text
RESET-9C — Legacy Runtime Inventory and Retirement Decision
```
