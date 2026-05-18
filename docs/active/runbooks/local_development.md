# Local Development Runbook

Status: ACTIVE

This runbook defines the local-first workflow for safe repository work.

## Before Editing

1. Start from a clean branch.
2. Sync local state with GitHub before editing.
3. Run `git status`.
4. Read the active docs relevant to the change:
   - architecture: `docs/active/architecture_current_state.md`
   - governance: `docs/active/governance_v2.md`
   - contracts: `docs/active/contracts/pipeline_contracts.md`
   - repository navigation: `docs/active/repository_structure.md`
5. Confirm the requested scope.

## Safety Boundaries

Do not make runtime, code, test, data, generated report, pipeline output, Decision Engine, reporting semantic, or data contract changes unless the task explicitly requests them.

For documentation-only work, do not touch:

- `scripts/`
- `tests/`
- `data/`
- `reports/`
- `.github/`
- runtime configuration

## During Editing

Keep edits small and source-of-truth oriented.

Prefer updating active documentation over adding parallel doctrine files. Use archive documents only as historical evidence unless `docs/active/` explicitly references them.

## Before Review

Run:

```bash
git status
git diff --stat
git diff --check
```

For staged changes, also run:

```bash
git diff --cached --check
```

Review the diff before asking for commit approval.

## Commit Rule

Do not commit until human review approves the diff.

Do not push unless explicitly requested.
