# Batch A — Documentation Authority Detachment Closeout

## 1. Purpose

Batch A detaches superseded active documentation from the active v2 source-of-truth path after RESET-1 and RESET-2.

The purpose is to make the RESET-1 canonical documentation set the only active documentation baseline under `docs/active/`, while preserving older active documents as legacy reference material.

## 2. Scope

Batch A is documentation-only.

Allowed:

- move superseded `docs/active/` documents to `docs/legacy/active_superseded/`;
- add a legacy README;
- update README onboarding references;
- add this closeout document.

Forbidden:

- code changes;
- test changes;
- CSV/data changes;
- generated outputs;
- report changes;
- workflow changes;
- runtime behavior changes;
- pipeline runs;
- SEC diagnostics;
- live SEC/network calls.

## 3. Result

Superseded active documents were moved to:

```text
docs/legacy/active_superseded/
```

The canonical active documentation surface now remains the RESET-1 v2 baseline under:

```text
docs/active/
```

README onboarding was updated to point to the RESET-1 canonical active documents and to clarify that superseded active documentation is preserved as legacy reference only.

## 4. Authority Statement

The following are the active v2 planning documents:

- `docs/active/project_charter.md`
- `docs/active/product_vision.md`
- `docs/active/roles_and_responsibilities.md`
- `docs/active/pm_operating_model.md`
- `docs/active/functional_analysis.md`
- `docs/active/technical_architecture.md`
- `docs/active/data_architecture.md`
- `docs/active/source_data_strategy.md`
- `docs/active/pipeline_contract.md`
- `docs/active/decision_engine_contract.md`
- `docs/active/reporting_contract.md`
- `docs/active/testing_strategy.md`
- `docs/active/repository_structure.md`
- `docs/active/backlog.md`
- `docs/active/roadmap.md`

Superseded documents under `docs/legacy/active_superseded/` are preserved for traceability only. They do not override the active v2 baseline unless explicitly carried forward by a current active document.

## 5. Backlog Impact Assessment

Backlog impact assessment:
- No new backlog items identified.

Batch A satisfies the documentation authority detachment path already identified by RESET-2.

## 6. Validation

Documentation-safe validation requested:

```bash
git diff --check
git status
```

Result: Not run through the GitHub connector because Batch A was executed through GitHub API file creation, tree commits, and file updates rather than a local working tree.

No tests were run because no code or tests were changed. No pipeline was run. No SEC diagnostics were run. No live SEC calls were made. No generated outputs were created.

## 7. Final Status

Batch A status: COMPLETE FOR REVIEW

Recommended next action: review and merge Batch A, then sync local `main` before deciding between RESET-3 codebase bootstrap or an additional root-governance update for `AGENTS.md`.
