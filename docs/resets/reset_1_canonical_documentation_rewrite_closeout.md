# RESET-1 — Canonical Documentation Rewrite Closeout

## 1. Purpose

RESET-1 converts the RESET-0 rebuild decision into a concise canonical v2 documentation baseline.

RESET-1 is documentation-only. It does not implement v2. It does not move, archive, delete, or regenerate files.

## 2. Created or Updated Canonical Documents

RESET-1 creates or updates the following canonical active documents:

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

## 3. Authority Statement

The RESET-1 documents are the v2 planning baseline. Legacy active documents, sprint documents, audits, archive documents, old code, old tests, old CSVs, generated reports, and workflows remain preserved but are not the v2 implementation authority unless explicitly carried forward by the new canonical documents.

## 4. Development Status

New feature work on the old active architecture remains paused.

Old Python files are reference-only for v2. v2 Python files must be newly written after contracts and structure are approved.

## 5. SEC/Fundamentals Status

Standalone SEC-7F remains paused. SEC and fundamentals knowledge should feed RESET-8 after v2 core pipeline contracts and source-data contracts exist.

## 6. Backlog Impact Assessment

Backlog impact assessment:
- New reset backlog direction captured in `docs/active/backlog.md`.

No runtime backlog implementation is authorized by RESET-1.

## 7. Validation

Documentation-safe validation requested:

```bash
git diff --check
git status
```

Result: Not run through the GitHub connector because RESET-1 was executed through GitHub API file creation and updates rather than a local working tree.

No tests were run because no code or tests were changed. No pipeline was run. No SEC diagnostics were run. No live SEC calls were made. No generated outputs were created.

## 8. Final RESET-1 Decision

RESET-1 status: COMPLETE FOR REVIEW

Recommended next action: RESET-2 — Repository Structure and Archive Plan

Development status: Pause new feature work on old active architecture

SEC-7F status: Pause standalone SEC-7F; absorb into v2 source-data roadmap

Old code status: Reference-only for v2

Old documentation status: Legacy source material unless explicitly carried forward

Old data/CSV status: Not v2 source-of-truth unless reapproved

Archive/delete status: Not authorized in RESET-1

Pipeline integration status: Blocked until v2 structure, contracts, fixtures, and skeleton exist
